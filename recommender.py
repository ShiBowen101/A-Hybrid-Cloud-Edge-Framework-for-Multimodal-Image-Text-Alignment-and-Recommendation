import os
import numpy as np
import logging
import time
import json
import requests
from typing import List, Dict, Any, Optional
from config import PROCESSING_RESULTS_DIR, TOP_K_INITIAL, TOP_K_FINAL, QWEN_API_KEY, QWEN_RERANK_MODEL, QWEN_GENERATION_MODEL, IMAGE_LIBRARY_DIR, EMBEDDING_DIMENSION
from embedding import EmbeddingProcessor
from image_describer import QwenImageDescriber
from utils import setup_logging
import pickle
import torch

logger = logging.getLogger(__name__)

class ImageRecommendationEngine:
    """图片推荐引擎，根据朋友圈文案推荐合适的图片"""

    def __init__(self):
        """初始化推荐引擎"""
        setup_logging()
        self.embedder = EmbeddingProcessor()
        self.image_paths = []
        self.index = None
        self.index_dimension = 0
        self.describer = QwenImageDescriber()
        self.api_key = QWEN_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self._load_index()
        self.recommendation_cache = {}
        self.rerank_cache = {}
        self.CACHE_TTL = 300

    def _load_index(self):
        """加载预处理生成的索引"""
        try:
            embeddings_path = os.path.join(PROCESSING_RESULTS_DIR, "train_embeddings.npy")
            paths_path = os.path.join(PROCESSING_RESULTS_DIR, "image_paths.pkl")
            index_path = os.path.join(PROCESSING_RESULTS_DIR, "sklearn_index.pkl")

            for p in [embeddings_path, paths_path, index_path]:
                if not os.path.exists(p):
                    logger.error(f"缺少文件: {p}")
                    logger.error("请先运行 preprocessor.py 处理图库")
                    raise FileNotFoundError(f"缺少文件: {p}")

            with open(paths_path, "rb") as f:
                self.image_paths = pickle.load(f)

            image_embeddings = np.load(embeddings_path)
            if len(image_embeddings) > 0:
                self.index_dimension = image_embeddings.shape[1]
                if self.index_dimension != EMBEDDING_DIMENSION:
                    logger.error(f"嵌入文件维度不匹配: {self.index_dimension} != {EMBEDDING_DIMENSION}")
                    raise ValueError(f"嵌入文件维度不匹配: {self.index_dimension}")
                logger.info(f"检测到嵌入向量维度: {self.index_dimension}")

            with open(index_path, "rb") as f:
                self.index = pickle.load(f)

            index_dimension = self.index.n_features_in_
            logger.info(f"索引维度: {index_dimension}")

            if self.index_dimension != index_dimension:
                logger.error(f"维度不匹配: 嵌入向量维度={self.index_dimension}, 索引维度={index_dimension}")
                logger.error("请删除旧索引文件并重新运行 preprocessor.py")
                self.image_paths = []
                self.index = None
                self.index_dimension = 0
                raise ValueError("索引维度不匹配")

            logger.info(f"成功加载索引，包含 {len(self.image_paths)} 张图片")
            self._warmup_index()

        except Exception as e:
            logger.exception(f"加载索引失败: {str(e)}")
            self.image_paths = []
            self.index = None
            self.index_dimension = 0
            raise RuntimeError("推荐引擎初始化失败，请检查预处理结果") from e

    def _warmup_index(self):
        """预热索引，提高首次查询性能"""
        logger.info("预热索引...")
        warmup_start = time.time()

        warmup_queries = ["蓝天白云", "美食", "旅行", "工作", "周末"]
        for query in warmup_queries:
            try:
                embedding = self.embedder.text_to_embedding(query)
                embedding = embedding.reshape(1, -1).astype('float32')
                if embedding.shape[1] != self.index.n_features_in_:
                    logger.error(f"维度不匹配: 查询向量维度={embedding.shape[1]}, 索引维度={self.index.n_features_in_}")
                    continue
                norms = np.linalg.norm(embedding, axis=1, keepdims=True)
                norms[norms == 0] = 1
                embedding = embedding / norms
                self.index.kneighbors(embedding, n_neighbors=5, return_distance=True)
            except Exception as e:
                logger.debug(f"预热查询 '{query}' 时出错: {str(e)}")

        warmup_time = time.time() - warmup_start
        logger.info(f"索引预热完成，耗时 {warmup_time:.2f}秒")

    def _generate_rerank_prompt(self, post_text: str, image_description: str) -> str:
        return f"""请严格按以下要求评估图片与朋友圈文案的匹配度：
            
【朋友圈文案】
{post_text}

【图片描述】
{image_description}

【评估要求】
1. 从三个维度评分(1-10分)：
   - 内容相关性：图片内容是否匹配文案描述
   - 情感一致性：图片氛围是否符合文案情感
   - 社交适宜性：是否适合在朋友圈发布
2. 给出总体匹配度(1-10分)
3. 用一句话说明推荐理由

请按此JSON格式输出，不要包含其他内容：
{{"content_relevance": 8, "emotional_consistency": 9, "social_appropriateness": 7, "overall_score": 8, "reason": "推荐理由"}}
"""

    def _parse_rerank_response(self, response: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    return json.loads(json_str)
            except:
                pass
        logger.warning(f"无法解析精排响应: {response[:100]}...")
        return None

    def _call_qwen_rerank_api(self, post_text: str, image_description: str) -> Optional[Dict[str, Any]]:
        cache_key = f"{post_text.strip()}|{image_description.strip()}"
        current_time = time.time()
        if cache_key in self.rerank_cache:
            result, timestamp = self.rerank_cache[cache_key]
            if current_time - timestamp < 86400:
                return result

        prompt = self._generate_rerank_prompt(post_text, image_description)
        api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Client": "python-sdk"
        }
        payload = {
            "model": QWEN_RERANK_MODEL,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            "parameters": {
                "result_format": "message"
            }
        }
        try:
            start_time = time.time()
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            if response.status_code != 200:
                logger.error(f"Qwen精排API调用失败 [状态码: {response.status_code}]: {response.text}")
                return None
            result = response.json()
            if "output" not in result or "choices" not in result["output"] or not result["output"]["choices"]:
                logger.error(f"Qwen API响应格式错误: {result}")
                return None
            content = result["output"]["choices"][0]["message"]["content"].strip()
            score_data = self._parse_rerank_response(content)
            if score_data and all(k in score_data for k in ["content_relevance", "emotional_consistency", "social_appropriateness", "overall_score", "reason"]):
                elapsed = time.time() - start_time
                logger.debug(f"精排API调用成功 (耗时: {elapsed:.2f}s)")
                self.rerank_cache[cache_key] = (score_data, current_time)
                return score_data
            logger.warning(f"评分数据格式不正确: {content[:100]}...")
            return None
        except Exception as e:
            logger.exception(f"调用Qwen精排API时出错: {str(e)}")
            return None

    def recommend_images(self, post_text: str, top_k: int = None, rerank_k: int = None) -> List[Dict[str, Any]]:
        if not self.image_paths or self.index is None:
            logger.error("推荐引擎未正确初始化，请检查预处理结果")
            raise RuntimeError("推荐引擎未正确初始化，请检查预处理结果")

        normalized_text = post_text.strip().lower()
        cache_key = f"{normalized_text}|{top_k}|{rerank_k}"
        current_time = time.time()
        if cache_key in self.recommendation_cache:
            result, timestamp = self.recommendation_cache[cache_key]
            if current_time - timestamp < self.CACHE_TTL:
                logger.info(f"使用缓存结果 for: '{post_text[:30]}{'...' if len(post_text) > 30 else ''}'")
                return result

        if not post_text.strip():
            raise ValueError("朋友圈文案不能为空")

        if top_k is None:
            top_k = TOP_K_INITIAL
        if rerank_k is None:
            rerank_k = TOP_K_FINAL

        logger.info(f"收到推荐请求: '{post_text[:50]}{'...' if len(post_text) > 50 else ''}'")
        logger.debug(f"参数: top_k={top_k}, rerank_k={rerank_k}")

        start_time = time.time()
        logger.debug("阶段1: 快速初筛 - 文本嵌入与向量检索")
        post_embedding = self.embedder.text_to_embedding(post_text)
        if len(post_embedding.shape) == 1:
            post_embedding = post_embedding.reshape(1, -1).astype('float32')
        if post_embedding.shape[1] != self.index.n_features_in_:
            logger.error(f"维度不匹配: 查询向量维度={post_embedding.shape[1]}, 索引维度={self.index.n_features_in_}")
            logger.error("请删除旧索引文件并重新运行 preprocessor.py")
            raise ValueError(f"维度不匹配: 查询向量维度={post_embedding.shape[1]}, 索引维度={self.index.n_features_in_}")
        norms = np.linalg.norm(post_embedding, axis=1, keepdims=True)
        norms[norms == 0] = 1
        post_embedding = post_embedding / norms
        distances, indices = self.index.kneighbors(post_embedding, n_neighbors=min(top_k, len(self.image_paths)), return_distance=True)
        candidate_indices = indices[0]
        candidate_distances = distances[0]
        candidates = []
        for i, img_idx in enumerate(candidate_indices):
            img_path = self.image_paths[img_idx]
            filename = os.path.splitext(os.path.basename(img_path))[0]
            similarity_score = 10 - candidate_distances[i] * 10
            candidates.append({
                "index": img_idx,
                "path": img_path,
                "filename": filename,
                "distance": candidate_distances[i],
                "similarity_score": similarity_score
            })
        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        logger.debug(f"初筛完成，获取 {len(candidates)} 个候选（已按相似度排序）")
        logger.debug("阶段2: Qwen API精排 - 多维度评分")
        rerank_results = []
        api_available = bool(self.api_key and self.api_key != "your-qwen-api-key-here")
        batch_size = 5
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            for candidate in batch:
                try:
                    img_path = candidate["path"]
                    filename = candidate["filename"]
                    image_description = self.describer.descriptions.get(filename, "默认描述")
                    if api_available:
                        score_data = self._call_qwen_rerank_api(post_text, image_description)
                        if score_data:
                            details = {
                                "content_relevance": score_data.get("content_relevance", 0),
                                "emotional_consistency": score_data.get("emotional_consistency", 0),
                                "social_appropriateness": score_data.get("social_appropriateness", 0)
                            }
                            rerank_results.append({
                                "image_path": img_path,
                                "filename": filename,
                                "score": float(score_data["overall_score"]),
                                "reason": score_data.get("reason", "图片与文案匹配度良好"),
                                "details": details,
                                "image_description": image_description
                            })
                        else:
                            rerank_results.append({
                                "image_path": img_path,
                                "filename": filename,
                                "score": candidate["similarity_score"],
                                "reason": "基于内容相似度的推荐",
                                "details": {
                                    "content_relevance": 0,
                                    "emotional_consistency": 0,
                                    "social_appropriateness": 0
                                },
                                "image_description": image_description
                            })
                    else:
                        rerank_results.append({
                            "image_path": img_path,
                            "filename": filename,
                            "score": candidate["similarity_score"],
                            "reason": "基于内容相似度的推荐",
                            "details": {
                                "content_relevance": 0,
                                "emotional_consistency": 0,
                                "social_appropriateness": 0
                            },
                            "image_description": image_description
                        })
                except Exception as e:
                    logger.error(f"精排图片 {img_path} 时出错: {str(e)}")
                time.sleep(0.1)
        logger.debug("阶段3: 结果排序与返回")
        rerank_results.sort(key=lambda x: x.get("score", x.get("similarity_score", 0)), reverse=True)
        final_results = rerank_results[:rerank_k]
        process_time = time.time() - start_time
        logger.info(f"推荐完成，共处理 {len(candidates)} 个候选，返回 {len(final_results)} 个结果，耗时 {process_time:.2f}秒")
        for i, result in enumerate(final_results, 1):
            logger.info(f"推荐 #{i}: {result['filename']} | 分数: {result['score']:.1f} | 理由: {result['reason']}")
        self.recommendation_cache[cache_key] = (final_results, current_time)
        return final_results

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态信息 - 确保返回一致的结构"""
        return {
            "status": "running" if (self.image_paths and self.index) else "error",
            "image_count": len(self.image_paths) if self.image_paths else 0,
            "description_count": len(self.describer.descriptions) if hasattr(self.describer, 'descriptions') else 0,
            "embedding_model": "3e-base",
            "embedding_dimension": EMBEDDING_DIMENSION if 'EMBEDDING_DIMENSION' in globals() else self.index_dimension,
            "index_dimension": self.index.n_features_in_ if self.index else 0,
            "description_model": QWEN_GENERATION_MODEL,
            "rerank_model": QWEN_RERANK_MODEL,
            "top_k_initial": TOP_K_INITIAL,
            "top_k_final": TOP_K_FINAL,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "image_library_path": IMAGE_LIBRARY_DIR
        }
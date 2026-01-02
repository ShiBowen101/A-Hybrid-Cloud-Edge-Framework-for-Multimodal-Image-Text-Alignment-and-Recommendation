"""
webapp.py - Web应用主程序
"""

import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
import time  # 添加缺失的time导入

# 必须在导入其他库前设置环境变量
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免多进程问题
os.environ["OMP_NUM_THREADS"] = "4"  # 限制OpenMP线程数
os.environ["MKL_NUM_THREADS"] = "4"  # 限制MKL线程数

# 现在可以安全导入其他库
from config import PROJECT_ROOT, IMAGE_LIBRARY_DIR
from recommender import ImageRecommendationEngine
from image_describer import QwenImageDescriber
from utils import setup_logging

# 设置日志
logger = setup_logging()

# 创建Flask应用
app = Flask(__name__,
            template_folder=os.path.join(PROJECT_ROOT, 'templates'),
            static_folder=os.path.join(PROJECT_ROOT, 'static'))

# 全局变量
recommender = None
describer = None

# ===== 直接初始化系统，而不是使用 @app.before_first_request =====
try:
    # 初始化推荐引擎
    recommender = ImageRecommendationEngine()
    logger.info("推荐引擎初始化成功")

    # 初始化图片描述生成器
    describer = QwenImageDescriber()
    logger.info("图片描述生成器初始化成功")

    logger.info("==================================================")
    logger.info("启动朋友圈图片推荐Web应用")
    logger.info(f"访问地址: http://localhost:{os.getenv('PORT', 5000)}")
    logger.info("==================================================")
except Exception as e:
    logger.exception("系统初始化失败")
    # 如果初始化失败，可以考虑退出应用
    import sys
    sys.exit(1)

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/system-status')
def system_status():
    """获取系统状态"""
    if recommender is None:
        return jsonify({
            "status": "error",
            "message": "推荐引擎未初始化"
        }), 500

    status = recommender.get_system_status()
    return jsonify(status)

@app.route('/generate-descriptions', methods=['POST'])
def generate_descriptions():
    """生成/更新图片描述"""
    if describer is None:
        return jsonify({
            "status": "error",
            "message": "图片描述生成器未初始化"
        }), 500

    force_regenerate = request.form.get('force_regenerate', 'false') == 'true'

    try:
        # 生成描述
        result = describer.generate_descriptions(force_regenerate=force_regenerate)

        return jsonify({
            "status": "success",
            "total_images": result['total_images'],
            "new_descriptions": result['new_descriptions'],
            "skipped": result['skipped'],
            "failed": result['failed']
        })
    except Exception as e:
        logger.exception("生成描述失败")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    """推荐图片"""
    if recommender is None:
        return jsonify({
            "error": "推荐引擎未初始化"
        }), 500

    post_text = request.form.get('post_text', '').strip()
    top_k = int(request.form.get('top_k', 3))

    if not post_text:
        return jsonify({
            "error": "朋友圈文案不能为空"
        }), 400

    try:
        results = recommender.recommend_images(post_text, top_k=top_k)

        # 转换结果为前端可用格式
        formatted_results = []
        for i, result in enumerate(results):
            # 提取文件名
            filename = os.path.basename(result['image_path'])

            formatted_results.append({
                "id": i + 1,
                "filename": filename,
                "score": round(result['score'], 1),
                "reason": result['reason'],
                "content_relevance": result['details'].get('content_relevance', 0),
                "emotional_consistency": result['details'].get('emotional_consistency', 0),
                "social_appropriateness": result['details'].get('social_appropriateness', 0),
                "image_description": result['image_description']
            })

        return jsonify({
            "results": formatted_results
        })
    except Exception as e:
        logger.exception("推荐失败")
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/images/<path:filename>')
def serve_image(filename):
    """提供图片服务"""
    return send_from_directory(IMAGE_LIBRARY_DIR, filename)

if __name__ == '__main__':
    # 确保目录存在
    os.makedirs(os.path.join(PROJECT_ROOT, 'static', 'css'), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, 'static', 'js'), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, 'templates'), exist_ok=True)

    # 记录开始时间（修复点：添加这一行）
    start_time = time.time()

    # 运行应用
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

    # 记录处理时间
    process_time = time.time() - start_time
    logger.info(f"应用运行完成，总耗时: {process_time:.2f}秒")

"""
recommender.py - 核心推荐引擎
"""

import os
import numpy as np
import logging
import time
import json
from typing import List, Dict, Any, Optional
from config import PROCESSING_RESULTS_DIR, TOP_K_INITIAL, TOP_K_FINAL, QWEN_API_KEY, QWEN_RERANK_MODEL, \
    QWEN_GENERATION_MODEL, IMAGE_LIBRARY_DIR
from embedding import EmbeddingProcessor
from image_describer import QwenImageDescriber
from utils import setup_logging
import pickle
import requests

logger = logging.getLogger(__name__)


class ImageRecommendationEngine:
    """图片推荐引擎，根据朋友圈文案推荐合适的图片"""

    def __init__(self):
        """初始化推荐引擎"""
        setup_logging()
        self.embedder = EmbeddingProcessor()
        self.image_paths = []
        self.index = None
        self.describer = QwenImageDescriber()
        self.api_key = QWEN_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self._load_index()  # 加载索引
        # 添加结果缓存
        self.recommendation_cache = {}
        self.rerank_cache = {}
        self.CACHE_TTL = 300  # 5分钟缓存有效期

    def _load_index(self):
        """加载预处理生成的索引"""
        try:
            # 检查必要文件是否存在
            embeddings_path = os.path.join(PROCESSING_RESULTS_DIR, "image_embeddings.npy")
            paths_path = os.path.join(PROCESSING_RESULTS_DIR, "image_paths.pkl")
            index_path = os.path.join(PROCESSING_RESULTS_DIR, "sklearn_index.pkl")

            if not all(os.path.exists(p) for p in [embeddings_path, paths_path, index_path]):
                logger.error("缺少必要的预处理文件！")
                logger.error("请先运行 preprocessor.py 处理图库")
                return

            # 加载图片路径
            with open(paths_path, "rb") as f:
                import pickle
                self.image_paths = pickle.load(f)

            # 加载Scikit-learn索引
            with open(index_path, "rb") as f:
                import pickle
                self.index = pickle.load(f)

            logger.info(f"成功加载索引，包含 {len(self.image_paths)} 张图片")

            # 预热索引（提高首次查询性能）
            self._warmup_index()

        except Exception as e:
            logger.exception(f"加载索引失败: {str(e)}")
            raise RuntimeError("推荐引擎初始化失败，请检查预处理结果") from e

    def _warmup_index(self):
        """预热索引，提高首次查询性能"""
        logger.info("预热索引...")
        warmup_start = time.time()

        # 使用一些常见查询预热
        warmup_queries = [
            "蓝天白云",
            "美食",
            "旅行",
            "工作",
            "周末"
        ]

        for query in warmup_queries:
            try:
                # 生成嵌入向量
                embedding = self.embedder.text_to_embedding(query)
                embedding = embedding.reshape(1, -1).astype('float32')

                # 归一化
                norms = np.linalg.norm(embedding, axis=1, keepdims=True)
                norms[norms == 0] = 1
                embedding = embedding / norms

                # 执行搜索
                self.index.kneighbors(embedding, n_neighbors=5, return_distance=True)
            except Exception as e:
                logger.debug(f"预热查询 '{query}' 时出错: {str(e)}")

        warmup_time = time.time() - warmup_start
        logger.info(f"索引预热完成，耗时 {warmup_time:.2f}秒")

    def _generate_rerank_prompt(self, post_text: str, image_description: str) -> str:
        """
        生成用于Qwen API精排的提示词

        参数:
            post_text: 用户输入的朋友圈文案
            image_description: 图片的语义描述

        返回:
            格式化的提示词
        """
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
        """
        解析Qwen API的精排响应

        参数:
            response: 模型返回的原始响应

        返回:
            解析后的评分字典，或None（如果解析失败）
        """
        try:
            # 尝试直接解析JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试从响应中提取JSON部分
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
        """
        调用Qwen API进行精排评分（使用正确格式）

        参数:
            post_text: 朋友圈文案
            image_description: 图片描述

        返回:
            评分结果字典，或None
        """
        # 创建缓存键
        cache_key = f"{post_text.strip()}|{image_description.strip()}"

        # 检查缓存
        current_time = time.time()
        if cache_key in self.rerank_cache:
            result, timestamp = self.rerank_cache[cache_key]
            if current_time - timestamp < 86400:  # 24小时缓存
                return result

        prompt = self._generate_rerank_prompt(post_text, image_description)

        # 正确的API端点（注意包含 /api/v1/ 路径）
        api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

        # 构建正确的请求头
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Client": "python-sdk"
        }

        # 构建正确的请求体（阿里云百炼平台要求的格式）
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
            # 使用正确的API端点
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=30
            )

            # 检查响应状态
            if response.status_code != 200:
                logger.error(f"Qwen精排API调用失败 [状态码: {response.status_code}]: {response.text}")
                return None

            # 解析响应
            result = response.json()

            # 检查API响应是否包含错误
            if "output" not in result or "choices" not in result["output"] or not result["output"]["choices"]:
                logger.error(f"Qwen API响应格式错误: {result}")
                return None

            # 提取内容
            content = result["output"]["choices"][0]["message"]["content"].strip()

            # 解析评分
            score_data = self._parse_rerank_response(content)
            if score_data:
                # 验证评分数据
                if all(k in score_data for k in ["content_relevance", "emotional_consistency",
                                                 "social_appropriateness", "overall_score", "reason"]):
                    # 记录API调用信息
                    elapsed = time.time() - start_time
                    logger.debug(f"精排API调用成功 (耗时: {elapsed:.2f}s)")

                    # 保存到缓存
                    self.rerank_cache[cache_key] = (score_data, current_time)

                    return score_data

            logger.warning(f"评分数据格式不正确: {content[:100]}...")
            return None

        except Exception as e:
            logger.exception(f"调用Qwen精排API时出错: {str(e)}")
            return None

    def recommend_images(self, post_text: str, top_k: int = None, rerank_k: int = None) -> List[Dict[str, Any]]:
        """
        为朋友圈文案推荐图片
        """
        # 创建缓存键
        normalized_text = post_text.strip().lower()
        cache_key = f"{normalized_text}|{top_k}|{rerank_k}"

        # 检查缓存
        current_time = time.time()
        if cache_key in self.recommendation_cache:
            result, timestamp = self.recommendation_cache[cache_key]
            if current_time - timestamp < self.CACHE_TTL:
                logger.info(f"使用缓存结果 for: '{post_text[:30]}{'...' if len(post_text) > 30 else ''}'")
                return result

        if not self.image_paths or self.index is None:
            raise RuntimeError("推荐引擎未正确初始化，请检查预处理结果")

        if not post_text.strip():
            raise ValueError("朋友圈文案不能为空")

        if top_k is None:
            top_k = TOP_K_INITIAL
        if rerank_k is None:
            rerank_k = TOP_K_FINAL

        logger.info(f"收到推荐请求: '{post_text[:50]}{'...' if len(post_text) > 50 else ''}'")
        logger.debug(f"参数: top_k={top_k}, rerank_k={rerank_k}")

        start_time = time.time()

        # === 阶段1: 快速初筛 ===
        logger.debug("阶段1: 快速初筛 - 文本嵌入与向量检索")

        # 生成文案的嵌入向量（自动使用GPU如果可用）
        post_embedding = self.embedder.text_to_embedding(post_text)

        # 确保数据维度正确
        if len(post_embedding.shape) == 1:
            post_embedding = post_embedding.reshape(1, -1).astype('float32')

        # 归一化（用于余弦相似度）
        norms = np.linalg.norm(post_embedding, axis=1, keepdims=True)
        norms[norms == 0] = 1
        post_embedding = post_embedding / norms

        # 使用Scikit-learn进行搜索
        distances, indices = self.index.kneighbors(
            post_embedding,
            n_neighbors=min(top_k, len(self.image_paths)),
            return_distance=True
        )

        # 获取候选图片并按相似度排序
        candidate_indices = indices[0]
        candidate_distances = distances[0]

        candidates = []
        for i, img_idx in enumerate(candidate_indices):
            img_path = self.image_paths[img_idx]
            # 从路径中获取文件名（不含扩展名）
            filename = os.path.splitext(os.path.basename(img_path))[0]
            similarity_score = 10 - candidate_distances[i] * 10
            candidates.append({
                "index": img_idx,
                "path": img_path,
                "filename": filename,  # 添加文件名
                "distance": candidate_distances[i],
                "similarity_score": similarity_score
            })

        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        logger.debug(f"初筛完成，获取 {len(candidates)} 个候选（已按相似度排序）")

        # === 阶段2: Qwen API精排 ===
        logger.debug("阶段2: Qwen API精排 - 多维度评分")

        rerank_results = []
        api_available = bool(self.api_key and self.api_key != "your-qwen-api-key-here")

        # 批量处理（减少API调用次数）
        batch_size = 5
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]

            for candidate in batch:
                try:
                    img_path = candidate["path"]
                    # 获取文件名（不含扩展名）
                    filename = candidate["filename"]

                    # 从describer中获取描述（使用文件名）
                    image_description = self.describer.descriptions.get(filename)
                    if not image_description:
                        logger.warning(f"跳过图片 {img_path} (无描述)")
                        continue

                    if api_available:
                        score_data = self._call_qwen_rerank_api(post_text, image_description)
                        if score_data:
                            rerank_results.append({
                                "image_path": img_path,
                                "filename": filename,
                                "score": float(score_data["overall_score"]),
                                "reason": score_data.get("reason", "图片与文案匹配度良好"),
                                "details": {
                                    "content_relevance": score_data.get("content_relevance", 0),
                                    "emotional_consistency": score_data.get("emotional_consistency", 0),
                                    "social_appropriateness": score_data.get("social_appropriateness", 0)
                                },
                                "image_description": image_description
                            })
                        else:
                            logger.warning(f"评分数据无效，使用初筛分数: {img_path}")
                            rerank_results.append({
                                "image_path": img_path,
                                "filename": filename,
                                "score": candidate["similarity_score"],
                                "reason": "基于内容相似度的推荐",
                                "details": {},
                                "image_description": image_description
                            })
                    else:
                        rerank_results.append({
                            "image_path": img_path,
                            "filename": filename,
                            "score": candidate["similarity_score"],
                            "reason": "基于内容相似度的推荐",
                            "details": {},
                            "image_description": image_description
                        })

                except Exception as e:
                    logger.error(f"精排图片 {img_path} 时出错: {str(e)}")

                time.sleep(0.1)  # 减少等待时间

        # === 阶段3: 结果排序与返回 ===
        logger.debug("阶段3: 结果排序与返回")

        # 按精排分数降序排序
        rerank_results.sort(key=lambda x: x.get("score", x.get("similarity_score", 0)), reverse=True)

        # 只返回前rerank_k个结果
        final_results = rerank_results[:rerank_k]

        # 记录处理时间
        process_time = time.time() - start_time
        logger.info(
            f"推荐完成，共处理 {len(candidates)} 个候选，返回 {len(final_results)} 个结果，耗时 {process_time:.2f}秒")

        # 记录详细结果
        for i, result in enumerate(final_results, 1):
            logger.info(f"推荐 #{i}: {result['filename']} | 分数: {result['score']:.1f} | 理由: {result['reason']}")

        # 保存到缓存
        self.recommendation_cache[cache_key] = (final_results, current_time)

        return final_results

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态信息"""
        return {
            "status": "running",
            "image_count": len(self.image_paths),
            "description_count": len(self.describer.descriptions),
            "embedding_model": "GPT2",
            "description_model": QWEN_GENERATION_MODEL,
            "rerank_model": QWEN_RERANK_MODEL,
            "top_k_initial": TOP_K_INITIAL,
            "top_k_final": TOP_K_FINAL,
            "image_library_path": IMAGE_LIBRARY_DIR
        }
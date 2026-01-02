import os
import logging
import pickle
import numpy as np
from datetime import datetime, timedelta
from config import PROCESSING_RESULTS_DIR, CACHE_TTL_HOURS
from utils import setup_logging
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# 嵌入维度定义（与 config.py 保持一致）
EMBEDDING_DIMENSION = 768  # m3e-base 默认维度

class SentenceBERTEmbeddingModel:

    def __init__(self):
        model_path = os.environ.get('SENTENCE_TRANSFORMERS_HOME', 'D:/Anaconda/envs/pytorch/Lib/site-packages/sentence-transformers')
        model_name = 'm3e-base'
        full_path = os.path.join(model_path, model_name)
        logger.info(f"尝试加载本地Sentence-BERT模型: {full_path}")
        try:
            self.model = SentenceTransformer(
                full_path,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                trust_remote_code=False  # 提高安全性
            )
            logger.info(f"Sentence-BERT模型加载成功 (运行在{'cuda' if torch.cuda.is_available() else 'cpu'})")
        except Exception as e:
            logger.error(f"加载Sentence-BERT模型失败: {str(e)}")
            raise
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_embedding_dimension(self) -> int:

        return self.model.get_sentence_embedding_dimension()  # 返回768

    def encode(self, text: str, convert_to_tensor: bool = False) -> np.ndarray:

        try:
            embedding = self.model.encode(text, convert_to_tensor=convert_to_tensor, device=self.device)
            if convert_to_tensor:
                return embedding
            return embedding.numpy() if hasattr(embedding, 'numpy') else embedding
        except Exception as e:
            logger.error(f"生成文本嵌入失败: {str(e)}")
            raise

class DummyModel:

    def encode(self, text: str, convert_to_tensor: bool = False) -> np.ndarray:
        logger.warning(f"使用备用模型为文本生成零向量嵌入: {text[:50]}...")
        return np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)

    def get_embedding_dimension(self) -> int:

        return EMBEDDING_DIMENSION

class EmbeddingProcessor:

    def __init__(self):
        setup_logging()
        self.text_model = None
        self.cache_file = os.path.join(PROCESSING_RESULTS_DIR, 'cache', 'embedding_cache.pkl')
        self.cache = {}
        self._load_text_model()
        self._load_cache()

    def _load_text_model(self):

        logger.info("尝试加载Sentence-BERT文本嵌入模型 (100%离线)")
        try:
            self.text_model = SentenceBERTEmbeddingModel()
            model_dimension = self.text_model.get_embedding_dimension()
            if model_dimension != EMBEDDING_DIMENSION:
                logger.error(f"模型维度({model_dimension})与配置维度({EMBEDDING_DIMENSION})不一致")
                raise ValueError(f"模型维度不匹配: {model_dimension} != {EMBEDDING_DIMENSION}")
            logger.info(f"嵌入维度验证通过: {EMBEDDING_DIMENSION}")
            logger.info("Sentence-BERT文本嵌入模型加载成功")
        except Exception as e:
            logger.exception("加载Sentence-BERT模型失败，将使用备用模型")
            self.text_model = DummyModel()
            logger.warning(f"使用备用模型，嵌入维度: {EMBEDDING_DIMENSION}")

    def _load_cache(self):

        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    cache_time = data.get('timestamp')
                    if cache_time and datetime.now() - cache_time < timedelta(hours=CACHE_TTL_HOURS):
                        self.cache = {k: v for k, v in data.get('cache', {}).items() if v.shape[0] == EMBEDDING_DIMENSION}
                        logger.info(f"成功加载嵌入缓存，包含 {len(self.cache)} 个有效{EMBEDDING_DIMENSION}维条目")
                        return
            self.cache = {}
            logger.info("未找到有效缓存，将创建新缓存")
        except Exception as e:
            logger.warning(f"加载缓存失败，将创建新缓存: {str(e)}")
            self.cache = {}

    def _save_cache(self):

        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump({
                    'timestamp': datetime.now(),
                    'cache': self.cache
                }, f)
            logger.info(f"嵌入缓存已保存至: {self.cache_file}")
        except Exception as e:
            logger.error(f"保存嵌入缓存失败: {str(e)}")

    def text_to_embedding(self, text: str) -> np.ndarray:

        if not text or not isinstance(text, str):
            logger.warning(f"无效输入文本: {text}")
            return np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)

        cache_key = ' '.join(text.strip().lower().split())
        if cache_key in self.cache:
            logger.debug(f"缓存命中: {cache_key[:50]}...")
            return self.cache[cache_key]

        try:
            embedding = self.text_model.encode(text)
            if embedding.shape[0] != EMBEDDING_DIMENSION:
                logger.error(f"嵌入维度不匹配: {embedding.shape[0]} != {EMBEDDING_DIMENSION}")
                raise ValueError(f"嵌入维度不匹配: {embedding.shape[0]}")
            self.cache[cache_key] = embedding
            self._save_cache()
            return embedding
        except Exception as e:
            logger.error(f"生成嵌入失败: {str(e)}")
            return np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)

    def image_description_to_embedding(self, description: str) -> np.ndarray:

        return self.text_to_embedding(description)
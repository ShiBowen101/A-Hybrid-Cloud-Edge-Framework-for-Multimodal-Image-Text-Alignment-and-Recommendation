

import os
import sys

# 获取项目根目录（相对于当前文件）
if getattr(sys, 'frozen', False):
    # 如果是打包后的可执行文件
    PROJECT_ROOT = os.path.dirname(sys.executable)
else:
    # 如果是普通Python脚本
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 图库配置 - 使用相对路径

IMAGE_LIBRARY_DIR = "Train"
# 处理结果目录 - 使用相对路径

PROCESSING_RESULTS_DIR = "processing_results"
# 图片描述JSON文件 - 使用相对路径
IMAGE_DESCRIPTIONS_JSON = "processing_results/train_descriptions.json"

# 缓存配置
CACHE_DIR = os.path.join(PROCESSING_RESULTS_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_TTL_HOURS = 24  # 缓存有效期(小时)

# 文本嵌入模型配置
TEXT_EMBEDDING_MODEL = "D:/Anaconda/envs/pytorch/Lib/site-packages/transformers/models/gpt2"
EMBEDDING_DIMENSION = 768

# 推荐参数
TOP_K_INITIAL = 20  # 初筛返回的候选图片数量
TOP_K_FINAL = 5     # 精排后最终返回数量

# Qwen API配置
QWEN_API_KEY = "sk-a36d47d58b6f4e8387e91cb61cab0e26"
QWEN_GENERATION_MODEL = "qwen-vl-plus"  # 图像描述生成模型
QWEN_RERANK_MODEL = "qwen-max"          # 用于精排的语言模型

# 日志配置
LOG_LEVEL = "INFO"

# 添加环境变量，确保使用本地模型
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免多进程问题
# 服务器配置
API_HOST = "127.0.0.1"
API_PORT = 5000
API_RELOAD = True

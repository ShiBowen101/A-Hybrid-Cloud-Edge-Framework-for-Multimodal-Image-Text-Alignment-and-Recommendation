
import os
import logging
from config import LOG_LEVEL
from typing import List

def setup_logging():

    logging.basicConfig(
        level=LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 配置第三方库的日志级别
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("sklearn").setLevel(logging.WARNING)

    return logging.getLogger(__name__)

def list_image_files(directory: str) -> List[str]:

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    image_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(root, file))

    return image_files

def get_file_size_mb(file_path: str) -> float:
    """获取文件大小(以MB为单位)"""
    return os.path.getsize(file_path) / (1024 * 1024)

def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.1f}s"
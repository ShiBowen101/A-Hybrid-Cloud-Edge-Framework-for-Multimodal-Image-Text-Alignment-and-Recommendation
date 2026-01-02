

import os
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from config import (
    API_HOST, API_PORT, API_RELOAD, LOG_LEVEL,
    IMAGE_LIBRARY_DIR, PROCESSING_RESULTS_DIR,
    IMAGE_DESCRIPTIONS_JSON
)
from recommender import ImageRecommendationEngine
from image_describer import QwenImageDescriber
from utils import setup_logging
import glob


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免多进程问题


setup_logging()
logger = logging.getLogger(__name__)

# 初始化FastAPI应用
app = FastAPI(
    title="朋友圈图片智能推荐API",
    description="基于Qwen API的图片描述与推荐系统",
    version="1.0.0"
)

# 初始化推荐引擎
try:
    recommendation_engine = ImageRecommendationEngine()
    logger.info("推荐引擎初始化成功")
except Exception as e:
    logger.exception("初始化推荐引擎失败")
    raise

# 初始化图片描述生成器
try:
    image_describer = QwenImageDescriber()
    logger.info("图片描述生成器初始化成功")
except Exception as e:
    logger.exception("初始化图片描述生成器失败")
    raise

# 请求模型
class RecommendationRequest(BaseModel):
    post_text: str
    top_k: int = 5  # 可选参数，默认5

class DescriptionGenerationRequest(BaseModel):
    force_regenerate: bool = False  # 是否强制重新生成所有描述

# 响应模型
class RecommendationResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

class DescriptionGenerationResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, int]] = None

class SystemStatusResponse(BaseModel):
    status: str
    data: Optional[Dict[str, Any]] = None

class ConfigResponse(BaseModel):
    status: str
    data: Optional[Dict[str, Any]] = None

def list_image_files(directory: str) -> List[str]:
    """列出目录中的所有图片文件"""
    if not os.path.exists(directory):
        return []

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, f'*{ext}')))
        image_files.extend(glob.glob(os.path.join(directory, f'*{ext.upper()}')))

    return image_files

@app.get("/system-status", response_model=SystemStatusResponse, tags=["系统"])
async def system_status():
    """获取系统状态信息"""
    try:
        status = recommendation_engine.get_system_status()

        # 确保返回一致的结构
        response_data = {
            "status": status.get("status", "error"),
            "data": {
                "image_count": status.get("image_count", 0),
                "description_count": status.get("description_count", 0),
                "embedding_model": status.get("embedding_model", "unknown"),
                "embedding_dimension": status.get("embedding_dimension", 0),
                "index_dimension": status.get("index_dimension", 0),
                "description_model": status.get("description_model", "unknown"),
                "rerank_model": status.get("rerank_model", "unknown"),
                "top_k_initial": status.get("top_k_initial", 20),
                "top_k_final": status.get("top_k_final", 5),
                "device": status.get("device", "unknown"),
                "image_library_path": status.get("image_library_path", "unknown")
            }
        }

        return response_data
    except Exception as e:
        logger.exception("获取系统状态失败")
        return {
            "status": "error",
            "message": str(e),
            "data": {
                "image_count": 0,
                "description_count": 0,
                "embedding_model": "unknown",
                "embedding_dimension": 0,
                "index_dimension": 0,
                "description_model": "unknown",
                "rerank_model": "unknown",
                "top_k_initial": 20,
                "top_k_final": 5,
                "device": "unknown",
                "image_library_path": "unknown"
            }
        }

@app.post("/generate-descriptions", response_model=DescriptionGenerationResponse, tags=["描述生成"])
async def generate_descriptions(request: DescriptionGenerationRequest, background_tasks: BackgroundTasks):
    """
    生成或更新图片描述

    - **force_regenerate**: 是否强制重新生成所有描述 (默认False)
    """
    try:
        # 在后台任务中执行描述生成
        def generate_descriptions_task():
            try:
                result = image_describer.generate_descriptions(
                    force_regenerate=request.force_regenerate
                )
                logger.info(f"图片描述生成完成: {result}")
                return result
            except Exception as e:
                logger.exception("后台任务中生成图片描述失败")
                return {
                    "total_images": 0,
                    "new_descriptions": 0,
                    "skipped": 0,
                    "failed": 0
                }

        background_tasks.add_task(generate_descriptions_task)

        total_images = len(list_image_files(IMAGE_LIBRARY_DIR))
        return {
            "status": "processing",
            "message": "图片描述生成任务已启动，请稍后检查",
            "data": {
                "new_descriptions": 0,
                "total_images": total_images
            }
        }

    except Exception as e:
        logger.exception("生成图片描述时发生未处理的异常")
        return {
            "status": "error",
            "message": f"服务器内部错误: {str(e)}",
            "data": {
                "new_descriptions": 0,
                "total_images": 0
            }
        }

@app.post("/recommend", response_model=RecommendationResponse, tags=["推荐"])
async def get_recommendations(request: RecommendationRequest):

    try:
        # 验证输入
        if not request.post_text or len(request.post_text.strip()) < 2:
            return {
                "status": "error",
                "message": "朋友圈文案不能为空且至少2个字符"
            }

        if request.top_k < 1 or request.top_k > 10:
            return {
                "status": "error",
                "message": "top_k必须在1-10之间"
            }

        # 获取推荐
        recommendations = recommendation_engine.recommend_images(
            post_text=request.post_text,
            top_k=request.top_k
        )

        # 转换结果格式
        results = []
        for rec in recommendations:
            results.append({
                "image_path": rec.get("image_path", ""),
                "filename": rec.get("filename", "unknown"),
                "score": rec.get("score", 0.0),
                "reason": rec.get("reason", ""),
                "details": rec.get("details", {
                    "content_relevance": 0,
                    "emotional_consistency": 0,
                    "social_appropriateness": 0
                }),
                "image_description": rec.get("image_description", "")[:500] + "..." if len(rec.get("image_description", "")) > 500 else rec.get("image_description", "")
            })

        return {
            "status": "success",
            "message": f"找到 {len(results)} 个推荐结果",
            "data": {
                "results": results
            }
        }

    except Exception as e:
        logger.exception("推荐处理过程中发生未处理的异常")
        return {
            "status": "error",
            "message": f"服务器内部错误: {str(e)}",
            "data": {
                "results": []
            }
        }

@app.get("/config", response_model=ConfigResponse, tags=["系统"])
async def get_config():

    return {
        "status": "success",
        "data": {
            "image_library_dir": IMAGE_LIBRARY_DIR,
            "descriptions_json": IMAGE_DESCRIPTIONS_JSON,
            "processing_results_dir": PROCESSING_RESULTS_DIR,
            "top_k_initial": 20,  # 与recommender.py中的TOP_K_INITIAL保持一致
            "top_k_final": 5,     # 与recommender.py中的TOP_K_FINAL保持一致
            "api_host": API_HOST,
            "api_port": API_PORT,
            "log_level": LOG_LEVEL,
            "description_model": "qwen-vl-plus",
            "rerank_model": "qwen-max"
        }
    }

@app.get("/health", response_model=SystemStatusResponse, tags=["系统"])
async def health_check():

    try:
        status = recommendation_engine.get_system_status()
        return {
            "status": "success",
            "data": status
        }
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return {
            "status": "error",
            "message": "系统未正确初始化",
            "data": {
                "image_count": 0,
                "description_count": 0,
                "embedding_model": "unknown",
                "embedding_dimension": 0,
                "index_dimension": 0,
                "description_model": "unknown",
                "rerank_model": "unknown",
                "top_k_initial": 20,
                "top_k_final": 5,
                "device": "unknown",
                "image_library_path": "unknown"
            }
        }

if __name__ == "__main__":
    import uvicorn

    logger.info("="*50)
    logger.info("启动朋友圈图片推荐API服务")
    logger.info(f"访问地址: http://{API_HOST}:{API_PORT}")
    logger.info(f"Swagger文档: http://{API_HOST}:{API_PORT}/docs")
    logger.info(f"ReDoc文档: http://{API_HOST}:{API_PORT}/redoc")
    logger.info("="*50)

    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
        log_level=LOG_LEVEL.lower(),
        workers=1
    )
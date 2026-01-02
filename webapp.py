"""
webapp.py - Web应用主程序
"""

import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
import time
import torch

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

# ===== 直接初始化系统 =====
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
    import sys
    sys.exit(1)

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/system-status')
def system_status():
    """获取系统状态 - 确保返回一致的结构"""
    try:
        if recommender is None:
            return jsonify({
                "status": "error",
                "message": "推荐引擎未初始化",
                "data": {
                    "image_count": 0,
                    "description_count": 0
                }
            }), 500

        status = recommender.get_system_status()

        # 确保返回的数据结构一致
        response_data = {
            "status": "success",
            "data": {
                "image_count": status.get("image_count", 0),
                "description_count": status.get("description_count", 0),
                "embedding_model": status.get("embedding_model", "unknown"),
                "device": status.get("device", "unknown")
            }
        }

        return jsonify(response_data)
    except Exception as e:
        logger.exception("获取系统状态失败")
        return jsonify({
            "status": "error",
            "message": str(e),
            "data": {
                "image_count": 0,
                "description_count": 0
            }
        }), 500

@app.route('/generate-descriptions', methods=['POST'])
def generate_descriptions():
    """生成/更新图片描述"""
    if describer is None:
        return jsonify({
            "status": "error",
            "message": "图片描述生成器未初始化",
            "data": {
                "total_images": 0,
                "new_descriptions": 0,
                "skipped": 0,
                "failed": 0
            }
        }), 500

    force_regenerate = request.form.get('force_regenerate', 'false') == 'true'

    try:
        # 生成描述
        result = describer.generate_descriptions(force_regenerate=force_regenerate)

        return jsonify({
            "status": "success",
            "data": {
                "total_images": result['total_images'],
                "new_descriptions": result['new_descriptions'],
                "skipped": result['skipped'],
                "failed": result['failed']
            }
        })
    except Exception as e:
        logger.exception("生成描述失败")
        return jsonify({
            "status": "error",
            "message": str(e),
            "data": {
                "total_images": 0,
                "new_descriptions": 0,
                "skipped": 0,
                "failed": 0
            }
        }), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    """推荐图片 - 确保返回一致的结构"""
    if recommender is None:
        return jsonify({
            "status": "error",
            "message": "推荐引擎未初始化",
            "data": {
                "results": []
            }
        }), 500

    post_text = request.form.get('post_text', '').strip()
    top_k = int(request.form.get('top_k', 3))

    if not post_text:
        return jsonify({
            "status": "error",
            "message": "朋友圈文案不能为空",
            "data": {
                "results": []
            }
        }), 400

    try:
        results = recommender.recommend_images(post_text, top_k=top_k)

        # 转换结果为前端可用格式
        formatted_results = []
        for i, result in enumerate(results):
            # 提取文件名
            filename = result['filename']

            # 确保details对象存在且有默认值
            details = result.get('details', {})
            content_relevance = details.get('content_relevance', 0)
            emotional_consistency = details.get('emotional_consistency', 0)
            social_appropriateness = details.get('social_appropriateness', 0)

            formatted_results.append({
                "id": i + 1,
                "filename": filename,
                "score": round(result['score'], 1),
                "reason": result['reason'],
                "content_relevance": content_relevance,
                "emotional_consistency": emotional_consistency,
                "social_appropriateness": social_appropriateness,
                "image_description": result['image_description']
            })

        return jsonify({
            "status": "success",
            "data": {
                "results": formatted_results
            }
        })
    except Exception as e:
        logger.exception("推荐失败")
        return jsonify({
            "status": "error",
            "message": str(e),
            "data": {
                "results": []
            }
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

    # 记录开始时间
    start_time = time.time()

    # 运行应用
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

    # 记录处理时间
    process_time = time.time() - start_time
    logger.info(f"应用运行完成，总耗时: {process_time:.2f}秒")


import os
import numpy as np
import logging
import time
from datetime import datetime
from config import IMAGE_LIBRARY_DIR, PROCESSING_RESULTS_DIR, EMBEDDING_DIMENSION
from embedding import EmbeddingProcessor
from sklearn.neighbors import NearestNeighbors
import pickle
import json
from image_describer import QwenImageDescriber  # 导入图片描述生成器

logger = logging.getLogger(__name__)

def get_filename_without_extension(file_path: str) -> str:

    return os.path.splitext(os.path.basename(file_path))[0]

def build_index(image_embeddings):

    logger.info("正在构建优化的Scikit-learn索引...")

    # 确保至少有一个邻居
    n_neighbors = min(100, len(image_embeddings))
    if n_neighbors < 1:
        logger.error("无法构建索引：没有有效的嵌入向量")
        raise ValueError("没有有效的嵌入向量，无法构建索引")

    # 使用更高效的算法和并行处理
    index = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric='cosine',
        algorithm='auto',
        n_jobs=-1
    )

    index.fit(image_embeddings)
    return index

def ensure_descriptions_exist():

    descriptions_path = os.path.join(PROCESSING_RESULTS_DIR, "train_descriptions.json")

    if not os.path.exists(descriptions_path):
        logger.info("图片描述文件不存在，正在生成...")

        # 确保目录存在
        os.makedirs(PROCESSING_RESULTS_DIR, exist_ok=True)

        # 检查目录权限
        if not os.access(PROCESSING_RESULTS_DIR, os.W_OK):
            logger.error(f"没有写入权限: {PROCESSING_RESULTS_DIR}")
            raise PermissionError(f"没有写入权限: {PROCESSING_RESULTS_DIR}")

        # 创建图片描述生成器
        describer = QwenImageDescriber()

        # 生成描述
        result = describer.generate_descriptions()

        # 检查是否成功生成
        if result["new_descriptions"] == 0 and result["skipped"] == 0 and result["failed"] == len(os.listdir(IMAGE_LIBRARY_DIR)):
            logger.error("无法生成图片描述，系统无法继续")
            raise RuntimeError("无法生成图片描述")

        # 再次检查描述文件是否存在
        if not os.path.exists(descriptions_path):
            logger.error(f"图片描述文件仍未生成: {descriptions_path}")
            raise FileNotFoundError(f"图片描述文件仍未生成: {descriptions_path}")

        logger.info(f"成功生成 {result['new_descriptions']} 个新描述")
        return True
    else:
        logger.info("图片描述文件已存在，跳过生成步骤")
        return False

def process_image_library():

    start_time = time.time()

    # 检查图库目录
    if not os.path.exists(IMAGE_LIBRARY_DIR):
        raise FileNotFoundError(f"图库目录不存在: {IMAGE_LIBRARY_DIR}")

    # 确保图片描述存在
    logger.info("步骤0: 确保图片描述文件存在")
    descriptions_generated = ensure_descriptions_exist()

    # 初始化组件
    embedder = EmbeddingProcessor()

    # 获取所有图片文件
    image_files = [
        os.path.join(IMAGE_LIBRARY_DIR, f)
        for f in os.listdir(IMAGE_LIBRARY_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ]

    logger.info(f"==================================================")
    logger.info(f"开始图库预处理")
    logger.info(f"图库路径: {IMAGE_LIBRARY_DIR}")
    logger.info(f"处理结果将保存至: {PROCESSING_RESULTS_DIR}")
    logger.info(f"==================================================")
    logger.info(f"找到 {len(image_files)} 张图片需要处理")

    # 步骤1: 加载图片描述
    logger.info("步骤1: 加载图片描述")

    # 确保描述文件存在
    descriptions_path = os.path.join(PROCESSING_RESULTS_DIR, "train_descriptions.json")
    if not os.path.exists(descriptions_path):
        error_msg = f"图片描述文件不存在！请检查: {descriptions_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # 加载图片描述
    try:
        with open(descriptions_path, 'r', encoding='utf-8') as f:
            descriptions = json.load(f)
        logger.info(f"成功加载 {len(descriptions)} 个图片描述")
    except Exception as e:
        logger.error(f"加载图片描述文件失败: {str(e)}")
        raise

    # 步骤2: 生成嵌入向量
    logger.info("步骤2: 生成图片语义特征")

    image_embeddings = []
    image_paths = []

    # 处理所有图片
    for i, image_path in enumerate(image_files, 1):
        # 获取文件名（不含扩展名）
        filename = get_filename_without_extension(image_path)

        # 检查是否有描述
        if filename in descriptions:
            description = descriptions[filename]

            try:
                # 生成嵌入向量
                embedding = embedder.image_description_to_embedding(description)

                # 检查维度是否匹配
                if len(embedding) != EMBEDDING_DIMENSION:
                    logger.warning(f"嵌入维度不匹配: {len(embedding)} != {EMBEDDING_DIMENSION}")
                    # 调整维度
                    if len(embedding) > EMBEDDING_DIMENSION:
                        embedding = embedding[:EMBEDDING_DIMENSION]
                    else:
                        padding = np.zeros(EMBEDDING_DIMENSION - len(embedding))
                        embedding = np.concatenate([embedding, padding])

                image_embeddings.append(embedding)
                image_paths.append(os.path.normpath(image_path))

                logger.debug(f"生成嵌入向量 {i}/{len(image_files)}: {os.path.basename(image_path)}")
            except Exception as e:
                logger.error(f"为图片 {image_path} 生成嵌入向量时出错: {str(e)}")
        else:
            logger.warning(f"跳过图片 {image_path} (无描述)")

        # 每10张图片报告一次进度
        if i % 10 == 0 or i == len(image_files):
            logger.info(f"嵌入向量生成进度: {i}/{len(image_files)} ({i/len(image_files)*100:.1f}%)")

    # 转换为numpy数组
    if len(image_embeddings) > 0:
        image_embeddings = np.array(image_embeddings)
        logger.info(f"成功生成 {len(image_embeddings)} 个嵌入向量")
        logger.info(f"嵌入向量维度: {image_embeddings.shape[1]}")
    else:
        logger.error("没有有效的嵌入向量，无法继续处理")
        return {
            "total_images": len(image_files),
            "processed_images": 0,
            "embedding_count": 0,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "processing_time": time.time() - start_time,
            "error": "没有有效的嵌入向量"
        }

    # 步骤3: 构建索引
    logger.info("步骤3: 构建Scikit-learn索引")
    index = build_index(image_embeddings)

    # 步骤4: 保存处理结果
    logger.info("步骤4: 保存处理结果")

    # 保存嵌入向量
    embeddings_path = os.path.join(PROCESSING_RESULTS_DIR, "train_embeddings.npy")
    np.save(embeddings_path, image_embeddings)

    # 保存图片路径
    paths_path = os.path.join(PROCESSING_RESULTS_DIR, "image_paths.pkl")
    with open(paths_path, "wb") as f:
        pickle.dump(image_paths, f)

    # 保存索引
    index_path = os.path.join(PROCESSING_RESULTS_DIR, "sklearn_index.pkl")
    with open(index_path, "wb") as f:
        pickle.dump(index, f)

    # 记录处理时间
    process_time = time.time() - start_time
    logger.info(f"图库预处理完成，总耗时: {process_time:.2f}秒")
    logger.info(f"有效图片数量: {len(image_embeddings)} / {len(image_files)}")
    logger.info(f"处理结果已保存至: {PROCESSING_RESULTS_DIR}")

    return {
        "total_images": len(image_files),
        "processed_images": len(image_embeddings),
        "embedding_count": len(image_embeddings),
        "index_size": index_path,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "processing_time": process_time
    }

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        logger.info("开始图库预处理...")
        process_image_library()
        logger.info("图库预处理完成！")
    except Exception as e:
        logger.exception("图库预处理失败")
        raise

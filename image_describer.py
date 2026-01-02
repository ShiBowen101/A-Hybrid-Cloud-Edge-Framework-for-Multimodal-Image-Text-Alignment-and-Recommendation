

import os
import base64
import json
import logging
import time
import requests
from config import IMAGE_LIBRARY_DIR, IMAGE_DESCRIPTIONS_JSON, QWEN_API_KEY, QWEN_GENERATION_MODEL
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def encode_image_to_base64(image_path: str) -> str:

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_filename_without_extension(file_path: str) -> str:

    return os.path.splitext(os.path.basename(file_path))[0]

class QwenImageDescriber:


    def __init__(self):

        self.api_key = QWEN_API_KEY
        self.descriptions = {}

        # 确保目录存在
        os.makedirs(os.path.dirname(IMAGE_DESCRIPTIONS_JSON), exist_ok=True)

        # 设置离线模式
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.load_descriptions()

    def load_descriptions(self):

        try:
            if os.path.exists(IMAGE_DESCRIPTIONS_JSON):
                with open(IMAGE_DESCRIPTIONS_JSON, 'r', encoding='utf-8') as f:
                    # 转换为以文件名（不含扩展名）为键的字典
                    descriptions = json.load(f)
                    self.descriptions = {}
                    for k, v in descriptions.items():
                        if os.path.isabs(k):  # 如果是完整路径
                            filename = get_filename_without_extension(k)
                            self.descriptions[filename] = v
                        else:
                            self.descriptions[k] = v
                logger.info(f"成功加载 {len(self.descriptions)} 个图片描述")
            else:
                logger.info("未找到图片描述文件，将创建新文件")
        except Exception as e:
            logger.warning(f"加载图片描述失败，将创建新文件: {str(e)}")

    def save_descriptions(self):

        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(IMAGE_DESCRIPTIONS_JSON), exist_ok=True)

            # 确保有写入权限
            if os.path.exists(IMAGE_DESCRIPTIONS_JSON):
                if not os.access(IMAGE_DESCRIPTIONS_JSON, os.W_OK):
                    logger.warning(f"没有写入权限: {IMAGE_DESCRIPTIONS_JSON}")
                    # 尝试修复权限
                    try:
                        os.chmod(IMAGE_DESCRIPTIONS_JSON, 0o644)
                        logger.info(f"已修复 {IMAGE_DESCRIPTIONS_JSON} 的权限")
                    except:
                        logger.error(f"无法修复 {IMAGE_DESCRIPTIONS_JSON} 的权限")

            # 确保路径是规范化格式，并转换为文件名（不含扩展名）
            normalized_descriptions = {}
            for k, v in self.descriptions.items():
                # 如果已经是简单文件名，直接使用
                if not os.path.isabs(k) and os.sep not in k:
                    normalized_descriptions[k] = v
                else:
                    # 如果是完整路径，提取文件名（不含扩展名）
                    normalized_descriptions[get_filename_without_extension(k)] = v

            # 确保目录存在
            os.makedirs(os.path.dirname(IMAGE_DESCRIPTIONS_JSON), exist_ok=True)

            with open(IMAGE_DESCRIPTIONS_JSON, 'w', encoding='utf-8') as f:
                json.dump(normalized_descriptions, f, ensure_ascii=False, indent=2)
            logger.info(f"图片描述已保存至 {IMAGE_DESCRIPTIONS_JSON}，共 {len(normalized_descriptions)} 个条目")
            return True
        except Exception as e:
            logger.error(f"保存图片描述失败: {str(e)}")
            logger.error(f"尝试保存到: {IMAGE_DESCRIPTIONS_JSON}")
            return False

    def generate_description_prompt(self) -> str:

        return """请为这张图片生成简洁而详细的中文描述，重点关注：
        
1. 图片中的主要场景和物体
2. 整体氛围和情感基调
3. 适合的社交分享场景

描述应流畅自然，避免使用编号、标题或Markdown格式。不要包含"图片中"、"这张照片"等冗余词语。直接描述内容，适合用于朋友圈图片推荐系统。请直接输出描述文本，不要包含其他内容。"""

    def call_qwen_api(self, image_path: str) -> Optional[str]:


        base64_image = encode_image_to_base64(image_path)


        api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"


        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Client": "python-sdk"
        }


        payload = {
            "model": QWEN_GENERATION_MODEL,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": f"image/jpeg;base64,{base64_image}"
                            },
                            {
                                "text": self.generate_description_prompt()
                            }
                        ]
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
                logger.error(f"Qwen API调用失败 [状态码: {response.status_code}]: {response.text}")
                return None

            # 解析响应
            result = response.json()

            # 检查API响应是否包含错误
            if "output" not in result or "choices" not in result["output"] or not result["output"]["choices"]:
                logger.error(f"Qwen API响应格式错误: {result}")
                return None

            # 处理content可能为列表的情况
            content = result["output"]["choices"][0]["message"]["content"]
            # 如果content是列表，将其转换为字符串
            if isinstance(content, list):
                text_content = ""
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_content += item["text"] + " "
                    elif isinstance(item, str):
                        text_content += item + " "
                description = text_content.strip()
            else:
                description = content.strip()

            # 清理描述内容 - 移除换行符和Markdown格式
            description = description.replace('\n', ' ').replace('\r', ' ')
            # 移除编号和标题
            import re
            description = re.sub(r'\d+\.\s*\*\*.*\*\*:\s*', '', description)
            description = re.sub(r'-\s*', '', description)
            # 移除多余的空格
            description = re.sub(r'\s+', ' ', description)
            # 移除引号中的特殊字符问题
            description = description.replace('\"', '"')

            # 记录API调用信息
            elapsed = time.time() - start_time
            logger.debug(f"Qwen API调用成功 (耗时: {elapsed:.2f}s): {description[:100]}{'...' if len(description) > 100 else ''}")

            return description

        except Exception as e:
            logger.exception(f"调用Qwen API时出错: {str(e)}")
            return None

    def get_description(self, image_path: str) -> Optional[str]:

        # 规范化路径
        image_path = os.path.normpath(image_path)
        # 获取文件名（不含扩展名）
        filename = get_filename_without_extension(image_path)

        # 检查是否已有描述
        if filename in self.descriptions:
            return self.descriptions[filename]

        # 如果没有描述，调用API生成
        description = self.call_qwen_api(image_path)

        # 如果生成成功，保存到缓存
        if description:
            self.descriptions[filename] = description
            self.save_descriptions()

        return description

    def generate_descriptions(self, force_regenerate: bool = False) -> Dict[str, int]:

        # 检查图库目录
        if not os.path.exists(IMAGE_LIBRARY_DIR):
            raise FileNotFoundError(f"图库目录不存在: {IMAGE_LIBRARY_DIR}")

        # 确保目录存在
        os.makedirs(os.path.dirname(IMAGE_DESCRIPTIONS_JSON), exist_ok=True)

        # 获取所有图片文件
        image_files = [
            os.path.join(IMAGE_LIBRARY_DIR, f)
            for f in os.listdir(IMAGE_LIBRARY_DIR)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]

        logger.info(f"==================================================")
        logger.info(f"开始图库预处理")
        logger.info(f"图库路径: {IMAGE_LIBRARY_DIR}")
        logger.info(f"处理结果将保存至: {IMAGE_DESCRIPTIONS_JSON}")
        logger.info(f"==================================================")
        logger.info(f"找到 {len(image_files)} 张图片需要处理")

        # 统计信息
        total = len(image_files)
        new_descriptions = 0
        skipped = 0
        failed = 0

        # 生成描述
        for i, image_path in enumerate(image_files, 1):
            # 获取文件名（不含扩展名）
            filename = get_filename_without_extension(image_path)

            # 检查是否需要跳过
            if not force_regenerate and filename in self.descriptions:
                skipped += 1
                continue

            logger.info(f"处理图片 {i}/{total}: {os.path.basename(image_path)}")

            try:
                description = self.call_qwen_api(image_path)
                if description:
                    # 使用文件名（不含扩展名）作为键
                    self.descriptions[filename] = description
                    new_descriptions += 1
                else:
                    failed += 1
                    logger.warning(f"无法生成图片 {os.path.basename(image_path)} 的描述")
            except Exception as e:
                logger.error(f"处理图片 {image_path} 时出错: {str(e)}")
                failed += 1

            # 避免API调用过于频繁
            time.sleep(0.5)

            # 每5张图片保存一次
            if i % 5 == 0 or i == total:
                success = self.save_descriptions()
                if success:
                    logger.info(f"处理进度: {i}/{total} ({i/total*100:.1f}%) | 新描述: {new_descriptions} | 失败: {failed}")
                else:
                    logger.warning(f"保存进度失败: {i}/{total} ({i/total*100:.1f}%) | 新描述: {new_descriptions} | 失败: {failed}")

        # 最后保存一次
        success = self.save_descriptions()
        if not success:
            logger.error("无法保存图片描述文件，请检查权限和磁盘空间")

        # 记录报告
        logger.info(f"图片描述生成完成报告")
        logger.info(f"==================")
        logger.info(f"- 图库路径: {IMAGE_LIBRARY_DIR}")
        logger.info(f"- 总图片数量: {total}")
        logger.info(f"- 新生成描述: {new_descriptions}")
        logger.info(f"- 跳过已有描述: {skipped}")
        logger.info(f"- 失败数量: {failed}")
        logger.info(f"- 描述保存至: {IMAGE_DESCRIPTIONS_JSON}")
        logger.info(f"- 当前总描述数: {len(self.descriptions)}")
        logger.info(f"")

        return {
            "total_images": total,
            "new_descriptions": new_descriptions,
            "skipped": skipped,
            "failed": failed
        }
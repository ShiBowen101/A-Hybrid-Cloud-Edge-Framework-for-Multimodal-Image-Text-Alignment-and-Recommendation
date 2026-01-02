# 朋友圈图片推荐系统

## 项目简介

这个项目是一个基于 AI 的朋友圈图片推荐系统，旨在根据用户输入的朋友圈文案（如“蓝天白云”）从本地图库中推荐匹配的图片。系统使用 Qwen API 生成图片描述，使用嵌入模型（如 m3e-base）生成文本向量，通过余弦相似度进行初筛，并使用 Qwen API 进行精排评分。适用于社交媒体图片匹配、内容推荐等场景。

主要功能：
- 自动生成图片描述。
- 生成语义嵌入向量。
- 基于文案推荐图片，包括内容相关性、情感一致性和社交适宜性评分。
- 支持离线模式和 GPU 加速。

技术栈：
- Python 3.8+
- sentence-transformers (嵌入模型)
- transformers & safetensors (模型加载)
- scikit-learn (索引构建)
- flask (Web 服务)
- Qwen API (描述生成和精排)

## 环境要求

- Python 3.8+
- Anaconda 环境（推荐，用于管理依赖）
- CUDA 支持（可选，用于 GPU 加速）
- Hugging Face 模型文件（本地下载，用于离线模式）

## 安装依赖

1. 激活环境：
   ```bash
   D:\Anaconda\envs\pytorch\Scripts\activate
   ```

2. 安装依赖（使用 requirements.txt）：
   ```bash
   D:\Anaconda\envs\pytorch\Scripts\pip.exe install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

3. 下载模型文件（离线模式）：
   - 访问 https://huggingface.co/moka-ai/m3e-base
   - 下载 `model.safetensors`, `config.json`, `tokenizer.json`, `vocab.txt`, `1_Pooling/config.json`。
   - 放置到 `D:/Anaconda/envs/pytorch/Lib/site-packages/sentence-transformers/m3e-base`。
   - 设置环境变量：
     ```bash
     set SENTENCE_TRANSFORMERS_HOME=D:/Anaconda/envs/pytorch/Lib/site-packages/sentence-transformers
     ```

## 这个项目如何运行（详细）

### 步骤 1: 配置项目
- 编辑 `config.py`：
  - 设置 `IMAGE_LIBRARY_DIR = "Train"`（图库路径）。
  - 设置 `PROCESSING_RESULTS_DIR = "processing_results"`（结果保存路径）。
  - 设置 `QWEN_API_KEY = "your-qwen-api-key"`（Qwen API 密钥）。
  - 设置 `EMBEDDING_DIMENSION = 768`（m3e-base 模型维度）。
- 确保图库目录 `Train` 包含图片文件（.jpg, .png 等）。

### 步骤 2: 生成图片描述和嵌入向量
- 运行预处理脚本生成描述、嵌入和索引：
  ```bash
  D:\Anaconda\envs\pytorch\python.exe C:\Users\27752\Desktop\NLP\preprocessor.py
  ```
  - 这会生成 `train_descriptions.json`（图片描述）、`image_embeddings.npy`（嵌入向量）和 `sklearn_index.pkl`（索引）。
  - 如果描述文件已存在，跳过生成步骤。
  - 耗时约几分钟（2000 张图片）。

### 步骤 3: 启动 Web 应用
- 运行 Web 服务：
  ```bash
  D:\Anaconda\envs\pytorch\python.exe C:\Users\27752\Desktop\NLP\webapp.py
  ```
  - 访问 http://localhost:5000 测试界面。
  - 输入文案（如“蓝天白云”），点击推荐，查看结果。

### 步骤 4: 测试推荐
- 使用 POST 请求测试：
  ```python
  import requests
  response = requests.post("http://localhost:5000/recommend", json={"post_text": "蓝天白云", "top_k": 5})
  print(response.json())
  ```
  - 预期返回 5 个推荐图片，包括路径、分数和理由。

### 步骤 5: 常见问题排查
- 如果模型加载失败：
  - 检查路径和环境变量。
  - 清空缓存：`del processing_results\cache\embedding_cache.pkl`
- 如果推荐效果弱：
  - 检查 `train_descriptions.json` 的描述质量。
  - 重新运行 `preprocessor.py` 更新嵌入。

## 这个项目的工作原理

### 系统架构
- **模块结构**：
  - `config.py`：系统配置（路径、API 密钥、维度）。
  - `image_describer.py`：使用 Qwen API 生成图片描述。
  - `embedding.py`：使用 m3e-base 模型生成文本嵌入向量。
  - `preprocessor.py`：预处理图库，生成描述、嵌入和索引。
  - `recommender.py`：核心推荐引擎，初筛（向量搜索） + 精排（Qwen API 评分）。
  - `webapp.py`：Flask Web 接口，提供推荐服务。

### 工作流程
1. **预处理阶段**：
   - 加载图库图片。
   - 使用 Qwen API 生成图片描述，保存到 `train_descriptions.json`。
   - 使用 m3e-base 模型将描述转换为 768 维嵌入向量，保存到 `image_embeddings.npy`。
   - 使用 scikit-learn 构建索引（NearestNeighbors），保存到 `sklearn_index.pkl`。

2. **推荐阶段**：
   - 输入文案（如“蓝天白云”）。
   - 生成文案嵌入向量。
   - 初筛：使用索引进行余弦相似度搜索，返回 TOP_K_INITIAL=20 个候选图片。
   - 精排：使用 Qwen API 对候选图片评分（内容相关性、情感一致性、社交适宜性），返回 TOP_K_FINAL=5 个结果。
   - 输出：图片路径、分数、理由和细节。

3. **优化点**：
   - 离线模式：所有模型本地加载，支持 CUDA 加速。
   - 缓存：嵌入缓存减少重复计算。
   - 安全：使用 safetensors 格式避免漏洞。


# 多模态关系分析算法服务

## 1. 项目简介

本项目是一个基于 FastAPI 的多模态关系分析算法服务。

它使用 **InternVL3_5-14B** 模型，旨在分析RGB图像（图像1）、红外图像（图像2）和文本（文本1）之间的深层关系。服务能够执行配对分析（如 图像1-图像2）和总体关系判定，输出**等价、关联、因果、矛盾、顺承**五种关系之一，并提供详细的分析论证和综合事实推断。

本项目基于 `chat_tools_intern_multigpu.py` 单文件实现，支持多GPU部署、流式（SSE）与非流式API调用，并集成了会话管理和结果持久化（CSV）功能。

## 2. 技术栈

* **Python**: 3.10
* **Web框架**: FastAPI, Uvicorn
* **AI模型**: PyTorch, Transformers (Hugging Face)
* **模型**: InternVL3_5-14B
* **部署**: Docker

## 3. 系统要求

* **硬件**: 一台或多台 NVIDIA GPU（已在 4 卡环境下测试）
* **操作系统**: Linux
* **软件**: Python 3.10+, CUDA 12.1+
* **模型文件**: 需要您自行下载 `InternVL3_5-14B` 模型权重，并在 `.env` 文件中配置其路径。

## 4. 如何在本地启动开发

### 步骤 1: 克隆项目

```bash
git clone [您的项目GIT地址]
cd [您的项目目录]
````

### 步骤 2: 创建并激活Python虚拟环境

```bash
# 创建虚拟环境 (venv)
python3 -m venv venv

# 激活虚拟环境 (Linux/macOS)
source venv/bin/activate
```

### 步骤 3: 安装依赖

本项目使用 `requirements.txt` 管理依赖。

```bash
# 确保 pip 是最新的
pip install --upgrade pip

# 安装所有依赖
pip install -r requirements.txt
```

### 步骤 4: 配置环境变量

根据项目规范，环境变量应放在 `.env` 文件中。请复制 `.env.example` 并重命名为 `.env`，然后修改其内容。

```bash
# 复制示例文件
cp .env.example .env

# 编辑 .env 文件
nano .env
```

`.env` 文件内容应如下（**重点：必须修改 MODEL\_PATH**）：

```dotenv
# 1. 服务与模型配置
# 回调基础地址
CALLBACK_BASE_URL=[http://121.48.162.151:18000](http://121.48.162.151:18000)
# 模型路径 (必须修改为服务器上的实际路径)
MODEL_PATH=/path/to/your/models/InternVL3_5-14B

# 2. 多GPU与性能配置
# 指定可见的GPU (例如: 0,1,2) - 建议在启动命令前 export
# CUDA_VISIBLE_DEVICES=0,1,2,3,4
# HF多卡分片策略 (auto, balanced, balanced_low_0)
HF_DEVICE_MAP=balanced
# 每张GPU的显存占用比例 (0.0 -> 1.0)
GPU_MEM_FRACTION=1.0

# 3. 算法推理参数
IMG_INPUT_SIZE=448
RGB_MAX_BLOCKS=8
IR_MAX_BLOCKS=6
MAX_NEW_TOKENS=1024
STREAM_TIMEOUT=2.0

# 4. 服务配置 (本地开发)
HOST=0.0.0.0
PORT=8102
RELOAD=True

# 5. 数据持久化路径
# 会话缓存目录
SESSIONS_DIR=session_cache
# CSV日志与结果目录 (./ 代表当前目录)
DATA_DIR=./
```

### 步骤 5: 启动服务

在启动前，请确保您已在当前终端设置了 `CUDA_VISIBLE_DEVICES` 环境变量（如果 `.env` 中未设置）。

```bash
# (可选) 指定要使用的GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 运行主程序
python chat_tools_intern_multigpu.py
```

服务启动后，您可以访问 `http://127.0.0.1:8102/docs` 查看 API 文档。

## 5\. 如何离线部署项目 (Docker)

项目规范要求支持使用 Docker 运行及导出离线镜像。

### 步骤 1: 准备离线文件

确保以下文件与 `Dockerfile` 位于同一目录：

  * `chat_tools_intern_multigpu.py`
  * `requirements.txt`
  * `.env` (用于生产环境的配置)

### 步骤 2: 构建 Docker 镜像

```bash
docker build -t multi_modal_analysis:1.0 .
```

### 步骤 3: 运行 Docker 容器

在生产环境运行时，我们通过 `-v` (volume) 挂载模型、会话缓存和日志目录，并通过 `--env-file` 传递配置。

> **注意**：
>
> 1.  确保 `--gpus all` 标志已添加。
> 2.  确保 `.env` 文件中的 `MODEL_PATH` 指向容器内的挂载路径 (例如 `/models/InternVL3_5-14B`)。
> 3.  确保 `.env` 中的 `RELOAD` 设置为 `False`。
> 4.  确保 `DATA_DIR` 和 `SESSIONS_DIR` 指向容器内的挂载点 (例如 `/app/data_logs` 和 `/app/session_cache`)。

```bash
docker run -d \
    --name multi_modal_service \
    --gpus all \
    -p 8102:8102 \
    -v /path/on/host/models:/models \
    -v /path/on/host/session_cache:/app/session_cache \
    -v /path/on/host/csv_logs:/app/data_logs \
    --env-file ./.env \
    multi_modal_analysis:1.0
```

## 6\. 常见问题 (FAQ)

**Q: 启动时报错 `RuntimeError: 模型路径不存在`**
A: 请检查您的 `.env` 文件中的 `MODEL_PATH` 是否正确指向了 `InternVL3_5-14B` 模型的权重目录。

**Q: 出现 `torch.cuda.OutOfMemoryError` (OOM)**
A:

1.  代码内置了 OOM 自动降档重试机制（降低图像块数）。
2.  如果仍然 OOM，请尝试在 `.env` 中调小 `RGB_MAX_BLOCKS` 和 `IR_MAX_BLOCKS` 的值。
3.  检查 `GPU_MEM_FRACTION` 是否设置过高（如 `1.0`），可以适当调低（如 `0.9`）。

### 第一阶段：有网环境准备 (Online Preparation)

在将项目拷贝到离线环境之前，必须在有网的机器上完成以下 3 步。

#### 1\. 整理目录结构

确保您的项目根目录包含以下文件：

  * [cite_start]`chat_tools_intern_multigpu.py` [cite: 3]
    * `relation_cli_common.py`, `relation_cli_conflict.py`, `relation_cli_consistency.py`, `relation_cli_relation.py`
  * [cite_start]`.env` (由 `.env.example` 复制而来) [cite: 4]
  * [cite_start]`Dockerfile` [cite: 1]

> 提示：Docker 镜像现在不再内置模型权重，部署时必须将宿主机的 `InternVL3_5-14B-Instruct` 目录挂载到容器内的 `/model/InternVL3_5-14B-Instruct`；`.env` 中的 `MODEL_PATH` 应填写宿主机真实路径。

#### 2. 下载离线依赖 (使用本地 Pip/Conda)

**注意**：为了确保下载的依赖包与 Docker 内部环境 (Linux + Python 3.12) 兼容，请务必在 **Linux** 环境下使用 **Python 3.12** 进行下载。

**操作步骤**：

1.  **准备 Python 3.12 环境** (推荐使用 Conda):
    ```bash
    # 如果您还没有安装 Conda，可以使用项目根目录下的安装脚本：
    # bash Miniconda3-latest-Linux-x86_64.sh

    # 创建并激活 Python 3.12 环境
    conda create -n py312_downloader python=3.12 -y
    conda activate py312_downloader
    ```

2.  **执行下载命令**:
    ```bash
    # 1. 创建存放目录
    mkdir -p offline/packages

    # 2. 升级 pip
    pip install --upgrade pip

    # 3. 下载所有依赖到 offline/packages
    # 注意：这将下载适配当前操作系统(Linux)的 whl 包
    pip download -d offline/packages -r requirements.txt

    # 4. 复制 requirements.txt 到 offline 目录
    cp requirements.txt offline/requirements.txt
    
    echo '[SUCCESS] 离线资源准备完成！文件位于 offline/packages'
    ```

#### 3. 检查离线资源

执行完上述命令后，请检查 `offline` 目录：
1.  `offline/packages` 中应包含大量 `.whl` 文件（包括 `transformers-*.whl`）。
2.  `offline/requirements.txt` 应已自动生成，且不包含 git 链接。

-----

### 第二阶段：一键执行脚本 (`start.sh`)

请在项目根目录新建 `start.sh`，赋予执行权限 (`chmod +x start.sh`)。

这个脚本集成了**依赖冲突解决逻辑**、**本地运行**、**Docker 构建**和**离线导入**功能。

```bash
#!/bin/bash

# =================配置区域=================
IMAGE_NAME="multi_modal_analysis:1.0"
CONTAINER_NAME="multimodal_service"
EXPORT_DIR="docker-images-export"
TAR_NAME="multi_modal_analysis_1.0.tar"
# =========================================

# [cite_start]读取 .env 配置 (端口默认 8102) [cite: 1]
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi
PORT=${PORT:-8102}

# 帮助函数
function show_help {
    echo "用法: $0 {install|run|run:nogpu|build}"
    echo "  install   : 安装依赖 (自动识别离线/在线模式)"
    echo "  run       : 启动服务 (优先使用 Docker，无镜像则使用本地 Python)"
    echo "  build     : 构建并导出 Docker 镜像 (用于交付)"
    echo "  run:nogpu : (仅调试) 不使用 GPU 启动 Docker"
}

# 动作分发
ACTION=$1
if [ -z "$ACTION" ]; then
    show_help
    exit 1
fi

case "$ACTION" in
    "install")
        echo "[INFO] === 开始安装流程 ==="
        
        # 场景 A: 生产环境导入 Docker 镜像
        if [ -f "$EXPORT_DIR/$TAR_NAME" ]; then
            echo "[INFO] 检测到离线镜像包，正在导入 Docker 镜像..."
            docker load -i "$EXPORT_DIR/$TAR_NAME"
            echo "[SUCCESS] 镜像导入完成。"
        
        # 场景 B: 开发环境离线安装 Python 库
        elif [ -d "offline/packages" ]; then
            echo "[INFO] 检测到 offline/packages，进入【离线开发安装模式】..."
            
            # 1. 创建并激活虚拟环境
            if [ ! -d "venv" ]; then
                echo "[INFO] 创建虚拟环境..."
                python3 -m venv venv
            fi
            source venv/bin/activate

            # 2. 关键步骤：优先安装本地编译的 Transformers
            # 解决 sentence-transformers 依赖冲突的关键
            echo "[INFO] 步骤1/2: 优先安装本地 Transformers 核心库..."
            TRANSFORMER_WHL=$(find offline/packages -name "transformers-*.whl" | head -n 1)
            if [ -n "$TRANSFORMER_WHL" ]; then
                pip install "$TRANSFORMER_WHL"
            else
                echo "[ERROR] 未在 offline/packages 中找到 transformers whl 包！请检查准备工作。"
                exit 1
            fi

            # 3. 安装剩余依赖
            echo "[INFO] 步骤2/2: 安装剩余依赖..."
            if [ -f "offline/requirements.txt" ]; then
                pip install --no-index --find-links=offline/packages -r offline/requirements.txt
            else
                echo "[ERROR] 未找到 offline/requirements.txt，请确保已移除 git 链接并复制该文件。"
                exit 1
            fi
            
            echo "[SUCCESS] 离线依赖安装完成。"

        # 场景 C: 在线安装
        else
            echo "[WARN] 未发现离线资源，尝试在线安装..."
            pip install -r requirements.txt
        fi
        ;;

    "build")
        echo "[INFO] === 开始构建流程 ==="
        echo "[INFO] 构建 Docker 镜像: $IMAGE_NAME"
        docker build -t $IMAGE_NAME .
        
        echo "[INFO] 正在导出镜像到 $EXPORT_DIR (这可能需要几分钟)..."
        mkdir -p $EXPORT_DIR
        docker save -o "$EXPORT_DIR/$TAR_NAME" $IMAGE_NAME
        
        echo "[SUCCESS] 构建完成！交付文件位于: $EXPORT_DIR/$TAR_NAME"
        ;;

    "run"|"run:nogpu")
        echo "[INFO] === 准备启动服务 ==="
        
        # 检查是否应该用 Docker 启动
        if docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
            echo "[INFO] 模式: Docker 容器运行"
            
            # GPU 参数处理
            GPU_FLAG="--gpus all"
            if [ "$ACTION" == "run:nogpu" ]; then
                echo "[WARN] 警告: 未启用 GPU，推理速度将极慢"
                GPU_FLAG=""
            fi

            # 清理旧容器
            docker rm -f $CONTAINER_NAME 2>/dev/null
            
            # [cite_start]确保挂载目录存在 [cite: 4]
            mkdir -p session_cache data
            
            # 启动容器
            # 注意：-v 挂载路径必须根据实际情况调整。
            # MODEL_PATH 在 .env 中读取，必须指向宿主机真实路径
            echo "[INFO] 正在启动容器，端口映射: $PORT:8102"
            echo "[INFO] 容器内模型路径: $MODEL_PATH_IN_CONTAINER"
            
            docker run -d \
                --name $CONTAINER_NAME \
                $GPU_FLAG \
                -p $PORT:8102 \
                --env-file .env \
                -e MODEL_PATH=$MODEL_PATH_IN_CONTAINER \
                -v $(pwd)/session_cache:/app/session_cache \
                -v $(pwd)/data:/app/data \
                ${MOUNT_MODEL} \
                $IMAGE_NAME
            
            if [ $? -eq 0 ]; then
                echo "[SUCCESS] 服务已启动。查看日志: docker logs -f $CONTAINER_NAME"
            else
                echo "[ERROR] 容器启动失败。"
            fi
            
        else
            echo "[INFO] 模式: 本地 Python 运行 (未检测到 Docker 镜像)"
            if [ -d "venv" ]; then
                source venv/bin/activate
            fi
            
            # 检查模型路径
            if [ ! -d "$MODEL_PATH" ]; then
                echo "[ERROR] 模型路径不存在: $MODEL_PATH"
                echo "请修改 .env 文件中的 MODEL_PATH 指向本地实际路径。"
                exit 1
            fi
            
            echo "[INFO] 启动 FastAPI 服务..."
            export CUDA_VISIBLE_DEVICES=0 
            python chat_tools_intern_multigpu.py
        fi
        ;;

    *)
        show_help
        exit 1
        ;;
esac
```

    > 停止服务：执行 `./start.sh stop` 会直接停止并删除已运行的 `multimodal_service` 容器。

-----

### 第三阶段：离线验证全流程 (Verification Steps)

现在，您可以按照以下步骤模拟“从开发到交付”的全过程：

#### 1\. 离线开发环境验证 (Verify Development)

*场景：在一台没有网的 GPU 服务器上开发。*

1.  **拷贝文件**：将包含 `offline/` 文件夹的项目上传到服务器。
2.  **配置环境**：编辑 `.env`，将 `MODEL_PATH` 修改为服务器上存放 InternVL3.5 模型的真实路径。
3.  **安装依赖**：
    ```bash
    ./start.sh install
    ```
    *预期结果*：脚本会先安装 `transformers-*.whl`，然后再安装其他包，成功绕过依赖冲突报错。
4.  **运行服务**：
    ```bash
    ./start.sh run
    ```
    *预期结果*：服务通过本地 Python 启动，访问 `http://IP:8102/docs` 可见 Swagger UI。

#### 使用 CLI 模式进行离线推理

在构建好的镜像或本地环境中，可直接通过 `start.sh cli` 调用四个命令行脚本：

```bash
# 示例：判定冲突（使用命名参数）
./start.sh cli conflict --rgb_image_url /workspace/data/rgb.jpg --infrared_image_url /workspace/data/ir.jpg --text_json_url /workspace/data/text.json

# 示例：输出一致性结果（使用位置参数，等价于 python relation_cli_consistency.py <rgb> <ir> <text>）
./start.sh cli consistency /workspace/data/rgb.jpg /workspace/data/ir.jpg /workspace/data/text.json

# 示例：仅输出关系（位置参数）
./start.sh cli relation /workspace/data/rgb.jpg /workspace/data/ir.jpg /workspace/data/text.json
```

说明：

- `start.sh cli` 会优先在 Docker 镜像中运行脚本；若未检测到镜像则回退到本地 Python 运行。
- 脚本会挂载当前目录到容器内的 `/workspace`，请使用 `/workspace/...` 形式访问宿主机文件；`./data` 与 `./session_cache` 会分别映射为 `/app/data` 与 `/app/session_cache`。
- 当 `USE_HOST_MODEL=true` 时会把 `.env` 中的 `MODEL_PATH` 挂载到容器内的 `$MODEL_PATH_IN_CONTAINER`。

#### 2\. 离线打包 (Build & Export)

*场景：验证通过后，制作交付物。*

1.  **执行构建**：
    ```bash
    ./start.sh build
    ```
    *预期结果*：Docker 镜像构建成功，并在 `docker-images-export/` 目录下生成 `.tar` 文件。

#### 3\. 离线生产部署 (Production Deploy)

*场景：在客户的隔离环境服务器上部署。*

1.  **上传交付物**：将 `docker-images-export` 文件夹、`start.sh` 和 `.env` 上传到生产服务器。
2.  **配置生产环境**：
    * 修改 `.env` 中的 `MODEL_PATH` 为生产服务器上模型的路径（例如 `/data/models/InternVL3_5-14B-Instruct`）。
    * **注意**：镜像未包含模型，`start.sh`（默认 `USE_HOST_MODEL=true`）会将该宿主机路径挂载为容器内的 `/model/InternVL3_5-14B-Instruct` 并在容器中使用该路径进行加载。
3.  **安装（导入镜像）**：
    ```bash
    ./start.sh install
    ```
    *预期结果*：脚本检测到 tar 包，自动执行 `docker load`。
4.  **启动服务**：
    ```bash
    ./start.sh run
    ```
    *预期结果*：容器启动成功。使用 `docker logs -f multimodal_service` 查看日志，确保模型加载无误。
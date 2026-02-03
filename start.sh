#!/bin/bash

# =================配置区域=================
IMAGE_NAME="multi_modal_analysis:1.0"
CONTAINER_NAME="multimodal_service"
EXPORT_DIR="docker-images-export"
TAR_NAME="multi_modal_analysis_1.0.tar"
# 是否强制使用宿主机模型路径挂载（镜像不再内置模型，默认 true）
USE_HOST_MODEL=${USE_HOST_MODEL:-true}
# 容器内模型路径（仅通过挂载提供）
MODEL_PATH_IN_CONTAINER="/model/InternVL3_5-14B-Instruct"
# 基础镜像 (如果 Docker Hub 无法访问，请修改此处为国内镜像源)
# 例如: swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
BASE_IMAGE="nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04"
# =========================================

# [cite_start]读取 .env 配置 (端口默认 8102) [cite: 1]
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi
PORT=${PORT:-8102}

# 帮助函数
function show_help {
    echo "用法: $0 {install|run|run:nogpu|build|stop|cli}"
    echo "  install   : 安装依赖 (自动识别离线/在线模式)"
    echo "  run       : 启动服务 (优先使用 Docker，无镜像则使用本地 Python)"
    echo "  build     : 构建并导出 Docker 镜像 (用于交付)"
    echo "  run:nogpu : (仅调试) 不使用 GPU 启动 Docker"
    echo "  stop      : 停止并删除已运行的 Docker 容器"
    echo "  cli MODE  : 在容器或本地运行 CLI 脚本 (MODE=conflict|consistency|relation)"
    echo "              支持位置参数调用: $0 cli consistency <rgb> <ir> <text>"
    echo "环境变量: USE_HOST_MODEL=true (默认) 挂载宿主机模型目录，使用 .env 中的 MODEL_PATH"
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
        if [ -f "$TAR_NAME" ]; then
            echo "[INFO] 检测到当前目录下存在离线镜像包，正在导入 Docker 镜像..."
            docker load -i "$TAR_NAME"
            echo "[SUCCESS] 镜像导入完成。"
        elif [ -f "$EXPORT_DIR/$TAR_NAME" ]; then
            echo "[INFO] 检测到离线镜像包 ($EXPORT_DIR)，正在导入 Docker 镜像..."
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
                # 使用 --no-deps 避免自动安装依赖（如 torch），防止版本冲突
                pip install "$TRANSFORMER_WHL" --no-deps
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
        mkdir -p $EXPORT_DIR
        
        # 检查是否需要重新构建
        DO_BUILD=true
        if [ -f "$EXPORT_DIR/$TAR_NAME" ]; then
            read -p "[QUESTION] 镜像包 $EXPORT_DIR/$TAR_NAME 已存在。是否重新构建镜像？(默认 N) [y/N]: " REBUILD_CHOICE
            case "$REBUILD_CHOICE" in 
                [yY][eE][sS]|[yY]) 
                    DO_BUILD=true 
                    ;;
                *) 
                    DO_BUILD=false 
                    echo "[INFO] 跳过镜像构建与导出，仅更新配置文件..." 
                    ;;
            esac
        fi

        if [ "$DO_BUILD" = "true" ]; then
            # 检查基础镜像是否存在，不存在则尝试拉取
            if ! docker image inspect $BASE_IMAGE > /dev/null 2>&1; then
                echo "[INFO] 本地未找到基础镜像 $BASE_IMAGE，尝试拉取..."
                docker pull $BASE_IMAGE
                if [ $? -ne 0 ]; then
                    echo "[ERROR] 无法拉取基础镜像。如果您在中国大陆，可能需要配置 Docker 镜像加速。"
                    echo "建议: 1. 配置 /etc/docker/daemon.json 使用镜像加速器"
                    echo "      2. 或者修改脚本中的 BASE_IMAGE 变量为可访问的镜像源"
                    exit 1
                fi
            fi

            echo "[INFO] 构建 Docker 镜像: $IMAGE_NAME (Base: $BASE_IMAGE)"
            docker build --build-arg BASE_IMAGE=$BASE_IMAGE -t $IMAGE_NAME .
            
            echo "[INFO] 正在导出镜像到 $EXPORT_DIR (这可能需要几分钟)..."
            docker save -o "$EXPORT_DIR/$TAR_NAME" $IMAGE_NAME
        fi
        
        echo "[INFO] 正在复制配置文件到 $EXPORT_DIR..."
        # 强制更新脚本和文档，确保交付版本一致
        cp -f start.sh "$EXPORT_DIR/"
        [ -f deploy.md ] && cp -f deploy.md "$EXPORT_DIR/"
        [ -f docker-compose.yml ] && cp -f docker-compose.yml "$EXPORT_DIR/"
        [ -f Dockerfile ] && cp -f Dockerfile "$EXPORT_DIR/"
        [ -f requirements.txt ] && cp -f requirements.txt "$EXPORT_DIR/"
        ls relation_cli_*.py >/dev/null 2>&1 && cp -f relation_cli_*.py "$EXPORT_DIR/"

        # 特殊处理 .env：如果目标目录已有 .env，则不覆盖，防止丢失生产配置
        if [ -f "$EXPORT_DIR/.env" ]; then
            echo "[WARN] 目标目录已存在 .env 文件，跳过覆盖以保留现有配置。"
        elif [ -f .env ]; then
            cp .env "$EXPORT_DIR/"
        elif [ -f .env.example ]; then
            cp .env.example "$EXPORT_DIR/.env"
        fi
        
        echo "[SUCCESS] 构建完成！交付文件位于: $EXPORT_DIR"
        ;;

    "run"|"run:nogpu")
        echo "[INFO] === 准备启动服务 ==="
        
        # 检查是否应该用 Docker 启动
        if docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
            echo "[INFO] 模式: Docker 容器运行"
            
            # GPU 参数处理
            if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
                # 将 0,1 转换为 device=0,1 格式
                GPU_FLAG="--gpus \"device=$CUDA_VISIBLE_DEVICES\""
                echo "[INFO] 检测到指定 GPU: $CUDA_VISIBLE_DEVICES (Docker 参数: $GPU_FLAG)"
            else
                GPU_FLAG="--gpus all"
                echo "[INFO] 未指定 GPU，容器将使用所有可用 GPU (--gpus all)"
            fi

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
            MOUNT_MODEL=""
            if [ "$USE_HOST_MODEL" = "true" ]; then
                if [ -z "$MODEL_PATH" ]; then
                    echo "[ERROR] USE_HOST_MODEL=true 但 .env 中未设置 MODEL_PATH";
                    exit 1
                fi
                if [ ! -d "$MODEL_PATH" ]; then
                    echo "[ERROR] 宿主机模型路径不存在: $MODEL_PATH";
                    exit 1
                fi
                echo "[INFO] 挂载宿主机模型: $MODEL_PATH -> $MODEL_PATH_IN_CONTAINER"
                MOUNT_MODEL="-v \"$MODEL_PATH\":$MODEL_PATH_IN_CONTAINER"
            else
                echo "[ERROR] 镜像已不包含模型，请设置 USE_HOST_MODEL=true 并确保 .env 中的 MODEL_PATH 指向宿主机模型目录"
                exit 1
            fi
            
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
            # 如果环境变量未设置，默认使用 0
            export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
            echo "[INFO] 本地运行使用 GPU: $CUDA_VISIBLE_DEVICES"
            python chat_tools_intern_multigpu.py
        fi
        ;;

    "stop")
        echo "[INFO] === 停止服务流程 ==="
        
        # 1. 停止 Docker 容器
        docker rm -f $CONTAINER_NAME >/dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "[SUCCESS] 已停止并删除 Docker 容器: $CONTAINER_NAME"
        else
            echo "[INFO] 未发现运行中的 Docker 容器。"
        fi

        # 2. 停止本地 Python 进程
        # 查找包含 chat_tools_intern_multigpu.py 的进程，排除 grep 自身
        PIDS=$(ps -ef | grep "chat_tools_intern_multigpu.py" | grep -v grep | awk '{print $2}')
        if [ -n "$PIDS" ]; then
            echo "[INFO] 发现本地 Python 服务进程 (PID: $PIDS)，正在停止..."
            # shellcheck disable=SC2086
            kill -9 $PIDS
            echo "[SUCCESS] 本地服务已停止。"
        else
            echo "[INFO] 未发现本地运行的服务进程。"
        fi
        ;;

    "cli")
        # 用法: ./start.sh cli conflict --rgb_image_url ...
        shift
        CLI_MODE=$1
        shift || true

        case "$CLI_MODE" in
            "conflict") CLI_SCRIPT="relation_cli_conflict.py" ;;
            "consistency") CLI_SCRIPT="relation_cli_consistency.py" ;;
            "relation") CLI_SCRIPT="relation_cli_relation.py" ;;
            ""|*)
                echo "[ERROR] CLI 模式缺失或不支持。用法: $0 cli {conflict|consistency|relation} <args>";
                exit 1
                ;;
        esac

        # 保持与 run 逻辑一致的 GPU/模型挂载处理
        if docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
            if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
                GPU_FLAG="--gpus \"device=$CUDA_VISIBLE_DEVICES\""
                echo "[INFO] CLI 使用指定 GPU: $CUDA_VISIBLE_DEVICES"
            else
                GPU_FLAG="--gpus all"
                echo "[INFO] CLI 未指定 GPU，默认使用全部 GPU"
            fi

            MOUNT_MODEL=""
            if [ "$USE_HOST_MODEL" = "true" ]; then
                if [ -z "$MODEL_PATH" ]; then
                    echo "[ERROR] USE_HOST_MODEL=true 但 .env 中未设置 MODEL_PATH"; exit 1; fi
                if [ ! -d "$MODEL_PATH" ]; then
                    echo "[ERROR] 宿主机模型路径不存在: $MODEL_PATH"; exit 1; fi
                echo "[INFO] 挂载宿主机模型: $MODEL_PATH -> $MODEL_PATH_IN_CONTAINER"
                MOUNT_MODEL="-v \"$MODEL_PATH\":$MODEL_PATH_IN_CONTAINER"
            else
                echo "[ERROR] 镜像已不包含模型，请设置 USE_HOST_MODEL=true 并确保 .env 中的 MODEL_PATH 指向宿主机模型目录"
                exit 1
            fi

            # 挂载常用目录以便读取输入/输出
            mkdir -p session_cache data
            docker run --rm \
                $GPU_FLAG \
                --env-file .env \
                -e MODEL_PATH=$MODEL_PATH_IN_CONTAINER \
                -v $(pwd)/session_cache:/app/session_cache \
                -v $(pwd)/data:/app/data \
                -v $(pwd):/workspace \
                ${MOUNT_MODEL} \
                $IMAGE_NAME \
                python $CLI_SCRIPT "$@"
        else
            echo "[INFO] 未检测到 Docker 镜像，回退到本地运行"
            if [ -d "venv" ]; then
                source venv/bin/activate
            fi
            python $CLI_SCRIPT "$@"
        fi
        ;;

    *)
        show_help
        exit 1
        ;;
esac
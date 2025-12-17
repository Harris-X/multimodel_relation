# 使用带有 CUDA 12.1 支持的 Ubuntu 22.04 基础镜像
# 这是一个兼顾大小和兼容性的选择，适合深度学习推理
ARG BASE_IMAGE=nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
FROM ${BASE_IMAGE}

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # 默认端口，与 .env.example 保持一致
    PORT=8102 \
    # 设置时区为上海，方便日志查看
    TZ=Asia/Shanghai \
    # 镜像内置模型路径（默认已拷贝 InternVL3_5-14B-Instruct）
    MODEL_PATH=/model/InternVL3_5-14B

# 设置工作目录
WORKDIR /app

# 1. 安装系统级依赖
# git: 用于安装 git+https 的依赖
# libgl1/libglib2.0: opencv-python 运行必须的库
# python3.12: 项目要求的 Python 版本
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    libgl1 \
    libglib2.0-0 \
    tzdata \
    curl \
    && ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime \
    && echo ${TZ} > /etc/timezone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 建立 python 软链接，确保 python 命令指向 python3.12
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# 安装 pip (为 Python 3.12)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# 2. 复制依赖文件并安装
# 先复制 requirements.txt 利用 Docker 缓存层
COPY requirements.txt .
# [新增] 复制离线资源目录
COPY offline /app/offline

# 升级 pip 并安装依赖
# 逻辑：优先检查离线包，如果存在则离线安装，否则在线安装
# 使用 --ignore-installed 避免 "Cannot uninstall distutils installed package" 错误 (如 blinker 1.4)
RUN python -m pip install --upgrade pip && \
    if [ -d "/app/offline/packages" ]; then \
        echo "[INFO] Found offline packages, installing from local..." && \
        # 1. 优先安装 transformers wheel (解决依赖冲突)
        find /app/offline/packages -name "transformers-*.whl" -exec pip install {} --no-deps --ignore-installed \; && \
        # 2. 安装剩余依赖 (使用 offline/requirements.txt)
        if [ -f "/app/offline/requirements.txt" ]; then \
            pip install --no-index --find-links=/app/offline/packages -r /app/offline/requirements.txt --ignore-installed; \
        else \
            pip install --no-index --find-links=/app/offline/packages -r requirements.txt --ignore-installed; \
        fi \
    else \
        echo "[INFO] No offline packages found, installing from PyPI..." && \
        python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --ignore-installed; \
    fi

# 3. 复制项目代码与模型权重（将 InternVL3_5-14B-Instruct 打包进镜像）
COPY chat_tools_intern_multigpu.py .
COPY .env.example .
COPY InternVL3_5-14B-Instruct /model/InternVL3_5-14B

# 4. 创建必要的目录 (避免运行时权限问题)
RUN mkdir -p session_cache && chmod 777 session_cache

# 5. 声明端口
EXPOSE 8102

# 6. 启动命令
# 默认加载 .env 文件（如果用户挂载了）
# 这里不直接写 CMD python ... 而是允许外部覆盖
CMD ["python", "chat_tools_intern_multigpu.py"]
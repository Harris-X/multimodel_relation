#!/usr/bin/env bash
set -euo pipefail

# ---------- Pip 镜像 ----------
mkdir -p ~/.pip
cat > ~/.pip/pip.conf <<'EOF'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple
extra-index-url = https://pypi.tuna.tsinghua.edu.cn/simple
timeout = 120
retries = 8
EOF
echo "[OK] pip 源已写入 ~/.pip/pip.conf (aliyun + tuna)"

# ---------- Conda 镜像 ----------
mkdir -p ~/.conda
cat > ~/.conda/.condarc <<'EOF'
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
  - defaults
show_channel_urls: true
EOF
echo "[OK] conda 源已写入 ~/.conda/.condarc (aliyun + tuna)"

# ---------- HuggingFace 镜像 (国内代理) ----------
HF_ENDPOINT_URL=${HF_ENDPOINT_URL:-https://hf-mirror.com}
export HF_ENDPOINT="$HF_ENDPOINT_URL"
if ! grep -q "HF_ENDPOINT" ~/.bashrc 2>/dev/null; then
  echo "export HF_ENDPOINT=\"$HF_ENDPOINT_URL\"" >> ~/.bashrc
  echo "[OK] 已将 HF_ENDPOINT 写入 ~/.bashrc: $HF_ENDPOINT_URL"
else
  echo "[OK] 已检测到 ~/.bashrc 中的 HF_ENDPOINT，跳过写入"
fi
echo "[OK] HuggingFace 镜像已设置 (HF_ENDPOINT=$HF_ENDPOINT_URL)"

# ---------- Docker 镜像加速 ----------
DAEMON_JSON=/etc/docker/daemon.json
sudo mkdir -p /etc/docker
if [ -f "$DAEMON_JSON" ]; then
  sudo cp "$DAEMON_JSON" "${DAEMON_JSON}.bak.$(date +%Y%m%d%H%M%S)"
fi
sudo tee "$DAEMON_JSON" >/dev/null <<'EOF'
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://dockerproxy.com",
    "https://mirror.baidubce.com"
  ]
}
EOF
echo "[OK] Docker 镜像加速已写入 $DAEMON_JSON (daocloud + dockerproxy)"

# ---------- 安装 Docker (若缺失) ----------
if ! command -v docker >/dev/null 2>&1; then
  echo "[INFO] 未检测到 docker，开始安装 docker.io"
  sudo apt-get update -y
  sudo apt-get install -y docker.io docker-compose-plugin
  if command -v systemctl >/dev/null 2>&1; then
    sudo systemctl enable --now docker || true
  fi
  echo "[OK] Docker 已安装"
else
  echo "[OK] 已检测到 docker 命令，跳过安装"
fi

# ---------- 启动/重启 Docker ----------
if command -v docker >/dev/null 2>&1; then
  if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files | grep -q '^docker.service'; then
    sudo systemctl daemon-reload || true
    sudo systemctl restart docker || true
  elif command -v service >/dev/null 2>&1; then
    sudo service docker restart || true
  else
    echo "[WARN] 未检测到 systemctl/service，无法自动重启 Docker"
  fi

  if docker info >/dev/null 2>&1; then
    echo "[OK] Docker 引擎可用"
  else
    echo "[WARN] Docker 引擎未启动或不可用。若为 WSL/Docker Desktop，请手动启动 Docker Desktop。"
  fi
else
  echo "[WARN] 未检测到 docker 命令，跳过重启 (请先安装 Docker)"
fi

# ---------- 安装 Miniconda (自动接受协议) ----------
CONDA_DIR=${CONDA_DIR:-/opt/miniconda3}
CONDA_BIN="$CONDA_DIR/bin/conda"
INSTALLER=/tmp/Miniconda3-latest-Linux-x86_64.sh

if command -v conda >/dev/null 2>&1; then
  echo "[OK] 已检测到 conda 命令，跳过 Miniconda 安装"
elif [ -x "$CONDA_BIN" ]; then
  echo "[OK] 已检测到 Miniconda: $CONDA_BIN"
else
  echo "[INFO] 未检测到 Miniconda，准备安装到 $CONDA_DIR"
  if [ ! -f "$INSTALLER" ]; then
    echo "[INFO] 下载 Miniconda 安装包"
    curl -L "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -o "$INSTALLER"
  fi
  sudo bash "$INSTALLER" -b -p "$CONDA_DIR"
  sudo chown -R "$USER":"$USER" "$CONDA_DIR" 2>/dev/null || true
  echo "[OK] Miniconda 已安装"
  if ! grep -q "$CONDA_DIR/bin" ~/.bashrc 2>/dev/null; then
    echo "export PATH=\"$CONDA_DIR/bin:\$PATH\"" >> ~/.bashrc
    echo "[OK] 已将 $CONDA_DIR/bin 写入 ~/.bashrc"
  fi
fi

echo "[DONE] 国内源配置完成。"
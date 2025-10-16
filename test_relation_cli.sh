#!/usr/bin/env bash
set -euo pipefail

# 找到 Conda 的基础安装路径并初始化
# 下面这行代码会找到你的 conda 安装位置，例如 /root/miniconda3
__conda_setup="$('/root/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/root/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/root/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

# 现在可以正常激活环境了
conda activate mr


# 避免 numexpr 线程数告警
export NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS:-512}

# 批量测试 relation CLI 脚本。
# 用法: ./test_relation_cli.sh [dataset_id]
# 若未传入 dataset_id，默认使用 1 号样本。

DATASET_ID=${1:-2}
BASE_DIR="/root/autodl-tmp/multimodel_relation/test"
RGB_IMAGE="${BASE_DIR}/merge_img/rgb/${DATASET_ID}.jpg"
IR_IMAGE="${BASE_DIR}/merge_img/thermal/${DATASET_ID}.jpg"
TEXT_JSON="${BASE_DIR}/description_with_label/${DATASET_ID}.json"
LABEL_JSON="${BASE_DIR}/label/${DATASET_ID}.json"  # 仅供参考

if [[ ! -f "${RGB_IMAGE}" ]]; then
	echo "找不到 RGB 图像: ${RGB_IMAGE}" >&2
	exit 1
fi

if [[ ! -f "${IR_IMAGE}" ]]; then
	echo "找不到红外图像: ${IR_IMAGE}" >&2
	exit 1
fi

if [[ ! -f "${TEXT_JSON}" ]]; then
	echo "找不到文本 JSON: ${TEXT_JSON}" >&2
	exit 1
fi

if [[ ! -f "${LABEL_JSON}" ]]; then
	echo "找不到文本 JSON: ${LABEL_JSON}" >&2
	exit 1
fi

# echo "===== 运行 conflict 模式 ====="
# python relation_cli_conflict.py \
# 	--rgb_image_url "${RGB_IMAGE}" \
# 	--infrared_image_url "${IR_IMAGE}" \
# 	--text_json_url "${TEXT_JSON}" \
# 	--label "${LABEL_JSON}" \
# 	--pretty

# echo
# echo "===== 运行 relation 模式 ====="
# python relation_cli_relation.py \
# 	--rgb_image_url "${RGB_IMAGE}" \
# 	--infrared_image_url "${IR_IMAGE}" \
# 	--text_json_url "${TEXT_JSON}" \
# 	--label "${LABEL_JSON}" \
# 	--pretty

# echo
# echo "===== 运行 consistency 模式 ====="
# python relation_cli_consistency.py \
# 	--rgb_image_url "${RGB_IMAGE}" \
# 	--infrared_image_url "${IR_IMAGE}" \
# 	--text_json_url "${TEXT_JSON}" \
# 	--label "${LABEL_JSON}" \
# 	--pretty

echo
echo "===== 运行 conflict 模式 (位置参数) ====="
python relation_cli_conflict.py \
	"${RGB_IMAGE}" \
	"${IR_IMAGE}" \
	"${TEXT_JSON}" \
	"${LABEL_JSON}" \
	--pretty

echo
echo "===== 运行 relation 模式 (位置参数) ====="
python relation_cli_relation.py \
	"${RGB_IMAGE}" \
	"${IR_IMAGE}" \
	"${TEXT_JSON}" \
	"${LABEL_JSON}" \
	--pretty

echo
echo "===== 运行 consistency 模式 (位置参数) ====="
python relation_cli_consistency.py \
	"${RGB_IMAGE}" \
	"${IR_IMAGE}" \
	"${TEXT_JSON}" \
	"${LABEL_JSON}" \
	--pretty

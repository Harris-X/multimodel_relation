#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
relation_cli_full.py

命令行工具：输出总体关系 + 一致性结论 + 是否矛盾。
"""
from __future__ import annotations

import argparse
import json
import sys

from relation_cli_common import execute_mode
from chat_tools_intern_multigpu import classify_consistency_relation


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="多模态关系分析 CLI - 全量信息模式")
    p.add_argument("--rgb_image_url", required=False, help="RGB 图像本地路径或 URL")
    p.add_argument("--infrared_image_url", required=False, help="红外图像本地路径或 URL")
    p.add_argument("--text_json_url", required=False, help="文本 JSON 路径或 URL (需包含 text 字段)")
    p.add_argument("rgb_image_url_pos", nargs="?", help="RGB 图像路径 (位置参数)")
    p.add_argument("infrared_image_url_pos", nargs="?", help="红外图像路径 (位置参数)")
    p.add_argument("text_json_url_pos", nargs="?", help="文本 JSON 路径 (位置参数)")
    p.add_argument("--label", required=False, help="对应的label")
    p.add_argument("label_pos", nargs="?", help="标签的 JSON 文件路径")
    p.add_argument("--pretty", default=True,action="store_true", help="添加分隔符突出显示结果")
    return p


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    rgb_url = args.rgb_image_url or args.rgb_image_url_pos
    infrared_url = args.infrared_image_url or args.infrared_image_url_pos
    text_url = args.text_json_url or args.text_json_url_pos

    if not (rgb_url and infrared_url and text_url):
        parser.error("必须提供 RGB、红外与文本路径，可使用位置或命名参数")

    try:
        result = execute_mode(
            "full",
            rgb_image_url=rgb_url,
            infrared_image_url=infrared_url,
            text_json_url=text_url,
        )
    except Exception as e:  # noqa: BLE001
        print(f"【错误】==== {e}", file=sys.stderr)
        sys.exit(1)

    final_relation = result.get("final_relation") or "未解析"
    consistency_result = result.get("consistency_result") or "未提取"
    conflict_flag = "是" if result.get("is_conflict") else "否"
    consistency_result = classify_consistency_relation(final_relation, consistency_result)

    label = args.label or args.label_pos
    with open(label, 'r') as file:
        label_json = json.load(file)
    consistency_result_label = label_json.get("consistency_result")
    conflict_final_conflict = label_json.get("conflict_final_conflict")

    

    summary_lines = [
        "【任务】==== 一致性认知判断",
        f"【RGB 图像】==== {rgb_url}",
        f"【红外图像】==== {infrared_url}",
        f"【文本 JSON】==== {text_url}",
        f"【是否冲突】==== {conflict_flag}",
        f"【冲突歧义检测结果】==== {consistency_result}",
        f"------------------标签------------------",
        f"【标签 是否冲突】==== {conflict_final_conflict}",
        f"【标签 冲突歧义检测结果】==== {consistency_result_label}",
    ]

    output = "\n".join(summary_lines)
    if args.pretty:
        border = "=" * 48
        output = f"{border}\n{output}\n{border}"

    print(output)


if __name__ == "__main__":  # pragma: no cover
    main()

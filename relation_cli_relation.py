#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
relation_cli_relation.py

命令行工具：输出总体关系类型。
"""
from __future__ import annotations

import argparse
import json
import sys

from relation_cli_common import execute_mode


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="多模态关系分析 CLI - 总体关系模式")
    p.add_argument("--rgb_image_url", required=False, help="RGB 图像本地路径或 URL")
    p.add_argument("--infrared_image_url", required=False, help="红外图像本地路径或 URL")
    p.add_argument("--text_json_url", required=False, help="文本 JSON 路径或 URL (需包含 text 字段)")
    p.add_argument("rgb_image_url_pos", nargs="?", help="RGB 图像路径 (位置参数)")
    p.add_argument("infrared_image_url_pos", nargs="?", help="红外图像路径 (位置参数)")
    p.add_argument("text_json_url_pos", nargs="?", help="文本 JSON 路径 (位置参数)")
    # p.add_argument("--label", required=False, help="对应的label")
    # p.add_argument("label_pos", nargs="?", help="标签的 JSON 文件路径")
    # p.add_argument("--pretty",default=True, action="store_true", help="添加分隔符突出显示结果")
    return p


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    rgb_url = args.rgb_image_url or args.rgb_image_url_pos
    infrared_url = args.infrared_image_url or args.infrared_image_url_pos
    text_url = args.text_json_url or args.text_json_url_pos
    # label = args.label or args.label_pos
    # with open(label, 'r') as file:
    #     label_json = json.load(file)
    # rgb1_rgb2 = label_json.get("relation_label_rgb1_rgb2")
    # rgb1_text1 = label_json.get("relation_label_rgb1_text1")
    # rgb2_text1 = label_json.get("relation_label_rgb2_text1")
    # final = label_json.get("relation_final_relation")

    if not (rgb_url and infrared_url and text_url):
        parser.error("必须提供 RGB、红外与文本路径，可使用位置或命名参数")

    try:
        result = execute_mode(
            "relation",
            rgb_image_url=rgb_url,
            infrared_image_url=infrared_url,
            text_json_url=text_url,
        )
    except Exception as e:  # noqa: BLE001
        print(f"【错误】==== {e}", file=sys.stderr)
        sys.exit(1)

    final_relation = result.get("final_relation") or "未解析"
    pair_rel = result.get("pair_relations") or {}

    def pair_value(key: str) -> str:
        value = pair_rel.get(key)
        return value or "未解析"

    summary_lines = [
        "【任务】==== 推演关系",
        f"【RGB 图像】==== {rgb_url}",
        f"【红外图像】==== {infrared_url}",
        f"【文本 JSON】==== {text_url}",
        f"【图像1-图像2 关系】==== {pair_value('图像1-图像2')}",
        f"【图像1-文本1 关系】==== {pair_value('图像1-文本1')}",
        f"【图像2-文本1 关系】==== {pair_value('图像2-文本1')}",
        f"【总体关系】==== {final_relation}",
        # f"-----------------标签------------------",
        # f"【标签 图像1-图像2 关系】==== {rgb1_rgb2}",
        # f"【标签 图像1-文本1 关系】==== {rgb1_text1}",
        # f"【标签 图像2-文本1 关系】==== {rgb2_text1}",
        # f"【标签 总体关系】==== {final}",
    ]

    output = "\n".join(summary_lines)
    if True:
        border = "=" * 48
        output = f"{border}\n{output}\n{border}"

    print(output)


if __name__ == "__main__":  # pragma: no cover
    main()

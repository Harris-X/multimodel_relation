#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
relation_cli_conflict.py

命令行工具：仅判断是否存在矛盾关系。
"""
from __future__ import annotations

import argparse
import sys

from relation_cli_common import execute_mode


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="多模态关系分析 CLI - 冲突判定模式")
    p.add_argument("--rgb_image_url", required=False, help="RGB 图像本地路径或 URL")
    p.add_argument("--infrared_image_url", required=False, help="红外图像本地路径或 URL")
    p.add_argument("--text_json_url", required=False, help="文本 JSON 路径或 URL (需包含 text 字段)")
    p.add_argument("rgb_image_url_pos", nargs="?", help="RGB 图像路径 (位置参数)")
    p.add_argument("infrared_image_url_pos", nargs="?", help="红外图像路径 (位置参数)")
    p.add_argument("text_json_url_pos", nargs="?", help="文本 JSON 路径 (位置参数)")
    p.add_argument("--pretty", default=True, action="store_true", help="添加分隔符突出显示结果")
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
            "conflict",
            rgb_image_url=rgb_url,
            infrared_image_url=infrared_url,
            text_json_url=text_url,
        )
    except Exception as e:  # noqa: BLE001
        print(f"【错误】==== {e}", file=sys.stderr)
        sys.exit(1)

    conflict_flag = "是" if result.get("conflict") else "否"
    pair_rel = result.get("pair_relations") or {}

    def pair_conflict_label(key: str) -> str:
        value = pair_rel.get(key)
        if value is None:
            return "未解析"
        text = str(value)
        return "是" if any(token in text for token in ("矛盾", "冲突")) else "否"

    summary_lines = [
        "【任务】==== 冲突判定 (conflict)",
        f"【RGB 图像】==== {rgb_url}",
        f"【红外图像】==== {infrared_url}",
        f"【文本 JSON】==== {text_url}",
        f"【图像1-图像2 是否冲突】==== {pair_conflict_label('图像1-图像2')}",
        f"【图像1-文本1 是否冲突】==== {pair_conflict_label('图像1-文本1')}",
        f"【图像2-文本1 是否冲突】==== {pair_conflict_label('图像2-文本1')}",
        f"【总体是否矛盾】==== {conflict_flag}",
    ]

    output = "\n".join(summary_lines)
    if args.pretty:
        border = "=" * 48
        output = f"{border}\n{output}\n{border}"

    print(output)


if __name__ == "__main__":  # pragma: no cover
    main()

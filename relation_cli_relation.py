#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
relation_cli_relation.py

命令行工具：输出总体关系类型。
"""
from __future__ import annotations

import argparse
import sys

from relation_cli_common import execute_mode


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="多模态关系分析 CLI - 总体关系模式")
    p.add_argument("--rgb_image_url", required=True, help="RGB 图像本地路径或 URL")
    p.add_argument("--infrared_image_url", required=True, help="红外图像本地路径或 URL")
    p.add_argument("--text_json_url", required=True, help="文本 JSON 路径或 URL (需包含 text 字段)")
    p.add_argument("--pretty", action="store_true", help="添加分隔符突出显示结果")
    return p


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = execute_mode(
            "relation",
            rgb_image_url=args.rgb_image_url,
            infrared_image_url=args.infrared_image_url,
            text_json_url=args.text_json_url,
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
        "【任务】==== 推演关系 (relation)",
        f"【RGB 图像】==== {args.rgb_image_url}",
        f"【红外图像】==== {args.infrared_image_url}",
        f"【文本 JSON】==== {args.text_json_url}",
        f"【图像1-图像2 关系】==== {pair_value('图像1-图像2')}",
        f"【图像1-文本1 关系】==== {pair_value('图像1-文本1')}",
        f"【图像2-文本1 关系】==== {pair_value('图像2-文本1')}",
        f"【总体关系】==== {final_relation}",
    ]

    output = "\n".join(summary_lines)
    if args.pretty:
        border = "=" * 48
        output = f"{border}\n{output}\n{border}"

    print(output)


if __name__ == "__main__":  # pragma: no cover
    main()

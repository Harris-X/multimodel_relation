#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
relation_cli_full.py

命令行工具：输出总体关系 + 一致性结论 + 是否矛盾。
"""
from __future__ import annotations

import argparse
import sys

from relation_cli_common import execute_mode


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="多模态关系分析 CLI - 全量信息模式")
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
            "full",
            rgb_image_url=args.rgb_image_url,
            infrared_image_url=args.infrared_image_url,
            text_json_url=args.text_json_url,
        )
    except Exception as e:  # noqa: BLE001
        print(f"【错误】==== {e}", file=sys.stderr)
        sys.exit(1)

    final_relation = result.get("final_relation") or "未解析"
    consistency_result = result.get("consistency_result") or "未提取"
    conflict_flag = "是" if result.get("is_conflict") else "否"

    summary_lines = [
        "【任务】==== 一致性认知判断",
        f"【RGB 图像】==== {args.rgb_image_url}",
        f"【红外图像】==== {args.infrared_image_url}",
        f"【文本 JSON】==== {args.text_json_url}",
        f"【是否冲突】==== {conflict_flag}",
        f"【一致性结论】==== {consistency_result}",
    ]

    output = "\n".join(summary_lines)
    if args.pretty:
        border = "=" * 48
        output = f"{border}\n{output}\n{border}"

    print(output)


if __name__ == "__main__":  # pragma: no cover
    main()

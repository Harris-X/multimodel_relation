#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
relation_cli.py

命令行多模态关系分析工具（基于 chat_tools_intern_multigpu 中的模型与解析逻辑）。

支持三种模式：
 1) conflict  -> 仅判断是否 "矛盾" (final_relation == 矛盾)，输出 JSON: {"conflict": bool, "final_relation": str}
 2) relation  -> 输出推演总体关系 final_relation: {"final_relation": str}
  3) full      -> 输出 final_relation + consistency_result + 是否矛盾: {"final_relation": str, "consistency_result": str|None, "is_conflict": bool}

输入统一通过本地路径（或 http(s) URL）：
  --rgb_image_url        RGB 图路径/URL
  --infrared_image_url   红外图路径/URL
  --text_json_url        文本 JSON 路径/URL（需包含 key: text，可选 label / consistency_result）

示例：
  python relation_cli.py conflict \
      --rgb_image_url /root/.../rgb/10.jpg \
      --infrared_image_url /root/.../infrared/10.jpg \
      --text_json_url /root/.../description_with_label/10.json

  python relation_cli.py relation --rgb_image_url ... --infrared_image_url ... --text_json_url ...
  python relation_cli.py full      --rgb_image_url ... --infrared_image_url ... --text_json_url ...

依赖：需已安装 torch、transformers，以及仓库内的 chat_tools_intern_multigpu.py。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import uuid
import re
from typing import Optional

import requests

# 为了减少重复代码，尝试导入已有服务模块的工具与全局变量
try:
    import chat_tools_intern_multigpu as svc
except Exception as e:  # noqa: BLE001
    print(f"导入 chat_tools_intern_multigpu 失败: {e}", file=sys.stderr)
    sys.exit(1)

import torch  # 延迟到这里再导入以便更早报错

# ------------- 辅助：下载 / 读取文本 JSON -------------

def _load_text_json(path_or_url: str) -> dict:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        resp = requests.get(path_or_url, timeout=60)
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return {"text": resp.text}
    if not os.path.exists(path_or_url):
        raise FileNotFoundError(f"文本 JSON 路径不存在: {path_or_url}")
    with open(path_or_url, "r", encoding="utf-8") as f:
        return json.load(f)

# ------------- 模型加载 -------------

def ensure_model_loaded():
    """与服务端生命周期一致：若未加载则加载模型到 svc.model_globals。"""
    if svc.model_globals.get("model") is not None and svc.model_globals.get("tokenizer") is not None:
        return
    print("[MODEL] 正在加载模型 ...", file=sys.stderr)
    tokenizer, model, dtype = svc.load_model()  # 原文件定义
    svc.model_globals["tokenizer"] = tokenizer
    svc.model_globals["model"] = model
    svc.model_globals["dtype"] = dtype
    print("[MODEL] 模型加载完成", file=sys.stderr)

# ------------- 推理核心（简化一次性版本） -------------

def run_infer(rgb_path: str, ir_path: str, text: str, extra_content: Optional[str] = None, max_new_tokens: Optional[int] = None) -> str:
    """复用 svc.chat 简化：保证一次性输出纯文本 response。"""
    ensure_model_loaded()
    # svc.chat 接口签名: (image1_path, image2_path, text, extra_content=None, history=None, return_history=False)
    # 我们不维护多轮历史，这里 return_history=False
    if max_new_tokens is not None:
        # 设置临时环境变量影响 svc.chat 内部读取的 MAX_NEW_TOKENS
        os.environ["MAX_NEW_TOKENS"] = str(max_new_tokens)
    try:
        resp = svc.chat(rgb_path, ir_path, text, extra_content, history=None, return_history=False)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"模型推理失败: {e}") from e
    return resp

# ------------- 关系与一致性解析 -------------

def parse_relations_and_consistency(model_output: str):
    parsed = svc.check_label_re(model_output)
    overall_inference = svc.extract_overall_inference(model_output)
    rels = {tuple(sorted(r["entities"])): r["type"] for r in parsed.get("relationships", [])}
    final_key = tuple(sorted(['图片1', '图片2', '文本1']))
    pred_raw = rels.get(final_key)
    final_relation = svc.normalize_relation_name(pred_raw)
    return final_relation, overall_inference

# ------------- 图像获取（支持 URL） -------------

def _obtain_image(path_or_url: str, temp_dir: str, suffix: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        resp = requests.get(path_or_url, timeout=120)
        resp.raise_for_status()
        local_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{suffix}.jpg")
        with open(local_path, "wb") as f:
            f.write(resp.content)
        return local_path
    if not os.path.exists(path_or_url):
        raise FileNotFoundError(f"图像路径不存在: {path_or_url}")
    return path_or_url

# ------------- 主执行逻辑 -------------

def execute(mode: str, rgb_image_url: str, infrared_image_url: str, text_json_url: str, extra_content: Optional[str], max_new_tokens: Optional[int]):
    with tempfile.TemporaryDirectory() as temp_dir:
        rgb_path = _obtain_image(rgb_image_url, temp_dir, "rgb")
        ir_path = _obtain_image(infrared_image_url, temp_dir, "ir")
        text_json = _load_text_json(text_json_url)
        main_text = (text_json.get("text") or "").strip()
        if not main_text:
            raise ValueError("文本 JSON 缺少有效 'text' 字段")
        # 运行推理
        output_text = run_infer(rgb_path, ir_path, main_text, extra_content=extra_content, max_new_tokens=max_new_tokens)
        final_relation, consistency_result = parse_relations_and_consistency(output_text)
        is_conflict = (final_relation == "矛盾")
        if mode == "conflict":
            return {"conflict": bool(is_conflict), "final_relation": final_relation}
        if mode == "relation":
            return {"final_relation": final_relation}
        if mode == "full":
            return {"final_relation": final_relation, "consistency_result": consistency_result, "is_conflict": bool(is_conflict)}
        raise ValueError(f"未知 mode: {mode}")

# ------------- CLI -------------

def build_arg_parser():
    p = argparse.ArgumentParser(description="多模态关系分析 CLI (conflict | relation | full)")
    p.add_argument("mode", choices=["conflict", "relation", "full"], help="执行模式：conflict=是否矛盾, relation=总体关系, full=总体关系+一致性")
    p.add_argument("--rgb_image_url", required=True, help="RGB 图像本地路径或 URL")
    p.add_argument("--infrared_image_url", required=True, help="红外图像本地路径或 URL")
    p.add_argument("--text_json_url", required=True, help="文本 JSON 路径或 URL (需包含 text 字段)")
    p.add_argument("--extra_content", help="可选附加指令/用户内容")
    p.add_argument("--max_new_tokens", type=int, default=None, help="覆盖默认生成最大 token 数")
    p.add_argument("--pretty", action="store_true", help="美化输出 JSON")
    return p


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = execute(args.mode, args.rgb_image_url, args.infrared_image_url, args.text_json_url, args.extra_content, args.max_new_tokens)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": str(e)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":  # pragma: no cover
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
relation_cli_common.py

共享的命令行推理工具函数，供单模式 CLI 脚本复用。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid

import requests

try:
    import chat_tools_intern_multigpu as svc
except Exception as e:  # noqa: BLE001
    print(f"导入 chat_tools_intern_multigpu 失败: {e}", file=sys.stderr)
    raise

import torch  # noqa: F401  # 延迟导入以尽早暴露环境问题

__all__ = [
    "load_text_json",
    "ensure_model_loaded",
    "run_infer",
    "parse_relations_and_consistency",
    "obtain_image",
    "execute_mode",
]


# ------------- 辅助：下载 / 读取文本 JSON -------------

def load_text_json(path_or_url: str) -> dict:
    if path_or_url.startswith(("http://", "https://")):
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
    tokenizer, model, dtype = svc.load_model()
    svc.model_globals["tokenizer"] = tokenizer
    svc.model_globals["model"] = model
    svc.model_globals["dtype"] = dtype
    print("[MODEL] 模型加载完成", file=sys.stderr)


# ------------- 推理核心（简化一次性版本） -------------

def run_infer(
    rgb_path: str,
    ir_path: str,
    text: str,
) -> str:
    """复用 svc.chat 简化：保证一次性输出纯文本 response。"""
    ensure_model_loaded()
    try:
        resp = svc.chat(
            rgb_path,
            ir_path,
            text,
            history=None,
            return_history=False,
        )
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"模型推理失败: {e}") from e
    return resp


# ------------- 关系与一致性解析 -------------

def parse_relations_and_consistency(model_output: str):
    parsed = svc.check_label_re(model_output)
    overall_inference = svc.extract_overall_inference(model_output)

    rels = parsed.get("relationships", [])
    pair_relations: dict[str, str] = {}
    normalized_pairs: dict[str, str] = {}

    def normalize_entity(name: str) -> str:
        return name.replace("图片", "图像").strip()

    final_relation = None
    final_key = tuple(sorted([normalize_entity(n) for n in ("图像1", "图像2", "文本1")]))

    for entry in rels:
        entities = entry.get("entities") or []
        norm_entities = [normalize_entity(e) for e in entities]
        raw_type = entry.get("type")
        normalized = svc.normalize_relation_name(raw_type)
        if len(norm_entities) == 3 and tuple(sorted(norm_entities)) == final_key:
            final_relation = normalized
        elif len(norm_entities) == 2:
            key = "-".join(sorted(norm_entities))
            pair_relations[key] = raw_type
            normalized_pairs[key] = normalized or raw_type

    return final_relation, overall_inference, normalized_pairs


# ------------- 图像获取（支持 URL） -------------

def obtain_image(path_or_url: str, temp_dir: str, suffix: str) -> str:
    if path_or_url.startswith(("http://", "https://")):
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

def execute_mode(
    mode: str,
    rgb_image_url: str,
    infrared_image_url: str,
    text_json_url: str,
):
    with tempfile.TemporaryDirectory() as temp_dir:
        rgb_path = obtain_image(rgb_image_url, temp_dir, "rgb")
        ir_path = obtain_image(infrared_image_url, temp_dir, "ir")
        text_json = load_text_json(text_json_url)
        main_text = (text_json.get("text") or "").strip()
        if not main_text:
            raise ValueError("文本 JSON 缺少有效 'text' 字段")
        output_text = run_infer(rgb_path, ir_path, main_text)
        final_relation, consistency_result, pair_relations = parse_relations_and_consistency(output_text)
        is_conflict = final_relation == "矛盾"
        if mode == "conflict":
            return {
                "conflict": bool(is_conflict),
                "final_relation": final_relation,
                "pair_relations": pair_relations,
            }
        if mode == "relation":
            return {
                "final_relation": final_relation,
                "pair_relations": pair_relations,
            }
        if mode == "full":
            return {
                "final_relation": final_relation,
                "consistency_result": consistency_result,
                "is_conflict": bool(is_conflict),
                "pair_relations": pair_relations,
            }
        raise ValueError(f"未知 mode: {mode}")

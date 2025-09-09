import json
import os
import torch
from transformers import AutoProcessor, Glm4vForConditionalGeneration
from PIL import Image
import re
import csv
import asyncio
import httpx # 用于服务间API调用
import uvicorn # 用于启动服务
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from enum import Enum
from typing import Optional
import tempfile
import uuid

# ==============================================================================
# 1. 服务与模型配置
# ==============================================================================

# 从环境变量读取配置
gpu_id = os.getenv("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
MODEL_PATH = os.getenv("MODEL_PATH", "/path/to/your/GLM-4.1V-9B-Thinking") # !<-- 请务必修改为你的模型路径

# 全局变量,用于在服务启动时加载并持有模型
model_globals = {
    "processor": None,
    "model": None
}

# ==============================================================================
# 2. FastAPI 应用与 Pydantic 模型定义
# ==============================================================================

def load_model():
    """加载模型与处理器到全局变量"""
    print("开始加载模型...")
    if not os.path.isdir(MODEL_PATH):
        raise RuntimeError(f"模型路径不存在: {MODEL_PATH}。请检查 MODEL_PATH 环境变量或代码中的路径。")
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if not torch.cuda.is_available():
        dtype = torch.float32

    model = Glm4vForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    print("模型加载完成。")
    return processor, model

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI应用的生命周期事件,用于在启动时加载模型"""
    model_globals["processor"], model_globals["model"] = await run_in_threadpool(load_model)
    yield
    # 在这里可以添加服务关闭时需要清理的资源
    model_globals.clear()
    print("服务已关闭,资源已清理。")

app = FastAPI(
    title="多模态关系分析算法服务",
    description="集成GLM-4V模型,提供图像与文本关系分析,并根据文档回调指定接口。",
    version="1.1.0",
    lifespan=lifespan
)

# --- Pydantic 模型定义 ---
class UpdateDatasetBody(BaseModel):
    rgb_infrared_relation: str
    text_infrared_relation: str
    rgb_text_relation: str
    final_relation: str
    accuracy: float

class ProjectStatusEnum(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class UpdateProjectBody(BaseModel):
    status: ProjectStatusEnum
    infer_relation_accuracy: float
    consistency_cognition_accuracy: float
    equivalence_relationship_accuracy: float
    conflict_relationship_accuracy: float
    relation_accuracy: float

# ==============================================================================
# 3. 原始算法核心逻辑 (稍作修改以适应服务)
# ==============================================================================
prompt = (
"""# 角色与任务
你是一名专精于多模态信息分析的军事情报分析师。你的核心任务是精准分析给定的两张军事图像和一段军事文本之间的关系。

# 输入信息
- 图像1
- 图像2
- 文本1

# 执行步骤
1.  独立分析:
    - 图像 (每张): 识别主体（如:敌军士兵、坦克）、其明确的行动意图（如:持枪逮捕、道路行进）及场景。分析图像内各主体间的关系。
    - 文本: 提取核心要素,包括时间、地点、人物、事件等细节。

2.  配对关系分析:
    - 基于独立分析的结果,判断以下三组配对的具体逻辑关系:
    - 图像1 - 图像2
    - 图像1 - 文本1
    - 图像2 - 文本1
    - 每组关系必须从【等价、关联、因果、矛盾】四种类型中选择一种。

3.  总体关系判定:
    - 综合上述三组配对关系,对【图像1-图像2-文本1】给出一个总体的关系判定,同样从四种类型中选择。

4.  生成报告:
    - 根据下文定义的【输出格式】生成最终分析报告,报告必须包含所有分析结论和支撑理由。

# 关系定义

- 等价: 描述的核心事实、主体和事件完全相同,不存在任何一方对另一方信息的扩展。
  - 判定依据: 若三对关系均为等价,或其中两对为等价,则总体关系可判定为等价。
- 关联: 描述的核心事件相关,但在范围、细节或视角上存在差异。例如,一方描述“一辆坦克”,另一方描述“装甲部队”,后者范围更广。
  - 判定依据: 若有两对关系为关联,另外一对关系为关联/等价,则总体关系可判定为关联。
- 因果: 一方是原因,另一方是结果,存在明确的时间或逻辑先后顺序（比如行动与状态）。例如,文本描述了行动或者时间在前-“我方发起轰炸”,图像展示状态或者时间在后-“轰炸后的场景”。
  - 判定依据: 若有一对关系为因果,另外两对关系为因果/等价/关联,则总体关系可判定为因果。
- 矛盾: 描述的核心事实存在直接冲突。包括但不限于:
  - 数量冲突: 图像显示2辆坦克,文本称只有1辆。
  - 行为冲突: 图像显示敌军在行进,文本称未发现敌军踪迹。
  - 状态冲突: 图像显示坦克完好,文本称其已被击毁。
  - 判定依据: 若有一对关系为矛盾,另外两对关系为矛盾/等价/关联,则总体关系可判定为矛盾。

# 输出格式

【标准格式】
图像1-图像2关系:[关系类型]
图像1-文本1关系:[关系类型]
图像2-文本1关系:[关系类型]
图像1-图像2-文本1总体关系:[关系类型]

分析过程:
1.  信息描述:
    - 图像1内容:[对图像1的简洁描述]
    - 图像2内容:[对图像2的简洁描述]
    - 文本1内容:[对文本1的核心内容概括]
2.  关系论证:
    - [对三组配对关系和总体关系的详细分析和论证]

【特殊格式:当任意一对关系被判定为“矛盾”时】

图像1-图像2关系:[关系类型]
图像1-文本1关系:[关系类型]
图像2-文本1关系:[关系类型]

最相关联的两者是:[图像1和图像2 / 图像1和文本1 / 图像2和文本1]
信息相斥的模态是:[图像1 / 图像2 / 文本1]
综合事实推断:[基于最相关联的两者,得出的一个综合事实结论]

图像1-图像2-文本1总体关系:矛盾

分析过程:
1.  信息描述:
    - 图像1内容:[对图像1的简洁描述]
    - 图像2内容:[对图像2的简洁描述]
    - 文本1内容:[对文本1的核心内容概括]
2.  关系论证:
    - [详细分析为何存在矛盾,并论证为何某两者最相关,以及为何某个模态信息相斥]

# 核心分析准则

- 语义优先: 所有判断只关注内容和语义,完全忽略图像的色彩、风格、成像技术（如“可见光”与“热成像”）等表现形式。
- 敌方视角: 所有输入信息（图、文）均是对敌方情况的记录（包括士兵、军事装备、军事车辆等）。
- 同义词兼容: 在分析中要考虑到同义词或上下位词（例如:“坦克”与“装甲车”；“草地”与“植被”；“敌军”涵盖“敌军人员、坦克、装甲车等”）。
- 语气肯定: 所有分析和结论都必须使用肯定、明确的语气,严禁使用“可能”、“也许”、“或许”等不确定性词汇。
- 避免主体替换: 在判定三对关系中,要确定同一图像的主体一致的表示（例如,“坦克”与“军事工程车”）。
"""
)

def build_initial_messages(image1: Image.Image, image2: Image.Image, prompt_text: str, extra_text: str | None):
    messages_content = [
        {"type": "image", "image": image1},
        {"type": "image", "image": image2},
    ]
    if extra_text:
        # 将文本放在提示词之前,以符合某些模型的最佳实践
        messages_content.append({"type": "text", "text": f"文本1: {extra_text}\n\n{prompt_text}"})
    else:
        messages_content.append({"type": "text", "text": prompt_text})

    return [{"role": "user", "content": messages_content}]

def chat(image1_path: str, image2_path: str, text: str):
    """
    模型推理函数,接收文件路径和文本,返回模型的原始输出。
    """
    # 从全局变量获取模型
    processor = model_globals["processor"]
    model = model_globals["model"]

    if not processor or not model:
        raise RuntimeError("模型尚未加载。")

    image1 = Image.open(image1_path).convert("RGB")
    image2 = Image.open(image2_path).convert("RGB")

    messages = build_initial_messages(image1, image2, prompt, text)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            repetition_penalty=1.05,
        )
    
    new_tokens = generated_ids[0][inputs["input_ids"].shape[1]:]
    output_text = processor.decode(new_tokens, skip_special_tokens=True).strip()
    
    return output_text

def extract_first_paragraph_after_answer(s: str) -> str:
    m = re.search(r'<answer>(.*?)(?:</answer>|$)', s, flags=re.S)
    block = m.group(1).strip() if m else s.strip()
    if not block:
        return block
    p = re.search(r'分析过程', block)
    if p:
        return block[:p.start()].strip()
    parts = re.split(r'\n\s*\n', block, maxsplit=1)
    return parts[0].strip()

def check_label_re(response:str):
    first_para = extract_first_paragraph_after_answer(response)
    norm_text = first_para.replace('—', '-').replace('－', '-').replace('–', '-')
    ENTITY = r'(?:图像|图片|文本)\s*\d+'
    SEP = r'\s*-\s*'
    
    def _normalize_rel_type(t: str) -> str:
        t = (t or "").strip()
        t = re.sub(r'[。；;,,！!？?\s]+$', '', t)
        t = re.sub(r'关系$', '', t)
        synonyms = {
            "相同": "等价", "一致": "等价", "相符": "等价",
            "相关": "关联", "联系": "关联",
            "冲突": "矛盾", "相斥": "矛盾", "相悖": "矛盾", "相矛盾": "矛盾"
        }
        return synonyms.get(t, t)

    pair_colon_pattern = re.compile(rf'({ENTITY}){SEP}({ENTITY})\s*关系\s*[::]\s*([\u4e00-\u9fa5]+)')
    triple_colon_pattern = re.compile(rf'({ENTITY}){SEP}({ENTITY}){SEP}({ENTITY})\s*(?:三者关系|总体关系|总关系|整体关系)\s*[::]\s*([\u4e00-\u9fa5]+)')
    
    result = {"relationships": []}
    seen = set()

    def push_rel(ents: list[str], rtype: str):
        ents = [re.sub(r'\s+', '', e).replace("图像", "图片") for e in ents]
        rtype = _normalize_rel_type(rtype)
        # 排序以确保 ('图片1', '文本1') 和 ('文本1', '图片1') 是同一个键
        key = (tuple(sorted(ents)), rtype)
        if key in seen:
            return
        seen.add(key)
        result["relationships"].append({"entities": ents, "type": rtype})

    for a, b, rtype in pair_colon_pattern.findall(norm_text):
        push_rel([a, b], rtype)
    for a, b, c, rtype in triple_colon_pattern.findall(norm_text):
        push_rel([a, b, c], rtype)

    return result

# ==============================================================================
# 4. API 端点定义
# ==============================================================================

# --- A. 文档中定义的两个数据更新接口 ---

@app.put(
    "/v1/consistency/infer/{project_id}/{dataset_id}",
    summary="[回调接收] 更新单条数据数据集"
)
async def update_single_dataset(project_id: int, dataset_id: int, body: UpdateDatasetBody):
    """
    **作用**: (模拟)接收算法分析完单条数据后回调的结果,用于更新推导结果和准确率。
    """
    print(f"--- 接收到回调请求 ---")
    print(f"项目ID (project_id): {project_id}, 数据集ID (dataset_id): {dataset_id}")
    print(f"回调内容 (Body): {body.dict()}")
    print(f"--- 回调处理完毕 ---")
    
    # 在实际应用中,这里会执行数据库更新等操作
    
    return {"code": 200, "message": "success", "data": None}

@app.put(
    "/v1/consistency/project/{project_id}",
    summary="[回调接收] 更新任务结果"
)
async def update_project_status(project_id: int, body: UpdateProjectBody):
    """
    **作用**: (模拟)当所有测试数据集的一致性结果都跑完了后调用该接口来更新这一批数据的各项指标。
    """
    print(f"--- 接收到任务最终结果更新请求 ---")
    print(f"项目ID (project_id): {project_id}")
    print(f"回调内容 (Body): {body.dict()}")
    print(f"--- 任务更新处理完毕 ---")

    # 在实际应用中,这里会执行数据库更新等操作
    
    return {"code": 200, "message": "success", "data": None}


# --- B. 新增的、集成了算法的核心分析接口 ---

@app.post(
    "/v1/consistency/analyze_and_update",
    summary="[算法执行] 分析数据并自动回调更新接口"
)
async def analyze_consistency(
    project_id: int = Form(...),
    dataset_id: int = Form(...),
    rgb_image: UploadFile = Form(...),
    infrared_image: UploadFile = Form(...),
    text: str = Form(...)
):
    """
    **作用**: 接收RGB图像、红外图像和文本,调用GLM-4V模型进行关系分析,
    然后将分析结果自动回调给 `/v1/consistency/infer/{project_id}/{dataset_id}` 接口。
    """
    # 创建一个临时目录来存放上传的文件
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # 保存上传的图片到临时文件
            rgb_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{rgb_image.filename}")
            with open(rgb_path, "wb") as f:
                f.write(await rgb_image.read())

            ir_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{infrared_image.filename}")
            with open(ir_path, "wb") as f:
                f.write(await infrared_image.read())

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

        try:
            # 在线程池中运行耗时的模型推理
            print(f"开始为 project:{project_id} dataset:{dataset_id} 进行模型推理...")
            model_response = await run_in_threadpool(chat, rgb_path, ir_path, text)
            print(f"模型推理完成。原始输出: \n{model_response}")

            # 解析模型输出
            parsed_result = check_label_re(model_response)
            print(f"解析后的关系: {json.dumps(parsed_result, ensure_ascii=False)}")
            
            # 从解析结果中提取所需的关系,用于构建回调的body
            rels = {
                tuple(sorted(r["entities"])): r["type"]
                for r in parsed_result.get("relationships", [])
            }

            rgb_ir_key = tuple(sorted(['图片1', '图片2']))
            text_ir_key = tuple(sorted(['文本1', '图片2']))
            rgb_text_key = tuple(sorted(['图片1', '文本1']))
            final_key = tuple(sorted(['图片1', '图片2', '文本1']))

            # 准备回调数据
            update_data = UpdateDatasetBody(
                rgb_infrared_relation=rels.get(rgb_ir_key, "未知"),
                text_infrared_relation=rels.get(text_ir_key, "未知"),
                rgb_text_relation=rels.get(rgb_text_key, "未知"),
                final_relation=rels.get(final_key, "未知"),
                accuracy=1.0  # 准确率暂时设为1.0,因为单次推理无此概念
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"模型推理或结果解析失败: {e}")

    # 使用 httpx 异步调用回调接口
    # 注意: 这里的 host 和 port 需要根据你实际部署情况修改
    # 'http://127.0.0.1:8000' 是 uvicorn 默认的地址
    callback_url = f"http://127.0.0.1:8000/v1/consistency/infer/{project_id}/{dataset_id}"
    try:
        async with httpx.AsyncClient() as client:
            print(f"正在向 {callback_url} 发送回调...")
            response = await client.put(callback_url, json=update_data.dict())
            response.raise_for_status() # 如果状态码不是 2xx,则会抛出异常
    
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"回调失败: 无法连接到更新接口 at {e.request.url!r}.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"回调接口返回错误: {e.response.status_code} - {e.response.text}"
        )

    return {
        "code": 200,
        "message": "分析完成并已触发回调",
        "analysis_result": {
            "parsed_relations": parsed_result,
            "callback_data": update_data.dict(),
            "raw_model_output": model_response
        }
    }


# ==============================================================================
# 5. 服务启动入口
# ==============================================================================

if __name__ == "__main__":
    # 启动FastAPI应用
    # host="0.0.0.0" 使服务可以被局域网内其他机器访问
    # reload=True 会在代码变动时自动重启服务,方便开发调试
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
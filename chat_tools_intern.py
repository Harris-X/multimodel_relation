import json
import os
import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import re
import csv
import asyncio
import httpx
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from enum import Enum
from typing import Optional
import tempfile
import uuid
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from threading import RLock
from urllib.parse import urlparse

# ==============================================================================
# 1. 服务与模型配置
# ==============================================================================

# 从环境变量读取配置
# gpu_id = os.getenv("CUDA_VISIBLE_DEVICES", "3")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# 不要强行覆盖 CUDA_VISIBLE_DEVICES，交给部署环境配置
# 修改模型路径
MODEL_PATH = r"/home/user/xieqiuhao/multimodel_relation/downloaded_model/InternVL3_5-14B"
# 全局变量,用于在服务启动时加载并持有模型
model_globals = {
    "tokenizer": None,
    "model": None,
    "dtype": None,  # 记录模型期望的输入精度
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size: int):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file: str, input_size=448, max_num=12) -> torch.Tensor:
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# ==============================================================================
# 2. FastAPI 应用与 Pydantic 模型定义
# ==============================================================================

def load_model():
    """加载模型与处理器到全局变量"""
    print("开始加载模型...")
    if not os.path.isdir(MODEL_PATH):
        raise RuntimeError(f"模型路径不存在: {MODEL_PATH}。请检查 MODEL_PATH 环境变量或代码中的路径。")
    
    # 获取tokenizer（GitHub格式不需要 AutoProcessor）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
    
    # 确定使用的数据类型
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    # 加载 GitHub 格式模型，使用 .chat 接口
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    # 思考模式提示词
    # try:
    #     model.system_message = R1_SYSTEM_PROMPT
    # except Exception:
    #     pass
     
    print("模型加载完成。")
    return tokenizer, model, dtype

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI应用的生命周期事件,用于在启动时加载模型"""
    model_globals["tokenizer"], model_globals["model"], model_globals["dtype"] = await run_in_threadpool(load_model)
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
    rgb_infrared_relation: str # rgb 和红外的关系
    text_infrared_relation: str # 文本 和红外的关系
    rgb_text_relation: str # rgb 和文本的关系
    final_relation: str # 最终关系
    accuracy: float # 检测准确率 1/0 比对 label 标签
    consistency_result: str # 一致性认知结果, 手动创建
    consistency_result_accuracy: float # 一致性认知结果准确率, 手动创建

class ProjectStatusEnum(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class UpdateProjectBody(BaseModel):
    status: ProjectStatusEnum
    infer_relation_accuracy: float # 总体推演关系准确率
    consistency_cognition_accuracy: float # 一致性认知准确率
    equivalence_relationship_accuracy: float # 等价关系准确率
    conflict_relationship_accuracy: float # 矛盾关系准确率
    causation_relationship_accuracy: float # 因果关系准确率
    relation_accuracy: float # 关联关系准确率

# ---------------- 新增：批量请求体定义 ----------------
class DatasetItem(BaseModel):
    dataset_id: str
    rgb_image_url: str
    infrared_image_url: str
    text_url: str  # 指向 JSON，内部含 key='text' 和 'label'

class BatchInferBody(BaseModel):
    project_id: Optional[int] = None
    datasets: list[DatasetItem]

# ---------------- 新增：关系归一化工具 ----------------
_REL_SYNONYMS = {
    "相同": "等价", "一致": "等价", "相符": "等价",
    "相关": "关联", "联系": "关联",
    "冲突": "矛盾", "相斥": "矛盾", "相悖": "矛盾", "相矛盾": "矛盾",
    "因果关系": "因果"
}
_VALID_REL = {"等价", "关联", "因果", "矛盾"}

def normalize_relation_name(name: str) -> Optional[str]:
    if not name:
        return None
    s = re.sub(r'[。；;,,！!？?\s]+$', '', name.strip())
    s = re.sub(r'关系$', '', s)
    s = _REL_SYNONYMS.get(s, s)
    return s if s in _VALID_REL else None

def is_http_url(s: Optional[str]) -> bool:
    try:
        return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))
    except Exception:
        return False

# ==============================================================================
# 3. 原始算法核心逻辑 (稍作修改以适应服务)
# ==============================================================================
prompt = (
"""# 角色与任务
你是一名专精于多模态信息分析的军事情报分析师。你的核心任务是精准分析给定的两张军事图像和一段军事文本之间的关系。

# 输入信息
- 图像1(RGB图像)
- 图像2(红外图像)
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
    - 每组关系必须从【等价、关联、因果、矛盾】四种关系类型中选择一种。

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
  - 状态冲突: 图像显示坦克完好,文本称其已被击毁。或者图像显示坦克冒烟行进,文本称工程车在灭火。
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

图像1-图像2-文本1总体关系:矛盾
综合事实推断:[描述最相关联的两者给出的事实]
最相关联的两者是:[图像1和图像2 / 图像1和文本1 / 图像2和文本1]
信息相斥的模态是:[图像1 / 图像2 / 文本1]


分析过程:
1.  信息描述:
    - 图像1内容:[对图像1的简洁描述]
    - 图像2内容:[对图像2的简洁描述]
    - 文本1内容:[对文本1的核心内容概括]
2.  关系论证:
    - [简洁分析为何存在矛盾,并论证为何某两者最相关,以及为何某个模态信息相斥]

# 核心分析准则

- 语义优先: 所有判断只关注内容和语义,完全忽略图像的色彩、风格、成像技术（如“可见光”与“热成像”）等表现形式。
- 敌方视角: 所有输入信息（图、文）均是对敌方情况的记录（包括士兵、军事装备、军事车辆等）。
- 同义词兼容: 在分析中要考虑到同义词或上下位词（例如:“坦克”与“装甲车”；“草地”与“植被”；“敌军”涵盖“敌军人员、坦克、装甲车等”）。
- 语气肯定: 所有分析和结论都必须使用肯定、明确的语气,严禁使用“可能”、“也许”、“或许”、“似乎”、“貌似”、“好像”等不确定性词汇。
- 避免主体替换: 在判定三对关系中,要确定同一图像的主体一致的表示（例如,“坦克”与“军事工程车”）。
- 细节支持: 在论证过程中,要尽量使用具体细节（如数量、行为、状态等）来支持关系判定（如“坦克冒烟行进”与“战场没有坦克”）,而非笼统描述。
- RGB-红外图片主体一致: 假设两张图片的主体是一致的,即它们描述的是同一场景或事件的不同视角或时间点。红外图片会将主体呈现红黄色是因为主体有热量，而不是因为着火。
"""
)

# 添加思考模式的系统提示词
R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()

def build_initial_messages(instruction_prompt: str, user_text: str | None):
    # 与 README 保持一致：使用 Image-1/2 标注
    content = f"Image-1: <image>\nImage-2: <image>\n"
    if user_text:
        content += f"Text-1: {user_text}\n\n"
    content += instruction_prompt
    return content

def chat(image1_path: str, image2_path: str, text: str):
    """
    模型推理函数,接收文件路径和文本,返回模型的原始输出。
    """
    # 从全局变量获取模型
    tokenizer = model_globals["tokenizer"]
    model = model_globals["model"]
    in_dtype = model_globals.get("dtype", torch.bfloat16)

    if not tokenizer or not model:
        raise RuntimeError("模型尚未加载。")

    # 载入并切片两张图像
    pixel_values1 = load_image(image1_path, max_num=12)
    pixel_values2 = load_image(image2_path, max_num=12)
    num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0).to(in_dtype)
    if torch.cuda.is_available():
        pixel_values = pixel_values.cuda(non_blocking=True)

    # 组装问题串（两个 <image> 必须与 num_patches_list 对齐）
    question = build_initial_messages(prompt, text)
    # print(f"[DEBUG] num_patches_list={num_patches_list}")
    # print(f"[DEBUG] question preview:\n{question[:300]}")

    generation_config = dict(
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.6,
        repetition_penalty=1.05
    )

    # 使用 InternVL 的 .chat
    response, _ = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=question,
        generation_config=generation_config,
        num_patches_list=num_patches_list,
        history=None,
        return_history=True
    )
    return response

def extract_first_paragraph_after_answer(s: str) -> str:
    # 思考模式会产生 <think>...</think> 块，答案在后面
    # 使用 re.S 使 '.' 可以匹配换行符
    # (?:...) 是一个非捕获组
    m = re.search(r'(?:</think>)?(.*)', s, flags=re.S)
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


# 新增: 提取“综合事实推断”内容
def extract_overall_inference(text: str) -> Optional[str]:
    """
    从模型输出中抽取 '综合事实推断:' 后的单行内容。
    若未找到返回 None。
    """
    # 允许前面有全角/半角冒号
    m = re.search(r'综合事实推断[:：]\s*(.+)', text)
    if m:
        # 截断到该行结束（去掉后续可能的附加说明）
        return m.group(1).strip()
    return None

# ==============================================================================
# 4. API 端点定义
# ==============================================================================

# --- A. 持久化配置与辅助函数 ---

# 定义持久化文件名
DATASET_RESULTS_FILE = "dataset_results.csv"
PROJECT_RESULTS_FILE = "project_results.csv"
# 文件操作锁
file_lock = RLock()

def _update_or_append_csv(filepath: str, header: list[str], new_row: dict, key_fields: list[str]):
    """
    一个线程安全的函数,用于更新或追加CSV行。
    如果文件或表头不正确,则会重新创建。
    """
    with file_lock:
        rows = []
        file_exists = os.path.exists(filepath)
        header_correct = False

        if file_exists:
            try:
                with open(filepath, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    # 检查表头
                    current_header = next(reader)
                    if current_header == header:
                        header_correct = True
                        # 读取所有行到内存
                        rows = list(csv.DictReader(open(filepath, 'r', newline='', encoding='utf-8')))
            except (StopIteration, FileNotFoundError, Exception):
                 # 文件为空或损坏,标记为需要重写
                header_correct = False

        # 如果文件不存在或表头不匹配,则准备重写
        if not file_exists or not header_correct:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
            rows = [] # 清空旧数据

        # 查找是否需要更新
        key_to_find = {k: str(new_row[k]) for k in key_fields}
        found_and_updated = False
        for i, row in enumerate(rows):
            # 检查主键是否匹配
            if all(row.get(k) == key_to_find[k] for k in key_fields):
                rows[i].update(new_row)
                found_and_updated = True
                break
        
        # 如果没有找到,则追加
        if not found_and_updated:
            rows.append(new_row)

        # 将更新后的所有内容写回文件
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)


# --- B. 文档中定义的两个数据更新接口 ---

@app.put(
    "/v1/consistency/infer/{project_id}/{dataset_id}",
    summary="[回调接收] 更新单条数据数据集"
)
async def update_single_dataset(project_id: int, dataset_id: int, body: UpdateDatasetBody):
    """
    接收算法分析回调结果并写入CSV文件。
    """
    print(f"--- 接收到回调请求 ---")
    print(f"项目ID: {project_id}, 数据集ID: {dataset_id}")
    print(f"回调内容: {body.dict()}")
    
    header = ["project_id", "dataset_id"] + list(UpdateDatasetBody.__annotations__.keys())
    new_row = {"project_id": project_id, "dataset_id": dataset_id, **body.dict()}
    
    await run_in_threadpool(_update_or_append_csv, DATASET_RESULTS_FILE, header, new_row, ["project_id", "dataset_id"])
    
    return {"code": 200, "message": "success", "data": body.dict()}

@app.get(
    "/v1/consistency/infer/{project_id}/{dataset_id}",
    summary="[查询] 获取单条数据最新分析结果"
)
async def get_single_dataset(project_id: int, dataset_id: int):
    with file_lock:
        if not os.path.exists(DATASET_RESULTS_FILE):
            raise HTTPException(status_code=404, detail="结果文件不存在。")
        
        with open(DATASET_RESULTS_FILE, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("project_id") == str(project_id) and row.get("dataset_id") == str(dataset_id):
                    return {"code": 200, "message": "success", "result": row}
    
    raise HTTPException(status_code=404, detail="结果未找到。")

@app.put(
    "/v1/consistency/infer/{project_id}",
    summary="[回调接收] 更新项目级聚合结果"
)
async def update_project_summary(project_id: int, body: UpdateProjectBody):
    """
    接收项目级（汇总统计）回调结果并写入CSV文件。
    """
    print(f"--- 接收到项目级回调 ---")
    print(f"项目ID: {project_id}")
    print(f"项目汇总内容: {body.dict()}")

    header = ["project_id"] + list(UpdateProjectBody.__annotations__.keys())
    new_row = {"project_id": project_id, **body.dict()}

    await run_in_threadpool(_update_or_append_csv, PROJECT_RESULTS_FILE, header, new_row, ["project_id"])

    return {"code": 200, "message": "success", "data": new_row}

@app.get(
    "/v1/consistency/infer/{project_id}",
    summary="[查询] 获取项目下全部结果（含项目级汇总）"
)
async def list_project_datasets(project_id: int):
    with file_lock:
        # 读取项目汇总
        project_summary = None
        if os.path.exists(PROJECT_RESULTS_FILE):
            with open(PROJECT_RESULTS_FILE, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("project_id") == str(project_id):
                        project_summary = row
                        break
        
        # 读取该项目下的所有数据集结果
        items = []
        if os.path.exists(DATASET_RESULTS_FILE):
            with open(DATASET_RESULTS_FILE, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("project_id") == str(project_id):
                        items.append(row)

    return {
        "code": 200,
        "project_id": project_id,
        "project_summary": project_summary,
        "count": len(items),
        "results": items
    }

# --- C. 新增的、集成了算法的核心分析接口 ---

@app.post(
    "/v1/consistency/analyze_and_update",
    summary="[算法执行] 分析数据并自动回调更新接口"
)
async def analyze_consistency(
    project_id: int = Form(...),
    dataset_id: int = Form(...),

    # 现在既支持 http(s) URL 也支持服务器本地路径
    rgb_image_url: Optional[str] = Form(None),
    infrared_image_url: Optional[str] = Form(None),
    text_json_url: Optional[str] = Form(None),

    # 兼容：文件直传 + 原始文本
    rgb_image: Optional[UploadFile] = None,
    infrared_image: Optional[UploadFile] = None,
    text: Optional[str] = Form(None),
):
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # RGB
                if rgb_image_url:
                    if is_http_url(rgb_image_url):
                        resp = await client.get(rgb_image_url); resp.raise_for_status()
                        rgb_path = os.path.join(temp_dir, f"{uuid.uuid4()}_rgb.jpg")
                        with open(rgb_path, "wb") as f: f.write(resp.content)
                    else:
                        if not os.path.exists(rgb_image_url):
                            raise HTTPException(status_code=400, detail=f"RGB图像路径不存在: {rgb_image_url}")
                        rgb_path = rgb_image_url
                elif rgb_image is not None:
                    rgb_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{rgb_image.filename}")
                    with open(rgb_path, "wb") as f: f.write(await rgb_image.read())
                else:
                    raise HTTPException(status_code=400, detail="缺少 RGB 图像：请提供 rgb_image_url 或上传 rgb_image 文件")

                # IR
                if infrared_image_url:
                    if is_http_url(infrared_image_url):
                        resp = await client.get(infrared_image_url); resp.raise_for_status()
                        ir_path = os.path.join(temp_dir, f"{uuid.uuid4()}_ir.jpg")
                        with open(ir_path, "wb") as f: f.write(resp.content)
                    else:
                        if not os.path.exists(infrared_image_url):
                            raise HTTPException(status_code=400, detail=f"红外图像路径不存在: {infrared_image_url}")
                        ir_path = infrared_image_url
                elif infrared_image is not None:
                    ir_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{infrared_image.filename}")
                    with open(ir_path, "wb") as f: f.write(await infrared_image.read())
                else:
                    raise HTTPException(status_code=400, detail="缺少红外图像：请提供 infrared_image_url 或上传 infrared_image 文件")

                # 文本
                if text_json_url:
                    if is_http_url(text_json_url):
                        t_resp = await client.get(text_json_url); t_resp.raise_for_status()
                        try:
                            j = t_resp.json()
                        except Exception:
                            j = {"text": t_resp.text}
                    else:
                        if not os.path.exists(text_json_url):
                            raise HTTPException(status_code=400, detail=f"文本JSON路径不存在: {text_json_url}")
                        with open(text_json_url, "r", encoding="utf-8") as jf:
                            j = json.load(jf)
                    final_text = (j.get("text") or "").strip()
                    if not final_text:
                        raise HTTPException(status_code=400, detail=f"文本JSON中未找到有效的 'text' 字段")
                elif text is not None:
                    final_text = text
                else:
                    raise HTTPException(status_code=400, detail="缺少文本：请提供 text_json_url 或 text 原文")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"输入处理失败: {e}")

        try:
            # 在线程池中运行耗时的模型推理
            print(f"开始为 project:{project_id} dataset:{dataset_id} 进行模型推理...")
            model_response = await run_in_threadpool(chat, rgb_path, ir_path, final_text)
            print(f"模型推理完成。原始输出: \n{model_response}")

            # 解析模型输出
            parsed_result = check_label_re(model_response)
            overall_inference = extract_overall_inference(model_response)

            rels = {
                tuple(sorted(r["entities"])): r["type"]
                for r in parsed_result.get("relationships", [])
            }

            rgb_ir_key = tuple(sorted(['图片1', '图片2']))
            text_ir_key = tuple(sorted(['文本1', '图片2']))
            rgb_text_key = tuple(sorted(['图片1', '文本1']))
            final_key = tuple(sorted(['图片1', '图片2', '文本1']))

            consistency_result_value = overall_inference

            update_data = UpdateDatasetBody(
                rgb_infrared_relation=rels.get(rgb_ir_key, "未知"),
                text_infrared_relation=rels.get(text_ir_key, "未知"),
                rgb_text_relation=rels.get(rgb_text_key, "未知"),
                final_relation=rels.get(final_key, "未知"),
                accuracy=1.0,
                consistency_result=consistency_result_value,
                consistency_result_accuracy=1.0
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"模型推理或结果解析失败: {e}")

    callback_url = f"http://127.0.0.1:8000/v1/consistency/infer/{project_id}/{dataset_id}"
    try:
        async with httpx.AsyncClient() as client:
            print(f"正在向 {callback_url} 发送回调...")
            response = await client.put(callback_url, json=update_data.dict())
            response.raise_for_status()
            callback_resp_json = response.json()
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
        "result": {
            "parsed_relations": parsed_result,
            "overall_inference": overall_inference,
            "callback_data": update_data.dict(),
            "raw_model_output": model_response,
            "callback_response": callback_resp_json
        }
    }

@app.post(
    "/v1/consistency/batch_analyze",
    summary="[算法执行] 批量分析项目数据并写入项目级汇总/样本结果"
)
async def batch_infer_project(project_id: int, body: BatchInferBody):
    if body.project_id is not None and int(body.project_id) != int(project_id):
        raise HTTPException(status_code=400, detail="路径中的 project_id 与 body 不一致。")

    cls_total = {c: 0 for c in _VALID_REL}
    cls_correct = {c: 0 for c in _VALID_REL}
    per_item_results, errors = [], []

    async with httpx.AsyncClient(timeout=60.0) as client:
        with tempfile.TemporaryDirectory() as temp_dir:
            for item in body.datasets:
                dsid = str(item.dataset_id)
                try:
                    # RGB
                    if is_http_url(item.rgb_image_url):
                        r = await client.get(item.rgb_image_url); r.raise_for_status()
                        rgb_path = os.path.join(temp_dir, f"{uuid.uuid4()}_rgb.jpg")
                        with open(rgb_path, "wb") as f: f.write(r.content)
                    else:
                        if not os.path.exists(item.rgb_image_url):
                            raise HTTPException(status_code=400, detail=f"{dsid}: RGB路径不存在: {item.rgb_image_url}")
                        rgb_path = item.rgb_image_url

                    # IR
                    if is_http_url(item.infrared_image_url):
                        r = await client.get(item.infrared_image_url); r.raise_for_status()
                        ir_path = os.path.join(temp_dir, f"{uuid.uuid4()}_ir.jpg")
                        with open(ir_path, "wb") as f: f.write(r.content)
                    else:
                        if not os.path.exists(item.infrared_image_url):
                            raise HTTPException(status_code=400, detail=f"{dsid}: 红外路径不存在: {item.infrared_image_url}")
                        ir_path = item.infrared_image_url

                    # 文本 JSON
                    if is_http_url(item.text_url):
                        t = await client.get(item.text_url); t.raise_for_status()
                        try:
                            j = t.json()
                        except Exception:
                            j = {"text": t.text}
                    else:
                        if not os.path.exists(item.text_url):
                            raise HTTPException(status_code=400, detail=f"{dsid}: 文本JSON路径不存在: {item.text_url}")
                        with open(item.text_url, "r", encoding="utf-8") as jf:
                            j = json.load(jf)
                    text = (j.get("text") or "").strip()
                    label_raw = (j.get("label") or "").strip()
                    label = normalize_relation_name(label_raw)
                    if not text:
                        raise HTTPException(status_code=400, detail=f"{dsid}: 文本JSON缺少有效 'text'")

                    # 推理
                    model_output = await run_in_threadpool(chat, rgb_path, ir_path, text)

                    # 解析
                    parsed = check_label_re(model_output)
                    rels = {tuple(sorted(r["entities"])): r["type"] for r in parsed.get("relationships", [])}
                    final_key = tuple(sorted(['图片1', '图片2', '文本1']))
                    pred_raw = rels.get(final_key)
                    pred = normalize_relation_name(pred_raw)

                    # 与 analyze_consistency 一致：提取“综合事实推断”文本作为一致性结果
                    overall_inference = extract_overall_inference(model_output)

                    # 计算准确率
                    sample_acc = 0.0
                    if label in _VALID_REL and pred in _VALID_REL:
                        cls_total[label] += 1
                        if pred == label:
                            cls_correct[label] += 1
                            sample_acc = 1.0

                    # 写入样本 CSV
                    update_row = UpdateDatasetBody(
                        rgb_infrared_relation=rels.get(tuple(sorted(['图片1', '图片2'])), "未知"),
                        text_infrared_relation=rels.get(tuple(sorted(['文本1', '图片2'])), "未知"),
                        rgb_text_relation=rels.get(tuple(sorted(['图片1', '文本1'])), "未知"),
                        final_relation=pred or (pred_raw or "未知"),
                        accuracy=sample_acc,
                        consistency_result=overall_inference or (pred or (pred_raw or "未知")),
                        consistency_result_accuracy=1.0
                    )
                    header = ["project_id", "dataset_id"] + list(UpdateDatasetBody.__annotations__.keys())
                    new_row = {"project_id": project_id, "dataset_id": dsid, **update_row.dict()}
                    await run_in_threadpool(_update_or_append_csv, DATASET_RESULTS_FILE, header, new_row, ["project_id", "dataset_id"])

                    per_item_results.append({"dataset_id": dsid, "label": label or label_raw, "pred": pred or pred_raw or "未知", "accuracy": sample_acc})
                except Exception as e:
                    errors.append({"dataset_id": dsid, "error": str(e)})

    # 统计项目级
    per_class_acc = {c: (cls_correct[c] / cls_total[c]) if cls_total[c] > 0 else None for c in _VALID_REL}
    valid_accs = [v for v in per_class_acc.values() if v is not None]
    infer_relation_accuracy = sum(valid_accs) / len(valid_accs) if valid_accs else 0.0

    # 一致性认知准确率：按你的要求先置为 1
    consistency_cognition_accuracy = 1.0

    # 写入项目级 CSV
    proj_body = UpdateProjectBody(
        status=ProjectStatusEnum.COMPLETED,
        infer_relation_accuracy=infer_relation_accuracy,
        consistency_cognition_accuracy=consistency_cognition_accuracy,
        equivalence_relationship_accuracy=per_class_acc["等价"] or 0.0,
        conflict_relationship_accuracy=per_class_acc["矛盾"] or 0.0,
        causation_relationship_accuracy=per_class_acc["因果"] or 0.0,
        relation_accuracy=per_class_acc["关联"] or 0.0
    )
    proj_header = ["project_id"] + list(UpdateProjectBody.__annotations__.keys())
    proj_row = {"project_id": project_id, **proj_body.dict()}

    # 新增：通过 HTTP PUT 回调项目级接口
    callback_url = f"http://127.0.0.1:8000/v1/consistency/infer/{project_id}"
    async with httpx.AsyncClient() as client:
        resp = await client.put(callback_url, json=proj_body.dict())
        resp.raise_for_status()
        callback_resp_json = resp.json()

    # 返回项目级结果 + 每条样本简要
    return {
        "code": 200,
        "message": "批量分析完成，结果已写入项目与样本CSV。",
        "project_summary": proj_row,
        "items": per_item_results,
        "errors": errors,
        "project_callback_response": callback_resp_json  # 可选：返回回调响应
    }


# ==============================================================================
# 5. 服务启动入口
# ==============================================================================

if __name__ == "__main__":
    uvicorn.run("chat_tools_intern:app", host="0.0.0.0", port=8000, reload=True)
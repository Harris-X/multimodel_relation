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
base_url = "http://10.13.31.103:7000"
# 从环境变量读取配置（不要在代码里强行覆盖 CUDA_VISIBLE_DEVICES）
# 可在启动前导出:  export CUDA_VISIBLE_DEVICES=0,1,2
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # <- 删除这类硬编码
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
    """加载模型与处理器到全局变量（多卡自动分片）"""
    print("开始加载模型...")
    if not os.path.isdir(MODEL_PATH):
        raise RuntimeError(f"模型路径不存在: {MODEL_PATH}。请检查 MODEL_PATH 环境变量或代码中的路径。")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)

    # dtype 选择
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    # 多卡配置：device_map 与 max_memory
    device_map = os.environ.get("HF_DEVICE_MAP", "balanced")  # 强制均衡分片；需要时可改回 auto
    # 每卡可用显存比例，默认 0.9，可通过 GPU_MEM_FRACTION=0.85 覆盖
    mem_fraction = float(os.environ.get("GPU_MEM_FRACTION", "1.0"))
    max_memory = None
    if torch.cuda.is_available():
        max_memory = {}
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory // (1024 * 1024)  # MiB
            cap = int(total * mem_fraction)
            max_memory[i] = f"{cap}MiB"
        print(f"Detected {torch.cuda.device_count()} GPUs, max_memory per GPU: {max_memory}")

    # 使用 transformers(集成 accelerate) 的多 GPU 自动分片
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,  # 关闭低内存模式以支持多卡
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map,        # 关键：自动分片
        max_memory=max_memory         # 关键：限制每卡占用
    ).eval()
    # 打印分片结果，便于确认使用了哪些 GPU
    try:
        print("hf_device_map:", getattr(model, "hf_device_map", None))
    except Exception:
        pass
    # 注意：不要再调用 model.cuda()，否则会把模型挪到单卡
    print("模型加载完成（多GPU）")
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
    final_relation: str # 最终关系 # 缺乏一个labal
    actual_relation: str  # 新增：标准答案标签
    accuracy: float # 检测准确率 1/0 比对 label 标签
    consistency_result: str # 一致性认知结果, 手动创建
    consistency_result_accuracy: float # 一致性认知结果准确率, 手动创建
    raw_model_output: Optional[str] = None  # 新增：模型原始输出（可选）

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
    dataset_id: int
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
prompt = ("""
#任务 
作为一名多模态军事情报分析师，分析给定的敌方的RGB图像（图像1）、红外图像（图像2）和文本（文本1）之间的关系，图片和文本均为敌方的信息（图片中的人均为敌方士兵，坦克和装甲车均为敌方坦克和装甲车）。

# 分析逻辑与关系类型

1.  **独立分析**: 分别提取每张图像（主体、行为、场景）和文本（时间、地点、人物、事件）的核心信息。
2.  **配对分析**: 判断三组配对（图像1-图像2、图像1-文本1、图像2-文本1）的关系，必须从以下四种类型中选择：
      * **等价**: 核心事实、主体和事件完全相同。
      * **关联**: 核心事件相关，但在范围、细节或视角上存在差异。
      * **因果**: 一方是原因（如行动），另一方是结果（如状态），或者反过来。比如，由于敌方森林基地被袭击，所以派出士兵在森林巡逻，是因果关系；由于导弹袭击了坦克，坦克爆炸，是因果关系。
      * **矛盾**: 核心事实存在直接冲突（如数量、行为、状态）。
3.  **总体判定**: 综合配对分析结果，给出【图像1-图像2-文本1】的总体关系。判定遵循**最高优先级原则：因果> 矛盾 > 关联 > 等价**。 注意：若三组关系中出现一组因果，则可以直接据此判定总体关系为因果。

# 分析准则
  - **内容优先**: 忽略模态差异（如RGB与红外成像），只关注语义内容。假定两张图片描述的是同一场景。红外图像中的红黄色代表热量，不代表着火。
  - **主体一致**: 分析时需兼容同义词（如“坦克”与“装甲车”），但不能混淆不同主体（如“坦克”与“工程车”）。所有信息均针对敌方。
  - **论证严谨**: 结论必须明确，禁止使用“可能”、“似乎”等模糊词汇。论证需引用数量、行为等具体细节支撑。

# 输出格式
严格按照以下格式输出。

**【标准格式】(无矛盾时)**

图像1-图像2关系:[关系类型]
图像1-文本1关系:[关系类型]
图像2-文本1关系:[关系类型]
图像1-图像2-文本1总体关系:[关系类型]
综合事实推断:[描述三者给出的事实判断]
          
分析过程:
1. 信息描述:
   - 图像1内容:[图像1的简洁描述]
   - 图像2内容:[图像2的简洁描述]
   - 文本1内容:[文本1的核心内容概括]
2. 关系论证:
   - [对三组配对关系和总体关系的详细分析和论证]
   - [分析中出现原因、结果，或者可能原因、可能结果的，直接判定为因果关系，不需要特别直接、明确的因果关系]

**【特殊格式】(存在矛盾时)**

图像1-图像2关系:[关系类型]
图像1-文本1关系:[关系类型]
图像2-文本1关系:[关系类型]

图像1-图像2-文本1总体关系:矛盾
综合事实推断:[描述最相关联的两者给出的事实]
最相关联的两者是:[图像1和图像2 / 图像1和文本1 / 图像2和文本1]
信息相斥的模态是:[图像1 / 图像2 / 文本1]

分析过程:
1. 信息描述:
   - 图像1内容:[图像1的简洁描述]
   - 图像2内容:[图像2的简洁描述]
   - 文本1内容:[文本1的核心内容概括]
2. 关系论证:
   - [简洁分析为何存在矛盾,并论证为何某两者最相关,以及为何某个模态信息相斥]
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

def _first_cuda_device_from_hf_map(model) -> torch.device:
    # 从 hf_device_map 找到第一块 CUDA 设备，避免固定用 cuda:0
    devs = getattr(model, "hf_device_map", None)
    if isinstance(devs, dict):
        for v in devs.values():
            if isinstance(v, str) and v.startswith("cuda:"):
                return torch.device(v)
            if isinstance(v, int):
                return torch.device(f"cuda:{v}")
    # 回退
    return torch.device("cuda:0")

def chat(image1_path: str, image2_path: str, text: str):
    tokenizer = model_globals["tokenizer"]
    model = model_globals["model"]
    in_dtype = model_globals.get("dtype", torch.bfloat16)
    if not tokenizer or not model:
        raise RuntimeError("模型尚未加载。")

    pixel_values = None
    try:
        pixel_values1 = load_image(image1_path, max_num=8)   # 可酌情降为 8/6 以减小激活
        pixel_values2 = load_image(image2_path, max_num=6)
        num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
        pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0).to(in_dtype)

        if torch.cuda.is_available():
            dev0 = _first_cuda_device_from_hf_map(model)
            # 把输入直接放到模型首块 GPU，避免先到 cuda:0 再搬运
            pixel_values = pixel_values.to(dev0, non_blocking=True)

        question = build_initial_messages(prompt, text)
        generation_config = dict(
            max_new_tokens=512,   # 适度调低生成长度可明显降显存峰值
            do_sample=False,
            temperature=0.6,
            repetition_penalty=1.05
        )

        with torch.inference_mode():
            chat_ret = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False   # 关闭历史返回，减少内存保留
            )
        # 兼容返回签名：可能是 str 或 (str, history)
        if isinstance(chat_ret, tuple):
            response = chat_ret[0]
        else:
            response = chat_ret
        return response
    finally:
        try:
            del pixel_values1, pixel_values2
        except Exception:
            pass
        if pixel_values is not None:
            del pixel_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

def _serialize_for_csv(value: Optional[str]) -> str:
    """
    将长文本安全序列化为单行，以便写入CSV:
    - 换行 -> \n
    - 回车 -> \n
    - 制表 -> \t
    - 其余保持原样
    """
    if value is None:
        return ""
    s = str(value)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", "\\n")
    s = s.replace("\t", "\\t")
    return s

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

# 从本地 CSV 读取指定项目与数据集的已存在结果，转换成 UpdateDatasetBody
def _load_existing_update_rows(project_id: int, dataset_ids: list[str]) -> dict[str, UpdateDatasetBody]:
    result: dict[str, UpdateDatasetBody] = {}
    if not os.path.exists(DATASET_RESULTS_FILE):
        return result
    wanted = set(dataset_ids)
    with file_lock:
        try:
            with open(DATASET_RESULTS_FILE, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # if row.get("project_id") != str(project_id):
                    #     continue
                    dsid = row.get("dataset_id")
                    if dsid not in wanted:
                        continue
                    # 构建 UpdateDatasetBody
                    body = UpdateDatasetBody(
                        rgb_infrared_relation=row.get("rgb_infrared_relation", "未知"),
                        text_infrared_relation=row.get("text_infrared_relation", "未知"),
                        rgb_text_relation=row.get("rgb_text_relation", "未知"),
                        final_relation=row.get("final_relation", "未知"),
                        actual_relation=row.get("actual_relation", "未知"),
                        accuracy=row.get("accuracy", 0),
                        consistency_result=row.get("consistency_result", "None"),
                        consistency_result_accuracy=row.get("consistency_result_accuracy", 0),
                        raw_model_output=row.get("raw_model_output")
                    )
                    result[dsid] = body
        except Exception:
            # 读取异常则视为无缓存
            return {}
    return result

# --- B. 文档中定义的两个数据更新接口 ---

@app.put(
    "/v1/consistency/infer/{project_id}/{dataset_id}",
    summary="[回调接收] 模拟更新单条数据数据集"
)
async def update_single_dataset(project_id: int, dataset_id: int, body: UpdateDatasetBody):
    """
    接收算法分析回调结果并写入CSV文件。
    """
    print(f"--- 接收到回调请求 ---")
    print(f"项目ID: {project_id}, 数据集ID: {dataset_id}")
    print(f"回调内容: {body.dict()}")
     
    header = ["project_id", "dataset_id"] + list(UpdateDatasetBody.__annotations__.keys())
    body_dict = body.dict()
    # 序列化原始输出为单行，避免CSV换行干扰
    if "raw_model_output" in body_dict:
        body_dict["raw_model_output"] = _serialize_for_csv(body_dict.get("raw_model_output"))
    new_row = {"project_id": project_id, "dataset_id": dataset_id, **body_dict}
     
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
    summary="[回调接收] 模拟更新项目级聚合结果"
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
                    consistency_result_label = (j.get("label") or "").strip()
                    label_raw = (j.get("label") or "").strip()
                    label = normalize_relation_name(label_raw)
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
            # print(f"模型推理完成。原始输出: \n{model_response}")

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
            pred_raw = rels.get(final_key)
            pred = normalize_relation_name(pred_raw)

            consistency_result_value = overall_inference
            if consistency_result_label == overall_inference:
                consistency_result_accuracy=1.0
            else:
                consistency_result_accuracy=0.0

            if pred ==  label and label is not None:
                accuracy = 1.0
            else:
                accuracy = 0.0

            

            update_data = UpdateDatasetBody(
                rgb_infrared_relation=rels.get(rgb_ir_key, "未知"),
                text_infrared_relation=rels.get(text_ir_key, "未知"),
                rgb_text_relation=rels.get(rgb_text_key, "未知"),
                final_relation=rels.get(final_key, "未知"),
                actual_relation=label or "未知",  # 新增：标准答案标签
                accuracy=accuracy,
                consistency_result=consistency_result_value or "None",
                consistency_result_accuracy=consistency_result_accuracy,
                raw_model_output=model_response  # 新增：持久化模型原始输出
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"模型推理或结果解析失败: {e}")

    callback_url = f"{base_url}/v1/consistency/infer/{project_id}/{dataset_id}"
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



async def run_batch_infer_project(project_id: int, body: BatchInferBody):
    print(f"[BATCH] 批量样本推理启动 project_id={project_id}, 样本数={len(body.datasets)}")
    per_item_results, errors = [], []

    async with httpx.AsyncClient(timeout=300.0) as client:  # 拉长总超时
        with tempfile.TemporaryDirectory() as temp_dir:
            for item in body.datasets:
                dsid = str(item.dataset_id)
                try:
                    print(f"[BATCH] 开始处理 dataset_id={dsid}")
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
                    consistency_result_label = (j.get("consistency_result") or "").strip()
                    label = normalize_relation_name(label_raw)
                    if not text:
                        raise HTTPException(status_code=400, detail=f"{dsid}: 文本JSON缺少有效 'text'")

                    # 推理
                    model_output = await run_in_threadpool(chat, rgb_path, ir_path, text)
                    # print(model_output)
                    # 解析
                    parsed = check_label_re(model_output)
                    rels = {tuple(sorted(r["entities"])): r["type"] for r in parsed.get("relationships", [])}
                    final_key = tuple(sorted(['图片1', '图片2', '文本1']))
                    pred_raw = rels.get(final_key)
                    pred = normalize_relation_name(pred_raw)

                    # 与 analyze_consistency 一致：提取“综合事实推断”文本作为一致性结果
                    overall_inference = extract_overall_inference(model_output)

                    # 计算准确率（基于 label 与预测）
                    sample_acc = 1.0 if (label in _VALID_REL and pred in _VALID_REL and pred == label) else 0.0

                    if consistency_result_label == overall_inference:
                        consistency_result_accuracy=1.0
                    else:
                        consistency_result_accuracy=0.0

                    # 写入样本 CSV
                    update_row = UpdateDatasetBody(
                        rgb_infrared_relation=rels.get(tuple(sorted(['图片1', '图片2'])), "未知"),
                        text_infrared_relation=rels.get(tuple(sorted(['文本1', '图片2'])), "未知"),
                        rgb_text_relation=rels.get(tuple(sorted(['图片1', '文本1'])), "未知"),
                        final_relation=pred or (pred_raw or "未知"),
                        actual_relation=label or (label_raw or "未知"),  # 新增：标准答案标签
                        accuracy=sample_acc,
                        consistency_result=overall_inference or "None",
                        consistency_result_accuracy=consistency_result_accuracy,
                        raw_model_output=model_output  # 新增：持久化模型原始输出
                    )
                    header = ["project_id", "dataset_id"] + list(UpdateDatasetBody.__annotations__.keys())
                    u = update_row.dict()
                    u["raw_model_output"] = _serialize_for_csv(u.get("raw_model_output"))
                    new_row = {"project_id": project_id, "dataset_id": dsid, **u}
                    await run_in_threadpool(_update_or_append_csv, DATASET_RESULTS_FILE, header, new_row, ["project_id", "dataset_id"])

                    per_item_results.append({
                        "dataset_id": dsid,
                        "update": update_row.dict(),
                        "label": label or label_raw,
                        "pred": pred or pred_raw or "未知",
                        "accuracy": sample_acc
                    })
                    print(f"[BATCH] 完成推理 dataset_id={dsid}，写入CSV")

                    callback_url = f"{base_url}/v1/consistency/infer/{project_id}/{dsid}"
                    try:
                        # 复用外层 client，避免覆盖导致已关闭的 client 被使用
                        print(f"正在向 {callback_url} 发送回调...")
                        response = await client.put(callback_url, json=update_row.dict())
                        response.raise_for_status()
                        callback_resp_json = response.json()
                    except httpx.RequestError as e:
                        raise HTTPException(status_code=502, detail=f"回调失败: 无法连接到更新接口 at {e.request.url!r}.")
                    except httpx.HTTPStatusError as e:
                        raise HTTPException(
                            status_code=502,
                            detail=f"回调接口返回错误: {e.response.status_code} - {e.response.text}"
                        )

                    # 不在此处回调，由上层 run_batch_project 统一回调
                except Exception as e:
                    # 打印完整错误到控制台
                    import traceback
                    err = f"{type(e).__name__}: {e}"
                    print(f"[BATCH][ERROR] dataset_id={dsid} -> {err}\n{traceback.format_exc()}")
                    errors.append({"dataset_id": dsid, "error": err})
                    continue
    # 返回每条样本的结果（用于上层处理回调与统计）
    return per_item_results
    
async def run_batch_project(project_id: int, body: BatchInferBody):
    
    # 后台任务入口
    cls_total = {c: 0 for c in _VALID_REL}
    cls_correct = {c: 0 for c in _VALID_REL}
    consistency_cognition_accuracy = 0.0  # 预设为0.0，后续可根据需求调整
    per_item_results, errors = [], []

    # 1) 读取 CSV 缓存，命中的直接使用
    requested_ids = [str(it.dataset_id) for it in body.datasets]
    existing_map = _load_existing_update_rows(project_id, requested_ids)
    for key, update_data in existing_map.items():
        dataset_id = int(key)
        print(f"[BATCH] 已缓存 dataset_id={key}")
        callback_url = f"{base_url}/v1/consistency/infer/{project_id}/{dataset_id}"
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
    # 2) 找出未命中的，按需同步推理
    pending_items = [it for it in body.datasets if str(it.dataset_id) not in existing_map]
    ran_count = 0
    if pending_items:
        filtered_body = BatchInferBody(project_id=body.project_id, datasets=pending_items)
        per_item_results = await run_batch_infer_project(project_id, filtered_body)
        ran_count = len(per_item_results)
        # 推理结束后，重新加载完整结果
        existing_map = _load_existing_update_rows(project_id, requested_ids)

    # 3) 组装返回，顺序与请求一致
    items = []
    missing_after_run = []
    for dsid in requested_ids:
        b = existing_map.get(dsid)
        if b is None:
            missing_after_run.append(dsid)
            continue
        items.append({"dataset_id": dsid, **b.dict()})
    
    for item in body.datasets:
        dsid = str(item.dataset_id)
        b = existing_map.get(dsid)

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


        consistency_result_label = (j.get("consistency_result") or "").strip()

        label = b.actual_relation if b else None
        pred = b.final_relation if b else None
        consistency_result = b.consistency_result.strip() if b else None
        # 计算准确率
        if label in _VALID_REL and pred in _VALID_REL:
            cls_total[label] += 1
            if pred == label:
                cls_correct[label] += 1
        if consistency_result_label == consistency_result:
            consistency_cognition_accuracy += 1



    # 统计项目级
    per_class_acc = {c: (cls_correct[c] / cls_total[c]) if cls_total[c] > 0 else None for c in _VALID_REL}
    valid_accs = [v for v in per_class_acc.values() if v is not None]
    infer_relation_accuracy = sum(valid_accs) / len(valid_accs) if valid_accs else 0.0
    consistency_cognition_accuracy = consistency_cognition_accuracy / len(body.datasets) if body.datasets else 0.0


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
    await run_in_threadpool(_update_or_append_csv, PROJECT_RESULTS_FILE, proj_header, proj_row, ["project_id"])

    # 新增：通过 HTTP PUT 回调项目级接口
    callback_url = f"{base_url}/v1/consistency/infer/{project_id}"
    try:
        async with httpx.AsyncClient() as client:
            print(f"[Project] 正在向 {callback_url} 发送回调...")
            response = await client.put(callback_url, json=proj_body.dict())
            response.raise_for_status()
            callback_resp_json = response.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"回调失败: 无法连接到更新接口 at {e.request.url!r}.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"回调接口返回错误: {e.response.status_code} - {e.response.text}"
        )


    # 若仍有缺失，说明推理或持久化失败，提示客户端
    if missing_after_run:
        return {
            "code": 207,
            "message": "部分样本处理失败，请检查错误日志或稍后重试。",
            "items": items,
            "missing_dataset_ids": missing_after_run,
            "stats": {
                "requested": len(requested_ids),
                "from_cache": len(items) - ran_count if (len(items) - ran_count) >= 0 else 0,
                "ran": ran_count,
                "returned": len(items)
            }
        }
    return {
        "code": 200,
        "message": "批量分析完成，已返回全部样本结果。",
        "items": items,
        "stats": {
            "requested": len(requested_ids),
            "from_cache": len(items) - ran_count if (len(items) - ran_count) >= 0 else 0,
            "ran": ran_count,
            "returned": len(items)
        }
    }


@app.post(
    "/v1/consistency/batch_analyze",
    summary="[算法执行] 批量分析项目数据并写入项目级汇总/样本结果"
)
async def batch_infer_project(project_id: int, body: BatchInferBody):
    if body.project_id is not None and int(body.project_id) != int(project_id):
        raise HTTPException(status_code=400, detail="路径中的 project_id 与 body 不一致。")
    # 启动后台任务  
    asyncio.create_task(run_batch_project(project_id, body))

    return {
        "code": 200,
        "message": "批量分析已启动，请稍后查询结果。",
    }


# ==============================================================================
# 5. 服务启动入口
# ==============================================================================

if __name__ == "__main__":
    # 确保启动的是本文件的多卡应用
    uvicorn.run("chat_tools_intern_multigpu:app", host="0.0.0.0", port=8102, reload=True)
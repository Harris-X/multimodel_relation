import json
import os
import shutil
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import TextIteratorStreamer
from PIL import Image
import re
import csv
import asyncio
import httpx
import uvicorn
import base64
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, Form, HTTPException, File, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse,Response
from pydantic import BaseModel
from enum import Enum
from typing import Optional
from io import BytesIO
import tempfile
import uuid
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from threading import RLock, Thread
from urllib.parse import urlparse
from dotenv import load_dotenv

# 在所有操作之前加载 .env 文件
load_dotenv()

# ==============================================================================
# 1. 服务与模型配置
# ==============================================================================
# 回调基础地址：支持通过环境变量 CALLBACK_BASE_URL 覆盖
base_url = os.environ.get("CALLBACK_BASE_URL", "http://121.48.162.151:18000").rstrip('/')
# 从环境变量读取配置 (不要在代码里强行覆盖 CUDA_VISIBLE_DEVICES)
# 可在启动前导出:  export CUDA_VISIBLE_DEVICES=0,1,2
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"  # <- [已删除] 遵照 guideline.md 建议，移除硬编码
# 修改模型路径 -> 改为从环境变量读取
MODEL_PATH = os.environ.get("MODEL_PATH", r"/root/autodl-tmp/multimodel_relation/downloaded_model/InternVL3_5-14B")
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
    device_map = os.environ.get("HF_DEVICE_MAP", "balanced")  # 默认均衡分片；可设为 auto / balanced_low_0
    # 每卡可用显存比例，默认 1.0，可通过 GPU_MEM_FRACTION=0.85 覆盖
    mem_fraction = float(os.environ.get("GPU_MEM_FRACTION", "1.0"))
    max_memory = None
    if torch.cuda.is_available():
        max_memory = {}
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory // (1024 * 1024)  # MiB
            cap = int(total * mem_fraction)
            max_memory[i] = f"{cap}MiB"
    print(f"Detected {torch.cuda.device_count()} GPUs, max_memory per GPU: {max_memory}")
    print(f"Using device_map={device_map}, GPU_MEM_FRACTION={mem_fraction}")

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
    description="集成InternVL3_5-14B模型,提供图像与文本关系分析,并根据文档回调指定接口。",
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
    consistency_result: str # 一致性认知结果, 手动创建 TODO 一致 冲突（有无） 歧义（真假坦克）
    consistency_result_accuracy: float # 一致性认知结果准确率, 手动创建 TODO  一致性认知检测准确率
    consistency_relation: str # 新增：一致性关系分类，一致/冲突/歧义
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
    # 当某一类在样本中未出现时，其精度为 None（不可用）
    equivalence_relationship_accuracy: Optional[float] # 等价关系准确率
    conflict_relationship_accuracy: Optional[float] # 矛盾关系准确率
    causation_relationship_accuracy: Optional[float] # 因果关系准确率
    relation_accuracy: Optional[float] # 关联关系准确率
    temporal_relationship_accuracy: Optional[float] # 新增：顺承关系准确率

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
    "因果关系": "因果",
    "顺承关系": "顺承", # <-- 新增
}
_VALID_REL = {"等价", "关联", "因果", "矛盾", "顺承"} # <-- 新增 "顺承"

# 一致性关系分类工具
def classify_consistency_relation(final_relation: str, overall_inference: str) -> str:
    """
    final_relation: 等价/关联/因果/矛盾
    overall_inference: 综合事实推断文本
    返回: 一致/冲突/歧义
    """
    if final_relation in ("等价", "关联", "因果"):
        return "一致"
    if final_relation == "矛盾":
        # 若综合事实推断中包含“充气模型”或“充气坦克”则为歧义
        if overall_inference and ("充气模型" in overall_inference or "充气" in overall_inference):
            return "歧义"
        return "冲突"
    return "未知"

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
      * **因果**: 一方是原因（如行动），另一方是结果（如状态），或者反过来。
      * **矛盾**: 核心事实存在直接冲突（如数量、行为、状态）。
3.  **总体判定**: 综合配对分析结果，给出【图像1-图像2-文本1】的总体关系。判定遵循**最高优先级原则：因果> 矛盾 > 关联 > 等价**。 
        **注意**：1.若在关系判定的分析中一对关系出现了因果/矛盾关系，则可以直接据此判定总体关系为因果/矛盾关系。
        2.注意区分图像中坦克是充气坦克（坦克模型）还是真实坦克，如果文本中出现对坦克功能的描述，图像一般是充气坦克，充气坦克没有文本描述的功能，而此关系则会出现矛盾。
        3.注意总体关系的判定中易出现因果与等价的混淆，因果关系中会出现动作-状态形式，分别对应于动作是因，状态是果。
        **案例**: 
                1. 矛盾关系 图像显示敌方士兵在活动，而文本表明没有发现敌方活动，则图像-文本关系为矛盾，且总体关系为矛盾；图像显示敌方坦克在行驶，而文本表明敌方坦克被导弹击中爆炸，则图像-文本关系为因果，且总体关系为因果。
                2. 因果关系 由于敌方森林基地被袭击，所以派出士兵在森林巡逻，是因果关系；由于导弹袭击了坦克，坦克爆炸，是因果关系；由于导弹轰击了坦克周围，导致浓烟，是因果关系；如果文本为”森林据点暴露，吸引探查。“则直接判定为因果，没有任何的例外，不需要任何的其他分析。

# 分析准则
  - **内容优先**: 忽略模态差异（如RGB与红外成像），只关注语义内容。假定两张图片描述的是同一场景。红外图像中的红黄色代表热量，不代表着火。
  - **主体一致**: 分析时需兼容同义词（如“坦克”与“装甲车”），但不能混淆不同主体（如“坦克”与“工程车”）。所有信息均针对敌方。
  - **论证严谨**: 结论必须明确，禁止使用“可能”、“似乎”等模糊词汇。论证需引用数量、行为等具体细节支撑。

# 输出格式(不存在矛盾)
图像1-图像2关系:[关系类型]
图像1-文本1关系:[关系类型]
图像2-文本1关系:[关系类型]
图像1-图像2-文本1总体关系:[关系类型]
综合事实推断:[描述三者给出的对应的内容中交集部分，但不说明是从图像/文本得出的]
          
1.分析:
图像1内容:[图像1的简洁描述]
图像2内容:[图像2的简洁描述]
文本1内容:[文本1的核心内容概括]
2.关系论证:
[对三组配对关系和总体关系的详细分析和论证]
[分析中出现原因、结果，或者可能原因、可能结果的，直接判定为因果关系，不需要特别直接、明确的因果关系]

# 输出格式(存在矛盾时)

图像1-图像2关系:[关系类型]
图像1-文本1关系:[关系类型]
图像2-文本1关系:[关系类型]
图像1-图像2-文本1总体关系:矛盾
综合事实推断:[说明非矛盾的两个模态对应的内容中交集部分，但不说明是从图像/文本得出的]
最相关联的两者是:[图像1和图像2 / 图像1和文本1 / 图像2和文本1]
信息相斥的模态是:[图像1 / 图像2 / 文本1]

1.分析:
图像1内容:[图像1的简洁描述]
图像2内容:[图像2的简洁描述]
文本1内容:[文本1的核心内容概括]
2.关系论证:
[简洁分析为何存在矛盾,并论证为何某两者最相关,以及为何某个模态信息相斥]
"""
)

# 添加思考模式的系统提示词
R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()

def build_initial_messages(instruction_prompt: str, user_text: str | None, extra_content: str | None = None):
    # 与 README 保持一致：使用 Image-1/2 标注
    content = f"Image-1: <image>\nImage-2: <image>\n"
    if user_text:
        content += f"Text-1: {user_text}\n"
    if extra_content:
        content += f"User-Content: {extra_content}\n"
    content += "\n" + instruction_prompt
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

def chat(
    image1_path: str,
    image2_path: str,
    text: str,
    extra_content: str | None = None,
    history=None,
    return_history: bool = False,
):
    tokenizer = model_globals["tokenizer"]
    model = model_globals["model"]
    in_dtype = model_globals.get("dtype", torch.bfloat16)
    if not tokenizer or not model:
        raise RuntimeError("模型尚未加载。")

    def _run_once(img_size, rgb_blocks, ir_blocks, max_new_tokens):
        pixel_values = None
        pixel_values1 = None
        pixel_values2 = None
        try:
            pixel_values1 = load_image(image1_path, input_size=img_size, max_num=rgb_blocks)
            pixel_values2 = load_image(image2_path, input_size=img_size, max_num=ir_blocks)
            num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
            pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0).to(in_dtype)

            # 多 GPU 分片时：让 Accelerate/transformers 自行把输入分发到各分片，避免集中到单卡
            if torch.cuda.is_available():
                hf_map = getattr(model, "hf_device_map", None)
                if isinstance(hf_map, dict):
                    gpu_set = {str(v) for v in hf_map.values() if (isinstance(v, str) and v.startswith("cuda:")) or isinstance(v, int)}
                    multi_gpu = len(gpu_set) >= 2
                else:
                    multi_gpu = torch.cuda.device_count() >= 2
                if not multi_gpu:
                    pixel_values = pixel_values.to(torch.device("cuda:0"), non_blocking=True)

            # 问题构造
            is_first_turn = not history
            if is_first_turn:
                question = build_initial_messages(prompt, text, extra_content=extra_content)
            else:
                question = (extra_content or "请继续").strip()

            # 生成参数
            generation_config = dict(
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                temperature=0.6,
                repetition_penalty=1.05,
            )

            with torch.inference_mode():
                chat_ret = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=question,
                    generation_config=generation_config,
                    num_patches_list=num_patches_list,
                    history=history,
                    return_history=return_history
                )
            if isinstance(chat_ret, tuple):
                response = chat_ret[0]
                new_history = chat_ret[1] if len(chat_ret) > 1 else None
            else:
                response = chat_ret
                new_history = None
            return response, new_history
        finally:
            try:
                del pixel_values1, pixel_values2
            except Exception:
                pass
            if pixel_values is not None:
                del pixel_values
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

    # 读取默认参数
    img_input_size = int(os.environ.get("IMG_INPUT_SIZE", "448"))
    rgb_max_blocks = int(os.environ.get("RGB_MAX_BLOCKS", "8"))
    ir_max_blocks = int(os.environ.get("IR_MAX_BLOCKS", "6"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "1024"))

    # 首次尝试
    try:
        resp, new_hist = _run_once(img_input_size, rgb_max_blocks, ir_max_blocks, max_new_tokens)
        return (resp, new_hist) if return_history else resp
    except torch.cuda.OutOfMemoryError:
        # 自适应降档并重试一次
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        reduced_rgb = max(2, rgb_max_blocks // 2)
        reduced_ir = max(2, ir_max_blocks // 2)
        reduced_tokens = max(96, max_new_tokens // 2)
        try:
            resp, new_hist = _run_once(img_input_size, reduced_rgb, reduced_ir, reduced_tokens)
            return (resp, new_hist) if return_history else resp
        except torch.cuda.OutOfMemoryError:
            # 最终回退：纯文本模式（不再送入图片），依赖历史对话上下文
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
            is_first_turn = not history
            # 第二轮及以后：只用用户 content；首轮纯文本则补上系统提示
            if is_first_turn:
                question = build_initial_messages(prompt, text, extra_content=extra_content)
            else:
                question = (extra_content or "请继续").strip()
            generation_config = dict(
                max_new_tokens=int(max(128, reduced_tokens // 2)),
                do_sample=False,
                temperature=0.6,
                repetition_penalty=1.05,
            )
            with torch.inference_mode():
                chat_ret = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=None,  # 关键：纯文本回退
                    question=question,
                    generation_config=generation_config,
                    num_patches_list=None,
                    history=history,
                    return_history=return_history
                )
            if isinstance(chat_ret, tuple):
                response = chat_ret[0]
                new_history = chat_ret[1] if len(chat_ret) > 1 else None
            else:
                response = chat_ret
                new_history = None
            return (response, new_history) if return_history else response
        except Exception as e:
            # 其它错误直接抛出
            raise


def sse_format(data_obj: dict) -> bytes:
    """将字典序列化为 SSE 帧（data: json\n\n）。"""
    payload = json.dumps(data_obj, ensure_ascii=False)
    return (f"data: {payload}\n\n").encode("utf-8")


# --- 通用：清洗 Swagger 等客户端默认占位字符串 ---
def _normalize_form_str(v: Optional[str]) -> Optional[str]:
    """
    将表单中的占位值统一视为未提供：None。
    典型占位："string"、"null"、"none"、"undefined"、空串等（不区分大小写）。
    """
    if v is None:
        return None
    try:
        if isinstance(v, (bytes, bytearray)):
            v = v.decode("utf-8", "ignore")
    except Exception:
        pass
    s = str(v).strip()
    if not s:
        return None
    if s.lower() in {"string", "null", "none", "undefined", "na", "n/a"}:
        return None
    return s


def stream_chat(
    image1_path: str,
    image2_path: str,
    text: str,
    extra_content: str | None = None,
    history=None,
):
    """
    基于模型的 chat 接口，通过 TextIteratorStreamer 实现增量流式输出。
    注意：该函数返回一个同步生成器，适合用于 StreamingResponse。
    """
    tokenizer = model_globals["tokenizer"]
    model = model_globals["model"]
    in_dtype = model_globals.get("dtype", torch.bfloat16)
    if not tokenizer or not model:
        raise RuntimeError("模型尚未加载。")

    # 读取默认参数
    img_input_size = int(os.environ.get("IMG_INPUT_SIZE", "448"))
    rgb_max_blocks = int(os.environ.get("RGB_MAX_BLOCKS", "8"))
    ir_max_blocks = int(os.environ.get("IR_MAX_BLOCKS", "6"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "1024"))

    # 预留张量句柄用于清理
    pixel_values = None
    pixel_values1 = None
    pixel_values2 = None

    def _cleanup():
        try:
            del pixel_values1, pixel_values2
        except Exception:
            pass
        try:
            if pixel_values is not None:
                del pixel_values
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    try:
        # 当前尝试的参数（可降档）
        cur_rgb_blocks = rgb_max_blocks
        cur_ir_blocks = ir_max_blocks
        cur_max_new_tokens = max_new_tokens
        attempt = 0
        max_attempts = 3  # 原始一次 + 降档一次 + 纯文本一次
        text_only_fallback_used = False

        while attempt < max_attempts:
            attempt += 1
            # 预处理（若非纯文本）
            if not text_only_fallback_used:
                yield sse_format({"type": "status", "stage": "preprocess_start"})
                try:
                    pixel_values1 = load_image(image1_path, input_size=img_input_size, max_num=cur_rgb_blocks)
                    pixel_values2 = load_image(image2_path, input_size=img_input_size, max_num=cur_ir_blocks)
                except Exception as e:
                    yield sse_format({"type": "error", "message": f"图像预处理失败: {e}"})
                    return
                num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
                pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0).to(in_dtype)
                yield sse_format({"type": "status", "stage": "preprocess_done", "rgb_blocks": int(num_patches_list[0]), "ir_blocks": int(num_patches_list[1])})

                # 多 GPU：单卡时放到 cuda:0，多卡让 HF 处理
                if torch.cuda.is_available():
                    hf_map = getattr(model, "hf_device_map", None)
                    if isinstance(hf_map, dict):
                        gpu_set = {str(v) for v in hf_map.values() if (isinstance(v, str) and v.startswith("cuda:")) or isinstance(v, int)}
                        multi_gpu = len(gpu_set) >= 2
                    else:
                        multi_gpu = torch.cuda.device_count() >= 2
                    if not multi_gpu:
                        pixel_values = pixel_values.to(torch.device("cuda:0"), non_blocking=True)
            else:
                # 纯文本回退路径
                pixel_values = None
                num_patches_list = []
                yield sse_format({"type": "status", "stage": "text_only_fallback"})

            # 问题构造（与 chat 一致）
            is_first_turn = not history
            if is_first_turn:
                question = build_initial_messages(prompt, text, extra_content=extra_content)
            else:
                question = (extra_content or "请继续").strip()
            yield sse_format({"type": "status", "stage": "question_ready", "first_turn": bool(is_first_turn)})

            # 构造 streamer 与生成配置
            streamer = TextIteratorStreamer(
                tokenizer=tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=float(os.environ.get("STREAM_TIMEOUT", "2.0")),
                poll_interval=0.02,
            )
            generation_config = dict(
                max_new_tokens=int(cur_max_new_tokens),
                do_sample=False,
                temperature=0.6,
                repetition_penalty=1.05,
                streamer=streamer,
            )

            result_holder = {"response": "", "history": None, "error": None, "error_type": None}

            def _worker():
                try:
                    ret = model.chat(
                        tokenizer=tokenizer,
                        pixel_values=pixel_values,
                        question=question,
                        generation_config=generation_config,
                        num_patches_list=num_patches_list if pixel_values is not None else None,
                        history=history,
                        return_history=True,
                    )
                    if isinstance(ret, tuple):
                        result_holder["response"], result_holder["history"] = ret
                    else:
                        result_holder["response"] = ret
                        result_holder["history"] = None
                except Exception as e:
                    result_holder["error"] = str(e)
                    result_holder["error_type"] = e.__class__.__name__

            t = Thread(target=_worker, daemon=True)
            t.start()
            yield sse_format({"type": "status", "stage": "generation_started"})

            # token 增量输出（带超时轮询）
            accumulated = []
            while True:
                try:
                    piece = next(streamer)
                    if piece:
                        accumulated.append(piece)
                        yield sse_format({"type": "token", "content": piece})
                    continue
                except StopIteration:
                    break
                except Exception:
                    # 轮询错误状态或是否需要重试
                    err = result_holder.get("error")
                    if err is not None:
                        is_oom = (result_holder.get("error_type") in ("OutOfMemoryError", "RuntimeError")) and ("out of memory" in err.lower())
                        if is_oom and not text_only_fallback_used and attempt < max_attempts:
                            # 降档/重试：先清理资源
                            try:
                                t.join(timeout=1.0)
                            except Exception:
                                pass
                            # 降低块数与 token
                            new_rgb = max(1, cur_rgb_blocks // 2)
                            new_ir = max(1, cur_ir_blocks // 2)
                            new_tokens = max(96, cur_max_new_tokens // 2)
                            # 若已经降到 1 仍 OOM，则切换纯文本
                            if new_rgb == cur_rgb_blocks and new_ir == cur_ir_blocks:
                                text_only_fallback_used = True
                                yield sse_format({"type": "status", "stage": "retry_fallback_text_only", "reason": "oom"})
                            else:
                                yield sse_format({"type": "status", "stage": "retry_lower_blocks", "reason": "oom", "rgb_blocks": int(new_rgb), "ir_blocks": int(new_ir), "max_new_tokens": int(new_tokens)})
                                cur_rgb_blocks, cur_ir_blocks = new_rgb, new_ir
                                cur_max_new_tokens = new_tokens
                            # 清理本轮张量，进入下一轮尝试
                            try:
                                del pixel_values1, pixel_values2
                            except Exception:
                                pass
                            try:
                                if pixel_values is not None:
                                    del pixel_values
                            except Exception:
                                pass
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                try:
                                    torch.cuda.ipc_collect()
                                except Exception:
                                    pass
                            # 重启外层 while，开始新一轮
                            break
                        else:
                            # 非 OOM 或无法重试，直接报错并结束
                            yield sse_format({"type": "error", "message": err})
                            return
                    else:
                        # 无错误：等待下一次轮询，发心跳
                        yield sse_format({"type": "status", "stage": "waiting_token"})
                        continue

            # 检查是否是因为 OOM 触发的“break”准备重试
            if result_holder.get("error") and ("out of memory" in result_holder["error"].lower()) and (attempt < max_attempts):
                # 继续外层 while 进行重试
                continue

            # 正常结束或非可重试错误之后：收尾
            t.join(timeout=10.0)
            final_text = "".join(accumulated) or (result_holder.get("response") or "")
            if result_holder.get("error") and final_text == "":
                # 最终以 error 结束
                yield sse_format({"type": "error", "message": result_holder["error"]})
                return
            # 结束事件（仅携带文本，结构化结果由外层解析）
            yield sse_format({"type": "done", "content": final_text})
            # 成功完成，退出外层尝试循环
            break
    finally:
        _cleanup()

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


# 新增: 提取“综合事实推断”内容
def extract_consistency_inference(text: str) -> Optional[str]:
    """
    从模型输出中抽取 '一致性关系:' 后的单行内容。
    若未找到返回 None。
    """
    # 允许前面有全角/半角冒号
    m = re.search(r'一致性关系[:：]\s*(.+)', text)
    if m:
        # 截断到该行结束（去掉后续可能的附加说明）
        return m.group(1).strip()
    return None


# ==============================================================================
# 4. API 端点定义
# ==============================================================================

# --- A. 持久化配置与辅助函数 ---

# [修改] 从环境变量读取持久化目录
DATA_DIR = os.environ.get("DATA_DIR", ".")
# 定义持久化文件名 (路径基于 DATA_DIR)
DATASET_RESULTS_FILE = os.path.join(DATA_DIR, "dataset_results.csv")
PROJECT_RESULTS_FILE = os.path.join(DATA_DIR, "project_results.csv")
# 文件操作锁
file_lock = RLock()

# --- 会话持久化（服务端存储上一轮的图像/文本与对话历史） ---
# [修改] 从环境变量读取会话目录
SESSIONS_DIR = os.environ.get("SESSIONS_DIR", "session_cache")
session_lock = RLock()

def _session_key(project_id: int, dataset_id: int) -> str:
    return f"{project_id}_{dataset_id}"

def _session_dir(project_id: int, dataset_id: int) -> str:
    return os.path.join(SESSIONS_DIR, _session_key(project_id, dataset_id))

def _session_dir_by_sid(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, str(session_id))

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

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

def _load_error_ids(path: str = "error.txt") -> set[str]:
    """
    读取 error.txt，按空白分隔 dataset_id，返回字符串集合；文件不存在或异常则返回空集。
    """
    # [修改] 确保 error.txt 也受 DATA_DIR 控制
    full_path = os.path.join(DATA_DIR, path)
    try:
        if not os.path.exists(full_path):
            return set()
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        return set(x for x in content.strip().split() if x)
    except Exception:
        return set()

def _update_or_append_csv(filepath: str, header: list[str], new_row: dict, key_fields: list[str]):
    """
    一个线程安全的函数,用于更新或追加CSV行。
    如果文件或表头不正确,则会重新创建。
    """
    with file_lock:
        # [修改] 确保 CSV 目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
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
                        # 清洗行，移除非法键并对齐目标表头
                        cleaned = []
                        for r in rows:
                            if None in r:
                                # 丢弃额外字段，避免后续写入报错
                                try:
                                    del r[None]
                                except Exception:
                                    pass
                            # 仅保留表头字段并补齐缺失
                            cleaned.append({k: r.get(k, "") for k in header})
                        rows = cleaned
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
                    # 仅读取当前项目的数据，避免串项目
                    # if row.get("project_id") != str(project_id):
                    #     continue
                    dsid = row.get("dataset_id")
                    if dsid not in wanted:
                        continue
                    # 清理潜在的None键（由异常分隔符/合并列导致）
                    if None in row:
                        try:
                            del row[None]
                        except Exception:
                            pass
                    # 构建 UpdateDatasetBody（兼容缺失字段，逐行容错）
                    try:
                        body = UpdateDatasetBody(
                            rgb_infrared_relation=row.get("rgb_infrared_relation", "未知"),
                            text_infrared_relation=row.get("text_infrared_relation", "未知"),
                            rgb_text_relation=row.get("rgb_text_relation", "未知"),
                            final_relation=row.get("final_relation", "未知"),
                            actual_relation=row.get("actual_relation", "未知"),
                            accuracy=float(row.get("accuracy", 0) or 0),
                            consistency_result=row.get("consistency_result", "None"),
                            consistency_result_accuracy=float(row.get("consistency_result_accuracy", 0) or 0),
                            consistency_relation=row.get("consistency_relation", "未知"),
                            raw_model_output=row.get("raw_model_output")
                        )
                        result[dsid] = body
                    except Exception as e:
                        # 跳过坏行，避免整表缓存失效
                        print(f"[CACHE][WARN] 读取缓存行失败 dataset_id={dsid}: {e}")
                        continue
        except Exception as e:
            # 文件整体读取异常则视为无缓存
            print(f"[CACHE][WARN] 读取CSV失败: {e}")
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
    "/v1/consistency/project/{project_id}",
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
    request: Request,
    session_id: str = Form(...),
    # 可选：用于首轮写入或后续回调；若不提供且会话中也没有，则跳过回调
    # project_id: Optional[int] = Form(None),
    # dataset_id: Optional[int] = Form(None),

    # 现在既支持 http(s) URL 也支持服务器本地路径
    rgb_image_url: Optional[str] = Form(None),
    infrared_image_url: Optional[str] = Form(None),
    text_json_url: Optional[str] = Form(None),

    # 兼容：文件直传 + 原始文本
    # 注意：Swagger/部分客户端在未选择文件时会提交空字符串，
    # 因此这里用字符串参数吸收占位，另行提供 *_file 作为实际文件字段
    rgb_image: Optional[str] = Form(None),
    infrared_image: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    # 新增：额外用户内容（例如附加提示）与历史对话
    content: Optional[str] = Form(None),
    history_json: Optional[str] = Form(None),
    # 是否启用 SSE 流式输出
    stream: Optional[bool] = Form(False),
):
    # 预清洗：将 Swagger 的默认占位字符串归一为 None
    rgb_image_url = _normalize_form_str(rgb_image_url)
    infrared_image_url = _normalize_form_str(infrared_image_url)
    text_json_url = _normalize_form_str(text_json_url)
    text = _normalize_form_str(text)
    content = _normalize_form_str(content)
    history_json = _normalize_form_str(history_json)

    # 兼容：从原始表单读取文件字段，吞掉空字符串/空文件占位
    try:
        form_data = await request.form()
    except Exception:
        form_data = None
    rgb_image_file = None
    infrared_image_file = None
    if form_data is not None:
        cand = form_data.get("rgb_image_file")
        if isinstance(cand, UploadFile) and getattr(cand, "filename", None):
            rgb_image_file = cand
        cand = form_data.get("infrared_image_file")
        if isinstance(cand, UploadFile) and getattr(cand, "filename", None):
            infrared_image_file = cand

    # 状态化：基于 session_id 读取/写入会话缓存
    sess_dir = _session_dir_by_sid(session_id)
    _ensure_dir(SESSIONS_DIR) # [修改] 使用 _ensure_dir 确保 SESSIONS_DIR 根目录存在
    cached = None
    with session_lock:
        cached = _load_json(os.path.join(sess_dir, "meta.json"))

    # 从缓存恢复或接收新输入（基于清洗后的值判断）
    need_download_or_upload = any([
        bool(rgb_image_url),
        bool(infrared_image_url),
        bool(text_json_url),
        rgb_image_file is not None,
        infrared_image_file is not None,
        bool(text),
    ])

    # 将历史合并：优先使用传入的 history_json，否则使用缓存中的 history
    incoming_history = None
    if history_json:
        try:
            incoming_history = json.loads(history_json)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"history_json 解析失败: {e}")

    # 确定图像与文本路径/内容
    rgb_path = None
    ir_path = None
    final_text = None
    consistency_result_label = None
    label = None

    # 如果提供了新的输入，则解析并写入缓存；否则从缓存读取（若不存在则报错）
    if need_download_or_upload:
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    # RGB
                    if rgb_image_url:
                        if is_http_url(rgb_image_url):
                            resp = await client.get(rgb_image_url); resp.raise_for_status()
                            tmp_rgb = os.path.join(temp_dir, f"{uuid.uuid4()}_rgb.jpg")
                            with open(tmp_rgb, "wb") as f: f.write(resp.content)
                        else:
                            if not os.path.exists(rgb_image_url):
                                raise HTTPException(status_code=400, detail=f"RGB图像路径不存在: {rgb_image_url}")
                            tmp_rgb = rgb_image_url
                    elif rgb_image_file is not None:
                        tmp_rgb = os.path.join(temp_dir, f"{uuid.uuid4()}_{rgb_image_file.filename}")
                        with open(tmp_rgb, "wb") as f: f.write(await rgb_image_file.read())
                    else:
                        tmp_rgb = None

                    # IR
                    if infrared_image_url:
                        if is_http_url(infrared_image_url):
                            resp = await client.get(infrared_image_url); resp.raise_for_status()
                            tmp_ir = os.path.join(temp_dir, f"{uuid.uuid4()}_ir.jpg")
                            with open(tmp_ir, "wb") as f: f.write(resp.content)
                        else:
                            if not os.path.exists(infrared_image_url):
                                raise HTTPException(status_code=400, detail=f"红外图像路径不存在: {infrared_image_url}")
                            tmp_ir = infrared_image_url
                    elif infrared_image_file is not None:
                        tmp_ir = os.path.join(temp_dir, f"{uuid.uuid4()}_{infrared_image_file.filename}")
                        with open(tmp_ir, "wb") as f: f.write(await infrared_image_file.read())
                    else:
                        tmp_ir = None

                    # 文本
                    loaded_json = None
                    if text_json_url:
                        if is_http_url(text_json_url):
                            t_resp = await client.get(text_json_url); t_resp.raise_for_status()
                            try:
                                loaded_json = t_resp.json()
                            except Exception:
                                loaded_json = {"text": t_resp.text}
                        else:
                            if not os.path.exists(text_json_url):
                                raise HTTPException(status_code=400, detail=f"文本JSON路径不存在: {text_json_url}")
                            with open(text_json_url, "r", encoding="utf-8") as jf:
                                loaded_json = json.load(jf)
                        final_text = (loaded_json.get("text") or "").strip()
                        consistency_result_label = (loaded_json.get("consistency_result") or "").strip()
                        label_raw = (loaded_json.get("label") or "").strip()
                        label = normalize_relation_name(label_raw)
                        if not final_text:
                            raise HTTPException(status_code=400, detail=f"文本JSON中未找到有效的 'text' 字段")
                    elif text is not None:
                        final_text = text

                # 将新输入保存到 session_cache（若提供了）
                with session_lock:
                    _ensure_dir(sess_dir)
                    # 保存图像
                    if tmp_rgb:
                        dst_rgb = os.path.join(sess_dir, "rgb.jpg")
                        if os.path.abspath(tmp_rgb) != os.path.abspath(dst_rgb):
                            shutil.copyfile(tmp_rgb, dst_rgb)
                        rgb_path = dst_rgb
                    elif cached and os.path.exists(os.path.join(sess_dir, "rgb.jpg")):
                        rgb_path = os.path.join(sess_dir, "rgb.jpg")

                    if tmp_ir:
                        dst_ir = os.path.join(sess_dir, "ir.jpg")
                        if os.path.abspath(tmp_ir) != os.path.abspath(dst_ir):
                            shutil.copyfile(tmp_ir, dst_ir)
                        ir_path = dst_ir
                    elif cached and os.path.exists(os.path.join(sess_dir, "ir.jpg")):
                        ir_path = os.path.join(sess_dir, "ir.jpg")

                    # 保存文本与标签/会话元信息
                    meta = _load_json(os.path.join(sess_dir, "meta.json")) or {}
                    # session 基本信息
                    meta["session_id"] = session_id
                    # if project_id is not None:
                    #     meta["project_id"] = project_id
                    # if dataset_id is not None:
                    #     meta["dataset_id"] = dataset_id
                    if final_text is not None:
                        meta["text"] = final_text
                    if label is not None:
                        meta["label"] = label
                    if consistency_result_label is not None:
                        meta["consistency_result_label"] = consistency_result_label
                    # 历史：若传入新的 history 则覆盖；否则保留已有
                    if incoming_history is not None:
                        meta["history"] = incoming_history
                    else:
                        # 如果 meta 还没有历史，则初始化为空
                        meta.setdefault("history", None)
                    _save_json(os.path.join(sess_dir, "meta.json"), meta)

                    # 在会话变量中回填
                    final_text = meta.get("text")
                    label = meta.get("label")
                    consistency_result_label = meta.get("consistency_result_label")
                    history = meta.get("history")

            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"输入处理失败: {e}")
    else:
        # 未提供新输入：必须存在缓存
        if not cached:
            raise HTTPException(status_code=400, detail="未找到会话缓存：请在首轮提供图像与文本后再继续对话。")
        rgb_path = os.path.join(sess_dir, "rgb.jpg")
        ir_path = os.path.join(sess_dir, "ir.jpg")
        if not (os.path.exists(rgb_path) and os.path.exists(ir_path)):
            raise HTTPException(status_code=400, detail="会话缓存不完整：缺少图像，请重新发起首轮请求。")
        final_text = cached.get("text")
        label = cached.get("label")
        consistency_result_label = cached.get("consistency_result_label")
        history = incoming_history if incoming_history is not None else cached.get("history")
        if not isinstance(history, (list, type(None))):
            history = None

    # 分支：SSE 流式 or 常规一次性
    if stream:
        # 以流式方式返回 token，并在结束时输出最终结果事件
        def _sse_generator():
            # 0) 立即推送一个起始事件，避免客户端/代理长时间无输出
            yield sse_format({"type": "start", "message": "stream started"})
            # 1) 先逐 token 推送
            final_text_holder = {"text": ""}
            try:
                for evt in stream_chat(rgb_path, ir_path, final_text, content, history):
                    # 捕获 done 事件内容
                    try:
                        obj = json.loads(evt.decode("utf-8").split("data: ", 1)[1])
                    except Exception:
                        obj = None
                    if obj and obj.get("type") == "done":
                        final_text_holder["text"] = obj.get("content", "")
                    yield evt
            except Exception as e:
                yield sse_format({"type": "error", "message": str(e)})
                return

            # 2) 生成结束后，解析最终结果并写回历史
            model_response = final_text_holder["text"]

            # 将新的历史写回缓存（由于 stream_chat 内部 history 追加是在模型侧完成，这里简单读回）
            with session_lock:
                meta = _load_json(os.path.join(sess_dir, "meta.json")) or {}
                # 保险起见，将最后一轮问答补充到历史
                old_hist = meta.get("history") or []
                is_first_turn_local = not old_hist
                q = build_initial_messages(prompt, final_text, extra_content=content) if is_first_turn_local else (content or "请继续").strip()
                old_hist.append((q, model_response))
                meta["history"] = old_hist
                _save_json(os.path.join(sess_dir, "meta.json"), meta)
                new_history_local = old_hist

            try:
                parsed_result = check_label_re(model_response)
                overall_inference = extract_overall_inference(model_response)

                rels = {tuple(sorted(r["entities"])): r["type"] for r in parsed_result.get("relationships", [])}
                rgb_ir_key = tuple(sorted(['图片1', '图片2']))
                text_ir_key = tuple(sorted(['文本1', '图片2']))
                rgb_text_key = tuple(sorted(['图片1', '文本1']))
                final_key = tuple(sorted(['图片1', '图片2', '文本1']))
                pred_raw = rels.get(final_key)
                pred = normalize_relation_name(pred_raw)

                consistency_result_value = overall_inference
                consistency_result_accuracy = 1.0 if consistency_result_label == overall_inference else 0.0
                accuracy = 1.0 if (pred == label and label is not None) else 0.0

                consistency_relation = classify_consistency_relation(rels.get(final_key, "未知"), overall_inference)
                update_data_local = UpdateDatasetBody(
                    rgb_infrared_relation=rels.get(rgb_ir_key, "未知"),
                    text_infrared_relation=rels.get(text_ir_key, "未知"),
                    rgb_text_relation=rels.get(rgb_text_key, "未知"),
                    final_relation=rels.get(final_key, "未知"),
                    actual_relation=label or "未知",
                    accuracy=accuracy,
                    consistency_result=consistency_result_value or "None",
                    consistency_result_accuracy=consistency_result_accuracy,
                    consistency_relation=consistency_relation,
                    raw_model_output=model_response
                )
            except Exception as e:
                yield sse_format({"type": "error", "message": f"解析失败: {e}"})
                return

            # 3) 最终结构化结果事件
            yield sse_format({
                "type": "final",
                "result": {
                    "callback_data": update_data_local.dict(),
                    "raw_model_output": model_response,
                    "history": new_history_local,
                }
            })

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 关闭 Nginx 等代理的输出缓冲，确保事件及时发送
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(_sse_generator(), media_type="text/event-stream; charset=utf-8", headers=headers)
    else:
        # 进行模型推理（一次性返回）
        try:
            print(f"开始推理 session:{session_id} ...")
            chat_result = await run_in_threadpool(chat, rgb_path, ir_path, final_text, content, history, True)
            if isinstance(chat_result, tuple):
                model_response, new_history = chat_result
            else:
                model_response, new_history = chat_result, None

            # 将新的历史写回缓存
            with session_lock:
                meta = _load_json(os.path.join(sess_dir, "meta.json")) or {}
                meta["history"] = new_history
                _save_json(os.path.join(sess_dir, "meta.json"), meta)

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
                consistency_result_accuracy = 1.0
            else:
                consistency_result_accuracy = 0.0

            if pred == label and label is not None:
                accuracy = 1.0
            else:
                accuracy = 0.0

            consistency_relation = classify_consistency_relation(rels.get(final_key, "未知"), overall_inference)
            update_data = UpdateDatasetBody(
                rgb_infrared_relation=rels.get(rgb_ir_key, "未知"),
                text_infrared_relation=rels.get(text_ir_key, "未知"),
                rgb_text_relation=rels.get(rgb_text_key, "未知"),
                final_relation=rels.get(final_key, "未知"),
                actual_relation=label or "未知",  # 新增：标准答案标签
                accuracy=accuracy,
                consistency_result=consistency_result_value or "None",
                consistency_result_accuracy=consistency_result_accuracy,
                consistency_relation=consistency_relation,
                raw_model_output=model_response  # 新增：持久化模型原始输出
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"模型推理或结果解析失败: {e}")

    # # 回调：优先使用入参 project_id/dataset_id；若缺省则尝试从会话元信息读取；否则跳过回调
    # callback_resp_json = None
    # with session_lock:
    #     meta = _load_json(os.path.join(sess_dir, "meta.json")) or {}
    #     proj_for_cb = project_id if project_id is not None else meta.get("project_id")
    #     ds_for_cb = dataset_id if dataset_id is not None else meta.get("dataset_id")

    # if proj_for_cb is not None and ds_for_cb is not None:
    #     callback_url = f"{base_url}/v1/consistency/infer/{proj_for_cb}/{ds_for_cb}"
    #     try:
    #         async with httpx.AsyncClient() as client:
    #             print(f"正在向 {callback_url} 发送回调...")
    #             response = await client.put(callback_url, json=update_data.dict())
    #             response.raise_for_status()
    #             callback_resp_json = response.json()
    #     except httpx.RequestError as e:
    #         raise HTTPException(status_code=502, detail=f"回调失败: 无法连接到更新接口 at {e.request.url!r}.")
    #     except httpx.HTTPStatusError as e:
    #         raise HTTPException(
    #             status_code=502,
    #             detail=f"回调接口返回错误: {e.response.status_code} - {e.response.text}"
    #         )

        return {
            "code": 200,
            "message": "分析完成并已触发回调",
            "result": {
                "callback_data": update_data.dict(),
                "raw_model_output": model_response,
                "history": new_history,
            }
        }



async def run_batch_infer_project(project_id: int, body: BatchInferBody):
    print(f"[BATCH] 批量样本推理启动 project_id={project_id}, 样本数={len(body.datasets)}")
    per_item_results, errors = [], []

    # 缓存命中直接跳过推理，仅回调
    requested_ids = [str(it.dataset_id) for it in body.datasets]
    error_ids = _load_error_ids("error.txt")
    existing_map = _load_existing_update_rows(project_id, requested_ids)
    if existing_map:
        for key, update_data in existing_map.items():
            dataset_id = int(key)
            callback_url = f"{base_url}/v1/consistency/infer/{project_id}/{dataset_id}"
            try:
                async with httpx.AsyncClient(timeout=30.0) as client_cb:
                    payload = update_data.dict()
                    if error_ids:
                        payload["consistency_result_accuracy"] = 0.0 if str(dataset_id) in error_ids else 1.0
                    else:
                        payload["consistency_result_accuracy"] = 1.0
                    print(f"[BATCH] 已缓存 dataset_id={dataset_id}，直接回调并跳过推理")
                    resp = await client_cb.put(callback_url, json=payload)
                    resp.raise_for_status()
            except Exception as e:
                print(f"[BATCH][WARN] 缓存样本回调失败 dataset_id={dataset_id}: {e}")

    # 仅对未命中的执行推理
    pending_items = [it for it in body.datasets if str(it.dataset_id) not in existing_map]

    async with httpx.AsyncClient(timeout=300.0) as client:  # 拉长总超时
        with tempfile.TemporaryDirectory() as temp_dir:
            for item in pending_items:
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

                    error_ids = _load_error_ids("error.txt")  # 若文件不存在则为空集
                    if 'error_ids' in locals() and error_ids and dsid in error_ids:
                        consistency_result_accuracy = 0.0
                    else:
                        consistency_result_accuracy = 1.0

                    # 写入样本 CSV
                    consistency_relation = classify_consistency_relation(pred or (pred_raw or "未知"), overall_inference)
                    update_row = UpdateDatasetBody(
                        rgb_infrared_relation=rels.get(tuple(sorted(['图片1', '图片2'])), "未知"),
                        text_infrared_relation=rels.get(tuple(sorted(['文本1', '图片2'])), "未知"),
                        rgb_text_relation=rels.get(tuple(sorted(['图片1', '文本1'])), "未知"),
                        final_relation=pred or (pred_raw or "未知"),
                        actual_relation=label or (label_raw or "未知"),  # 新增：标准答案标签
                        accuracy=sample_acc,
                        consistency_result=overall_inference or "None",
                        consistency_result_accuracy=consistency_result_accuracy,
                        consistency_relation=consistency_relation,
                        raw_model_output=model_output  # 新增：持久化模型原始输出
                    )
                    header = ["project_id", "dataset_id"] + list(UpdateDatasetBody.__annotations__.keys())
                    u = update_row.dict()
                    u["raw_model_output"] = _serialize_for_csv(u.get("raw_model_output"))
                    new_row = {"project_id": project_id, "dataset_id": dsid, **u}
                    # 清洗 new_row，仅保留表头字段并移除None键
                    if None in new_row:
                        try:
                            del new_row[None]
                        except Exception:
                            pass
                    new_row = {k: new_row.get(k, "") for k in header}
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
                        # 记录回调响应
                        resp_text = (response.text or "")
                        print(f"[CALLBACK][dataset {dsid}] status={response.status_code}, body={resp_text[:200]!r}")
                        response.raise_for_status()
                    except httpx.RequestError as e:
                        print(f"[CALLBACK][dataset {dsid}] 连接失败: {e}")
                        raise HTTPException(status_code=502, detail=f"回调失败: 无法连接到更新接口 at {getattr(e, 'request', None) and e.request.url!r}.")
                    except httpx.HTTPStatusError as e:
                        print(f"[CALLBACK][dataset {dsid}] HTTP错误: status={e.response.status_code}, body={e.response.text[:200]!r}")
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
    # 注意：批量推理路径不使用会话缓存，避免占用磁盘与显存；单条接口负责多轮对话
    # 后台任务入口
    cls_total = {c: 0 for c in _VALID_REL}
    cls_correct = {c: 0 for c in _VALID_REL}
    consistency_cognition_accuracy = 0.0  # 预设为0.0，后续可根据需求调整
    per_item_results, errors = [], []

    # 1) 读取 CSV 缓存，命中的直接使用
    requested_ids = [str(it.dataset_id) for it in body.datasets]
    error_ids = _load_error_ids("error.txt")  # 若文件不存在则为空集
    existing_map = _load_existing_update_rows(project_id, requested_ids)
    for key, update_data in existing_map.items():
        dataset_id = int(key)
        print(f"[BATCH] 已缓存 dataset_id={key}")
        callback_url = f"{base_url}/v1/consistency/infer/{project_id}/{dataset_id}"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                print(f"正在向 {callback_url} 发送回调...")
                # 若存在 error.txt，则覆盖单条一致性认知准确率
                payload = update_data.dict()
                if 'error_ids' in locals() and error_ids:
                    payload["consistency_result_accuracy"] = 0.0 if str(dataset_id) in error_ids else 1.0
                # await asyncio.sleep(2)
                response = await client.put(callback_url, json=payload)
                resp_text = (response.text or "")
                print(f"[CALLBACK][cached dataset {dataset_id}] status={response.status_code}, body={resp_text[:200]!r}")
                response.raise_for_status()
        except Exception as e:
            # 降级为日志，避免中断整个批任务
            print(f"[BATCH][WARN] 缓存样本回调失败 dataset_id={dataset_id}: {e}")
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
        row = {"dataset_id": dsid, **b.dict()}
        if error_ids:
            row["consistency_result_accuracy"] = 0.0 if dsid in error_ids else 1.0
        items.append(row)
    

    # 读取每条样本的标签与一致性结果标签，用于项目级统计
    async with httpx.AsyncClient(timeout=120.0) as client:
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
                    print(f"[BATCH][WARN] 文本JSON路径不存在: {item.text_url}")
                    j = {}
                else:
                    with open(item.text_url, "r", encoding="utf-8") as jf:
                        j = json.load(jf)

            consistency_result_label = (j.get("consistency_result") or "").strip() if isinstance(j, dict) else ""

            label = b.actual_relation if b else None
            pred = b.final_relation if b else None
            consistency_result = (b.consistency_result or "").strip() if b else None
            # 计算准确率
            if label in _VALID_REL and pred in _VALID_REL:
                cls_total[label] += 1
                if pred == label:
                    cls_correct[label] += 1
            # consistency_cognition_accuracy 统计逻辑修改：只根据 error.txt 判断
            # 只在 error.txt 存在时，统计方式为：id 不在 error.txt 里为正确

    # 统计项目级
    per_class_acc = {c: (cls_correct[c] / cls_total[c]) if cls_total[c] > 0 else None for c in _VALID_REL}
    valid_accs = [v for v in per_class_acc.values() if v is not None]
    infer_relation_accuracy = sum(valid_accs) / len(valid_accs) if valid_accs else 0.0

    # consistency_cognition_accuracy 只根据 error.txt 判断
    if error_ids and len(requested_ids) > 0:
        # 项目级一致性认知准确率以 error.txt 为准
        correct_count = sum(1 for i in requested_ids if i not in error_ids)
        consistency_cognition_accuracy = correct_count / len(requested_ids)
    else:
        # 若 error.txt 不存在，保持原有逻辑
        consistency_cognition_accuracy = 0.0


    # 写入项目级 CSV
    proj_body = UpdateProjectBody(
        status=ProjectStatusEnum.COMPLETED,
        infer_relation_accuracy=infer_relation_accuracy,
        consistency_cognition_accuracy=consistency_cognition_accuracy,
        equivalence_relationship_accuracy=per_class_acc["等价"] or 0.0,
        conflict_relationship_accuracy=per_class_acc["矛盾"] or 0.0,
        causation_relationship_accuracy=per_class_acc["因果"] or 0.0,
        relation_accuracy=per_class_acc["关联"] or 0.0,
        temporal_relationship_accuracy=per_class_acc["顺承"] or 0.0  # <-- 新增
    )
    proj_header = ["project_id"] + list(UpdateProjectBody.__annotations__.keys())
    proj_row = {"project_id": project_id, **proj_body.dict()}
    await run_in_threadpool(_update_or_append_csv, PROJECT_RESULTS_FILE, proj_header, proj_row, ["project_id"])

    # 新增：通过 HTTP PUT 回调项目级接口
    callback_url = f"{base_url}/v1/consistency/project/{project_id}"
    try:
        async with httpx.AsyncClient() as client:
            print(f"[Project] 正在向 {callback_url} 发送回调...")
            response = await client.put(callback_url, json=proj_body.dict())
            resp_text = (response.text or "")
            print(f"[CALLBACK][project {project_id}] status={response.status_code}, body={resp_text[:200]!r}")
            response.raise_for_status()
    except httpx.RequestError as e:
        print(f"[CALLBACK][project {project_id}] 连接失败: {e}")
        raise HTTPException(status_code=502, detail=f"回调失败: 无法连接到更新接口 at {e.request.url!r}.")
    except httpx.HTTPStatusError as e:
        print(f"[CALLBACK][project {project_id}] HTTP错误: status={e.response.status_code}, body={e.response.text[:200]!r}")
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

# 附加：GET 版本 SSE 便于前端 EventSource 调用
@app.get(
    "/v1/consistency/stream",
    summary="[SSE] 使用会话缓存和给定 content 发起一轮对话（仅流式）"
)
async def stream_with_session(session_id: str, content: str):
    sess_dir = _session_dir_by_sid(session_id)
    with session_lock:
        meta = _load_json(os.path.join(sess_dir, "meta.json")) or {}
    if not meta:
        raise HTTPException(status_code=400, detail="未找到会话缓存，请先通过 POST 首轮上传图像与文本。")
    rgb_path = os.path.join(sess_dir, "rgb.jpg")
    ir_path = os.path.join(sess_dir, "ir.jpg")
    if not (os.path.exists(rgb_path) and os.path.exists(ir_path)):
        raise HTTPException(status_code=400, detail="会话缓存不完整：缺少图像。")
    final_text = meta.get("text")
    history = meta.get("history")

    def _gen():
        # 起始
        yield sse_format({"type": "start", "message": "stream started"})
        for evt in stream_chat(rgb_path, ir_path, final_text, content, history):
            yield evt
        # done 事件包含在 stream_chat 内部
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(_gen(), media_type="text/event-stream; charset=utf-8", headers=headers)

# ==============================================================================
# 6. 图生文以及图生图
# ==============================================================================
def load_image_from_base64(base64_str: str, input_size=448, max_num=12) -> torch.Tensor:
    if 'base64,' in base64_str:
        base64_str = base64_str.split('base64,')[-1]

    image_bytes = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(torch.bfloat16)
    
    return pixel_values

def chat4text(img_base64, is_consistency: bool = True):
    tokenizer = model_globals["tokenizer"]
    model = model_globals["model"]
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    pixel_values = load_image_from_base64(img_base64)   
    
    question = ""
    if is_consistency:
        question = '<image>\n你是一名战场分析官，我将提供敌军的情报照片，请描述这张图片，一段话。'
    else:
        question = '<image>\n你是一名战场分析官，我将提供敌军的情报照片，请对照片增加额外信息，使得文本和图片信息存在矛盾，然后再修改矛盾程度，使得与文本事实有偏离但不能太大，直接给出最终的结果，一段话。'
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    
    return response

@app.post(
    "/v1/consistency/generate/text",
    summary="[生成] 图生文"
)
async def img_generate_text(img_base64: str = Form(...)):
    return {"text": chat4text(img_base64)}

if __name__ == "__main__":
    # [修改] 遵照 guideline.md 建议，从环境变量读取配置
    port = int(os.environ.get("PORT", "8102"))
    host = os.environ.get("HOST", "0.0.0.0")
    # 仅在 RELOAD=true (不区分大小写) 时开启
    reload = os.environ.get("RELOAD", "True").lower() == "true"
    
    print(f"--- 启动服务 ---")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Reload: {reload}")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Sessions Dir: {SESSIONS_DIR}")
    print(f"Data/Logs Dir: {DATA_DIR}")
    
    uvicorn.run("chat_tools_intern_multigpu:app", host=host, port=port, reload=reload)
import json
import os
# 设置使用GPU 5
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from transformers import AutoProcessor, Glm4vForConditionalGeneration
import torch
from PIL import Image
import re
import csv  # 新增

MODEL_PATH = "/home/user/xieqiuhao/multimodel_relation/downloaded_model/GLM-4.1V-9B-Thinking"

# # 加载本地图片
# image_path1 = "/home/user/xieqiuhao/multimodel_relation/datasets/1-150/rgb_img/3.jpg"
# image_path2 = "/home/user/xieqiuhao/multimodel_relation/datasets/1-150/thermal_img/3.jpg"

prompt = (
    "现给出一对军事领域的图片（图片1、图片2）以及一段军事文本（文本1）,"
    "对于每张图像：识别图像目标主体的行动意图和场景，以及图像中各个主体之间的关系，比如敌军在持枪在逮捕、敌军在森林巡逻等；"
    "对于文本：仔细分析文本的正文细节内容，比如时间、地点、人物、事情、经过等。" # "只依据图片场景意思和文本语义内容，忽略图像不同表现形式（色彩、风格等）以及不同成像技术（可见光、热成像等），"
    "现分析它们三对之间（图像1-图像2，图像1-文本1，图像2-文本1）是什么关系，然后结合这三对关系给出总关系的判定（即对图像1-图像2-文本1的判断，输出一种关系），这四种输出关系类型分别是关联、因果、等价、矛盾。"
    "输出要求：首先给出三对关系类型，后结合这三对关系给出总关系的判定，然后给出分析，分析过程首先要说明图片/文本的具体内容，然后分析之间的关系。"
    "输出注意：如果其中一对图像对/图像-文本对被判断为矛盾关系时（不包括其他的关系），在原本的输出要求的后面（即为输出三对关系类型后面），输出对三者关系（分别对应图像1-图像2，图像1-文本1，图像2-文本1）中哪俩种是最相关联的判断，并输出是哪个模态表达的信息与其他俩个模态表达的信息相斥（如文本1表达的内容与图像1和图像2表达内容相斥），然后接着回答的开头输出对这俩种综合得到的事实。"
    "例如，当图像1识别出存在释放尾气的装甲坦克在行进，图像2识别出无尾气热力图的装甲坦克在行进，文本识别出充气式的坦克伪装成装甲坦克来混淆视听，则输出：图片1-图片2存在矛盾关系，图片2-文本1存在等价关系，图片1-文本1存在矛盾关系，图像1-图像2-文本1三者关系为矛盾关系，其中文本1表达的内容与图像1和图像2表达内容相斥，实际情况预测：敌方使用的是充气坦克。相应的图片/文本分析。"
    "关系判断时注意，1）所有关系的判断忽略表现形式上（如RGB图像-热力图的成像方式）的不同，只关注图像或者文本的表达意思；"
    "2）等价只要求图片对场景内容的意思（如图像1-图像2主体内容相同，场景相同，即为等价）或图片-文本对（图像1-文本1，图像2-文本1）表达的意思相同即可（例如图像1表达的预测是”1辆“坦克在行径，周围有周围有树木、地形，车辆旁有烟雾，而文本1表达是敌方坦克在行进，因此可以判断这段关系是等价关系，此处只要主体内容一致，即为等价关系）；"
    "2.1)对于图片1-图片2-文本1的等价关系的判断，若三对关系都是等价的或者其中2对关系是等价的，即可判断图片1-图片2-文本1之间的关系是等价关系；"
    "3）矛盾关系不关注格式、色彩、成像方式，只关注图像场景内容或者文本语义内容之间是否存在语义上的矛盾关系，比如二者同时某个主体都存在，但同一主体所在的场景或者做的事情不同，"
    "3.1）矛盾关系要注意主体的数量问题，比如图片1显示只有2辆坦克，图片2也显示只有2辆坦克，而文本却说只有1辆坦克。或者因绝对化语气（形容词、修饰词、量词等）比如人数量词等导致场景变换、行动意图、行动方式、与人员数量冲突等；"
    "3.2）矛盾关系要注意修饰词的变换问题及结论的问题，比如说图片1、2分别显示坦克在开炮，但文本显示战场安静静谧，未发现敌军踪迹。图片是坦克正在开炮（结论敌方在进攻），而文本却显示战场安静（结论敌方未精工），形容不对，事实不符，因而矛盾。"
    "4）关联关系的区分需要注意描绘范围的词，例如：图像1、2分别展示了一辆敌方坦克车辆处于燃烧状态，而文本1显示敌方装甲车被击毁，敌方装甲部队受到打击。范围从”一辆“坦克被击毁到敌方装甲部队受到打击，文本1的表达不局限在图片1、2内容上，因此是关联关系。"
    "5）因果关系要注意，图片1、图片2分别显示的内容是否与文本1所显示的内容成因果关系，文本1是否是图片1/2的原因，或者图片1/2是否是文本1的原因。"
    "6）所有的数据（图片1、图片2、文本1）都假设是对敌方的记录，不包含本方相关场景、文本的记录。"
)

# 可选文本；如无则仅图片+提示词
# text = None
# text = "密级：秘密  等级：紧急  时间：2023年5月10日  发报：军情局  收报：前线指挥部  抄送：各相关部门  主题：观察报告  正文：敌方坦克不在战场上"

def build_initial_messages(image1: Image.Image, image2: Image.Image, prompt_text: str, extra_text: str | None):
    if extra_text:
        prompt_text = f"{prompt_text} \n{extra_text}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image1},
                {"type": "image", "image": image2},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]

def load_model():
    # 加载模型与处理器
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    # 尝试优先bf16，失败则退回fp16
    dtype = torch.bfloat16
    if not torch.cuda.is_available():
        dtype = torch.float32
    model = Glm4vForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        torch_dtype=dtype,
        device_map="auto",
    )
    return processor, model

def chat(image1:str, image2:str, text:str, processor, model, eval:bool=True):
    # 加载图片
    image1 = Image.open(image1).convert("RGB")
    image2 = Image.open(image2).convert("RGB")

    # 初始化多轮消息（首轮带图）
    messages = build_initial_messages(image1, image2, prompt, text)

    def generate_and_print(current_messages: list[dict], max_new_tokens: int = 1024):
        inputs = processor.apply_chat_template(
            current_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.05,
            )
        # 仅取新生成片段
        new_tokens = generated_ids[0][inputs["input_ids"].shape[1]:]
        output_text = processor.decode(new_tokens, skip_special_tokens=False)
        print(f"\n模型：\n{output_text}\n")
        return output_text

    # 首轮回答（基于两张图 + 提示词）
    print("检测中，请稍等。")
    first_reply = generate_and_print(messages, max_new_tokens=5096)
    if not eval:
        print("你可以继续输入文本与模型对话，输入 exit/quit 退出")
        messages.append({"role": "assistant", "content": [{"type": "text", "text": first_reply}]})

        # 进入多轮对话循环（后续轮次仅追加文本）
        try:
            while True:
                user_input = input("你：").strip()
                if user_input.lower() in {"exit", "quit", "q"}:
                    print("已退出。")
                    break
                if not user_input:
                    continue

                messages.append({"role": "user", "content": [{"type": "text", "text": user_input}]})
                reply = generate_and_print(messages, max_new_tokens=5096)
                messages.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
        except KeyboardInterrupt:
            print("\n已中断。")
    return first_reply

def extract_first_paragraph_after_answer(s: str) -> str:
    # 提取 <answer> ... </answer> 中内容（若无则返回原文）
    m = re.search(r'<answer>(.*?)(?:</answer>|$)', s, flags=re.S)
    block = m.group(1).strip() if m else s.strip()
    if not block:
        return block
    # 优先在“分析过程”前截断，视为第一段
    p = re.search(r'分析过程', block)
    if p:
        return block[:p.start()].strip()
    # 否则按空行分段，取第一段
    parts = re.split(r'\n\s*\n', block, maxsplit=1)
    return parts[0].strip()


def check_label_re(response:str):
    first_para = extract_first_paragraph_after_answer(response)

    # 统一破折号/全角横线为半角-
    norm_text = first_para.replace('—', '-').replace('－', '-').replace('–', '-')

    # 实体：图片/图像/文本 + 可含空格的编号；连接符允许两侧空格
    ENTITY = r'(?:图像|图片|文本)\s*\d+'
    SEP = r'\s*-\s*'

    # 1) 成对关系：X - Y 存在 R 关系
    pair_pattern = re.compile(
        rf'({ENTITY}){SEP}({ENTITY})\s*(?:存在)\s*([\u4e00-\u9fa5]+)\s*关系'
    )

    # 2) 三者关系：X - Y - Z 三者关系为 R（也兼容 总关系为 / 整体关系为）
    triple_pattern = re.compile(
        rf'({ENTITY}){SEP}({ENTITY}){SEP}({ENTITY})\s*(?:三者关系为|总关系为|整体关系为)\s*([\u4e00-\u9fa5]+)'
    )

    # 3) 结论提取（兼容“实际情况预测/综合判断”）
    conclusion_pattern = re.compile(r'实际情况(?:预测|综合判断)?[，,：:\s]*(.*)')

    result = {
        "relationships": [],
        "conclusion": None
    }

    # 成对关系
    for a, b, rtype in pair_pattern.findall(norm_text):
        a = re.sub(r'\s+', '', a)  # 去空格：图片1、文本1 等
        b = re.sub(r'\s+', '', b)
        result["relationships"].append({
            "entities": [a, b],
            "type": rtype
        })

    # 三者关系
    for a, b, c, rtype in triple_pattern.findall(norm_text):
        a = re.sub(r'\s+', '', a)
        b = re.sub(r'\s+', '', b)
        c = re.sub(r'\s+', '', c)
        result["relationships"].append({
            "entities": [a, b, c],
            "type": rtype
        })

    # 结论
    m = conclusion_pattern.search(norm_text)
    if m:
        result["conclusion"] = m.group(1).strip()

    return result


def _norm_entity(name: str) -> str:
    # 统一“图像/图片”为“图片”，避免同义冲突
    name = name.strip()
    return re.sub(r'^图像', '图片', name)

def _disp_entity(name: str) -> str:
    # 展示时将“图片”显示为“图像”，与示例口径一致
    return re.sub(r'^图片', '图像', name)

def determine_contradiction(data: dict) -> dict:
    """
    输入形如：
    {
        "relationships": [
            {"entities": ["图片1","图片2"], "type": "等价"},
            {"entities": ["图片1","文本1"], "type": "矛盾"},
            {"entities": ["图片2","文本1"], "type": "矛盾"},
            {"entities": ["图像1","图像2","文本1"], "type": "矛盾"}
        ],
        "conclusion": "..."
    }
    输出：
    {
        "has_contradiction": True/False,
        "contradict_modality": "文本1" / "图像1" 等,
        "most_related_pair": {"entities":["图像1","图像2"], "type":"等价"} 或 None,
        "phrase": "其中文本1表达的内容与图像1和图像2表达内容相斥"
    }
    """
    rels = data.get("relationships", [])
    # 仅构建两两关系表，方便统计
    pair_type = {}  # frozenset(norm_a, norm_b) -> type
    nodes = set()

    for r in rels:
        ents = r.get("entities", [])
        rtype = r.get("type", "")
        ents_norm = [_norm_entity(e) for e in ents]
        if len(ents_norm) == 2:
            a, b = ents_norm
            pair_type[frozenset([a, b])] = rtype
            nodes.update([a, b])
        elif len(ents_norm) == 3:
            nodes.update(ents_norm)

    # 统计每个节点参与的“矛盾”条数
    contradict_count = {n: 0 for n in nodes}
    for key, t in pair_type.items():
        if t == "矛盾":
            a, b = tuple(key)
            contradict_count[a] += 1
            contradict_count[b] += 1

    result = {
        "has_contradiction": False,
        "contradict_modality": None,
        "most_related_pair": None,
        "phrase": None,
    }

    if not contradict_count or max(contradict_count.values(), default=0) == 0:
        return result  # 没有任何矛盾关系

    # 判定“矛盾的模态”= 参与矛盾最多的那个节点
    contradict_node = max(contradict_count, key=lambda k: contradict_count[k])
    if contradict_count[contradict_node] == 0:
        return result

    result["has_contradiction"] = True
    result["contradict_modality"] = _disp_entity(contradict_node)

    # 确定另外两个节点并排序（优先展示图像，再文本）
    others = sorted([n for n in nodes if n != contradict_node],
                    key=lambda x: (0 if x.startswith("图片") else 1, int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 99))
    others_disp = [_disp_entity(o) for o in others]
    result["phrase"] = f"其中{_disp_entity(contradict_node)}表达的内容与{others_disp[0]}和{others_disp[1]}表达内容相斥"

    # 选择“最相关联”的那一对（非矛盾，权重：等价>因果>关联）
    weights = {"等价": 3, "因果": 2, "关联": 1}
    candidates = []
    for key, t in pair_type.items():
        if t == "矛盾":
            continue
        a, b = tuple(key)
        candidates.append((weights.get(t, 0), t, a, b))

    # 优先不包含“矛盾模态”的那一对
    best = None
    for w, t, a, b in sorted(candidates, key=lambda x: x[0], reverse=True):
        if contradict_node not in (a, b):
            best = (t, a, b)
            break
    if not best and candidates:
        w, t, a, b = max(candidates, key=lambda x: x[0])
        best = (t, a, b)

    if best:
        t, a, b = best
        result["most_related_pair"] = {"entities": [_disp_entity(a), _disp_entity(b)], "type": t}

    return result


def check_answer(response: str, ground_truth: dict) -> dict:
    dict_response = check_label_re(response)
    rels = dict_response.get("relationships", [])

    # 优先找三者关系
    triple = next((r for r in rels if len(r.get("entities", [])) == 3), None)

    # 若缺失三者关系，基于成对关系回退一个总关系
    if triple is None:
        # 简单回退规则：有矛盾→矛盾；否则按优先级 因果 > 等价 > 关联 选最高
        pairs = [r for r in rels if len(r.get("entities", [])) == 2]
        types = [r["type"] for r in pairs]
        if "矛盾" in types:
            inferred = "矛盾"
        elif "因果" in types:
            inferred = "因果"
        elif "等价" in types:
            inferred = "等价"
        elif "关联" in types:
            inferred = "关联"
        else:
            inferred = None
        response_label = inferred
    else:
        response_label = triple["type"]

    check_truth = {}
    # 记录预测标签
    check_truth["pred_label"] = response_label
    check_truth["label"] = (response_label[:2] == ground_truth.get("label"))

    # 判断矛盾模态（仅当标注提供 error）
    if ground_truth.get("error") != None:
        filter_ans = determine_contradiction(dict_response)
        # 取“图像/文本”前缀的两个字
        cm = filter_ans.get("contradict_modality")
        error = cm[:2] if cm else None
        check_truth["pred_error"] = error
        check_truth["error"] = (error == ground_truth.get("error"))
    else:
        check_truth["pred_error"] = None
        check_truth["error"] = None

    return check_truth 

def read_json_file(file_path):
    """
    读取JSON文件并返回解析后的数据
    
    参数:
        file_path (str): JSON文件的路径
        
    返回:
        dict/list: 解析后的JSON数据，如果出错则返回None
    """
    try:
        # 打开并读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as file:
            # 解析JSON数据
            data = json.load(file)
            print(f"成功读取JSON文件: {file_path}")
            return data
            
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在")
    except json.JSONDecodeError:
        print(f"错误: 文件 '{file_path}' 不是有效的JSON格式")
    except PermissionError:
        print(f"错误: 没有权限读取文件 '{file_path}'")
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
    
    return None

def _natural_sort_key(s: str):
    # 自然排序：按文件名中的数字顺序排序（先去扩展名）
    s = os.path.splitext(s)[0]
    return [int(p) if p.isdigit() else p for p in re.split(r'(\d+)', s)]

def _build_stem_map(dir_path: str) -> dict:
    # 构建 {去扩展名: 文件名} 映射，便于按公共ID对齐
    files = [f for f in os.listdir(dir_path) if not f.startswith('.')]
    return {os.path.splitext(f)[0]: f for f in files}

def eval():
    base_datadir = r"/home/user/xieqiuhao/multimodel_relation/data_with_label/"
    rgb_dir = os.path.join(base_datadir, "rgb_img")
    ir_dir = os.path.join(base_datadir, "infrared_img")
    desc_dir = os.path.join(base_datadir, "description")

    rgb_map = _build_stem_map(rgb_dir)
    ir_map = _build_stem_map(ir_dir)
    desc_map = _build_stem_map(desc_dir)

    # 取三者公共ID并做自然排序，保证一一对应
    common_ids = sorted(set(rgb_map) & set(ir_map) & set(desc_map), key=_natural_sort_key)
    if not common_ids:
        print("未找到三者共同的样本ID，请检查文件名是否对应。")
        return

    # 输出CSV与日志目录
    out_csv = os.path.join(base_datadir, "eval_results.csv")
    log_dir = os.path.join(base_datadir, "eval_logs")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    header = [
        "id", "rgb_file", "infrared_file", "desc_file",
        "gt_label", "gt_error", "pred_label", "pred_error",
        "label_correct", "error_correct",
        "model_output"  # 新增：记录模型原始输出
    ]

    # 若CSV不存在或为空，则先写表头
    need_header = (not os.path.exists(out_csv)) or (os.path.getsize(out_csv) == 0)
    if need_header:
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    processor, model = load_model()

    # 评测范围（保留你原来的起始下标）
    num = len(common_ids)
    for i in range(9, num):
        sid = common_ids[i]
        rgb_file = rgb_map[sid]
        ir_file = ir_map[sid]
        desc_file = desc_map[sid]

        img1 = os.path.join(rgb_dir, rgb_file)
        img2 = os.path.join(ir_dir, ir_file)
        text_path = os.path.join(desc_dir, desc_file)

        json_data = read_json_file(text_path)
        if json_data is None:
            continue

        text = json_data.get("msg", "")
        response = chat(img1, img2, text, processor, model)
        result = check_answer(response, json_data)

        # 1) 模型原始输出写入独立日志（每个样本一个txt，及时写入）
        log_path = os.path.join(log_dir, f"{sid}.txt")
        try:
            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write(response)
        except Exception as e:
            print(f"写入日志失败 {log_path}: {e}")

        # 2) 将该样本评测结果立即追加写入CSV
        #    为避免多行文本破坏CSV行结构，这里将换行替换为 \n
        model_out_for_csv = response.replace("\r\n", "\n").replace("\n", r"\n")
        row = [
            sid,
            rgb_file,
            ir_file,
            desc_file,
            json_data.get("label"),
            json_data.get("error"),
            result.get("pred_label"),
            result.get("pred_error"),
            result.get("label"),
            result.get("error"),
            model_out_for_csv,
        ]
        try:
            with open(out_csv, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
                f.flush()
        except Exception as e:
            print(f"写入CSV失败 {out_csv}: {e}")

        print(f"样本 {sid} 结果已记录到CSV与日志。")

    print(f"评测完成。CSV: {out_csv}  日志目录: {log_dir}")



if __name__ == "__main__":
    # chat(image_path1,image_path2,text)
    eval()
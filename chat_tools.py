import json
import os
import sqlite3
# 设置使用GPU 5
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from transformers import AutoProcessor, Glm4vForConditionalGeneration
import torch
from PIL import Image
import re
import csv  # 新增


# LangChain 相关引用
from typing import Any, List, Optional
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_core.language_models.llms import LLM
from transformers import AutoProcessor, Glm4vForConditionalGeneration



MODEL_PATH = "/home/user/xieqiuhao/multimodel_relation/downloaded_model/GLM-4.1V-9B-Thinking"
DB_PATH = "./test.db"  # 定义数据库文件路径


# --- 新增：LangChain 自定义LLM包装类 ---
class CustomGLM(LLM):
    """
    针对本地部署的 GLM-4.1V 模型的 LangChain LLM 包装类。
    """
    processor: Any
    model: Any

    @property
    def _llm_type(self) -> str:
        return "custom_glm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """
        实现LLM的核心调用逻辑，用于Agent的文本生成。
        """
        # Agent的思考过程是纯文本的，因此我们只构建文本输入
        messages = [{"role": "user", "content": prompt}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.05,
            )

        # 解码生成的文本
        new_tokens = generated_ids[0][inputs["input_ids"].shape[1]:]
        output_text = self.processor.decode(new_tokens, skip_special_tokens=True).strip()

        # LangChain Agent依赖于特定的停止序列来分割思考和行动
        if stop is not None:
            for s in stop:
                if output_text.endswith(s):
                    output_text = output_text[:-len(s)]
        
        # 移除可能由模型生成的多余的 "Thought:" 标记，防止Agent解析混乱
        if "Thought:" in output_text:
            output_text = output_text.split("Thought:")[0]

        return output_text

# --- 新增：数据库设置函数 ---
def setup_database_from_schema(db_path: str):
    """
    根据 PDF 中的表结构，创建并初始化一个 SQLite 数据库。
    """
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # [cite_start]根据PDF文件中的表结构创建表 [cite: 12, 27, 36, 52, 53, 68, 69, 70, 71, 72, 78, 104]
    # 使用TEXT, INTEGER, REAL, BLOB来对应SQL数据类型
    # varchar -> TEXT, number -> INTEGER/REAL, timestamp -> DATETIME, enum -> INTEGER, string -> TEXT
    
    # 实体关系抽取项目表
    cursor.execute("""
    CREATE TABLE extract_projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at DATETIME,
        entity_total INTEGER,
        entity_correct_accuracy_number INTEGER,
        event_total INTEGER,
        event_accuracy REAL,
        relation_accuracy REAL
    );
    """)

    # 抽取文件表
    cursor.execute("""
    CREATE TABLE extract_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT,
        file_type TEXT,
        file_key TEXT,
        created_at DATETIME,
        label TEXT,
        project_id INTEGER
    );
    """)

    # 抽取实体表
    cursor.execute("""
    CREATE TABLE extract_entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        type TEXT,
        created_at DATETIME,
        is_event INTEGER,
        status INTEGER,
        file_id INTEGER
    );
    """)

    # 抽取关系表
    cursor.execute("""
    CREATE TABLE extract_relations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        from_id TEXT,
        to_id TEXT,
        created_at DATETIME,
        status INTEGER,
        file_id INTEGER
    );
    """)

    # 数据一致性项目表
    cursor.execute("""
    CREATE TABLE consistency_projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at DATETIME,
        infer_relation_accuracy REAL,
        consistency_cognition_accuracy_number INTEGER,
        equivalence_relationship_accur_number INTEGER,
        conflict_relationship_accuracy_number INTEGER,
        relation_accuracy REAL,
        dataset_ids TEXT
    );
    """)
    
    # 数据集表
    cursor.execute("""
    CREATE TABLE dataset (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        description TEXT,
        created_at DATETIME,
        rgb_file_id TEXT,
        infrared_file_id TEXT,
        text_file_id TEXT
    );
    """)

    # 多模态数据组表
    cursor.execute("""
    CREATE TABLE multi_model_data_groups (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER,
        rgb_file_id TEXT,
        image_file_id TEXT,
        text_file_id TEXT,
        created_at DATETIME,
        semantic_relation_accuracy REAL,
        extract_similarity REAL,
        fusion_similarity REAL,
        original_image TEXT,
        extract_image TEXT
    );
    """)

    # 为了演示效果，插入一些示例数据
    cursor.execute("""
    INSERT INTO extract_projects (id, entity_total, event_total, relation_accuracy) 
    VALUES (101, 1500, 200, 0.95), (102, 2500, 350, 0.92);
    """)
    cursor.execute("""
    INSERT INTO extract_entities (id, name, type, file_id) 
    VALUES (1, '敌军坦克', '装备', 1), (2, '士兵', '人员', 1), (3, '指挥中心', '设施', 2);
    """)
    cursor.execute("""
    INSERT INTO dataset (id, name, description) 
    VALUES (20, '沙漠风暴行动', '关于1991年海湾战争的数据集'), (21, '红旗军演', '多国空军联合演习数据集');
    """)

    conn.commit()
    conn.close()
    print(f"数据库 '{db_path}' 已创建并初始化。")




prompt = (
"""# 角色与任务
你是一名专精于多模态信息分析的军事情报分析师。你的核心任务是精准分析给定的两张军事图像和一段军事文本之间的关系。

# 输入信息
- 图像1
- 图像2
- 文本1

# 执行步骤
1.  独立分析:
    - 图像 (每张): 识别主体（如：敌军士兵、坦克）、其明确的行动意图（如：持枪逮捕、道路行进）及场景。分析图像内各主体间的关系。
    - 文本: 提取核心要素，包括时间、地点、人物、事件等细节。

2.  配对关系分析:
    - 基于独立分析的结果，判断以下三组配对的具体逻辑关系：
      - 图像1 - 图像2
      - 图像1 - 文本1
      - 图像2 - 文本1
    - 每组关系必须从【等价、关联、因果、矛盾】四种类型中选择一种。

3.  总体关系判定:
    - 综合上述三组配对关系，对【图像1-图像2-文本1】给出一个总体的关系判定，同样从四种类型中选择。

4.  生成报告:
    - 根据下文定义的【输出格式】生成最终分析报告，报告必须包含所有分析结论和支撑理由。

# 关系定义

- 等价: 描述的核心事实、主体和事件完全相同，不存在任何一方对另一方信息的扩展。
  - 判定依据: 若三对关系均为等价，或其中两对为等价，则总体关系可判定为等价。
- 关联: 描述的核心事件相关，但在范围、细节或视角上存在差异。例如，一方描述“一辆坦克”，另一方描述“装甲部队”，后者范围更广。
  - 判定依据: 若有两对关系为关联，另外一对关系为关联/等价，则总体关系可判定为关联。
- 因果: 一方是原因，另一方是结果，存在明确的时间或逻辑先后顺序（比如行动与状态）。例如，文本描述了行动或者时间在前-“我方发起轰炸”，图像展示状态或者时间在后-“轰炸后的场景”。
  - 判定依据: 若有一对关系为因果，另外两对关系为因果/等价/关联，则总体关系可判定为因果。
- 矛盾: 描述的核心事实存在直接冲突。包括但不限于：
  - 数量冲突: 图像显示2辆坦克，文本称只有1辆。
  - 行为冲突: 图像显示敌军在行进，文本称未发现敌军踪迹。
  - 状态冲突: 图像显示坦克完好，文本称其已被击毁。
  - 判定依据: 若有一对关系为矛盾，另外两对关系为矛盾/等价/关联，则总体关系可判定为矛盾。

# 输出格式

【标准格式】
图像1-图像2关系：[关系类型]
图像1-文本1关系：[关系类型]
图像2-文本1关系：[关系类型]
图像1-图像2-文本1总体关系：[关系类型]

分析过程：
1.  信息描述:
      - 图像1内容：[对图像1的简洁描述]
      - 图像2内容：[对图像2的简洁描述]
      - 文本1内容：[对文本1的核心内容概括]
2.  关系论证:
      - [对三组配对关系和总体关系的详细分析和论证]

【特殊格式：当任意一对关系被判定为“矛盾”时】

图像1-图像2关系：[关系类型]
图像1-文本1关系：[关系类型]
图像2-文本1关系：[关系类型]

最相关联的两者是：[图像1和图像2 / 图像1和文本1 / 图像2和文本1]
信息相斥的模态是：[图像1 / 图像2 / 文本1]
综合事实推断：[基于最相关联的两者，得出的一个综合事实结论]

图像1-图像2-文本1总体关系：矛盾

分析过程：
1.  信息描述:
      - 图像1内容：[对图像1的简洁描述]
      - 图像2内容：[对图像2的简洁描述]
      - 文本1内容：[对文本1的核心内容概括]
2.  关系论证:
      - [详细分析为何存在矛盾，并论证为何某两者最相关，以及为何某个模态信息相斥]

# 核心分析准则

- 语义优先: 所有判断只关注内容和语义，完全忽略图像的色彩、风格、成像技术（如“可见光”与“热成像”）等表现形式。
- 敌方视角: 所有输入信息（图、文）均是对敌方情况的记录（包括士兵、军事装备、军事车辆等）。
- 同义词兼容: 在分析中要考虑到同义词或上下位词（例如：“坦克”与“装甲车”；“草地”与“植被”；“敌军”涵盖“敌军人员、坦克、装甲车等”）。
- 语气肯定: 所有分析和结论都必须使用肯定、明确的语气，严禁使用“可能”、“也许”、“或许”等不确定性词汇。
- 避免主体替换: 在判定三对关系中，要确定同一图像的主体一致的表示（例如，“坦克”与“军事工程车”）。
"""
)



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
    # first_reply = generate_and_print(messages, max_new_tokens=2048)
    # --- 核心修改：对话部分集成 LangChain SQL Agent ---
    if not eval:
        print("你可以继续输入文本与模型对话，或询问关于数据库的问题。输入 exit/quit 退出")
        # 多轮对话不再直接调用 generate_and_print，而是使用Agent
        
        # 1. 初始化数据库
        setup_database_from_schema(DB_PATH)
        db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

        # 2. 封装本地LLM
        llm = CustomGLM(processor=processor, model=model)

        # 3. 创建SQL Agent
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True, # 开启详细模式，方便观察Agent思考过程
            handle_parsing_errors=True, # 增加鲁棒性
        )
        
        # 4. 进入Agent驱动的多轮对话循环
        try:
            while True:
                user_input = input("你：").strip()
                if user_input.lower() in {"exit", "quit", "q"}:
                    print("已退出。")
                    break
                if not user_input:
                    continue
                
                # 调用Agent执行任务
                try:
                    # 为Agent提供更明确的指令，引导它使用SQL工具
                    agent_prompt = f"""
                    你是一名智能助手。根据用户的问题，如果需要数据库中的信息来回答，请生成并执行正确的SQLite查询语句，然后根据查询结果给出最终答案。如果不需要，就直接回答。
                    用户问题: {user_input}
                    """
                    # --- 核心修改 ---
                    # 1. 使用 .invoke() 而不是 .run()
                    # 2. 传入一个字典 {"input": ...} 而不是一个字符串
                    result = agent_executor.invoke({"input": agent_prompt})
                    
                    # 3. 从返回的字典中获取 'output' 键的值
                    response_text = result.get("output", "抱歉，我无法找到答案。")
                    
                    print(f"\n模型：\n{response_text}\n")
                    # --- 修改结束 ---
                except Exception as e:
                    print(f"Agent执行出错: {e}")
                    print("抱歉，我暂时无法回答这个问题。")

        except KeyboardInterrupt:
            print("\n已中断。")
        finally:
            # 清理数据库文件
            if os.path.exists(DB_PATH):
                os.remove(DB_PATH)

    return first_reply


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

    # 统一破折号/全角横线为半角-
    norm_text = first_para.replace('—', '-').replace('－', '-').replace('–', '-')

    # 实体：图片/图像/文本 + 可含空格的编号；连接符允许两侧空格
    ENTITY = r'(?:图像|图片|文本)\s*\d+'
    SEP = r'\s*-\s*'

    # 关系类型标准化
    def _normalize_rel_type(t: str) -> str:
        t = (t or "").strip()
        # 去末尾标点与“关系”后缀
        t = re.sub(r'[。；;，,！!？?\s]+$', '', t)
        t = re.sub(r'关系$', '', t)
        # 同义词归一
        synonyms = {
            "相同": "等价", "一致": "等价", "相符": "等价",
            "相关": "关联", "联系": "关联",
            "冲突": "矛盾", "相斥": "矛盾", "相悖": "矛盾", "相矛盾": "矛盾"
        }
        return synonyms.get(t, t)

    # 旧式：X - Y 存在 R 关系
    pair_exist_pattern = re.compile(
        rf'({ENTITY}){SEP}({ENTITY})\s*(?:存在)\s*([\u4e00-\u9fa5]+)\s*关系'
    )
    # 新式：X - Y 关系：R
    pair_colon_pattern = re.compile(
        rf'({ENTITY}){SEP}({ENTITY})\s*关系\s*[：:]\s*([\u4e00-\u9fa5]+)'
    )

    # 旧式：X - Y - Z 三者/总/整体 关系为 R（R 后可带“关系”）
    triple_exist_pattern = re.compile(
        rf'({ENTITY}){SEP}({ENTITY}){SEP}({ENTITY})\s*'
        rf'(?:三者关系为|总关系为|整体关系为)\s*'
        rf'([\u4e00-\u9fa5]+)(?:关系)?'
    )
    # 新式：X - Y - Z (三者|总体|总|整体)关系：R
    triple_colon_pattern = re.compile(
        rf'({ENTITY}){SEP}({ENTITY}){SEP}({ENTITY})\s*'
        rf'(?:三者关系|总体关系|总关系|整体关系)\s*[：:]\s*'
        rf'([\u4e00-\u9fa5]+)'
    )

    # 结论提取（兼容“实际情况预测/综合判断”）
    conclusion_pattern = re.compile(r'实际情况(?:预测|综合判断)?[，,：:\s]*(.*)')

    result = {"relationships": [], "conclusion": None}

    # 去重集合
    seen = set()
    def push_rel(ents: list[str], rtype: str):
        ents = [re.sub(r'\s+', '', e) for e in ents]  # 去掉实体中的空格
        rtype = _normalize_rel_type(rtype)
        key = (tuple(ents), rtype)
        if key in seen:
            return
        seen.add(key)
        result["relationships"].append({"entities": ents, "type": rtype})

    # 成对关系
    for a, b, rtype in pair_exist_pattern.findall(norm_text):
        push_rel([a, b], rtype)
    for a, b, rtype in pair_colon_pattern.findall(norm_text):
        push_rel([a, b], rtype)

    # 三者关系
    for a, b, c, rtype in triple_exist_pattern.findall(norm_text):
        push_rel([a, b, c], rtype)
    for a, b, c, rtype in triple_colon_pattern.findall(norm_text):
        push_rel([a, b, c], rtype)

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

    # 标签归一比较（防止 None 与带“关系”的写法）
    def _norm_label_short(x: str | None) -> str | None:
        if not x:
            return None
        x = x.strip()
        x = re.sub(r'关系$', '', x)
        # 取前两字且限定到四类
        x2 = x[:2]
        return x2 if x2 in {"关联", "因果", "等价", "矛盾"} else None

    gt_label = _norm_label_short(ground_truth.get("label"))
    pr_label = _norm_label_short(response_label)

    check_truth = {}
    # 记录预测标签（保持未识别为 None）
    check_truth["pred_label"] = pr_label
    check_truth["label"] = (pr_label is not None and gt_label is not None and pr_label == gt_label)

    # 判断矛盾模态（仅当标注提供 error）
    if ground_truth.get("error") is not None:
        filter_ans = determine_contradiction(dict_response)
        cm = filter_ans.get("contradict_modality")
        error = cm[:2] if cm else None
        check_truth["pred_error"] = error
        check_truth["error"] = (error == ground_truth.get("error"))
    else:
        check_truth["pred_error"] = None
        check_truth["error"] = None

    # 把抽取到的 relationships 一并返回，便于外部记录
    check_truth["relationships"] = rels

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
        "relationships",
        "model_output",
    ]

    # 若CSV不存在或为空，则先写表头
    need_header = (not os.path.exists(out_csv)) or (os.path.getsize(out_csv) == 0)
    if need_header:
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    processor, model = load_model()

    # 评测统计（总体与分类别）
    metrics = {
        "total": 0,
        "correct": 0,
        "per_class": {}  # {label: {"total": x, "correct": y}}
    }

    # 评测范围（保留你原来的起始下标）
    num = len(common_ids)
    for i in range(0,num):
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

        text = json_data.get("text", "")
        response = chat(img1, img2, text, processor, model)
        result = check_answer(response, json_data)

        # relationships（本样本即时输出到控制台）
        rels = result.get("relationships", [])
        print(f"[{sid}] relationships: {json.dumps(rels, ensure_ascii=False)}")

        # 1) 模型原始输出写入独立日志（每个样本一个txt，及时写入）
        log_path = os.path.join(log_dir, f"{sid}.txt")
        try:
            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write(response)
        except Exception as e:
            print(f"写入日志失败 {log_path}: {e}")

        # 1.1) relationships 也单独保留为 JSON（及时写入）
        rel_path = os.path.join(log_dir, f"{sid}_relationships.json")
        try:
            with open(rel_path, "w", encoding="utf-8") as rf:
                json.dump(rels, rf, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"写入关系日志失败 {rel_path}: {e}")

        # 2) 将该样本评测结果立即追加写入CSV
        #    为避免多行文本破坏CSV行结构，这里将换行替换为 \n
        model_out_for_csv = response.replace("\r\n", "\n").replace("\n", r"\n")
        rels_for_csv = json.dumps(rels, ensure_ascii=False)
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
            rels_for_csv,
            model_out_for_csv,
        ]
        try:
            with open(out_csv, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
                f.flush()
        except Exception as e:
            print(f"写入CSV失败 {out_csv}: {e}")

        # 3) 统计（总体与分类别）
        gt_label = (json_data.get("label") or "").strip()
        pred_ok = bool(result.get("label"))
        if gt_label:
            # 仅用前两个字归一（如“等价/等价关系”统一到“等价”）
            cls = gt_label[:2]
            cls_stat = metrics["per_class"].setdefault(cls, {"total": 0, "correct": 0})
            cls_stat["total"] += 1
            if pred_ok:
                cls_stat["correct"] += 1

        metrics["total"] += 1
        if pred_ok:
            metrics["correct"] += 1

        print(f"样本 {sid} 结果已记录到CSV与日志。")

    # 评测完成后，计算并写出总体/分类别准确率
    def safe_div(a, b):
        return (a / b) if b else 0.0

    summary = {
        "overall": {
            "total": metrics["total"],
            "correct": metrics["correct"],
            "accuracy": round(safe_div(metrics["correct"], metrics["total"]), 6)
        },
        "per_class": {}
    }
    for cls, st in metrics["per_class"].items():
        summary["per_class"][cls] = {
            "total": st["total"],
            "correct": st["correct"],
            "accuracy": round(safe_div(st["correct"], st["total"]), 6)
        }

    # 写入 JSON 与 TXT 日志
    metrics_json = os.path.join(log_dir, "metrics.json")
    metrics_txt = os.path.join(log_dir, "metrics.txt")
    try:
        with open(metrics_json, "w", encoding="utf-8") as jf:
            json.dump(summary, jf, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"写入指标JSON失败 {metrics_json}: {e}")

    try:
        lines = []
        lines.append(f"Overall: {summary['overall']['correct']}/{summary['overall']['total']} "
                     f"acc={summary['overall']['accuracy']:.4f}")
        for cls, st in summary["per_class"].items():
            lines.append(f"{cls}: {st['correct']}/{st['total']} acc={st['accuracy']:.4f}")
        with open(metrics_txt, "w", encoding="utf-8") as tf:
            tf.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"写入指标TXT失败 {metrics_txt}: {e}")

    print("评测完成。")
    print(f"Overall acc: {summary['overall']['accuracy']:.4f} "
          f"({summary['overall']['correct']}/{summary['overall']['total']})")
    for cls, st in summary["per_class"].items():
        print(f"{cls} acc: {st['accuracy']:.4f} ({st['correct']}/{st['total']})")
    print(f"指标文件: {metrics_json} / {metrics_txt}")


if __name__ == "__main__":
    # # 加载本地图片
    image_path1 = "/home/user/xieqiuhao/multimodel_relation/datasets/1-150/rgb_img/3.jpg"
    image_path2 = "/home/user/xieqiuhao/multimodel_relation/datasets/1-150/thermal_img/3.jpg"
    # 可选文本；如无则仅图片+提示词
    # text = None
    text = "敌方坦克不在战场上"
    setup_database_from_schema("./test.db")
    processor, model = load_model()
    chat(image_path1,image_path2,text,processor, model,False)
    # eval()
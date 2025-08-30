import os
# 设置使用GPU 5
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from transformers import AutoProcessor, Glm4vForConditionalGeneration
import torch
from PIL import Image

MODEL_PATH = "/home/user/xieqiuhao/multimodel_relation/downloaded_model/GLM-4.1V-9B-Thinking"

# 加载本地图片
image_path1 = "/home/user/xieqiuhao/multimodel_relation/datasets/1-150/rgb_img/97.jpg"
image_path2 = "/home/user/xieqiuhao/multimodel_relation/datasets/1-150/thermal_img/95.jpg"

prompt = (
    "现给出一对图片对或者图片-文本对，对于图像：识别图像目标主体的行动意图和场景，以及图像中各个主体之间的关系，比如敌军在持枪在逮捕、敌军在森林巡逻等；对于文本：仔细分析文本的正文细节内容，比如时间、地点、人物、事情、经过等。"
    "只依据图片场景意思和文本语义内容，忽略图像不同表现形式（色彩、风格等）以及不同成像技术（可见光、热成像等），"
    "分析它们之间是什么关系，关系类型分别是关联、因果、等价、矛盾。"
    "输出要求：首先给出关系类型判断，然后给出分析，分析过程首先要说明图片/文本的具体内容，然后分析之间的关系。"
    "注意，1）所有关系的判断忽略表现形式上（如RGB图像-热力图的成像方式）的不同，只关注图像或者文本的表达意思；"
    "2）等价只要求图片场景内容的意思或文本表达的意思相同即可；"
    "3）矛盾关系不关注格式、色彩、成像方式，只关注图像场景内容或者文本语义内容之间是否存在语义上的矛盾关系，比如二者同时某个主体都存在，但同一主体所在的场景或者做的事情不同，"
    "或者因绝对化语气（形容词、修饰词、量词等）比如人数量词等导致场景、行动意图、行动方式、与人员数量冲突等；"
    "4）所有的数据都假设是对敌方的记录，不包含本方相关场景、文本的记录。"
)

# 可选文本；如无则仅图片+提示词
text = None
# text = "密级：秘密  等级：紧急  时间：2023年5月10日  发报：军情局  收报：前线指挥部  抄送：各相关部门  主题：观察报告  正文：  敌方全部人员正在城市中休整"

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
        }
    ]

def main():
    # 加载图片
    image1 = Image.open(image_path1).convert("RGB")
    image2 = Image.open(image_path2).convert("RGB")

    # 初始化多轮消息（首轮带图）
    messages = build_initial_messages(image1, image2, prompt, text)

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
                do_sample=True,
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
    first_reply = generate_and_print(messages, max_new_tokens=2048)
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
            reply = generate_and_print(messages, max_new_tokens=2048)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
    except KeyboardInterrupt:
        print("\n已中断。")

if __name__ == "__main__":
    main()
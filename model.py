import os
# 设置使用GPU 5
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from transformers import AutoProcessor, Glm4vForConditionalGeneration
import torch
from PIL import Image

MODEL_PATH = "/home/user/xieqiuhao/multimodel_relation/downloaded_model/GLM-4.1V-9B-Thinking"

# 加载本地图片
image_path1 = "/home/user/xieqiuhao/multimodel_relation/datasets/1-150/rgb_img/14.jpg"  # 替换为你的第一张图片路径
image_path2 = "/home/user/xieqiuhao/multimodel_relation/datasets/1-150/rgb_img/95.jpg"  # 替换为你的第二张图片路径

prompt = "现给出一对图片对或者图片-文本对，识别图像目标主体的行动意图和场景，以及图像中各个主体之间的关系或者文本的正文细节内容。只依据图片场景意思和文本语义内容，忽略图像不同表现形式（色彩、风格等）以及不同成像技术（可见光、热成像等），分析它们之间是什么关系，关系类型分别是关联、因果、等价、矛盾。可利用假设法、排除法等方式。输出要求：首先给出关系类型判断，然后给出分析。注意，1）所有关系的判断忽略表现形式上（如RGB图像-热力图的成像方式）的不同，只关注图像或者文本的表达意思；2）等价只要求图片场景内容的意思或文本表达的意思相同即可；3）矛盾关系不关注格式、色彩、成像方式，只关注图像场景内容或者文本语义内容之间是否存在语义上的矛盾关系，比如二者同时某个主体都存在，但同一主体所在的场景或者做的事情不同，或者因绝对化语气（修饰词、量词等）比如人数说法等导致场景、行动意图、行动方式、人员数量冲突等；4）所有的数据都假设是对敌方的记录，不包含本方相关场景、文本的记录。"

image1 = Image.open(image_path1).convert('RGB')
image2 = Image.open(image_path2).convert('RGB')
text = None
# text = "密级：秘密  等级：紧急  时间：2023年5月10日  发报：军情局  收报：前线指挥部  抄送：各相关部门  主题：观察报告  正文：  敌方全部人员正在城市中休整"
if text:
    interval = " 文本内容如下：\n"
    text = prompt + interval + text
else:
    text = prompt

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image1  # 直接使用PIL图像对象
            },
            {
                "type": "image",
                "image": image2  # 直接使用PIL图像对象
            },
            {
                "type": "text",
                "text": text
            }
        ],
    }
]
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
model = Glm4vForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=5096)
output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
print(output_text)

"""
测试 Qwen3-VL 的正确输入格式
"""
import json
from PIL import Image
from transformers import Qwen3VLProcessor, Qwen3VLForConditionalGeneration
import torch

# 加载处理器和模型
processor = Qwen3VLProcessor.from_pretrained(
    "./qwen3-vl-4b/unsloth/Qwen3-VL-4B-Instruct",
    trust_remote_code=True
)

# 测试一个样本
with open("train.jsonl", "r") as f:
    line = f.readline()
    item = json.loads(line)

print("原始数据:")
print(json.dumps(item, indent=2, ensure_ascii=False))

# 加载图像
image_path = item["image"]
try:
    image = Image.open(image_path).convert("RGB")
    print(f"\n✓ 图像加载成功: {image_path}, 尺寸: {image.size}")
except Exception as e:
    print(f"\n✗ 图像加载失败: {e}")
    image = Image.new("RGB", (224, 224), color="white")

# 方法1: 使用正确的对话格式
print("\n=== 方法1: 使用正确的对话格式 ===")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "请识别图片中的物体"}
        ]
    }
]

print("messages 格式:")
print(messages)

# 使用 apply_chat_template
text_input = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print(f"\napply_chat_template 结果:")
print(text_input)

# 处理输入
inputs = processor(
    images=image,
    text=text_input,
    return_tensors="pt"
)

print(f"\n处理后的 inputs:")
print(f"input_ids 形状: {inputs['input_ids'].shape}")
print(f"pixel_values 形状: {inputs['pixel_values'].shape}")

# 方法2: 直接使用 processor 处理对话
print("\n=== 方法2: 直接使用 processor 处理对话 ===")
inputs2 = processor(
    images=[image],
    text=[messages],
    return_tensors="pt",
    padding=True
)

print(f"\n直接处理对话结果:")
print(f"input_ids 形状: {inputs2['input_ids'].shape}")
print(f"pixel_values 形状: {inputs2['pixel_values'].shape}")

# 检查是否有 image_token_mask
if 'image_token_mask' in inputs2:
    print(f"image_token_mask 形状: {inputs2['image_token_mask'].shape}")
    print(f"image_token_mask 值: {inputs2['image_token_mask']}")

# 解码看看
print(f"\n解码 input_ids:")
decoded = processor.tokenizer.decode(inputs2['input_ids'][0])
print(decoded[:300] + "..." if len(decoded) > 300 else decoded)
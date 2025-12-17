"""
调试数据格式问题
"""
import json
from PIL import Image
from transformers import Qwen3VLProcessor

# 加载处理器
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

# 构建对话文本
conversations = item["conversations"]
text = ""
for msg in conversations:
    if msg["from"] == "user":
        text += f"<|im_start|>user\n{msg['value']}<|im_end|>\n"
    else:
        text += f"<|im_start|>assistant\n{msg['value']}<|im_end|>\n"

print(f"\n构建的文本:\n{text}")

# 使用 processor 处理
print("\n使用 processor 处理...")
inputs = processor(
    images=image,
    text=text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=256
)

print(f"\n处理后的 inputs 类型: {type(inputs)}")
print(f"input_ids 形状: {inputs['input_ids'].shape}")
print(f"input_ids 类型: {type(inputs['input_ids'])}")
print(f"attention_mask 形状: {inputs['attention_mask'].shape}")
print(f"pixel_values 形状: {inputs['pixel_values'].shape}")

# 检查是否有 image_token_mask
if hasattr(processor, 'image_token_mask'):
    print(f"image_token_mask: {processor.image_token_mask}")

# 解码看看
print(f"\n解码 input_ids:")
decoded = processor.tokenizer.decode(inputs['input_ids'][0])
print(decoded[:200] + "..." if len(decoded) > 200 else decoded)
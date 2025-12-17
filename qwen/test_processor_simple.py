"""
简单测试 Qwen3-VL processor 的正确用法
"""
from PIL import Image
from transformers import Qwen3VLProcessor
import torch

# 加载处理器
processor = Qwen3VLProcessor.from_pretrained(
    "./qwen3-vl-4b/unsloth/Qwen3-VL-4B-Instruct",
    trust_remote_code=True
)

# 加载测试图像
image = Image.open("test_images/cat01.jpeg").convert("RGB")

print("=== 测试不同输入格式 ===")

# 方法1: 使用 apply_chat_template 生成文本
print("\n方法1: 使用 apply_chat_template")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "请识别图片中的物体"}
        ]
    }
]

text_input = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print(f"生成的文本: {text_input[:100]}...")

inputs1 = processor(
    images=image,
    text=text_input,
    return_tensors="pt"
)

print(f"input_ids 形状: {inputs1['input_ids'].shape}")
print(f"pixel_values 形状: {inputs1['pixel_values'].shape}")

# 方法2: 直接使用文本和图像
print("\n方法2: 直接使用文本和图像")
# 从 apply_chat_template 的结果可以看到图像占位符
# 图像占位符是: <|vision_start|><|image_pad|><|vision_end|>
text_with_placeholder = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>请识别图片中的物体<|im_end|>\n<|im_start|>assistant\n"

inputs2 = processor(
    images=image,
    text=text_with_placeholder,
    return_tensors="pt"
)

print(f"input_ids 形状: {inputs2['input_ids'].shape}")
print(f"pixel_values 形状: {inputs2['pixel_values'].shape}")

# 解码看看
decoded = processor.tokenizer.decode(inputs2['input_ids'][0])
print(f"\n解码结果: {decoded[:200]}...")

# 检查是否有 image_token_mask
if 'image_token_mask' in inputs2:
    print(f"image_token_mask: {inputs2['image_token_mask'].shape}")
else:
    print("没有 image_token_mask")
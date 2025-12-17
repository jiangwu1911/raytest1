"""
测试加载 LoRA 权重后的模型
"""
import json
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from peft import PeftModel
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("=== 测试加载 LoRA 权重 ===")
    
    # 加载基础模型和处理器
    model_path = "./qwen3-vl-4b/unsloth/Qwen3-VL-4B-Instruct"
    lora_path = "./qwen3_vl_lora_cpu"
    
    logger.info("1. 加载基础模型和处理器...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    
    processor = Qwen3VLProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    logger.info("2. 加载 LoRA 权重...")
    try:
        model = PeftModel.from_pretrained(model, lora_path)
        logger.info("✓ LoRA 权重加载成功")
    except Exception as e:
        logger.error(f"LoRA 权重加载失败: {e}")
        logger.info("使用基础模型继续测试...")
    
    logger.info("3. 测试推理...")
    model.eval()
    
    # 读取测试样本
    with open("train.jsonl", "r") as f:
        line = f.readline()
        item = json.loads(line)
    
    # 加载图像
    image_path = item["image"]
    try:
        image = Image.open(image_path).convert("RGB")
        logger.info(f"✓ 图像加载成功: {image_path}")
    except Exception as e:
        logger.error(f"图像加载失败: {e}")
        image = Image.new("RGB", (224, 224), color="white")
    
    # 准备问题
    question_text = item["conversations"][0]["value"]
    if "<img>" in question_text and "</img>" in question_text:
        question_text = question_text.split("</img>")[1]
    
    # 准备输入
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question_text}
            ]
        }
    ]
    
    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = processor(
        images=image,
        text=text_input,
        return_tensors="pt"
    )
    
    # 生成回复
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
        )
    
    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]
    
    logger.info(f"测试图像: {image_path}")
    logger.info(f"问题: {question_text}")
    logger.info(f"生成回复: {generated_text}")
    
    # 对比原始答案
    original_answer = item["conversations"][1]["value"]
    logger.info(f"原始答案: {original_answer}")

if __name__ == "__main__":
    main()
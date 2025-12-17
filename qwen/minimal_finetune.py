"""
最小化 Qwen3-VL 微调测试
"""
import json
import torch
from PIL import Image
from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("=== 最小化 Qwen3-VL 微调测试 ===")
    
    # ========== 1. 加载模型和处理器 ==========
    logger.info("1. 加载模型和处理器...")
    
    model_path = "./qwen3-vl-4b/unsloth/Qwen3-VL-4B-Instruct"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        processor = Qwen3VLProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        logger.info(f"✓ 模型加载成功，设备: {model.device}")
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return
    
    # ========== 2. 配置 LoRA ==========
    logger.info("2. 配置 LoRA...")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,  # 非常小的 rank
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # 只训练少量模块
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # ========== 3. 准备单个训练样本 ==========
    logger.info("3. 准备训练样本...")
    
    # 读取第一个训练样本
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
    
    # 构建对话文本
    conversations = item["conversations"]
    text = ""
    for msg in conversations:
        if msg["from"] == "user":
            text += f"<|im_start|>user\n{msg['value']}<|im_end|>\n"
        else:
            text += f"<|im_start|>assistant\n{msg['value']}<|im_end|>\n"
    
    logger.info(f"训练文本长度: {len(text)} 字符")
    
    # ========== 4. 手动训练一步 ==========
    logger.info("4. 手动训练一步...")
    
    try:
        # 准备输入
        inputs = processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        ).to(model.device)
        
        # 设置标签（与输入相同，用于语言建模）
        inputs["labels"] = inputs["input_ids"].clone()
        
        logger.info(f"input_ids 形状: {inputs['input_ids'].shape}")
        logger.info(f"pixel_values 形状: {inputs['pixel_values'].shape}")
        
        # 前向传播
        model.train()
        outputs = model(**inputs)
        
        loss = outputs.loss
        logger.info(f"✓ 前向传播成功，损失: {loss.item():.4f}")
        
        # 反向传播
        loss.backward()
        logger.info("✓ 反向传播成功")
        
        # 简单优化（模拟一步训练）
        for param in model.parameters():
            if param.grad is not None:
                param.data -= 1e-4 * param.grad
        
        logger.info("✓ 参数更新成功")
        
        # ========== 5. 测试推理 ==========
        logger.info("5. 测试推理...")
        
        model.eval()
        
        # 使用相同的图像和问题
        test_image = image
        question = item["conversations"][0]["value"]
        
        # 准备输入
        messages = [
            {"role": "user", "content": question}
        ]
        
        text_input = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        test_inputs = processor(
            images=test_image,
            text=text_input,
            return_tensors="pt"
        ).to(model.device)
        
        # 生成回复
        with torch.no_grad():
            generated_ids = model.generate(
                **test_inputs,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
            )
        
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        logger.info(f"测试图像: {image_path}")
        logger.info(f"问题: {question}")
        logger.info(f"生成回复: {generated_text}")
        
        # 对比原始答案
        original_answer = item["conversations"][1]["value"]
        logger.info(f"原始答案: {original_answer}")
        
        # ========== 6. 保存模型 ==========
        logger.info("6. 保存 LoRA 权重...")
        
        model.save_pretrained("./qwen3_vl_lora_minimal")
        processor.save_pretrained("./qwen3_vl_lora_minimal")
        logger.info("✓ LoRA 权重已保存到 ./qwen3_vl_lora_minimal")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 尝试最基本的推理测试
        logger.info("尝试最基本推理测试...")
        try:
            model.eval()
            test_image = Image.new("RGB", (224, 224), color="white")
            test_text = "Hello"
            
            inputs = processor(
                images=test_image,
                text=test_text,
                return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                )
            
            generated_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            logger.info(f"基础测试生成: {generated_text}")
            
        except Exception as e2:
            logger.error(f"基础测试也失败: {e2}")

if __name__ == "__main__":
    main()
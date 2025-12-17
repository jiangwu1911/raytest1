"""
低显存 Qwen3-VL 微调方案
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
    logger.info("=== 低显存 Qwen3-VL 微调方案 ===")
    
    # ========== 1. 加载模型和处理器（使用 CPU） ==========
    logger.info("1. 加载模型和处理器（使用 CPU）...")
    
    model_path = "./qwen3-vl-4b/unsloth/Qwen3-VL-4B-Instruct"
    
    try:
        # 使用 CPU 加载，避免显存问题
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # 使用 float32 在 CPU 上
            device_map="cpu",  # 强制使用 CPU
            trust_remote_code=True
        )
        
        processor = Qwen3VLProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        logger.info(f"✓ 模型加载成功，设备: CPU")
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return
    
    # ========== 2. 配置 LoRA ==========
    logger.info("2. 配置 LoRA...")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=2,  # 非常小的 rank
        lora_alpha=4,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # 只训练最少模块
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # ========== 3. 测试推理（不训练） ==========
    logger.info("3. 测试推理能力...")
    
    try:
        model.eval()
        
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
        
        # 准备问题 - 提取纯文本部分（去掉<img>标签）
        question_text = item["conversations"][0]["value"]
        # 移除<img>标签部分，只保留文本
        if "<img>" in question_text and "</img>" in question_text:
            question_text = question_text.split("</img>")[1]

        # 准备输入 - 使用正确的 Qwen3-VL 格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question_text}
                ]
            }
        ]

        # 使用 apply_chat_template 生成文本
        text_input = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 使用 processor 处理
        test_inputs = processor(
            images=image,
            text=text_input,
            return_tensors="pt"
        )
        
        logger.info(f"input_ids 形状: {test_inputs['input_ids'].shape}")
        logger.info(f"pixel_values 形状: {test_inputs['pixel_values'].shape}")
        
        # 生成回复（在 CPU 上）
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
        logger.info(f"问题: {question_text}")
        logger.info(f"生成回复: {generated_text}")
        
        # 对比原始答案
        original_answer = item["conversations"][1]["value"]
        logger.info(f"原始答案: {original_answer}")
        
        # ========== 4. 尝试小规模训练 ==========
        logger.info("4. 尝试小规模训练（单步）...")
        
        try:
            model.train()
            
            # 准备训练数据 - 使用正确的 Qwen3-VL 对话格式
            conversations = item["conversations"]
            messages = []

            for msg in conversations:
                if msg["from"] == "user":
                    # 提取用户消息中的文本部分
                    user_text = msg["value"]
                    if "<img>" in user_text and "</img>" in user_text:
                        user_text = user_text.split("</img>")[1]

                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": user_text}
                        ]
                    })
                else:
                    # 助手消息只有文本
                    messages.append({
                        "role": "assistant",
                        "content": msg["value"]
                    })

            # 使用 apply_chat_template 生成完整的对话文本
            # 注意：对于训练，我们不需要 add_generation_prompt=True
            text_input = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            # 处理输入 - 禁用截断以避免图像令牌不匹配
            inputs = processor(
                images=image,
                text=text_input,
                return_tensors="pt",
                padding=True,
                truncation=False  # 禁用截断
            )
            
            inputs["labels"] = inputs["input_ids"].clone()
            
            # 前向传播
            outputs = model(**inputs)
            loss = outputs.loss
            logger.info(f"✓ 前向传播成功，损失: {loss.item():.4f}")
            
            # 反向传播
            loss.backward()
            logger.info("✓ 反向传播成功")
            
            # 简单优化
            for param in model.parameters():
                if param.grad is not None:
                    param.data -= 1e-5 * param.grad  # 更小的学习率
            
            logger.info("✓ 参数更新成功")
            
            # ========== 5. 保存模型 ==========
            logger.info("5. 保存 LoRA 权重...")
            
            model.save_pretrained("./qwen3_vl_lora_cpu")
            processor.save_pretrained("./qwen3_vl_lora_cpu")
            logger.info("✓ LoRA 权重已保存到 ./qwen3_vl_lora_cpu")
            
        except Exception as e:
            logger.error(f"训练失败（但推理成功）: {e}")
            logger.info("至少推理功能正常，可以保存模型配置")
            
            # 仍然保存模型配置
            model.save_pretrained("./qwen3_vl_lora_cpu")
            processor.save_pretrained("./qwen3_vl_lora_cpu")
            logger.info("✓ 模型配置已保存")
            
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
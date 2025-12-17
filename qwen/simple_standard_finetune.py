"""
Qwen3-VL 简化标准微调脚本
适合小规模测试和快速验证
"""
import os
import json
import torch
from PIL import Image
from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("=== Qwen3-VL 简化标准微调 ===")
    
    # ========== 1. 加载模型和处理器 ==========
    logger.info("1. 加载模型和处理器...")
    
    model_path = "./qwen3-vl-4b/unsloth/Qwen3-VL-4B-Instruct"
    
    # 使用 4-bit 量化减少显存使用
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
        r=8,  # 较小的 rank
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # ========== 3. 准备数据 ==========
    logger.info("3. 准备训练数据...")
    
    # 创建一个小型测试数据集
    test_data = [
        {
            "id": "test_001",
            "image": "test_images/cat01.jpeg",
            "conversations": [
                {"from": "user", "value": "<img>test_images/cat01.jpeg</img>这是什么动物？"},
                {"from": "assistant", "value": "这是一只小橘猫。"}
            ]
        },
        {
            "id": "test_002", 
            "image": "test_images/panda.jpeg",
            "conversations": [
                {"from": "user", "value": "<img>test_images/panda.jpeg</img>图片里是什么？"},
                {"from": "assistant", "value": "这是一只熊猫。"}
            ]
        }
    ]
    
    # 处理数据
    def process_example(example):
        # 加载图像
        try:
            image = Image.open(example["image"]).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224), color="white")
        
        # 构建对话
        conversations = example["conversations"]
        text = ""
        for msg in conversations:
            if msg["from"] == "user":
                text += f"<|im_start|>user\n{msg['value']}<|im_end|>\n"
            else:
                text += f"<|im_start|>assistant\n{msg['value']}<|im_end|>\n"
        
        # 使用 processor 处理
        inputs = processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256
        )
        
        # 设置标签
        inputs["labels"] = inputs["input_ids"].clone()
        
        return inputs
    
    # 创建数据集
    dataset_dict = {
        "id": [item["id"] for item in test_data],
        "image": [item["image"] for item in test_data],
        "conversations": [item["conversations"] for item in test_data]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    processed_dataset = dataset.map(
        process_example,
        remove_columns=["id", "image", "conversations"],
        batched=False
    )
    
    logger.info(f"✓ 数据集准备完成，大小: {len(processed_dataset)}")
    
    # ========== 4. 配置训练 ==========
    logger.info("4. 配置训练参数...")
    
    training_args = TrainingArguments(
        output_dir="./test_output",
        num_train_epochs=1,  # 只训练 1 个 epoch
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=2,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        max_grad_norm=0.3,
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,
        model=model,
        padding=True,
        max_length=256,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
    )
    
    # ========== 5. 训练 ==========
    logger.info("5. 开始训练（小规模测试）...")
    
    try:
        trainer.train()
        logger.info("✓ 训练完成")
        
        # ========== 6. 保存模型 ==========
        logger.info("6. 保存 LoRA 权重...")
        
        model.save_pretrained("./qwen3_vl_lora_simple")
        processor.save_pretrained("./qwen3_vl_lora_simple")
        logger.info("✓ LoRA 权重已保存")
        
        # ========== 7. 测试推理 ==========
        logger.info("7. 测试推理...")
        
        # 切换到评估模式
        model.eval()
        
        # 测试第一个样本
        test_item = test_data[0]
        test_image = Image.open(test_item["image"]).convert("RGB")
        question = test_item["conversations"][0]["value"]
        
        # 准备输入
        messages = [
            {"role": "user", "content": question}
        ]
        
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = processor(
            images=test_image,
            text=text,
            return_tensors="pt"
        ).to(model.device)
        
        # 生成回复
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        logger.info(f"测试图像: {test_item['image']}")
        logger.info(f"问题: {question}")
        logger.info(f"生成回复: {generated_text}")
        
        # 对比原始答案
        original_answer = test_item["conversations"][1]["value"]
        logger.info(f"原始答案: {original_answer}")
        
    except Exception as e:
        logger.error(f"训练或测试失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 尝试最基本的测试
        logger.info("尝试最基本测试...")
        try:
            model.eval()
            test_image = Image.new("RGB", (224, 224), color="white")
            test_text = "用户: 你好\n助手:"
            
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

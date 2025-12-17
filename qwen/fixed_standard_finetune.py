"""
修复后的 Qwen3-VL 标准微调脚本
解决数据格式和图像特征问题
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

def load_dataset_from_jsonl(file_path):
    """从 JSONL 文件加载数据集"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data

def create_dataset_object(data):
    """创建 HuggingFace Dataset 对象"""
    dataset_dict = {
        "id": [],
        "image_path": [],
        "conversations": []
    }
    
    for item in data:
        dataset_dict["id"].append(item["id"])
        dataset_dict["image_path"].append(item["image"])
        dataset_dict["conversations"].append(item["conversations"])
    
    return Dataset.from_dict(dataset_dict)

def process_function(example, processor):
    """处理单个样本 - 修复版本"""
    # 加载图像
    image_path = example["image_path"]
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.warning(f"无法加载图像 {image_path}: {e}")
        # 使用空白图像作为占位符
        image = Image.new("RGB", (224, 224), color="white")
    
    # 构建对话文本
    conversations = example["conversations"]
    text = ""
    for msg in conversations:
        if msg["from"] == "user":
            text += f"<|im_start|>user\n{msg['value']}<|im_end|>\n"
        else:
            text += f"<|im_start|>assistant\n{msg['value']}<|im_end|>\n"
    
    # 使用 processor 处理图像和文本
    # 关键修复：确保返回张量而不是字典
    inputs = processor(
        images=image,
        text=text,
        return_tensors="pt",
        padding="max_length",  # 使用 max_length 确保统一长度
        truncation=True,
        max_length=256
    )
    
    # 返回字典格式，确保所有值都是张量
    return {
        "input_ids": inputs["input_ids"][0],  # 去掉 batch 维度
        "attention_mask": inputs["attention_mask"][0],
        "pixel_values": inputs["pixel_values"][0],
        "labels": inputs["input_ids"][0].clone()  # 标签与输入相同
    }

def main():
    logger.info("=== 修复版 Qwen3-VL 标准微调 ===")
    
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
    
    # 加载训练数据
    train_data = load_dataset_from_jsonl("train.jsonl")
    train_dataset = create_dataset_object(train_data)
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    
    # 处理数据集 - 使用修复的函数
    def process_example(example):
        return process_function(example, processor)
    
    processed_dataset = train_dataset.map(
        process_example,
        remove_columns=["id", "image_path", "conversations"],
        batched=False
    )
    
    logger.info(f"✓ 数据集处理完成")
    
    # ========== 4. 自定义数据整理器 ==========
    logger.info("4. 创建自定义数据整理器...")
    
    class Qwen3VLDataCollator:
        def __init__(self, processor):
            self.processor = processor
            
        def __call__(self, features):
            # 手动整理批次
            batch = {
                "input_ids": torch.stack([f["input_ids"] for f in features]),
                "attention_mask": torch.stack([f["attention_mask"] for f in features]),
                "pixel_values": torch.stack([f["pixel_values"] for f in features]),
                "labels": torch.stack([f["labels"] for f in features]),
            }
            return batch
    
    data_collator = Qwen3VLDataCollator(processor)
    
    # ========== 5. 配置训练参数 ==========
    logger.info("5. 配置训练参数...")
    
    training_args = TrainingArguments(
        output_dir="./qwen3_vl_finetuned_fixed",
        num_train_epochs=1,  # 先训练 1 个 epoch 测试
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
        dataloader_num_workers=0,
    )
    
    # ========== 6. 创建 Trainer ==========
    logger.info("6. 创建 Trainer...")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
    )
    
    # ========== 7. 训练 ==========
    logger.info("7. 开始训练...")
    
    try:
        train_result = trainer.train()
        logger.info(f"✓ 训练完成，损失: {train_result.training_loss:.4f}")
        
        # ========== 8. 保存模型 ==========
        logger.info("8. 保存 LoRA 权重...")
        
        model.save_pretrained("./qwen3_vl_lora_fixed")
        processor.save_pretrained("./qwen3_vl_lora_fixed")
        logger.info("✓ LoRA 权重已保存")
        
        # ========== 9. 测试推理 ==========
        logger.info("9. 测试推理...")
        
        # 切换到评估模式
        model.eval()
        
        # 测试第一个样本
        test_item = train_data[0]
        test_image = Image.open(test_item["image"]).convert("RGB")
        question = test_item["conversations"][0]["value"]
        
        # 准备输入 - 使用 processor 的正确方式
        messages = [
            {"role": "user", "content": question}
        ]
        
        # 使用 apply_chat_template
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

if __name__ == "__main__":
    main()
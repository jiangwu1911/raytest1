"""
Qwen3-VL 标准微调脚本
使用 Transformers 库进行完整的多模态微调
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

# 设置日志
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
    """处理单个样本"""
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
    inputs = processor(
        images=image,
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # 准备标签（只对助手回复进行预测）
    # 这里简化处理，实际可能需要更复杂的标签掩码
    inputs["labels"] = inputs["input_ids"].clone()
    
    return inputs

def main():
    # ========== 1. 配置 ==========
    logger.info("开始 Qwen3-VL 标准微调")
    
    # 模型路径
    model_path = "./qwen3-vl-4b/unsloth/Qwen3-VL-4B-Instruct"
    
    # 量化配置（可选，减少显存使用）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    # ========== 2. 加载模型和处理器 ==========
    logger.info("加载模型和处理器...")
    
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
    
    logger.info(f"模型加载成功，设备: {model.device}")
    
    # ========== 3. 配置 LoRA ==========
    logger.info("配置 LoRA...")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "vision_model.encoder.layers.*.self_attn.*_proj"  # 视觉编码器的注意力层
        ],
        bias="none",
        modules_to_save=["lm_head", "embed_tokens"]  # 保存这些层的完整权重
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # ========== 4. 加载数据 ==========
    logger.info("加载数据集...")
    
    # 加载训练数据
    train_data = load_dataset_from_jsonl("train.jsonl")
    train_dataset = create_dataset_object(train_data)
    
    # 如果有验证数据
    val_dataset = None
    if os.path.exists("val.jsonl"):
        val_data = load_dataset_from_jsonl("val.jsonl")
        val_dataset = create_dataset_object(val_data)
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"验证集大小: {len(val_dataset)}")
    
    # 处理数据集
    def process_example(example):
        return process_function(example, processor)
    
    train_dataset = train_dataset.map(
        process_example,
        remove_columns=["id", "image_path", "conversations"],
        batched=False
    )
    
    if val_dataset:
        val_dataset = val_dataset.map(
            process_example,
            remove_columns=["id", "image_path", "conversations"],
            batched=False
        )
    
    # ========== 5. 配置训练参数 ==========
    logger.info("配置训练参数...")
    
    training_args = TrainingArguments(
        output_dir="./qwen3_vl_finetuned_standard",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_steps=5,
        eval_steps=50 if val_dataset else None,
        save_steps=100,
        save_total_limit=2,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
    )
    
    # ========== 6. 创建 Trainer ==========
    logger.info("创建 Trainer...")
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,
        model=model,
        padding=True,
        max_length=512,
        pad_to_multiple_of=8
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
    )
    
    # ========== 7. 训练 ==========
    logger.info("开始训练...")
    
    try:
        train_result = trainer.train()
        logger.info(f"训练完成，损失: {train_result.training_loss:.4f}")
        
        # ========== 8. 保存模型 ==========
        logger.info("保存模型...")
        
        # 保存 LoRA 权重
        model.save_pretrained("./qwen3_vl_lora_standard")
        processor.save_pretrained("./qwen3_vl_lora_standard")
        
        # 保存完整模型（合并权重）
        model = model.merge_and_unload()
        model.save_pretrained("./qwen3_vl_finetuned_standard")
        processor.save_pretrained("./qwen3_vl_finetuned_standard")
        
        logger.info("模型保存完成")
        
        # ========== 9. 测试推理 ==========
        logger.info("测试推理...")
        
        # 加载一个测试样本
        if len(train_data) > 0:
            test_item = train_data[0]
            image_path = test_item["image"]
            question = test_item["conversations"][0]["value"]
            
            # 加载图像
            test_image = Image.open(image_path).convert("RGB")
            
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
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            
            generated_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            logger.info(f"测试图像: {image_path}")
            logger.info(f"问题: {question}")
            logger.info(f"生成回复: {generated_text}")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 尝试简单的测试
        logger.info("尝试简单测试...")
        try:
            # 使用纯文本测试
            test_text = "用户: 你好\n助手:"
            test_image = Image.new("RGB", (224, 224), color="white")
            
            inputs = processor(
                images=test_image,
                text=test_text,
                return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=20,
                )
            
            generated_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            logger.info(f"简单测试生成: {generated_text}")
            
        except Exception as e2:
            logger.error(f"简单测试也失败: {e2}")

if __name__ == "__main__":
    main()
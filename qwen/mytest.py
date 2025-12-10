# unsloth_finetune_fixed.py
"""
修复后的 unsloth 微调脚本 - 完全避免图像处理
"""
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import json
import re

# ========== 1. 加载模型 ==========
print("使用 unsloth 加载模型...")
model_path = "./qwen3-vl-4b/unsloth/Qwen3-VL-4B-Instruct"

# Unsloth 会自动处理模型加载
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

print(f"✓ 模型加载成功，设备: {model.device}")

# ========== 2. 配置 LoRA ==========
print("配置 LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

print("✓ LoRA 配置完成")

# ========== 3. 加载数据 ==========
print("加载数据...")

def load_data_to_json(file_path, output_json="dataset.json"):
    """将 JSONL 转换为 unsloth 需要的格式，移除或替换图像标记"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            conversations = item["conversations"]
            
            # 构建对话文本，移除或替换图像标记
            text = ""
            for msg in conversations:
                msg_text = msg['value']
                # 替换 <img> 标记为文本描述，避免tokenizer尝试加载图像
                if "<img>" in msg_text:
                    # 方法1：完全移除图像标记
                    msg_text = re.sub(r'<img>.*?</img>', '[图像]', msg_text)
                    # 或者方法2：用纯文本描述代替
                    # msg_text = re.sub(r'<img>.*?</img>', '这是一张图片', msg_text)
                
                if msg["from"] == "user":
                    text += f"<|im_start|>user\n{msg_text}<|im_end|>\n"
                else:
                    text += f"<|im_start|>assistant\n{msg_text}<|im_end|>\n"
            
            data.append({"text": text})
    
    # 保存为 JSON 文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 转换完成，保存到: {output_json}")
    return output_json

# 转换数据
try:
    dataset_file = load_data_to_json("train.jsonl")
    
    # 加载数据集
    dataset = load_dataset("json", data_files=dataset_file, split="train")
    print(f"✓ 数据集加载成功，大小: {len(dataset)}")
    
    # 显示示例
    if len(dataset) > 0:
        print("\n示例数据:")
        print("-" * 40)
        sample_text = dataset[0]["text"]
        print(sample_text[:200] + "..." if len(sample_text) > 200 else sample_text)
        print("-" * 40)
        
except Exception as e:
    print(f"数据加载失败: {e}")
    # 如果没有数据文件，创建一个小示例（无图像标记）
    print("创建示例数据集...")
    data = [{"text": "<|im_start|>user\n描述一张图片<|im_end|>\n<|im_start|>assistant\n这是一张风景图片。<|im_end|>"}]
    dataset = load_dataset('json', data_files={'train': data})['train']

# ========== 4. 训练 ==========
print("\n开始训练...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,  # 先用较小的长度
    dataset_num_proc=1,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=1,  # 减小 batch size
        gradient_accumulation_steps=2,
        warmup_steps=2,
        max_steps=10,  # 先训练10步测试
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        save_strategy="no",  # 测试时不保存
    ),
)

print("训练配置完成，开始训练...")
try:
    trainer_stats = trainer.train()
    print(f"✓ 训练完成，损失: {trainer_stats.training_loss:.4f}")
    
    # ========== 5. 保存模型 ==========
    print("\n保存模型...")
    
    # 保存 LoRA 权重
    model.save_pretrained("qwen3_vl_lora_unsloth")
    tokenizer.save_pretrained("qwen3_vl_lora_unsloth")
    print("✓ LoRA 权重已保存")
    
    # 尝试保存完整模型（可选）
    try:
        model.save_pretrained_merged("qwen3_vl_finetuned", tokenizer, save_method="merged_16bit")
        print("✓ 完整模型已保存")
    except Exception as e:
        print(f"完整模型保存失败（可忽略）: {e}")
    
    # ========== 6. 测试模型 ==========
    print("\n测试模型生成...")
    
    # 切换到推理模式
    FastLanguageModel.for_inference(model)
    
    # 重要：使用 tokenizer 时禁用图像处理
    print("测试1: 纯文本推理（无图像标记）")
    print("-" * 40)
    
    # 使用纯文本，确保没有任何图像标记
    test_input = "<|im_start|>user\n请描述一张风景图片。<|im_end|>\n<|im_start|>assistant\n"
    
    print(f"测试输入: {test_input}")
    
    # 直接使用 tokenizer 处理文本
    inputs = tokenizer(test_input, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"完整响应: {response}")
    print(f"生成部分: {response[len(test_input):]}")
    print("-" * 40)
    
    # 测试2：使用训练数据中的格式
    print("\n测试2: 使用处理后的训练数据格式")
    print("-" * 40)
    
    if len(dataset) > 0:
        sample = dataset[0]["text"]
        # 提取用户部分，添加助手标记
        parts = sample.split("<|im_start|>assistant")
        if len(parts) > 0:
            test_input = parts[0] + "<|im_start|>assistant\n"
            
            print(f"测试输入: {test_input[:100]}...")
            
            inputs = tokenizer(test_input, return_tensors="pt").to("cuda")
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"完整响应: {response}")
            print(f"生成部分: {response[len(test_input):]}")
    
    print("-" * 40)
    
    print("\n" + "=" * 60)
    print("✅ Unsloth 微调成功完成！")
    print("=" * 60)
    
except Exception as e:
    print(f"训练失败: {e}")
    import traceback
    traceback.print_exc()
    
    # 更简单的测试
    print("\n尝试最简单的推理测试...")
    try:
        model.eval()
        # 确保没有任何图像标记
        test_input = "用户: 你好\n助手:"
        print(f"测试输入: {test_input}")
        
        inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"基础测试生成: {response}")
    except Exception as e2:
        print(f"基础测试失败: {e2}")
        print("这可能是因为tokenizer仍在尝试处理图像标记")
        print("尝试最简化的测试...")
        try:
            # 使用最简短的文本
            test_input = "Hello"
            inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"最简化测试: {response}")
        except Exception as e3:
            print(f"所有测试都失败: {e3}")

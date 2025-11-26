from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = "cpu"
    return device

def inference(text, model_path='saved_model/best_model', label=None, device=None):
    # 获取设备
    if device is None:
        device = get_device()
    
    # 加载tokenizer和模型
    print(f"正在从 {model_path} 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    # Move input tensors to the specified device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Set the model to evaluation mode and perform inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted probabilities
    probabilities = torch.softmax(logits, dim=1)
    pred_label_idx = torch.argmax(logits, dim=1).item()
    confidence = probabilities[0][pred_label_idx].item()
    
    # Map label index to meaningful text
    label_map = {0: "假新闻", 1: "真新闻"}
    predicted_label = label_map.get(pred_label_idx, f"未知标签 {pred_label_idx}")
    
    print(f"📰 文本: {text[:100]}...")
    print(f"🔮 预测结果: {predicted_label} (索引: {pred_label_idx})")
    print(f"📊 置信度: {confidence:.4f}")
    
    if label is not None:
        actual_label = label_map.get(label, f"未知标签 {label}")
        print(f"✅ 实际标签: {actual_label}")
        print(f"🎯 预测{'正确' if pred_label_idx == label else '错误'}")
    
    print("-" * 50)
    return pred_label_idx, confidence

# 批量推理函数
def batch_inference(texts, model_path='saved_model/best_model', labels=None, device=None):
    if device is None:
        device = get_device()
    
    # 加载tokenizer和模型
    print(f"正在从 {model_path} 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    
    # Tokenize all texts
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform batch inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get predictions
    probabilities = torch.softmax(logits, dim=1)
    pred_label_indices = torch.argmax(logits, dim=1).cpu().numpy()
    confidences = probabilities.max(dim=1).values.cpu().numpy()
    
    # Map label indices to meaningful text
    label_map = {0: "假新闻", 1: "真新闻"}
    
    print("📊 批量推理结果:")
    print("=" * 60)
    for i, (text, pred_idx, conf) in enumerate(zip(texts, pred_label_indices, confidences)):
        predicted_label = label_map.get(pred_idx, f"未知标签 {pred_idx}")
        print(f"{i+1}. 预测: {predicted_label} | 置信度: {conf:.4f}")
        print(f"   文本: {text[:80]}...")
        if labels is not None:
            actual_label = label_map.get(labels[i], f"未知标签 {labels[i]}")
            correct = "✅" if pred_idx == labels[i] else "❌"
            print(f"   实际: {actual_label} {correct}")
        print("-" * 40)
    
    return pred_label_indices, confidences

# 测试函数
def test_saved_model():
    """测试保存的模型是否正常工作"""
    print("🧪 测试保存的模型...")
    
    # 测试文本
    test_texts = [
        "Scientists have discovered a new breakthrough in renewable energy technology that could revolutionize the industry.",
        "BREAKING: Celebrities are hiding the secret to eternal youth from the public! You won't believe what they know!",
        "The government announced new economic policies today that aim to stimulate growth and create jobs.",
        "SHOCKING: Government cover-up of alien contact revealed by anonymous sources!"
    ]
    
    test_labels = [1, 0, 1, 0]  # 1=真新闻, 0=假新闻
    
    # 测试单个推理
    print("单个推理测试:")
    for i, text in enumerate(test_texts[:2]):
        inference(text, label=test_labels[i])
    
    # 测试批量推理
    print("\n批量推理测试:")
    batch_inference(test_texts, labels=test_labels)

if __name__ == "__main__":
    # 运行测试
    test_saved_model()
    
    # 示例：使用不同的模型路径
    # inference("Your text here", model_path='saved_model/final_model')
    # inference("Your text here", model_path='saved_model/inference_model')

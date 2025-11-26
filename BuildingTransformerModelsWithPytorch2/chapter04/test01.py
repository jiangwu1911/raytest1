import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import os

def get_device():
    device="cpu"
    if torch.cuda.is_available():
        device="cuda"
    elif torch.backends.mps.is_available():
        device='mps'
    else:
        device="cpu"
    return device

device = get_device()
print(device) 

df = pd.read_csv('train.tsv', sep='\t')
print("数据列名:", df.columns.tolist())
print("数据形状:", df.shape)
print("标签分布:")
print(df['label'].value_counts())

df = df[['text', 'label']].dropna()
df['label'] = df['label'].astype(float)

data=list(zip(df['text'].tolist(), df['label'].tolist()))

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# This function takes list of Texts, and Labels as Parameter
# This function return input_ids, attention_mask, and labels_out
def tokenize_and_encode(texts, labels):
    input_ids, attention_masks, labels_out = [], [], []
    for text, label in zip(texts, labels):
        encoded = tokenizer.encode_plus(text, max_length=512, padding='max_length', truncation=True)
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        labels_out.append(label)
    return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(labels_out)

# seprate the tuples
# generate two lists: a) containing texts, b) containing labels
texts, labels = zip(*data)

# train, validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

# tokenization
train_input_ids, train_attention_masks, train_labels = tokenize_and_encode(train_texts, train_labels)
val_input_ids, val_attention_masks, val_labels = tokenize_and_encode(val_texts, val_labels)

class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks, labels, num_classes=2):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.num_classes = num_classes
        self.one_hot_labels = self.one_hot_encode(labels, num_classes)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.one_hot_labels[idx]
        }

    @staticmethod
    def one_hot_encode(targets, num_classes):
        targets = targets.long()
        one_hot_targets = torch.zeros(targets.size(0), num_classes)
        one_hot_targets.scatter_(1, targets.unsqueeze(1), 1.0)
        return one_hot_targets
        
train_dataset = TextClassificationDataset(train_input_ids, train_attention_masks, train_labels)
val_dataset = TextClassificationDataset(val_input_ids, val_attention_masks, val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_dataloader = DataLoader(val_dataset, batch_size=8)

print(len(train_dataset))
print(len((val_dataset)))

item=next(iter(train_dataloader))
item_ids,item_mask,item_labels=item['input_ids'],item['attention_mask'],item['labels']
print ('item_ids, ',item_ids.shape, '\n',
       'item_mask, ',item_mask.shape, '\n',
       'item_labels, ',item_labels.shape, '\n',)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# 创建保存模型的目录
os.makedirs('saved_model', exist_ok=True)

num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps
  )
progress_bar = tqdm(range(num_training_steps))

best_accuracy = 0.0  # 用于保存最佳模型

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {epoch_loss/len(train_dataloader):.4f}")
    
    # 评估模型
    model.eval()
    preds = []
    out_label_ids = []

    for batch in eval_dataloader:
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits

        preds.extend(torch.argmax(logits.detach().cpu(), dim=1).numpy())
        out_label_ids.extend(torch.argmax(inputs["labels"].detach().cpu(),dim=1).numpy())
    
    accuracy = accuracy_score(out_label_ids, preds)
    f1 = f1_score(out_label_ids, preds, average='weighted')
    recall = recall_score(out_label_ids, preds, average='weighted')
    precision = precision_score(out_label_ids, preds, average='weighted')

    print(f"Epoch {epoch + 1}/{num_epochs} Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")

    # 保存最佳模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        # 使用accelerator保存（推荐方式）
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            'saved_model/best_model',
            save_function=accelerator.save
        )
        tokenizer.save_pretrained('saved_model/best_model')
        print(f"✅ 保存最佳模型，准确率: {accuracy:.4f}")

# 保存最终模型
print("正在保存最终模型...")
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(
    'saved_model/final_model',
    save_function=accelerator.save
)
tokenizer.save_pretrained('saved_model/final_model')
print("✅ 最终模型已保存")

# 保存训练完成的模型用于推理
print("正在保存推理模型...")
unwrapped_model.save_pretrained('saved_model/inference_model')
tokenizer.save_pretrained('saved_model/inference_model')
print("✅ 推理模型已保存")

print("\n🎉 训练完成！模型已保存到以下位置：")
print("   - saved_model/best_model/    (最佳性能模型)")
print("   - saved_model/final_model/   (最终训练模型)") 
print("   - saved_model/inference_model/ (推理专用模型)")

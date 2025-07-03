
import torch
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertModel, AdamW
import torch.nn as nn
import os
import random
import matplotlib.pyplot as plt


class ChineseNewsDataset(torch.utils.data.Dataset):
    """中文新闻分类数据集类"""
    def __init__(self, data_file, vocab_file=None, split='train'):
        # 读取数据文件
        self.data = []
        self.labels = []
        # 类别映射
        self.label_map = {
            '体育': 0, '财经': 1, '房产': 2, '家居': 3, '教育': 4,
            '科技': 5, '时尚': 6, '时政': 7, '游戏': 8, '娱乐': 9
        }
        
        print(f"正在加载数据集: {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 分割标签和文本
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        label, text = parts
                        if label in self.label_map:
                            self.data.append(text)
                            self.labels.append(self.label_map[label])
        print(f"✓ 成功加载数据集: {len(self.data)} 条数据")
        print(f"类别分布: {self.get_label_distribution()}")
        # 转换为HuggingFace Dataset格式
        dataset_dict = {
            'text': self.data,
            'label': self.labels
        }
        self.dataset = Dataset.from_dict(dataset_dict)

    def get_label_distribution(self):
        """获取标签分布"""
        from collections import Counter
        counter = Counter(self.labels)
        return dict(counter)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        row = self.dataset[i]
        return row['text'], row['label']


def collate_fn(data):
    """数据批处理函数"""
    texts = [i[0] for i in data]
    labels = [i[1] for i in data]

    # 编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=texts,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt',
        return_length=True,
        add_special_tokens=True
    )

    # input_ids:编码之后的数字
    # attention_mask:是补零的位置是0,其他位置是1
    # token_type_ids:第一个句子和特殊符号的位置是0,第二个句子的位置是1
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    token_type_ids = data['token_type_ids'].to(device)
    labels = torch.LongTensor(labels).to(device)

    return input_ids, attention_mask, token_type_ids, labels


class ChineseNewsModel(torch.nn.Module):
    """中文新闻分类模型"""
    def __init__(self, pretrained_model, num_classes=10):
        super().__init__()
        self.pretrained = pretrained_model
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = self.pretrained(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

        # 使用[CLS]标记的输出进行分类
        pooled_output = out.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


def train_model(model, train_loader, num_epochs=3):
    """训练模型，记录损失和准确率"""
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    epoch_losses = []
    epoch_accuracies = []
    step_losses = []
    step_accuracies = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            preds = logits.argmax(dim=1)
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += len(labels)
            total_loss += loss.item()

            # 记录每个step的loss和accuracy
            step_losses.append(loss.item())
            step_accuracies.append(correct / len(labels))

            if i % 50 == 0:
                accuracy = total_correct / total_samples
                avg_loss = total_loss / (i + 1)
                print(f"Step {i}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

            if i == 1000:
                break

        final_accuracy = total_correct / total_samples
        final_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} 完成 - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")

        # 记录每个epoch的loss和accuracy
        epoch_losses.append(final_loss)
        epoch_accuracies.append(final_accuracy)

    return step_losses, step_accuracies, epoch_losses, epoch_accuracies


def evaluate_model(model, test_loader, save_report_path=None):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(test_loader):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if i % 10 == 0:
                print(f"Testing batch {i}, Current accuracy: {correct/total:.4f}")

    accuracy = correct / total
    print(f"最终测试准确率: {accuracy:.4f}")
    from sklearn.metrics import classification_report
    label_names = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    print("\n分类报告:")
    report = classification_report(all_labels, all_preds, target_names=label_names, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=label_names))
    if save_report_path is not None:
        pd.DataFrame(report).transpose().to_csv(save_report_path)
    return accuracy


def main():
    """主函数"""
    print("开始中文新闻分类任务 - CUDA GPU版本...")
    
    # 检查CUDA可用性
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    if device == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(" 警告: 未检测到CUDA，将使用CPU训练（速度较慢）")
    
    # 1. 加载数据集
    print("\n1. 加载数据集...")
    train_file = "mydata/cnews.train.txt"
    test_file = "mydata/cnews.test.txt"
    val_file = "mydata/cnews.val.txt"
    
    if not os.path.exists(train_file):
        print(f" 训练文件不存在: {train_file}")
        return
    
    train_dataset = ChineseNewsDataset(train_file, split='train')
    test_dataset = ChineseNewsDataset(test_file, split='test')
    val_dataset = ChineseNewsDataset(val_file, split='validation')
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 2. 加载tokenizer (使用本地模型)
    print("\n2. 加载tokenizer (本地模型)...")
    global token
    try:
        token = BertTokenizer.from_pretrained('models/bert-base-chinese')
        print("✓ 成功加载本地tokenizer")
    except Exception as e:
        print(f"加载本地tokenizer失败: {e}")
        print("尝试从网络下载...")
        token = BertTokenizer.from_pretrained('bert-base-chinese')
        print("✓ 成功从网络下载tokenizer")
    
    print(f"Tokenizer: {token}")
    
    # 3. 创建数据加载器
    print("\n3. 创建数据加载器...")
    # GPU版本可以使用更大的批次大小
    batch_size = 32 if device == 'cuda' else 16
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size * 2,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size * 2,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False
    )
    
    # 测试数据加载器
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
        break
    print(f"数据加载器大小: {len(train_loader)}")
    print(f"批次大小: {batch_size}")
    print(f"输入形状: {input_ids.shape}, {attention_mask.shape}, {token_type_ids.shape}")
    print(f"标签: {labels}")
    print(f"数据设备: {input_ids.device}")
    print(f"解码示例: {token.decode(input_ids[0])}")
    
    # 4. 加载预训练模型 (使用本地模型)
    print("\n4. 加载预训练模型 (本地模型)...")
    try:
        pretrained = BertModel.from_pretrained('models/bert-base-chinese')
        print("✓ 成功加载本地预训练模型")
    except Exception as e:
        print(f" 加载本地预训练模型失败: {e}")
        print("尝试从网络下载...")
        pretrained = BertModel.from_pretrained('bert-base-chinese')
        print("✓ 成功从网络下载预训练模型")
    pretrained.to(device)
    print(f"预训练模型设备: {next(pretrained.parameters()).device}")

    for param in pretrained.parameters():
        param.requires_grad_(False)
    
    # 模型试算
    out = pretrained(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )
    print(f"预训练模型输出形状: {out.last_hidden_state.shape}")
    
    # 5. 创建下游任务模型
    print("\n5. 创建下游任务模型...")
    model = ChineseNewsModel(pretrained, num_classes=10)
    model.to(device)  # 移动到GPU
    print(f"模型设备: {next(model.parameters()).device}")
    
    # 测试模型
    output_shape = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    ).shape
    print(f"模型输出形状: {output_shape}")
    
    # 6. 训练模型
    print("\n6. 开始训练...")
    step_losses, step_accuracies, epoch_losses, epoch_accuracies = train_model(model, train_loader, num_epochs=5)

    # 保存训练过程日志
    pd.DataFrame({'step_loss': step_losses, 'step_accuracy': step_accuracies}).to_csv('step_log.csv', index=False)
    pd.DataFrame({'epoch_loss': epoch_losses, 'epoch_accuracy': epoch_accuracies}).to_csv('epoch_log.csv', index=False)

    # 绘制损失曲线并保存
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(step_losses, label='Step Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(step_accuracies, label='Step Accuracy')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

    # 绘制每个epoch的loss和accuracy并保存
    plt.figure()
    plt.plot(epoch_losses, label='Epoch Loss')
    plt.plot(epoch_accuracies, label='Epoch Accuracy')
    plt.xlabel('Epoch')
    plt.title('Epoch Loss & Accuracy')
    plt.legend()
    plt.savefig('epoch_curves.png')
    plt.close()

    # 7. 验证模型
    print("\n7. 开始验证...")
    val_accuracy = evaluate_model(model, val_loader, save_report_path='val_classification_report.csv')

    # 8. 测试模型
    print("\n8. 开始测试...")
    test_accuracy = evaluate_model(model, test_loader, save_report_path='test_classification_report.csv')

    print(f"\n最终结果:")
    print(f"验证集准确率: {val_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print("\n训练完成!")


if __name__ == "__main__":
    main() 
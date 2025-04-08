import os
import json
import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




#%% 信号处理模块
def preprocess_signal(raw, low=0.4, high=18, resample=100):
    """带通滤波 + 重采样"""
    if high <= low:
        raise ValueError(f"高通截止频率{high}Hz必须大于低通截止频率{low}Hz")
    raw.filter(low, high, fir_design='firwin')

    raw.resample(resample)
    return raw

#%% 正则化模块
class L2Regularizer:
    def __init__(self, model, lambda_reg=0.01):
        self.model = model
        self.lambda_reg = lambda_reg
        
    def regularize(self, loss):
        for param in self.model.parameters():
            if param.requires_grad:
                loss += self.lambda_reg * torch.norm(param, p=2)
        return loss

#%% 睡眠阶段映射
stage_mapping = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 4,
    'Sleep stage R': 5,
    'Movement time': -1,  # 标记为无效类别
    'Sleep stage ?': -1   # 标记为无效类别
}

#%% 数据集模块
class SleepDataset(Dataset):
    def __init__(self, edf_files, labels, seq_len=21, split='train', split_ratio=0.8, random_seed=42):
        """
        初始化睡眠数据集
        
        参数:
            edf_files: EDF文件路径列表
            labels: 标签文件路径列表
            seq_len: 序列长度
            split: 'train'或'val'，指定是训练集还是验证集
            split_ratio: 训练集比例，默认0.8
            random_seed: 随机种子，用于确保训练/验证集划分的可重复性
        """
        self.seq_len = seq_len
        self.split = split
        self.split_ratio = split_ratio
        self.random_seed = random_seed
        self.data, self.labels = [], []
        self.record_indices = []  # 记录每个文件的索引范围
        
        for file, label_path in zip(edf_files, labels):
            raw = mne.io.read_raw_edf(file, preload=True, include=['EEG Fpz-Cz','EOG horizontal'])
            raw = preprocess_signal(raw)
            # 使用mne内置方法解析注释
            annotations = mne.read_annotations(label_path)
            raw.set_annotations(annotations)  # 应用注释到原始数据
            print(f'裁剪前总时长: {raw.times[-1]:.1f}秒')
            tmin, tmax = 20000, raw.times[-1] - 20000
            raw = raw.crop(tmin=tmin, tmax=tmax)
            data = raw.get_data()  # (channels, time)、
            print(f'裁剪后时间范围: [{tmin}, {tmax}] 总时长: {tmax-tmin:.1f}秒')
            # 自动生成事件并过滤无效阶段
            events, event_id = mne.events_from_annotations(raw, chunk_duration=30.0)
            
            # 打印验证事件映射关系
            print(f'生成的事件映射关系：{event_id}')
            
            # 转换事件ID为阶段标签
            valid_events = []
            for event in events:
                # 通过event_id反向查找注释名称
                stage_name = next((k for k, v in event_id.items() if v == event[2]), None)
                mapped_label = stage_mapping.get(str(stage_name), -1) if stage_name else -1
                if mapped_label != -1:
                    event[2] = mapped_label
                    valid_events.append(event)
                else:
                    print(f'发现未识别的睡眠阶段: {stage_name}')
            events = np.array(valid_events)
            # 验证时间对齐
            max_time = raw.times[-1]
            
            # 基于events构建30秒分段的标签数组
            labels_30s = np.full(int(np.ceil(max_time / 30)), -1)
            for event in events:
                onset_sec = event[0] / raw.info['sfreq']
                idx = int(onset_sec // 30)
                if idx < len(labels_30s) and event[2] != -1:  # 过滤无效标签
                    labels_30s[idx] = event[2]
            
            # 过滤无效分段
            valid_labels = labels_30s[labels_30s != -1]
            print(f'成功解析{len(valid_labels)}个30秒分段')
            
            # 统计标签分布
            unique, counts = np.unique(valid_labels, return_counts=True)
            stage_names = ['W', '1', '2', '3', '4', 'R']
            print('标签分布统计:')
            for u, c in zip(unique, counts):
                print(f'{stage_names[u]}: {c} samples')
            
            # 将数据和标签分为训练集和验证集
            # 计算有效样本数量
            n_samples = len(valid_labels)
            if n_samples > 0:
                # 设置随机种子以确保可重复性
                np.random.seed(self.random_seed)
                # 生成随机索引
                indices = np.random.permutation(n_samples)
                # 计算训练集大小
                train_size = int(n_samples * self.split_ratio)
                
                # 根据split参数选择相应的索引
                if self.split == 'train':
                    selected_indices = indices[:train_size]
                else:  # 'val'
                    selected_indices = indices[train_size:]
                
                # 记录当前文件的起始索引
                start_idx = len(self.data)
                
                # 只添加选定的数据和标签
                self.data.append(torch.FloatTensor(data))
                self.labels.append(torch.LongTensor(valid_labels)[selected_indices])
                
                # 记录该记录的索引范围
                self.record_indices.append({
                    'start': start_idx,
                    'end': start_idx + len(selected_indices),
                    'indices': selected_indices
                })
                
                print(f'{self.split}集选择了{len(selected_indices)}个样本')

    def __len__(self):
        # 返回所有记录中选定索引的总数
        return sum(len(record['indices']) for record in self.record_indices)
    
    def __getitem__(self, idx):
        # 定位到具体记录
        rec_idx = 0
        remaining = idx
        
        # 遍历所有记录找到对应的rec_idx
        while rec_idx < len(self.record_indices):
            if remaining < len(self.record_indices[rec_idx]['indices']):
                break
            remaining -= len(self.record_indices[rec_idx]['indices'])
            rec_idx += 1
            
        # 检查是否找到有效记录
        if rec_idx >= len(self.record_indices):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self)}")
        
        # 获取该记录中的实际索引
        actual_idx = self.record_indices[rec_idx]['indices'][remaining]
        
        # 构建时序上下文
        start = actual_idx * 30 * 100
        end = start + (self.seq_len + 1) * 30 * 100
        
        # 验证时间范围
        if end > len(self.data[rec_idx][0]):
            end = len(self.data[rec_idx][0])
            start = end - (self.seq_len + 1) * 30 * 100
        
        assert start >= 0 and end <= len(self.data[rec_idx][0]), \
            f'时间范围越界: start={start}, end={end}, data_len={len(self.data[rec_idx][0])}'
            
        signal = self.data[rec_idx][:, start:end]  # (channels, time)
        
        # 确保标签长度与输入序列匹配
        # 注意：这里我们需要使用原始标签数组中的索引
        start_label_idx = actual_idx
        end_label_idx = start_label_idx + self.seq_len
        
        # 确保不超出标签数组范围
        if end_label_idx > len(self.labels[rec_idx]):
            end_label_idx = len(self.labels[rec_idx])
            start_label_idx = end_label_idx - self.seq_len
        
        # 获取标签序列
        labels = self.labels[rec_idx][start_label_idx:end_label_idx]
        
        # 如果标签不足，用最后一个标签填充
        if len(labels) < self.seq_len:
            last_label = labels[-1] if len(labels) > 0 else 0
            labels = torch.cat([labels, torch.full((self.seq_len - len(labels),), last_label, dtype=torch.long)])
        
        # 修改标签维度为单个时间点的标签
        label = labels[-1]  # 取最后一个时间点的标签
        
        return signal[:, :-30*100], label

#%% 模型架构
class SimpleSleepNet(nn.Module):
    def __init__(self, input_channels, hidden_dim=128, n_classes=6, seq_len=21): 
        super().__init__()
        self.seq_len = seq_len
        
        # 频谱特征提取
        self.spectral = nn.Sequential(
            nn.Conv1d(input_channels, 256, 15, padding=7),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(256, 512, 15, padding=7),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(3),    
        )
        
        # 时序建模
        self.gru = nn.GRU(512, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_dim*2, 1)
        # 初始化注意力权重
        nn.init.xavier_uniform_(self.attention.weight)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, n_classes)
        )
        
    def forward(self, x):
        assert x.size(-1) == 30*100*self.seq_len, \
            f"输入时间维度异常: {x.size(-1)} != {30*100*self.seq_len}"
        batch_size = x.size(0)
        
        # 频谱特征
        x = self.spectral(x)  # (batch, 64, time')
        x = x.permute(0,2,1)  # (batch, time', features)
        
        # 时序建模
        gru_out, _ = self.gru(x)  # (batch, seq, 2*hidden)
        attn = torch.softmax(self.attention(gru_out), dim=1)
        context = torch.sum(attn * gru_out, dim=1)
        
        # 分类
        return self.classifier(context)

#%% 训练循环
class SleepTrainer:
    def __init__(self, model, train_loader, val_loader, lr=1e-3, reg_lambda=0.01):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.scaler = GradScaler()
        
        # 计算类别权重以处理不平衡问题
        class_weights = self._calculate_class_weights(train_loader.dataset)
        print(f"应用的类别权重: {class_weights}")
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        
        self.regularizer = L2Regularizer(model, reg_lambda)
        self.best_f1 = 0
        self.patience = 5
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(f'runs/sleep_detection_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        preds, targets = [], []
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.to(device)
            y = y.to(device).flatten()
            
            self.optimizer.zero_grad()
            
            with autocast():
                out = self.model(x)
                loss = self.criterion(out, y)
                loss = self.regularizer.regularize(loss)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            preds.extend(torch.argmax(out, 1).cpu().numpy())
            targets.extend(y.cpu().numpy())
            
            # 记录每个batch的损失
            if batch_idx % 10 == 0:  # 每10个batch记录一次
                self.writer.add_scalar('Loss/train_batch', loss.item(), 
                                      len(self.train_loader) * self.current_epoch + batch_idx)
            
        f1 = f1_score(targets, preds, average='macro')
        acc = accuracy_score(targets, preds)
        
        # 记录每个epoch的指标
        self.writer.add_scalar('Loss/train_epoch', total_loss/len(self.train_loader), self.current_epoch)
        self.writer.add_scalar('Metrics/train_f1', f1, self.current_epoch)
        self.writer.add_scalar('Metrics/train_accuracy', acc, self.current_epoch)
        
        return total_loss/len(self.train_loader), f1
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        preds, targets, all_preds, all_targets = [], [], [], []
        total_loss = 0
        
        for x, y in self.val_loader:
            x = x.to(device)
            y = y.to(device).flatten()
            out = self.model(x)
            loss = self.criterion(out, y)
            total_loss += loss.item()
            
            batch_preds = torch.argmax(out, 1).cpu().numpy()
            batch_targets = y.cpu().numpy()
            
            preds.extend(batch_preds)
            targets.extend(batch_targets)
            all_preds.extend(batch_preds.tolist())
            all_targets.extend(batch_targets.tolist())

        # 添加对比输出
        #self._print_comparison(all_targets[:20], all_preds[:20])
        self._print_confusion_matrix(all_targets, all_preds)
        
        # 计算混淆矩阵并记录到TensorBoard
        cm = confusion_matrix(targets, preds, labels=[0,1,2,3,4,5])
        cm_figure = plt.figure(figsize=(10,10))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        stage_names = ['W', '1', '2', '3', '4', 'R']
        plt.xticks(np.arange(len(stage_names)), stage_names)
        plt.yticks(np.arange(len(stage_names)), stage_names)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # 添加数值标注
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        self.writer.add_figure('Confusion Matrix', cm_figure, self.current_epoch)
        plt.close()
        
        return {
            'loss': total_loss / len(self.val_loader),
            'f1': f1_score(targets, preds, average='macro'),
            'acc': accuracy_score(targets, preds),
            'kappa': cohen_kappa_score(targets, preds)
        }

    def _print_comparison(self, targets, preds):
        stage_names = ['W', '1', '2', '3', '4', 'R']  # 保持与有效类别一致
        print("\n样本对比:")
        print("Index\t真实\t预测\t状态")
        for i, (t, p) in enumerate(zip(targets, preds)):
            status = "✓" if t == p else "✗"
            print(f"{i+1}\t{stage_names[t]}\t{stage_names[p]}\t{status}")

    def _calculate_class_weights(self, dataset):
        """计算各类别的权重，用于处理类别不平衡问题"""
        # 收集所有标签
        all_labels = []
        for record_idx, record in enumerate(dataset.record_indices):
            record_labels = dataset.labels[record_idx]
            for idx in record['indices']:
                if idx < len(record_labels):  # 确保索引不越界
                    label_idx = record_labels[idx].item()
                    if label_idx != -1:  # 忽略无效标签
                        all_labels.append(label_idx)
        
        # 统计各类别样本数
        label_counts = np.bincount(all_labels, minlength=6)  # 确保包含所有6个类别
        
        # 打印类别分布
        stage_names = ['W', '1', '2', '3', '4', 'R']
        print("\n训练集类别分布:")
        for i, count in enumerate(label_counts):
            print(f"{stage_names[i]}: {count} samples")
        
        # 计算权重 (反比例权重)
        # 使用平滑处理避免除零错误
        smoothed_counts = label_counts + 1e-5
        weights = 1.0 / smoothed_counts
        
        # 归一化权重
        weights = weights / weights.sum() * len(weights)
        
        return torch.FloatTensor(weights).to(device)
    
    def _print_confusion_matrix(self, targets, preds):
        from sklearn.metrics import confusion_matrix
        # 过滤无效标签
        valid_indices = [i for i, t in enumerate(targets) if t != -1]
        filtered_targets = [targets[i] for i in valid_indices]
        filtered_preds = [preds[i] for i in valid_indices]
        
        # 按stage_mapping顺序生成标签
        stage_names = ['W', '1', '2', '3', '4', 'R']  # 保持与有效类别一致
        cm = confusion_matrix(filtered_targets, filtered_preds, labels=[0,1,2,3,4,5])
        
        print("\n混淆矩阵（行: 真实标签，列: 预测标签）:")
        print("   W   1   2   3   4   R")
        for i, row in enumerate(cm):
            print(f"{stage_names[i]} " + " ".join(f"{count:>4}" for count in row))
    
    def run(self, epochs=50):
        self.current_epoch = 0
        for epoch in range(epochs):
            self.current_epoch = epoch
            train_loss, train_f1 = self.train_epoch()
            val_metrics = self.validate()
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f} | Acc: {val_metrics['acc']:.4f}")
            print('-'*50)
            
            # 记录验证集指标
            self.writer.add_scalar('Loss/val', val_metrics.get('loss', 0), epoch)
            self.writer.add_scalar('Metrics/val_f1', val_metrics['f1'], epoch)
            self.writer.add_scalar('Metrics/val_accuracy', val_metrics['acc'], epoch)
            self.writer.add_scalar('Metrics/val_kappa', val_metrics['kappa'], epoch)
            
            # 早停与模型保存
            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                torch.save(self.model.state_dict(), 'best_model.pth')
                self.patience = 7
            else:
                self.patience -= 1
                if self.patience == 0:
                    print("Early stopping...")
                    break
        
        self.writer.close()

#%% 主程序
if __name__ == "__main__":
    # 数据准备
    edf_dir = os.environ.get("SLEEP_EDF_DIR", "sleep-edf-database-expanded-1.0.0/sleep-cassette")
    
    # 获取所有PSG文件
    psg_files = [f for f in os.listdir(edf_dir) if "PSG.edf" in f]
    
    # 构建对应的Hypnogram文件路径
    def get_hypno_path(psg_path):
        # 提取ID部分(破折号前的部分)
        file_id = os.path.basename(psg_path)[:6]
        # 查找匹配的Hypnogram文件
        hypno_files = [f for f in os.listdir(edf_dir) 
                      if f.startswith(file_id) and f.endswith("Hypnogram.edf")]
        if not hypno_files:
            raise FileNotFoundError(f"找不到与 {psg_path} 匹配的Hypnogram文件")
        return hypno_files[0]
    
    # 构建完整的文件路径列表
    edf_files = [os.path.join(edf_dir, f) for f in psg_files]
    label_files = [os.path.join(edf_dir, get_hypno_path(f)) for f in psg_files]
    
    # 设置训练参数
    seq_len = 21
    split_ratio = 0.8  # 每个记录内部80%用于训练，20%用于验证
    random_seed = 42
    
    # 创建训练集和验证集
    # 注意：现在我们使用相同的文件列表，但在SleepDataset内部进行分割
    train_set = SleepDataset(edf_files, label_files, seq_len=seq_len, 
                           split='train', split_ratio=split_ratio, random_seed=random_seed)
    val_set = SleepDataset(edf_files, label_files, seq_len=seq_len, 
                         split='val', split_ratio=split_ratio, random_seed=random_seed)
    
    print(f"训练集大小: {len(train_set)} 样本")
    print(f"验证集大小: {len(val_set)} 样本")
    
    # 数据加载器
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=32, num_workers=2, pin_memory=True)
    
    # 初始化模型
    # 使用固定的2个通道(EEG Fpz-Cz和EOG horizontal)
    model = SimpleSleepNet(input_channels=2, seq_len=seq_len)
    
    # 训练
    trainer = SleepTrainer(model, train_loader, val_loader, lr=1e-5, reg_lambda=0.01)
    trainer.run(epochs=50)
    
    # 最终评估
    model.load_state_dict(torch.load('best_model.pth'))
    final_metrics = trainer.validate()
    print("\nFinal Metrics:")
    print(f"F1: {final_metrics['f1']:.4f} | Acc: {final_metrics['acc']:.4f}")

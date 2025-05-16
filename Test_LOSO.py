import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import EEGDataset
from models.SSVEPFormer import SSVEPFormer
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def train_one_fold(train_subjects, test_subject, config):
    device = config['device']

    # —— 在 train_subjects 中划分出验证集 subjects —— #
    val_subjects = random.sample(train_subjects, config['val_n_subjects'])
    inner_train  = [s for s in train_subjects if s not in val_subjects]

    # 构建数据集
    train_ds = EEGDataset(
        config['data_root'], inner_train, session=config['session'],
        channel_indices=config['channels_sel'],
        highpass=config['highpass'],
        lowpass=config['lowpass'],
        filter_order=config['filter_order']
    )
    val_ds = EEGDataset(
        config['data_root'], val_subjects, session=config['session'],
        channel_indices=config['channels_sel'],
        highpass=config['highpass'],
        lowpass=config['lowpass'],
        filter_order=config['filter_order']
    )
    test_ds = EEGDataset(
        config['data_root'], [test_subject], session=config['session'],
        channel_indices=config['channels_sel'],
        highpass=config['highpass'],
        lowpass=config['lowpass'],
        filter_order=config['filter_order']
    )

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=config['batch_size'], shuffle=False)

    # 模型、损失、优化器
    model = SSVEPFormer(
        C=config['in_channels'],
        T=config['samples'],
        N=config['n_classes'],
        fs=config['fs'],
        freq_range=config['freq_range'],
        dropout=config['dropout']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['lr'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )

    best_val_loss = float('inf')
    patience_cnt  = 0

    # 训练 + 验证环节
    for epoch in range(1, config['epochs']+1):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)
        print(f"\nFold on subj {test_subject} — Epoch {epoch}/{config['epochs']}")
        for i, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % config['log_interval'] == 0 or i == total_batches:
                avg = running_loss / i
                print(f"\r  Train  Batch {i:3d}/{total_batches:3d}, loss={avg:.4f}", end='')
        print()

        # —— 验证集计算 Loss —— #
        model.eval()
        val_loss = 0.0
        count    = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                logits = model(x_val)
                l = criterion(logits, y_val)
                val_loss += l.item() * y_val.size(0)
                count += y_val.size(0)
        avg_val_loss = val_loss / count
        print(f"  Validation loss: {avg_val_loss:.4f}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_cnt = 0
            best_state = model.state_dict()  # 保存当前最优模型
        else:
            patience_cnt += 1
            if patience_cnt >= config['patience']:
                print(f"  Early stopping (no improvement for {config['patience']} epochs).")
                break

    # 恢复最优模型权重
    model.load_state_dict(best_state)

    # —— 测试 —— #
    correct = 0
    total   = 0
    model.eval()
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            preds = model(x_test).argmax(dim=1)
            correct += (preds == y_test).sum().item()
            total   += y_test.size(0)
    acc = 100.0 * correct / total
    print(f"  >> Test subject {test_subject}: Accuracy = {acc:.2f}%")
    return acc

if __name__ == "__main__":
    # ---- 配置 ----
    config = {
        'data_root':    "D:/14-AORIonline/DataSet",
        'session':       1,
        'in_channels':  32,         # 如果使用 channels_sel，设置成 len(channels_sel)
        'samples':     640,
        'fs':          128,
        'n_classes':     4,
        'dropout':     0.5,
        'freq_range':   (0.5, 5),

        'channels_sel': [9,10,11,12,13,14,17,18,19,20,21,22,
                         44,45,46,49,50,51,54,55,56,57,58,59] +
                        [25,26,30,63,62,27,29,64],

        'highpass':     0.5,
        'lowpass':      5.0,
        'filter_order': 4,

        'epochs':      100,
        'batch_size':  256,
        'lr':         0.001,
        'momentum':   0.9,
        'weight_decay':1e-4,
        'log_interval': 10,
        'val_n_subjects': 9,
        'patience':      5,

        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    subjects = list(range(1, 61))
    results  = []

    for test_subj in subjects:
        train_subj = [s for s in subjects if s != test_subj]
        acc = train_one_fold(train_subj, test_subj, config)
        results.append((test_subj, acc))

    # 保存结果到 CSV
    out_f = "loso_results.csv"
    with open(out_f, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['subject', 'accuracy'])
        writer.writerows(results)

    avg_acc = np.mean([acc for _, acc in results])
    std_acc = np.std([acc for _, acc in results])
    print(f"\nLOSO results saved to {out_f}")
    print(f"=== LOSO 平均准确率: {avg_acc:.2f}% ± {std_acc:.2f}% ===")

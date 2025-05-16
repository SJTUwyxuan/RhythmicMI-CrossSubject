import os
import csv
import time
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from dataset import EEGDataset
from models.SSVEPFormer import SSVEPFormer

# 设置随机种子以保证可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def generate_bootstrap_templates(data_np, n_groups=10, n_repeats=30):
    """
    为单个受试者生成 Bootstrap 平均模板。
    data_np: shape = (S, C, B, K)
    返回: shape = (S, C, n_repeats, K)
    """
    S, C, B, K = data_np.shape
    base = B // n_groups
    extras = B % n_groups
    group_indices = []
    start = 0
    for g in range(n_groups):
        size = base + (1 if g < extras else 0)
        group_indices.append(list(range(start, start + size)))
        start += size

    templates = np.zeros((S, C, n_repeats, K), dtype=data_np.dtype)
    for cls in range(K):
        trials = data_np[:, :, :, cls]
        pool_per_group = []
        for grp in group_indices:
            m = len(grp)
            base_cnt = n_repeats // m
            rem = n_repeats - base_cnt * m
            counts = [base_cnt] * m
            for i in range(rem): counts[i] += 1
            pool = []
            for idx, cnt in zip(grp, counts): pool += [idx] * cnt
            np.random.shuffle(pool)
            pool_per_group.append(pool)
        for r in range(n_repeats):
            sel_idxs = [pool_per_group[g][r] for g in range(n_groups)]
            templates[:, :, r, cls] = trials[:, :, sel_idxs].mean(axis=2)
    return templates


def generate_bootstrap_dataset(dataset, n_groups, n_repeats):
    """
    从 EEGTrialDataset 实例生成 bootstrap 数据集。
    返回 (X, y)，X 形状=(n_repeats*K, C, T)，y 形状=(n_repeats*K,)
    """
    data = dataset.data
    labels = dataset.labels
    unique = np.unique(labels)
    K = unique.size
    N, C, T = data.shape
    B = N // K
    data_np = np.zeros((T, C, B, K), dtype=data.dtype)
    for cls in unique:
        idxs = np.where(labels == cls)[0]
        block = data[idxs]
        data_np[:, :, :, cls] = block.transpose(2, 1, 0)
    tpl = generate_bootstrap_templates(data_np, n_groups, n_repeats)
    samples, sample_labels = [], []
    for r in range(n_repeats):
        for cls in unique:
            arr = tpl[:, :, r, cls]
            samples.append(arr.T)
            sample_labels.append(cls)
    X = np.stack(samples, axis=0)
    y = np.array(sample_labels, dtype=np.int64)
    return X, y

def train_one_fold_bootstrap(train_subjects, test_subject, config):
    device = config['device']
    val_subjects = random.sample(train_subjects, config['val_n_subjects'])
    inner_train = [s for s in train_subjects if s not in val_subjects]

def train_one_fold_bootstrap(train_subjects, test_subject, config):
    device = config['device']
    val_subjects = random.sample(train_subjects, config['val_n_subjects'])
    inner_train = [s for s in train_subjects if s not in val_subjects]

    train_data_list, train_label_list = [], []
    for subj in inner_train:
        ds = EEGDataset(config['data_root'], [subj], session=config['session'],
                             channel_indices=config['channels_sel'],
                             highpass=config['highpass'], lowpass=config['lowpass'],
                             filter_order=config['filter_order'])

        # === 修改：支持关闭 bootstrap 重采样 ===
        if config.get('bootstrap', True):
            Xb, yb = generate_bootstrap_dataset(ds, config['n_groups'], config['n_repeats'])
        else:
            Xb = ds.data  # 原始数据 (N, C, T)
            yb = ds.labels  # 原始标签 (N,)
        # === 结束修改 ===

        train_data_list.append(Xb)
        train_label_list.append(yb)

    X_train = np.concatenate(train_data_list, axis=0)
    y_train = np.concatenate(train_label_list, axis=0)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train)),
        batch_size=config['batch_size'], shuffle=True
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

    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=config['batch_size'], shuffle=False)

    model = SSVEPFormer(C=config['in_channels'], T=config['samples'], N=config['n_classes'],
                        fs=config['fs'], freq_range=config['freq_range'],
                        dropout=config['dropout']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'],
                          momentum=config['momentum'], weight_decay=config['weight_decay'])

    best_val_loss, patience_cnt = float('inf'), 0
    for epoch in range(1, config['epochs']+1):
        model.train(); total_loss=0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
        model.eval(); val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * y.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)
        if avg_val_loss < best_val_loss:
            best_val_loss, patience_cnt = avg_val_loss, 0
            best_state = model.state_dict()
        else:
            patience_cnt += 1
            if patience_cnt >= config['patience']:
                break
    model.load_state_dict(best_state)

    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


if __name__ == "__main__":
    config = {
        'data_root':    "D:/14-AORIonline/DataSet",
        'session':      1,
        'in_channels':  len([9,10,11,12,13,14,17,18,19,20,21,22,
                              44,45,46,49,50,51,54,55,56,57,58,59] +
                             [25,26,30,63,62,27,29,64]),
        'samples':      640,
        'fs':           128,
        'n_classes':    4,
        'dropout':      0.5,
        'freq_range':   (0.5,5),
        'channels_sel': [9,10,11,12,13,14,17,18,19,20,21,22,
                         44,45,46,49,50,51,54,55,56,57,58,59]+[25,26,30,63,62,27,29,64],
        'highpass':     0.5,
        'lowpass':      5.0,
        'filter_order': 4,
        'epochs':       100,
        'batch_size':   256,
        'lr':           0.001,
        'momentum':     0.9,
        'weight_decay': 1e-4,
        'patience':     5,
        'val_n_subjects':9,
        'device':       'cuda' if torch.cuda.is_available() else 'cpu',
        'bootstrap': True
    }
    subjects = list(range(1,61))
    # grid_n_groups = [3,5,10]
    # grid_n_repeats= [10,30,50,100,150,200]
    grid_n_groups = [1,3]
    grid_n_repeats= [100,150,200]
    summary = []

    for n_groups in grid_n_groups:
        for n_repeats in grid_n_repeats:
            print(f"Grid Search n_groups={n_groups}, n_repeats={n_repeats}")
            config['n_groups'] = n_groups
            config['n_repeats'] = n_repeats
            start = time.time()
            results = []
            for test_subj in subjects:
                train_subj = [s for s in subjects if s != test_subj]
                acc = train_one_fold_bootstrap(train_subj, test_subj, config)
                results.append((test_subj, acc))
                print(f"  Subj {test_subj}: {acc:.2f}%")
            elapsed = time.time() - start
            # 保存详细结果，末尾添加运行时长
            fname = f"bootstrap_ng{n_groups}_nr{n_repeats}.csv"
            with open(fname, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['subj', 'acc'])
                writer.writerows(results)
                writer.writerow([])
                writer.writerow(['elapsed_time_sec', f"{elapsed:.2f}"])
            avg = np.mean([a for _, a in results])
            std = np.std([a for _, a in results])
            summary.append((n_groups, n_repeats, avg, std, elapsed))
            print(f"  saved {fname}, avg {avg:.2f}% ± {std:.2f}%, time {elapsed:.1f}s\n")
    # 保存summary，包含时间
    summary_file = 'bootstrap_grid_summary.csv'
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n_groups', 'n_repeats', 'avg', 'std', 'elapsed_time_sec'])
        writer.writerows(summary)
    print(f"Summary saved to {summary_file}")

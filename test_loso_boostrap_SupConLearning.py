import os
import csv
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dataset import EEGDataset
from models.SSVEPFormer import SSVEPFormer

# 设置随机种子以保证可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def generate_bootstrap_templates(data_np, n_groups, n_repeats):
    """
    将单个受试者数据(data_np: S,C,B,K) 按 n_groups 和 n_repeats 构建 Bootstrap 平均模板
    返回 shape (S, C, n_repeats, K)
    """
    S, C, B, K = data_np.shape
    base = B // n_groups
    extras = B % n_groups
    groups = []
    start = 0
    for g in range(n_groups):
        size = base + (1 if g < extras else 0)
        groups.append(list(range(start, start + size)))
        start += size
    tpl = np.zeros((S, C, n_repeats, K), dtype=data_np.dtype)
    for cls in range(K):
        trials = data_np[:, :, :, cls]  # (S, C, B)
        pools = []
        for grp in groups:
            m = len(grp)
            base_cnt = n_repeats // m
            rem = n_repeats - base_cnt * m
            counts = [base_cnt] * m
            for i in range(rem): counts[i] += 1
            pool = []
            for idx, cnt in zip(grp, counts): pool += [idx] * cnt
            random.shuffle(pool)
            pools.append(pool)
        for r in range(n_repeats):
            idxs = [pools[g][r] for g in range(n_groups)]
            tpl[:, :, r, cls] = trials[:, :, idxs].mean(axis=2)
    return tpl


def generate_bootstrap_dataset(ds, n_groups, n_repeats):
    """
    从 EEGTrialDataset ds 生成 Bootstrap dataset
    返回 X (n_repeats*K, C, T), y (n_repeats*K,)
    """
    data, labels = ds.data, ds.labels
    uniq = np.unique(labels)
    K = uniq.size
    N, C, T = data.shape
    B = N // K
    arr = np.zeros((T, C, B, K), dtype=data.dtype)
    for cls in uniq:
        idxs = np.where(labels == cls)[0]
        block = data[idxs]  # (B, C, T)
        arr[:, :, :, cls] = block.transpose(2, 1, 0)
    tpl = generate_bootstrap_templates(arr, n_groups, n_repeats)
    samples, labs = [], []
    for r in range(n_repeats):
        for cls in uniq:
            samples.append(tpl[:, :, r, cls].T)
            labs.append(cls)
    X = np.stack(samples, axis=0)
    y = np.array(labs, dtype=np.int64)
    return X, y


class ContrastiveModel(nn.Module):
    """
    对比学习模型：SSVEPFormer encoder + projection head
    """
    def __init__(self, base_model, proj_dim):
        super().__init__()
        # encoder parts
        self.chancomb = base_model.chancomb
        self.encoder = base_model.encoder
        # record freq_idx for slicing
        self.freq_idx = getattr(base_model, 'freq_idx', None)
        # projection head: use flattened feature dimension (in_features of head.lin1)
        feat_dim = base_model.head.lin1.in_features
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, proj_dim)
        )

    def forward(self, x):
        # x: (B, C, T)
        Xf = torch.fft.rfft(x, dim=-1)  # (B, C, F_full)
        # slice frequencies
        if self.freq_idx is not None:
            Xf = Xf[..., self.freq_idx]
        z0 = self.chancomb(Xf)  # (B, 2*C, F)
        z1 = self.encoder(z0)   # (B, 2*C, F)
        flat = z1.view(z1.size(0), -1)  # (B, feat_dim)
        return self.proj(flat)

    # ... rest unchanged ...


def train_contrastive(model, train_loader, val_loader, config):
    """
    训练 ContrastiveModel，带验证集早停
    """
    device = config['device']
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=config['contrastive_lr'],
                          momentum=config['momentum'], weight_decay=config['weight_decay'])
    best_loss = float('inf')
    patience = 0
    for epoch in range(1, config['contrastive_epochs']+1):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            h = model(x)  # (B, proj_dim)
            hn = nn.functional.normalize(h, dim=1)
            sim = hn @ hn.T / config['temperature']
            labels = y.view(-1, 1)
            mask = (labels == labels.T).float() - torch.eye(len(y), device=device)
            exp_sim = torch.exp(sim) * (1 - torch.eye(len(y), device=device))
            logp = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
            loss = - (mask * logp).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.size(0)
        # 验证
        model.eval()
        val_loss = 0
        count = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                h = model(x)
                hn = nn.functional.normalize(h, dim=1)
                sim = hn @ hn.T / config['temperature']
                labels = y.view(-1, 1)
                mask = (labels == labels.T).float() - torch.eye(len(y), device=device)
                exp_sim = torch.exp(sim) * (1 - torch.eye(len(y), device=device))
                logp = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
                batch_loss = - (mask * logp).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
                val_loss += batch_loss.sum().item()
                count += y.size(0)
        avg_val = val_loss / count
        if avg_val < best_loss:
            best_loss = avg_val
            patience = 0
            best_state = model.state_dict()
        else:
            patience += 1
            if patience >= config['patience']:
                break
    model.load_state_dict(best_state)


def train_classification(encoder, train_loader, val_loader, test_loader, config):
    """
    冻结 encoder，用交叉熵训练分类头，带早停
    """
    device = config['device']
    # 新建完整模型，加载 encoder 权重
    model = SSVEPFormer(
        C=config['in_channels'], T=config['samples'], N=config['n_classes'],
        fs=config['fs'], freq_range=config['freq_range'], dropout=config['dropout']
    ).to(device)
    # 从 ContrastiveModel 提取对应子模块权重
    enc_sd = encoder.state_dict()
    # 加载 ChannelCombination 权重
    cc_sd = {k.split('chancomb.')[1]: v for k, v in enc_sd.items() if k.startswith('chancomb.')}
    model.chancomb.load_state_dict(cc_sd)
        # 加载 Encoder 模块权重
    enc_block_sd = {k.split('encoder.')[1]: v for k, v in enc_sd.items() if k.startswith('encoder.')}
    model.encoder.load_state_dict(enc_block_sd)
    for p in model.chancomb.parameters(): p.requires_grad = False
    for p in model.encoder.parameters(): p.requires_grad = False
    optimizer = optim.SGD(model.head.parameters(), lr=config['lr'],
                          momentum=config['momentum'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf'); patience = 0
    for epoch in range(1, config['clf_epochs']+1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward(); optimizer.step()
        # 验证
        model.eval(); val_loss = 0; count = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * y.size(0)
                count += y.size(0)
        avg_val = val_loss / count
        if avg_val < best_loss:
            best_loss, patience = avg_val, 0
            best_state = model.state_dict()
        else:
            patience += 1
            if patience >= config['patience']:
                break
    model.load_state_dict(best_state)
    # 测试
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def main():
    # 基本配置 + 超参数网格
    config = {
        'data_root': 'D:/14-AORIonline/DataSet',
        'session': 1,
        'in_channels': 32,
        'samples': 640,
        'fs': 128,
        'n_classes': 4,
        'dropout': 0.5,
        'freq_range': (0.5, 5),
        'channels_sel': [],
        'highpass': 0.5,
        'lowpass': 5.0,
        'filter_order': 4,
        'contrastive_epochs': 100,
        'contrastive_lr': 1e-3,
        'clf_epochs': 100,
        'lr': 1e-3,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'batch_size': 256,
        'temperature': 0.07,
        'patience': 5,
        'val_n_subjects': 9,
        'proj_dim': 128,
        'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        'bootstrap': False
    }
    # 填充 channels_sel 和 in_channels
    sel = [9,10,11,12,13,14,17,18,19,20,21,22,44,45,46,49,50,51,54,55,56,57,58,59]+[25,26,30,63,62,27,29,64]
    config['channels_sel'], config['in_channels'] = sel, len(sel)

    subjects = list(range(1,61))
    grid = {
        'n_groups': [1, 3, 5],
        'n_repeats': [100, 200]
    }
    summary = []

    for ng in grid['n_groups']:
        for nr in grid['n_repeats']:
            print(f"Grid ng={ng}, nr={nr}")
            config['n_groups'], config['n_repeats'] = ng, nr
            start = time.time()
            results = []
            for test_subj in subjects:
                train_subj = [s for s in subjects if s != test_subj]
                # 划分 train/val/test
                val_sub = random.sample(train_subj, config['val_n_subjects'])
                inner = [s for s in train_subj if s not in val_sub]
                # contrastive loader uses bootstrap of inner (train) and val sets separately
                # bootstrap train set
                ds_train_list, lb_train_list = [], []
                for subj in inner:
                    ds = EEGDataset(config['data_root'], [subj], config['session'],
                                         config['channels_sel'], config['highpass'], config['lowpass'], config['filter_order'])
                    # === 修改：支持关闭 bootstrap 重采样 ===
                    if config.get('bootstrap', True):
                        Xb_train, yb_train = generate_bootstrap_dataset(ds, ng, nr)
                    else:
                        Xb_train = ds.data  # 原始数据 (N, C, T)
                        yb_train  = ds.labels  # 原始标签 (N,)
                    # === 结束修改 ===
                    Xb_train, yb_train = generate_bootstrap_dataset(ds, ng, nr)
                    ds_train_list.append(Xb_train)
                    lb_train_list.append(yb_train)
                Xc_train = np.concatenate(ds_train_list, axis=0)
                yc_train = np.concatenate(lb_train_list, axis=0)
                train_tensor = TensorDataset(torch.from_numpy(Xc_train).float(), torch.from_numpy(yc_train))
                train_loader = DataLoader(train_tensor, batch_size=config['batch_size'], shuffle=True)
                # 构建 raw validation DataLoader 用于 contrastive 早停
                val_raw_ds = EEGDataset(
                    config['data_root'], val_sub, session=config['session'],
                    channel_indices=config['channels_sel'],
                    highpass=config['highpass'], lowpass=config['lowpass'],
                    filter_order=config['filter_order']
                )
                val_loader = DataLoader(
                    val_raw_ds, batch_size=config['batch_size'], shuffle=False
                )

                # 预训练 encoder
                base=SSVEPFormer(C=config['in_channels'],T=config['samples'],N=config['n_classes'],fs=config['fs'],freq_range=config['freq_range'],dropout=config['dropout'])
                model_con = ContrastiveModel(base, config['proj_dim'])
                train_contrastive(model_con, train_loader, val_loader, config)
                # 构建分类 loader：训练集使用 Bootstrap 增强的 inner 数据，验证/测试使用原始 trial
                # 训练集复用 contrastive 的 train_loader
                trL = train_loader
                # 验证集复用 raw validation loader
                vlL = val_loader
                # 测试集：原始 trial
                test_raw_ds = EEGDataset(
                    config['data_root'], [test_subj], session=config['session'],
                    channel_indices=config['channels_sel'],
                    highpass=config['highpass'], lowpass=config['lowpass'],
                    filter_order=config['filter_order']
                )
                teL = DataLoader(
                    test_raw_ds, batch_size=config['batch_size'], shuffle=False
                )
                # 下游分类测试，只训练分类头
                acc = train_classification(model_con, trL, vlL, teL, config)
                results.append((test_subj, acc))
                print(f"  Subj {test_subj}: {acc:.2f}%")
            # 记录并保存本组网格参数结果
            elapsed = time.time() - start
            fname = f"boost_ng{ng}_nr{nr}.csv"
            with open(fname, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['subject', 'accuracy'])
                writer.writerows(results)
                writer.writerow([])
                writer.writerow(['elapsed_time_sec', f"{elapsed:.1f}"])
            avg = np.mean([a for _, a in results])
            sd = np.std([a for _, a in results])
            summary.append((ng, nr, avg, sd, elapsed))
            print(f"Saved {fname}, avg {avg:.2f}% ± {sd:.2f}%, time {elapsed:.1f}s")
    # 保存汇总文件
    summary_file = 'boost_grid_summary.csv'
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n_groups', 'n_repeats', 'avg', 'std', 'elapsed_time_sec'])
        writer.writerows(summary)
    print(f"Summary saved to {summary_file}")
    # 控制台显示汇总结果
    print("Grid Search Summary:")
    print("n_groups | n_repeats | avg | std | elapsed_time_sec")
    for ng, nr, avg, sd, elapsed in summary:
        print(f"{ng} | {nr} | {avg:.2f} | {sd:.2f} | {elapsed:.1f}")

if __name__ == '__main__':
    main()


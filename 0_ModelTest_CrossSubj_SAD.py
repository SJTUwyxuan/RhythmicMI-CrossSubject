# ======== ModelTest_CrossSubj_SAD.py (GAN-style SAD + 原样式测试) ========

import os, time, datetime, random, csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
import argparse, yaml, importlib
import matplotlib.pyplot as plt

from dataset import EEGDataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',  type=str, required=True)
    p.add_argument('--config', type=str, required=True)
    return p.parse_args()

def load_config(path):
    if not os.path.exists(path):
        alt = os.path.join("configs", path)
        if os.path.exists(alt):
            path = alt
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_model_class(name):
    mod = importlib.import_module(f"models.{name}")
    return getattr(mod, name)

def train_one_fold(train_loader, val_loader, conf, ModelClass):
    device = conf.get('device','cpu')
    model = ModelClass(**conf['model']['params']).to(device)
    criterion_task = nn.CrossEntropyLoss().to(device)
    criterion_subj = nn.CrossEntropyLoss().to(device)

    optim_conf = conf['training']
    Opt = getattr(optim, optim_conf['optimizer']['name'])
    opts = optim_conf['optimizer']['params']
    if 'betas' in opts:
        opts['betas'] = tuple(opts['betas'])

    encoder_params = [p for n,p in model.named_parameters() if 'discriminator' not in n]
    disc_params    = [p for n,p in model.named_parameters() if 'discriminator' in n]
    optimizer_disc = Opt(disc_params, **opts)
    optimizer_enc  = Opt(encoder_params, **opts)

    subj_weight = conf.get('subject_loss_weight', 0.01)
    best_val, patience, best_state = float('inf'), 0, None
    train_losses, val_losses, subj_accs = [], [], []

    for epoch in range(1, optim_conf['epochs'] + 1):
        model.train()
        running_task, count = 0.0, 0
        correct_subj, total_subj = 0, 0

        for x, y, s in train_loader:
            x, y, s = x.to(device), y.to(device), s.to(device)

            # Step1: 更新判别器（冻结 encoder）
            for p in encoder_params: p.requires_grad = False
            for p in disc_params:    p.requires_grad = True
            optimizer_disc.zero_grad()
            with torch.no_grad():
                feat = model.extract_features(x)
            subj_out = model.discriminator(feat)
            loss_subj = criterion_subj(subj_out, s)
            loss_subj.backward()
            optimizer_disc.step()

            # Step2: 对抗更新 encoder（冻结判别器，最小化 task loss，最大化 subj loss）
            for p in encoder_params: p.requires_grad = True
            for p in disc_params:    p.requires_grad = False
            optimizer_enc.zero_grad()
            task_out, subj_out = model(x)
            loss_task     = criterion_task(task_out, y)
            loss_subj_adv = criterion_subj(subj_out, s)
            loss = loss_task - subj_weight * loss_subj_adv
            loss.backward()
            optimizer_enc.step()

            running_task += loss_task.item() * y.size(0)
            count       += y.size(0)
            pred_subj    = subj_out.argmax(dim=1)
            correct_subj += (pred_subj == s).sum().item()
            total_subj   += s.size(0)

        train_losses.append(running_task / count)
        subj_accs.append(100.0 * correct_subj / total_subj)

        # 验证
        model.eval()
        val_running, val_count = 0.0, 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                logits, _ = model(x_val)
                val_running += criterion_task(logits, y_val).item() * y_val.size(0)
                val_count   += y_val.size(0)
        val_losses.append(val_running / val_count)

        if val_losses[-1] < best_val:
            best_val, patience, best_state = val_losses[-1], 0, model.state_dict()
        else:
            patience += 1
            if patience >= optim_conf['patience']:
                break

        print(f"Epoch {epoch:02d} | train loss {train_losses[-1]:.4f} "
              f"| val loss {val_losses[-1]:.4f} | subj acc {subj_accs[-1]:.2f}%")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses, subj_accs

def main():
    args = parse_args()
    conf = load_config(args.config)
    ModelClass = get_model_class(args.model)

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    subjects      = list(range(1, 87))
    train_val_ids = subjects[30:]
    test_ids      = subjects[:30]

    dconf         = conf['data']
    ch_sel        = dconf.get('channels_sel', None)
    riemann_train = dconf.get('riemann_align_train', False)
    riemann_test  = dconf.get('riemann_align_test', False)

    # 计算对齐矩阵（如需）
    align_matrix = None
    if riemann_train:
        align_ds = EEGDataset(
            data_root       = dconf['data_root'],
            subject_ids     = train_val_ids,
            session         = dconf.get('session', 1),
            channel_indices = ch_sel,
            highpass        = dconf.get('highpass', None),
            lowpass         = dconf.get('lowpass', None),
            filter_order    = dconf.get('filter_order', 2),
            riemann_align   = True
        )
        align_matrix = align_ds.align_matrix

    # 划分 val/train
    val_ids   = random.sample(train_val_ids, conf['training']['val_n_subjects'])
    train_ids = [s for s in train_val_ids if s not in val_ids]

    # 创建 subject->idx 映射
    domain_id_map = {sid: idx for idx, sid in enumerate(train_ids)}
    conf['model']['params']['n_subjects'] = len(train_ids)

    per_subj = []
    for s in subjects:
        apply_align = riemann_train if s in train_val_ids else riemann_test
        ds = EEGDataset(
            data_root       = dconf['data_root'],
            subject_ids     = [s],
            session         = dconf.get('session', 1),
            channel_indices = ch_sel,
            highpass        = dconf.get('highpass', None),
            lowpass         = dconf.get('lowpass', None),
            filter_order    = dconf.get('filter_order', 2),
            riemann_align   = apply_align,
            align_matrix    = align_matrix
        )
        x_tensor = torch.from_numpy(ds.data).float()
        y_tensor = torch.from_numpy(ds.labels).long()
        if s in train_ids:
            d_tensor = torch.full((ds.data.shape[0],), domain_id_map[s], dtype=torch.long)
            per_subj.append(TensorDataset(x_tensor, y_tensor, d_tensor))
        else:
            per_subj.append(TensorDataset(x_tensor, y_tensor))

    bs = conf['training']['batch_size']
    train_ds = ConcatDataset([per_subj[i-1] for i in train_ids])
    val_ds   = ConcatDataset([per_subj[i-1] for i in val_ids])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  pin_memory=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, pin_memory=True, num_workers=2)

    start_time = time.time()
    model, train_losses, val_losses, subj_accs = train_one_fold(train_loader, val_loader, conf, ModelClass)
    train_time = time.time() - start_time

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 绘制训练/验证损失
    plt.figure()
    epochs = list(range(1, len(train_losses)+1))
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses,   marker='x', label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(f'{args.model} Loss Curve'); plt.legend(); plt.grid(True)
    loss_fig = f"{args.model}_Loss_{ts}.png"
    plt.savefig(loss_fig); plt.close()
    print(f"Saved loss curve to {loss_fig}")

    # —— 测试评估 —— #
    device = conf.get('device','cpu')
    model.eval()
    test_results = []
    with torch.no_grad():
        for s in test_ids:
            ds_s = per_subj[s-1]
            loader_s = DataLoader(ds_s, batch_size=bs, shuffle=False, pin_memory=True, num_workers=2)
            correct, total = 0, 0
            for batch in loader_s:
                x, y = batch[:2]
                x, y = x.to(device), y.to(device)
                output_test = model(x)
                class_logits_test = output_test[0] if isinstance(output_test, tuple) else output_test
                pred = class_logits_test.argmax(1)
                correct += (pred==y).sum().item()
                total   += y.size(0)
            acc_s = 100.0 * correct / total
            test_results.append((s, acc_s))
            print(f"Subject {s:02d} → {acc_s:.2f}%")

    acc_vals = [a for (_,a) in test_results]
    mean_acc = np.mean(acc_vals)
    std_acc  = np.std(acc_vals, ddof=1)
    print(f"Test avg acc: {mean_acc:.2f}% ± {std_acc:.2f}%")

    csv_name = f"{args.model}_Results_{ts}.csv"
    with open(csv_name, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['subject', 'accuracy'])
        for subj, a in test_results:
            writer.writerow([subj, f"{a:.2f}"])
        writer.writerow([])
        writer.writerow(['mean_accuracy', f"{mean_acc:.2f}"])
        writer.writerow(['std_accuracy',  f"{std_acc:.2f}"])
        writer.writerow(['train_time_s',   f"{train_time:.2f}"])
    print(f"Saved detailed results to {csv_name}")

if __name__ == "__main__":
    main()

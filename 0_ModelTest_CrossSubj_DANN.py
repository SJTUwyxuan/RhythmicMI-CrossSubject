import os, time, datetime, random, csv, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from torch.amp import autocast, GradScaler
import argparse, yaml, importlib
import matplotlib.pyplot as plt

from dataset import EEGDataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required=True, help='Model class name in models/<model>.py')
    p.add_argument('--config', type=str, required=True, help='Path to config, e.g., configs/FreqMLP.yaml')
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

def train_one_fold(train_loader, val_loader, conf, ModelClass, domain_enabled=False):
    device = conf.get('device', 'cpu')
    model = ModelClass(**conf['model']['params']).to(device)
    criterion_task = nn.CrossEntropyLoss().to(device)
    criterion_domain = nn.CrossEntropyLoss().to(device) if domain_enabled else None

    alpha = conf.get('training', {}).get('domain_loss_weight', 1.0)
    gamma = conf.get('training', {}).get('lambda_weight', 1.0)

    optim_conf = conf['training']
    Opt = getattr(optim, optim_conf['optimizer']['name'])
    opts = optim_conf['optimizer']['params']
    if 'betas' in opts:
        opts['betas'] = tuple(opts['betas'])
    optimizer = Opt(model.parameters(), **opts)
    scaler = GradScaler()

    n_batches = len(train_loader)
    total_steps = optim_conf['epochs'] * n_batches
    global_step = 0

    best_val, patience, best_state = float('inf'), 0, None
    train_losses, val_losses = [], []
    domain_loss_curve, domain_acc_curve = [], []

    for epoch in range(1, optim_conf['epochs'] + 1):
        model.train()
        running_loss, count = 0.0, 0
        domain_losses_epoch, domain_correct, domain_total = 0.0, 0, 0

        for batch in train_loader:
            if domain_enabled:
                x, y, d = batch
                d = d.to(device)
            else:
                x, y = batch
                d = None
            x, y = x.to(device), y.to(device)

            p = float(global_step) / float(total_steps)
            lambda_raw = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
            lambda_grl = gamma * lambda_raw

            optimizer.zero_grad()
            with autocast(device):
                output = model(x, lambda_grl) if domain_enabled else model(x)
                if isinstance(output, tuple):
                    class_logits, domain_logits = output
                else:
                    class_logits = output
                    domain_logits = None
                loss_task = criterion_task(class_logits, y)
                if domain_enabled:
                    loss_domain = criterion_domain(domain_logits, d)
                    loss = loss_task + alpha * loss_domain
                    domain_preds = domain_logits.argmax(1)
                    domain_correct += (domain_preds == d).sum().item()
                    domain_total += d.size(0)
                    domain_losses_epoch += loss_domain.item() * d.size(0)
                else:
                    loss = loss_task

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * y.size(0)
            count += y.size(0)
            global_step += 1

        train_losses.append(running_loss / count)

        model.eval()
        val_running, val_count = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x_val, y_val = batch[:2]
                x_val, y_val = x_val.to(device), y_val.to(device)
                output_val = model(x_val)
                class_logits_val = output_val[0] if isinstance(output_val, tuple) else output_val
                l = criterion_task(class_logits_val, y_val).item()
                val_running += l * y_val.size(0)
                val_count += y_val.size(0)
        val_losses.append(val_running / val_count)

        if domain_enabled and domain_total > 0:
            avg_domain_loss = domain_losses_epoch / domain_total
            avg_domain_acc = domain_correct / domain_total
            domain_loss_curve.append(avg_domain_loss)
            domain_acc_curve.append(avg_domain_acc)
            print(f"Epoch {epoch:02d} | domain loss {avg_domain_loss:.4f} | acc {avg_domain_acc*100:.2f}%")

        if val_losses[-1] < best_val:
            best_val, patience, best_state = val_losses[-1], 0, model.state_dict()
        else:
            patience += 1
            if patience >= optim_conf['patience']:
                break
        print(f"Epoch {epoch:02d} | train loss {train_losses[-1]:.4f} | val loss {val_losses[-1]:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Plot domain loss & acc
    if domain_enabled:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.figure()
        plt.plot(domain_loss_curve, label='Domain Loss', marker='o')
        plt.plot(domain_acc_curve, label='Domain Acc', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title(f"{ModelClass.__name__} Domain Loss and Acc")
        plt.legend()
        plt.grid(True)
        domain_fig = f"{ModelClass.__name__}_Domain_{ts}.png"
        plt.savefig(domain_fig)
        plt.close()
        print(f"Saved domain curve to {domain_fig}")

    return model, train_losses, val_losses, domain_loss_curve, domain_acc_curve


def main():
    args = parse_args()
    conf = load_config(args.config)
    ModelClass = get_model_class(args.model)

    model_instance = ModelClass(**conf['model']['params'])
    domain_enabled = hasattr(model_instance, 'domain_disc')


    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    subjects = list(range(1, 87))
    per_subj = []
    dconf = conf['data']

    train_val_ids = subjects[30:]
    test_ids      = subjects[:30]
    val_ids   = random.sample(train_val_ids, conf['training']['val_n_subjects'])
    train_ids = [s for s in train_val_ids if s not in val_ids]

    if domain_enabled:
        domain_id_map = {sid: idx for idx, sid in enumerate(train_ids)}
        conf['model']['params']['num_domains'] = len(train_ids)

    for s in subjects:
        ds = EEGDataset(
            dconf['data_root'], [s],
            session=dconf['session'],
            channel_indices=dconf['channels_sel'],
            highpass=dconf['highpass'],
            lowpass=dconf['lowpass'],
            filter_order=dconf['filter_order']
        )
        x_tensor = torch.from_numpy(ds.data).float()
        y_tensor = torch.from_numpy(ds.labels).long()
        if domain_enabled and s in train_ids:
            d_tensor = torch.full((ds.data.shape[0],), domain_id_map[s], dtype=torch.long)
            per_subj.append(TensorDataset(x_tensor, y_tensor, d_tensor))
        else:
            per_subj.append(TensorDataset(x_tensor, y_tensor))

    train_ds = ConcatDataset([per_subj[i-1] for i in train_ids])
    val_ds   = ConcatDataset([per_subj[i-1] for i in val_ids])
    bs       = conf['training']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  pin_memory=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, pin_memory=True, num_workers=2)

    start_time = time.time()
    model, train_losses, val_losses, domain_losses, domain_accs = train_one_fold(train_loader, val_loader, conf, ModelClass, domain_enabled)
    train_time = time.time() - start_time

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.figure()
    epochs = list(range(1, len(train_losses)+1))
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses,   marker='x', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{args.model} Loss Curve')
    plt.legend()
    plt.grid(True)
    loss_fig = f"{args.model}_Loss_{ts}.png"
    plt.savefig(loss_fig)
    plt.close()
    print(f"Saved loss curve to {loss_fig}")

    device = conf.get('device','cpu')
    test_results = []
    model.eval()
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

# ModelTest_WSLOOCV.py

import os, time, datetime, random, csv
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse, yaml, importlib
from dataset import EEGDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--model', type=str, required=True,
        help='Model class name under models/ (e.g., SSVEPFormer)'
    )
    p.add_argument(
        '--config', type=str, required=True,
        help='Path to YAML config (e.g., configs/SSVEPFormer.yaml)'
    )
    return p.parse_args()


def load_config(path):
    if not os.path.exists(path):
        candidate = os.path.join("configs", path)
        if os.path.exists(candidate):
            path = candidate
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_model_class(name):
    m = importlib.import_module(f"models.{name}")
    return getattr(m, name)


def within_subject_cv(subject_id, ds, ModelClass, cfg, device):
    """
    Within-subject 30-fold CV:
    for each fold, hold out one trial per class for testing and use the rest for training.
    Returns overall accuracy (%) on the subject.
    """
    X = torch.from_numpy(ds.data).float()   # [120, C, T]
    Y = torch.from_numpy(ds.labels).long()  # [120]
    N = int(Y.max().item()) + 1             # number of classes (e.g., 4)

    cls2idx = {c: (Y == c).nonzero(as_tuple=True)[0].tolist() for c in range(N)}
    folds = len(cls2idx[0])  # trials per class (e.g., 30)

    correct = 0
    total = 0

    bs = cfg['training']['batch_size']
    epochs = cfg['training']['epochs']
    patience = cfg['training']['patience']
    optim_nm = cfg['training']['optimizer']['name']
    optim_pr = cfg['training']['optimizer']['params']

    for fold in range(folds):
        test_idx = [cls2idx[c][fold] for c in range(N)]
        train_idx = [i for i in range(len(Y)) if i not in test_idx]

        train_ds = TensorDataset(X[train_idx], Y[train_idx])
        test_ds = TensorDataset(X[test_idx], Y[test_idx])

        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

        model = ModelClass(**cfg['model']['params']).to(device)
        Opt = getattr(optim, optim_nm)
        optimizer = Opt(model.parameters(), **optim_pr)
        criterion = nn.CrossEntropyLoss().to(device)

        best_val = float('inf')
        wait = 0
        best_state = None

        # No explicit validation split; use training loss for early stopping.
        for ep in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            cnt = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                l = criterion(model(x), y)
                l.backward()
                optimizer.step()
                train_loss += l.item() * y.size(0)
                cnt += y.size(0)
            avg_train = train_loss / cnt

            if avg_train < best_val:
                best_val, wait, best_state = avg_train, 0, model.state_dict()
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

    return 100.0 * correct / total


def main():
    args = parse_args()
    cfg = load_config(args.config)
    ModelC = get_model_class(args.model)

    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)

    data_conf = cfg['data']
    subjects = list(range(1, 101))
    results = []

    for s in subjects:
        ds = EEGDataset(
            data_conf['data_root'], [s],
            session=data_conf['session'],
            channel_indices=data_conf['channels_sel'],
            highpass=data_conf['highpass'],
            lowpass=data_conf['lowpass'],
            filter_order=data_conf['filter_order']
        )
        acc = within_subject_cv(s, ds, ModelC, cfg, device)
        print(f"Subject {s}: within-subject CV accuracy = {acc:.2f}%")
        results.append(acc)

    mean_acc = float(np.mean(results))
    std_acc = float(np.std(results, ddof=1))
    print(f"\nOverall within-subject average accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"{args.model}_WSLOOCV_{now}.csv"
    with open(out, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['subject', 'accuracy'])
        for s, a in zip(subjects, results):
            w.writerow([s, f"{a:.2f}"])
        w.writerow(['mean', f"{mean_acc:.2f}"])
        w.writerow(['std', f"{std_acc:.2f}"])
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()

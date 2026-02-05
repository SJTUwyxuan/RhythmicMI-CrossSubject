# MLModelTest_CrossSubj_GPU.py

import os, time, datetime, csv
import numpy as np
import torch
import argparse
import yaml
from dataset import EEGDataset
from models.SSMRRTDCA import SSMRRTDCA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--config', type=str, required=True,
        help='Path to YAML config (e.g., configs/SSMRRTDCA.yaml)'
    )
    return p.parse_args()


def load_config(path):
    if not os.path.exists(path):
        alt = os.path.join("configs", path)
        if os.path.exists(alt):
            path = alt
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dcfg = cfg['data']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    subjects = list(range(1, 101))
    test_ids = subjects[:30]
    train_ids = subjects[30:]
    # train_ids = [25, 26, 27, 28, 29, 30, 31, 33, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 57, 58, 59, 60, 63, 65, 66, 67, 68, 69, 72, 74, 76, 78]

    print(f"Training on subjects: {train_ids}")
    print(f"Testing on subjects: {test_ids}")

    ds_train = EEGDataset(
        dcfg['data_root'], train_ids,
        session=dcfg['session'],
        channel_indices=dcfg['channels_sel'],
        highpass=dcfg['highpass'],
        lowpass=dcfg['lowpass'],
        filter_order=dcfg['filter_order']
    )
    X_tr = ds_train.data    # (N, C, T)
    Y_tr = ds_train.labels

    # TDCA input layout: (T, C, block, class)
    n_tr, C, T = X_tr.shape
    classes = np.unique(Y_tr)
    N = len(classes)
    cls2idx = {c: np.where(Y_tr == c)[0].tolist() for c in classes}
    block = min(len(v) for v in cls2idx.values())

    eeg_train = np.zeros((T, C, block, N), dtype=np.float32)
    for ci, c in enumerate(classes):
        idxs = cls2idx[c][:block]
        for bi, ti in enumerate(idxs):
            eeg_train[:, :, bi, ci] = X_tr[ti].T

    train_tensor = torch.from_numpy(eeg_train).float().to(device)

    params = cfg['params']
    model = SSMRRTDCA(
        fs=params['fs'],
        subspace_num=params['subspace_num'],
        delay_num=params['delay_num'],
        freq_vec=params['freq_vec'],
        harmonic=params['harmonic'],
        device=device
    ).train(train_tensor)

    results = []
    start_time = time.time()
    for s in test_ids:
        ds = EEGDataset(
            dcfg['data_root'], [s],
            session=dcfg['session'],
            channel_indices=dcfg['channels_sel'],
            highpass=dcfg['highpass'],
            lowpass=dcfg['lowpass'],
            filter_order=dcfg['filter_order']
        )
        X = ds.data
        Y = ds.labels

        correct = 0
        for x, y in zip(X, Y):
            pred, _ = model.detection(torch.from_numpy(x.T).float().to(device))
            if pred - 1 == y:
                correct += 1

        acc = 100.0 * correct / len(Y)
        results.append((s, acc))
        print(f"Subject {s:02d} → Accuracy: {acc:.2f}%")

    acc_vals = [a for (_, a) in results]
    mean_acc = np.mean(acc_vals)
    std_acc = np.std(acc_vals, ddof=1)
    elapsed = time.time() - start_time
    print(f"Cross-subject TDCA accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%, Time: {elapsed:.2f}s")

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"SSMRRTDCA_CrossSubj_{now}.csv"
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['subject', 'accuracy'])
        w.writerows(results)
        w.writerow(['mean_accuracy', f"{mean_acc:.2f}"])
        w.writerow(['std_accuracy', f"{std_acc:.2f}"])
        w.writerow(['train_time_s', f"{elapsed:.2f}"])
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()

# MLModelTest_WSLOOCV_GPU.py

import os
import time
import datetime
import random
import csv

import numpy as np
import torch

import argparse
import yaml

from dataset import EEGDataset
from models.SSMRRTDCA import SSMRRTDCA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to YAML config (e.g., configs/SSMRRTDCA.yaml)'
    )
    return parser.parse_args()


def load_config(path):
    if not os.path.exists(path):
        cand = os.path.join("configs", path)
        if os.path.exists(cand):
            path = cand
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def within_subject_loocv_gpu(subject_id, ds, cfg, device):
    """
    Within-subject LOOCV (one trial per class held out per fold).
    GPU-accelerated TDCA training and inference.
    Returns accuracy (%) on the subject.
    """
    X = ds.data       # numpy (trials, chan, samples)
    Y = ds.labels     # numpy (trials,)
    trials, C, T = X.shape

    classes = np.unique(Y)
    N = len(classes)
    cls2idx = {c: np.where(Y == c)[0].tolist() for c in classes}
    trials_per_class = len(cls2idx[classes[0]])

    params = cfg['params']
    fs = params['fs']
    subspace_num = params['subspace_num']
    delay_num = params['delay_num']
    freq_vec = params['freq_vec']
    harmonic = params['harmonic']

    correct = 0
    total = 0

    for fold in range(trials_per_class):
        train_idx = []
        test_idx = []
        for c in classes:
            idx_list = cls2idx[c]
            test_i = idx_list[fold]
            test_idx.append(test_i)
            train_idx += [i for i in idx_list if i != test_i]

        # Arrange training data as (T, C, block, N)
        block = trials_per_class - 1
        eeg_train = np.zeros((T, C, block, N), dtype=np.float32)
        for ci, c in enumerate(classes):
            tr_idxs = [i for i in cls2idx[c] if i != cls2idx[c][fold]]
            for bi, ti in enumerate(tr_idxs):
                eeg_train[:, :, bi, ci] = X[ti].T

        train_tensor = torch.from_numpy(eeg_train).float().to(device)

        tdca = SSMRRTDCA(
            fs=fs,
            subspace_num=subspace_num,
            delay_num=delay_num,
            freq_vec=freq_vec,
            harmonic=harmonic,
            device=device
        ).train(train_tensor)

        for ti in test_idx:
            epoch = X[ti].T  # (T, C)
            epoch_tensor = torch.from_numpy(epoch).float().to(device)
            pred, _ = tdca.detection(epoch_tensor)
            if pred - 1 == Y[ti]:
                correct += 1
            total += 1

    return 100.0 * correct / total


def main():
    args = parse_args()
    cfg = load_config(args.config)

    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dcfg = cfg['data']
    subjects = list(range(1, 61))
    results = []

    start = time.time()
    for s in subjects:
        ds = EEGDataset(
            dcfg['data_root'], [s],
            session=dcfg['session'],
            channel_indices=dcfg['channels_sel'],
            highpass=dcfg['highpass'],
            lowpass=dcfg['lowpass'],
            filter_order=dcfg['filter_order']
        )
        acc = within_subject_loocv_gpu(s, ds, cfg, device)
        print(f"Subject {s}: WSLOOCV accuracy = {acc:.2f}")
        results.append(acc)

    elapsed = time.time() - start
    mean_acc = float(np.mean(results))
    std_acc = float(np.std(results, ddof=1))
    print(f"\nOverall WSLOOCV avg acc: {mean_acc:.2f}% ± {std_acc:.2f}%, Time: {elapsed:.2f}s")

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"SSMRRTDCA_WSLOOCV_GPU_{now}.csv"
    with open(out, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['subject', 'accuracy'])
        for s, a in zip(subjects, results):
            w.writerow([s, f"{a:.2f}"])
        w.writerow(['mean', f"{mean_acc:.2f}"])
        w.writerow(['std',  f"{std_acc:.2f}"])
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()

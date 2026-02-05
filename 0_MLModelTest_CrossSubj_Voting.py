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
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True,
                   help='Path to config file, e.g., configs/SSMRRTDCA.yaml')
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
    dconf = cfg['data']
    params = cfg['params']
    vote_cfg = cfg.get('voting', {})
    threshold = vote_cfg.get('threshold', 0.0)
    weighted = vote_cfg.get('weighted_vote', True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    subjects = list(range(1, 87))
    test_ids  = subjects[:30]
    train_ids = subjects[30:]
    print(f"Training on subjects: {train_ids}")
    print(f"Testing on subjects: {test_ids}")

    # Riemann alignment optionally
    riemann_train = dconf.get('riemann_align_train', False)
    riemann_test  = dconf.get('riemann_align_test', False)
    align_matrix = None
    if riemann_train:
        # compute align matrix from all training data
        ds_align = EEGDataset(
            dconf['data_root'], train_ids,
            session=dconf['session'],
            channel_indices=dconf['channels_sel'],
            highpass=dconf['highpass'],
            lowpass=dconf['lowpass'],
            filter_order=dconf['filter_order'],
            riemann_align=True,
            align_matrix=None
        )
        align_matrix = ds_align.align_matrix
        print("Computed Riemann align matrix from training subjects.")

    # Train TDCA per subject
    models = []
    for s in train_ids:
        ds = EEGDataset(
            dconf['data_root'], [s],
            session=dconf['session'],
            channel_indices=dconf['channels_sel'],
            highpass=dconf['highpass'],
            lowpass=dconf['lowpass'],
            filter_order=dconf['filter_order'],
            riemann_align=(riemann_train),
            align_matrix=align_matrix
        )
        X, Y = ds.data, ds.labels
        classes = np.unique(Y)
        S, C = X.shape[2], X.shape[1]
        K = len(classes)
        # pad/truncate to equal trials per class
        cls2idx = {c: np.where(Y == c)[0].tolist() for c in classes}
        B = min(len(idxs) for idxs in cls2idx.values())

        eeg = np.zeros((S, C, B, K), dtype=np.float32)
        for ci, c in enumerate(classes):
            idxs = cls2idx[c][:B]
            for bi, ti in enumerate(idxs):
                eeg[:, :, bi, ci] = X[ti].T
        eeg_tensor = torch.from_numpy(eeg).float().to(device)

        model = SSMRRTDCA(
            fs=params['fs'],
            subspace_num=params['subspace_num'],
            delay_num=params['delay_num'],
            freq_vec=params['freq_vec'],
            harmonic=params['harmonic'],
            device=device
        ).train(eeg_tensor)
        models.append(model)

    # Soft voting prediction
    results = []
    t0 = time.time()
    for s in test_ids:
        ds = EEGDataset(
            dconf['data_root'], [s],
            session=dconf['session'],
            channel_indices=dconf['channels_sel'],
            highpass=dconf['highpass'],
            lowpass=dconf['lowpass'],
            filter_order=dconf['filter_order'],
            riemann_align=(riemann_test),
            align_matrix=align_matrix
        )
        X, Y = ds.data, ds.labels
        correct = 0
        n_trials = X.shape[0]
        K = len(np.unique(Y))

        for trial_idx in range(n_trials):
            epoch = torch.from_numpy(X[trial_idx].T).float().to(device)
            # collect correlation vectors
            rhos = torch.stack([m.detection(epoch)[1] for m in models], dim=0)
            if weighted:
                rhos = torch.where(rhos >= threshold, rhos, torch.tensor(0.0, device=device))
                scores = rhos.sum(dim=0)
            else:
                votes = (rhos >= threshold).float()
                scores = votes.sum(dim=0)
            pred = torch.argmax(scores).item() + 1
            if pred - 1 == Y[trial_idx]:
                correct += 1

        acc = 100.0 * correct / n_trials
        results.append((s, acc))
        print(f"Subject {s:02d} → {acc:.2f}%")

    elapsed = time.time() - t0
    acc_vals = [a for _, a in results]
    mean_acc = np.mean(acc_vals)
    std_acc  = np.std(acc_vals, ddof=1)
    print(f"SoftVoting Cross-Subject TDCA avg acc: {mean_acc:.2f}% ± {std_acc:.2f}%, Time: {elapsed:.2f}s")

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"SSMRRTDCA_SoftVotingCrossSubj_{now}.csv"
    with open(out_csv, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['subject', 'accuracy'])
        for s, a in results:
            w.writerow([s, f"{a:.2f}"])
        w.writerow(['mean_accuracy', f"{mean_acc:.2f}"])
        w.writerow(['std_accuracy', f"{std_acc:.2f}"])
        w.writerow(['total_time_s', f"{elapsed:.2f}"])
    print(f"Results saved to {out_csv}")

if __name__ == '__main__':
    main()

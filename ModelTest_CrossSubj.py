# ======== ModelTest_CrossSubj.py ========
import os, time, datetime, random, csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from torch.amp import autocast, GradScaler
import argparse, yaml, importlib
import matplotlib.pyplot as plt
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm

# t-SNE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.io import savemat

from dataset import EEGDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',  type=str, required=True,
                   help='Model class name in models/<model>.py (e.g., FreqMLP, EEGNet)')
    p.add_argument('--config', type=str, required=True,
                   help='Path to config, e.g., configs/FreqMLP.yaml')
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
    device = conf.get('device', 'cpu')
    model = ModelClass(**conf['model']['params']).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    optim_conf = conf['training']
    Opt = getattr(optim, optim_conf['optimizer']['name'])
    opts = optim_conf['optimizer']['params']
    if 'betas' in opts:
        opts['betas'] = tuple(opts['betas'])
    optimizer = Opt(model.parameters(), **opts)
    scaler = GradScaler()

    best_val, patience, best_state = float('inf'), 0, None
    train_losses, val_losses = [], []
    epoch_times = []
    best_epoch = 1

    for epoch in range(1, optim_conf['epochs'] + 1):
        start_epoch_time = time.time()
        model.train()
        running_loss, count = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(device):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * y.size(0)
            count += y.size(0)

        train_losses.append(running_loss / count)

        model.eval()
        val_running, val_count = 0.0, 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                l = criterion(model(x_val), y_val).item()
                val_running += l * y_val.size(0)
                val_count += y_val.size(0)

        val_losses.append(val_running / val_count)
        epoch_times.append(time.time() - start_epoch_time)

        if val_losses[-1] < best_val:
            best_val, patience, best_state = val_losses[-1], 0, model.state_dict()
            best_epoch = epoch
        else:
            patience += 1
            if patience >= optim_conf['patience']:
                break

        print(f"Epoch {epoch:02d} | train loss {train_losses[-1]:.4f} | val loss {val_losses[-1]:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    avg_epoch_time = np.mean(epoch_times) if len(epoch_times) > 0 else 0.0
    return model, train_losses, val_losses, avg_epoch_time, best_epoch


# -------------------------- t-SNE helpers --------------------------
def _standardize(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd


def _pca_tsne(X: np.ndarray, perplexity=30, seed=0) -> np.ndarray:
    X = _standardize(X)
    pca_dim = min(50, X.shape[1])
    Xp = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
    Z = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(Xp)
    return Z


def _plot_tsne_by_class(Z: np.ndarray, Y: np.ndarray, title: str, out_png: str):
    # Soft but distinct palette
    palette = ["#4E79A7", "#F28E2B", "#59A14F", "#B07AA1"]
    class_names = [f"Class {i}" for i in range(4)]

    plt.figure(figsize=(6.2, 4.8))
    for k in range(4):
        idx = (Y == k)
        if np.any(idx):
            plt.scatter(Z[idx, 0], Z[idx, 1], s=10, c=palette[k],
                        label=class_names[k], alpha=0.85)

    plt.title(title)
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")

    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(out_png, dpi=200)
    plt.close()


def _collect_labels_only(per_subj, subject_ids, bs):
    Ys = []
    for s in subject_ids:
        ds_s = per_subj[s - 1]
        loader_s = DataLoader(ds_s, batch_size=bs, shuffle=False, num_workers=0)
        for _, y in loader_s:
            Ys.append(y.detach().cpu())
    return torch.cat(Ys, dim=0).numpy() if len(Ys) > 0 else None


def _find_last_linear(module: nn.Module):
    """Find the last nn.Linear inside a module (prefer Sequential order if exists)."""
    if module is None:
        return None
    if isinstance(module, nn.Sequential):
        for m in reversed(list(module)):
            if isinstance(m, nn.Linear):
                return m
            if isinstance(m, nn.Sequential):
                mm = _find_last_linear(m)
                if mm is not None:
                    return mm
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    return linears[-1] if len(linears) > 0 else None


def run_tsne_freqmlp_multilayer(model, per_subj, subject_ids, conf, out_prefix):
    """
    FreqMLP: save 4 nodes:
      1) input (raw time-domain),
      2) after_fft (one-sided spectrum; keep real or real+imag depending on model; apply freq_idx if exists),
      3) after_chancomb (output of channel-combination module; if disabled, fallback to after_fft),
      4) pre_logits embedding (input to the final classification Linear layer).
    Only class-colored plots + raw .mat.
    """
    device = conf.get('device', 'cpu')
    bs = conf['training']['batch_size']

    # ----------------- prepare hooks -----------------
    acts_out = {}   # store outputs (e.g., after_chancomb)
    acts_inp = {}   # store inputs (e.g., pre_logits)
    handles = []

    # (3) after_chancomb: output of model.chancomb if present
    has_chancomb = hasattr(model, "chancomb") and (getattr(model, "chancomb") is not None)
    if has_chancomb:
        acts_out["after_chancomb"] = []

        def _hook_out(name):
            def fn(module, inp, out):
                acts_out[name].append(out.detach().cpu())
            return fn

        handles.append(model.chancomb.register_forward_hook(_hook_out("after_chancomb")))

    # (4) pre_logits: input to the LAST Linear classifier
    prelogits_layer = None
    if hasattr(model, "head") and getattr(model, "head") is not None:
        if hasattr(model.head, "net"):
            prelogits_layer = _find_last_linear(model.head.net)
        if prelogits_layer is None:
            prelogits_layer = _find_last_linear(model.head)
    if prelogits_layer is None:
        prelogits_layer = _find_last_linear(model)

    if prelogits_layer is None:
        print("[t-SNE] FreqMLP multilayer: cannot find final Linear layer for pre_logits. Will skip pre_logits.")
    else:
        acts_inp["pre_logits"] = []

        def _hook_inp(name):
            def fn(module, inp, out):
                if isinstance(inp, (tuple, list)) and len(inp) > 0:
                    acts_inp[name].append(inp[0].detach().cpu())
            return fn

        handles.append(prelogits_layer.register_forward_hook(_hook_inp("pre_logits")))

    # ----------------- collect data -----------------
    input_list = []      # (1) raw input
    Xf_feat_list = []    # (2) after_fft (real or real+imag)
    Y_list = []

    # Ablation compatibility: infer whether to keep imag based on model attribute if present
    keep_imag = True
    if hasattr(model, "use_imag"):
        keep_imag = bool(getattr(model, "use_imag"))

    model.eval()
    with torch.no_grad():
        for s in subject_ids:
            ds_s = per_subj[s - 1]
            loader_s = DataLoader(ds_s, batch_size=bs, shuffle=False, num_workers=0)
            for x, y in loader_s:
                x = x.to(device)

                # (1) input (raw time-domain)
                input_list.append(x.detach().cpu())

                # (2) after_fft (explicit, aligned with model settings)
                try:
                    Xf = torch.fft.rfft(x, dim=-1)  # (B, C, F_full) complex
                    if getattr(model, "freq_idx", None) is not None:
                        Xf = Xf[..., model.freq_idx]  # keep band of interest

                    if keep_imag:
                        Xf_feat = torch.cat([Xf.real, Xf.imag], dim=1)  # (B, 2C, F)
                    else:
                        Xf_feat = Xf.real  # (B, C, F)

                    Xf_feat_list.append(Xf_feat.detach().cpu())
                except Exception:
                    pass

                _ = model(x)  # trigger hooks
                Y_list.append(y.detach().cpu())

    for h in handles:
        h.remove()

    if len(Y_list) == 0:
        print("[t-SNE] FreqMLP multilayer: no data collected. Skip.")
        return

    Y = torch.cat(Y_list, dim=0).numpy()

    feats = {}

    # Assemble (1) input
    if len(input_list) > 0:
        feats["input"] = torch.cat(input_list, dim=0)

    # Assemble (2) after_fft
    if len(Xf_feat_list) > 0:
        feats["after_fft"] = torch.cat(Xf_feat_list, dim=0)

    # Assemble (3) after_chancomb
    if "after_chancomb" in acts_out and len(acts_out["after_chancomb"]) > 0:
        feats["after_chancomb"] = torch.cat(acts_out["after_chancomb"], dim=0)
    else:
        # If chancomb is disabled/absent, use after_fft as a consistent placeholder
        if "after_fft" in feats:
            feats["after_chancomb"] = feats["after_fft"]

    # Assemble (4) pre_logits
    if "pre_logits" in acts_inp and len(acts_inp["pre_logits"]) > 0:
        feats["pre_logits"] = torch.cat(acts_inp["pre_logits"], dim=0)

    if len(feats) == 0:
        print("[t-SNE] FreqMLP multilayer: no features collected. Skip.")
        return

    def to_vec(t: torch.Tensor) -> np.ndarray:
        """
        - For 3D tensors (N, C, T/F): average over the last dimension -> (N, C)
        - For 2D tensors (N, D): keep as is
        - For >3D: flatten except batch
        """
        if t.ndim == 3:
            t = t.mean(dim=-1)
        elif t.ndim > 3:
            t = t.view(t.shape[0], -1)
        return t.numpy()

    for name, t in feats.items():
        X = to_vec(t)
        Z = _pca_tsne(X, perplexity=30, seed=0)

        mat_path = f"{out_prefix}_tsne_{name}.mat"
        savemat(mat_path, {
            "Z": Z.astype(np.float32),
            "Y": Y.astype(np.int32),
            "layer": np.array([name], dtype=object)
        })
        print(f"[t-SNE] Saved raw: {mat_path}")

        png_path = f"{out_prefix}_tsne_{name}_byClass.png"
        _plot_tsne_by_class(Z, Y, f"t-SNE ({name})", png_path)


def _get_final_feature_layer(model, model_name: str):
    """
    Return (layer_module, layer_tag) for "pre-logits features" extraction.
    The hook will capture inp[0] of this layer.
    """
    # Exact mappings (preferred)
    if model_name == "EEGNet":
        if hasattr(model, "dense"):
            return model.dense, "pre_logits"
    if model_name == "LMDA":
        if hasattr(model, "classifier"):
            return model.classifier, "pre_logits"
    if model_name == "FACTNet":
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            if len(model.classifier) >= 2:
                # classifier: [Flatten, Linear, Softmax]
                return model.classifier[1], "pre_logits"
    if model_name == "SSVEPFormer":
        if hasattr(model, "head") and hasattr(model.head, "lin2"):
            return model.head.lin2, "pre_logits"

    # Fallback heuristics (if names differ)
    if hasattr(model, "classifier"):
        return model.classifier, "pre_logits"
    if hasattr(model, "fc"):
        return model.fc, "pre_logits"
    if hasattr(model, "head"):
        # try last linear in head
        for m in reversed(list(model.head.modules())):
            if isinstance(m, nn.Linear):
                return m, "pre_logits"
    return None, None


def run_tsne_final_only(model, per_subj, subject_ids, conf, out_prefix, model_name):
    """
    Non-FreqMLP: only save the final pre-logits feature t-SNE (one figure + one .mat).
    """
    device = conf.get('device', 'cpu')
    bs = conf['training']['batch_size']

    layer, tag = _get_final_feature_layer(model, model_name)
    if layer is None:
        print(f"[t-SNE] Final-only: cannot find final feature layer for {model_name}. Skip.")
        return

    feats_list = []
    Y_list = []

    def hook_inp(module, inp, out):
        # inp[0] is the feature fed into the final classifier/linear
        feats_list.append(inp[0].detach().cpu())

    handle = layer.register_forward_hook(hook_inp)

    model.eval()
    with torch.no_grad():
        for s in subject_ids:
            ds_s = per_subj[s - 1]
            loader_s = DataLoader(ds_s, batch_size=bs, shuffle=False, num_workers=0)
            for x, y in loader_s:
                x = x.to(device)
                _ = model(x)
                Y_list.append(y.detach().cpu())

    handle.remove()

    if len(Y_list) == 0 or len(feats_list) == 0:
        print(f"[t-SNE] Final-only: no data collected for {model_name}. Skip.")
        return

    Y = torch.cat(Y_list, dim=0).numpy()
    X = torch.cat(feats_list, dim=0).numpy()  # (N, D)

    Z = _pca_tsne(X, perplexity=30, seed=0)

    # Save raw for MATLAB
    name = f"{model_name}_{tag}"
    mat_path = f"{out_prefix}_tsne_{name}.mat"
    savemat(mat_path, {
        "Z": Z.astype(np.float32),
        "Y": Y.astype(np.int32),
        "layer": np.array([name], dtype=object)
    })
    print(f"[t-SNE] Saved raw: {mat_path}")

    # Save plot
    png_path = f"{out_prefix}_tsne_{name}_byClass.png"
    _plot_tsne_by_class(Z, Y, f"t-SNE ({name})", png_path)


# ---------------------------------------------------------------------


def main():
    args = parse_args()
    conf = load_config(args.config)
    ModelClass = get_model_class(args.model)

    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)

    subjects = list(range(1, 101))
    test_ids      = subjects[:30]  # default test subjects: 1..30
    train_val_ids = subjects[30:]

    dconf = conf['data']
    riemann_train = dconf.get('riemann_align_train', False)
    riemann_test  = dconf.get('riemann_align_test', False)

    align_matrix = None
    if riemann_train:
        align_ds = EEGDataset(
            dconf['data_root'], train_val_ids,
            session=dconf['session'],
            channel_indices=dconf['channels_sel'],
            highpass=dconf['highpass'],
            lowpass=dconf['lowpass'],
            filter_order=dconf['filter_order'],
            riemann_align=True,
            align_matrix=None
        )
        align_matrix = align_ds.align_matrix

    per_subj = []
    for s in subjects:
        apply_align = riemann_train if (s in train_val_ids) else riemann_test

        if s in test_ids:
            time_shift = dconf.get('time_shift', 0.0)
            spatial_shuffle_p = dconf.get('spatial_shuffle_p', 0.0)
        else:
            time_shift = 0.0
            spatial_shuffle_p = 0.0

        ds = EEGDataset(
            dconf['data_root'], [s],
            session=dconf['session'],
            channel_indices=dconf['channels_sel'],
            highpass=dconf['highpass'],
            lowpass=dconf['lowpass'],
            filter_order=dconf['filter_order'],
            riemann_align=apply_align,
            align_matrix=align_matrix,
            time_shift=time_shift,
            spatial_shuffle_p=spatial_shuffle_p
        )

        per_subj.append(
            TensorDataset(torch.from_numpy(ds.data).float(),
                          torch.from_numpy(ds.labels).long())
        )

    val_ids   = random.sample(train_val_ids, conf['training']['val_n_subjects'])
    train_ids = [s for s in train_val_ids if s not in val_ids]

    train_ds = ConcatDataset([per_subj[s - 1] for s in train_ids])
    val_ds   = ConcatDataset([per_subj[s - 1] for s in val_ids])

    bs = conf['training']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, pin_memory=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, pin_memory=True, num_workers=2)

    start_time = time.time()
    model, train_losses, val_losses, avg_epoch_time, best_epoch = train_one_fold(
        train_loader, val_loader, conf, ModelClass
    )
    train_time = time.time() - start_time

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Loss curve
    plt.figure()
    epochs = list(range(1, len(train_losses) + 1))
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

    # ---- t-SNE visualization (optional) ----
    if conf.get("tsne", {}).get("enable", False):
        tsne_subjects = conf["tsne"].get("subjects", test_ids)  # default: all test subjects 1..30
        out_prefix = f"{args.model}_{ts}"

        if args.model == "FreqMLP":
            run_tsne_freqmlp_multilayer(model, per_subj, tsne_subjects, conf, out_prefix)
        else:
            run_tsne_final_only(model, per_subj, tsne_subjects, conf, out_prefix, args.model)

        print("Saved t-SNE figures and raw .mat files.")

    # Test
    device = conf.get('device', 'cpu')
    test_results = []
    model.eval()
    with torch.no_grad():
        total_infer_time = 0.0
        total_trials = 0
        for s in test_ids:
            ds_s = per_subj[s - 1]
            loader_s = DataLoader(ds_s, batch_size=bs, shuffle=False, pin_memory=True, num_workers=2)
            correct, total = 0, 0
            for x, y in loader_s:
                x, y = x.to(device), y.to(device)
                start_inf = time.time()
                pred = model(x).argmax(1)
                end_inf = time.time()
                total_infer_time += (end_inf - start_inf)
                total_trials += y.size(0)
                correct += (pred == y).sum().item()
                total += y.size(0)
            acc_s = 100.0 * correct / total
            test_results.append((s, acc_s))
            print(f"Subject {s:02d} → {acc_s:.2f}%")
        mean_infer_time = total_infer_time / total_trials

    acc_vals = [a for (_, a) in test_results]
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
        writer.writerow(['train_time_s',  f"{train_time:.2f}"])
        writer.writerow(['avg_epoch_time_s', f"{avg_epoch_time:.4f}"])
        writer.writerow(['early_stopped_epoch', best_epoch])
        writer.writerow(['mean_inference_time_per_trial_s', f"{mean_infer_time:.6f}"])
    print(f"Saved detailed results to {csv_name}")


if __name__ == "__main__":
    main()

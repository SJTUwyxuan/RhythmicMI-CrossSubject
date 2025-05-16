# main.py
import os, time, datetime, random, csv
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
import argparse, yaml, importlib
from dataset import EEGDataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',     type=str, required=True,
                   help='模型名，对应 models.<model>.py 中的类名')
    p.add_argument('--config',    type=str, required=True,
                   help='配置文件路径，比如 configs/SSVEPFormer.yaml')
    return p.parse_args()

def load_config(path):
    if not os.path.exists(path):
        candidate = os.path.join("configs", path)
        if os.path.exists(candidate):
            path = candidate
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_model_class(model_name):
    module = importlib.import_module(f"models.{model_name}")
    return getattr(module, model_name)

def train_one_fold(train_loader, val_loader, test_loader, config, ModelClass):
    # 动态实例化
    m_conf = config['model']['params']
    device = config.get('device', 'cpu')
    model = ModelClass(**m_conf).to(device)
    criterion = nn.CrossEntropyLoss()

    optim_conf = config['training']
    OptClass = getattr(optim, optim_conf['optimizer']['name'])  # torch.optim.SGD / Adam / AdamW
    opt_params = optim_conf['optimizer']['params']

    # 确保把 list 转 tuple（如 betas）
    if 'betas' in opt_params:
        opt_params['betas'] = tuple(opt_params['betas'])

    optimizer = OptClass(model.parameters(), **opt_params)
    
    best_val, patience = float('inf'), 0
    best_state = None

    for epoch in range(1, optim_conf['epochs'] + 1):
        model.train()
        for x,y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); optimizer.step()

        # 验证
        model.eval()
        val_loss, cnt = 0, 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                l = criterion(model(x_val), y_val).item()
                val_loss += l * y_val.size(0)
                cnt += y_val.size(0)
        avg_val = val_loss/cnt
        if avg_val < best_val:
            best_val, patience, best_state = avg_val, 0, model.state_dict()
        else:
            patience += 1
            if patience >= optim_conf['patience']:
                break

    if best_state:
        model.load_state_dict(best_state)

    # 测试
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            preds = model(x_test).argmax(1)
            correct += (preds == y_test).sum().item()
            total   += y_test.size(0)
    return 100.0 * correct / total

def main():
    args   = parse_args()
    conf   = load_config(args.config)
    ModelC = get_model_class(args.model)

    # —— 数据预处理 —— #
    data_conf = conf['data']
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    subjects = list(range(1,61))
    per_subj = []
    for s in subjects:
        ds = EEGDataset(
            data_conf['data_root'], [s],
            session=data_conf['session'],
            channel_indices=data_conf['channels_sel'],
            highpass=data_conf['highpass'],
            lowpass=data_conf['lowpass'],
            filter_order=data_conf['filter_order']
        )
        per_subj.append(
            TensorDataset(torch.from_numpy(ds.data).float(),
                          torch.from_numpy(ds.labels).long())
        )

    results = []
    start = time.time()
    for test_s in subjects:
        train_ids = [s for s in subjects if s!=test_s]
        val_ids   = random.sample(train_ids, conf['training']['val_n_subjects'])
        train_ds  = ConcatDataset([per_subj[i-1] for i in train_ids if i not in val_ids])
        val_ds    = ConcatDataset([per_subj[i-1] for i in val_ids])
        test_ds   = per_subj[test_s-1]

        loaders = {
            'train': DataLoader(train_ds, batch_size=conf['training']['batch_size'], shuffle=True,  pin_memory=True),
            'val':   DataLoader(val_ds,   batch_size=conf['training']['batch_size'], shuffle=False, pin_memory=True),
            'test':  DataLoader(test_ds,  batch_size=conf['training']['batch_size'], shuffle=False, pin_memory=True)
        }

        acc = train_one_fold(loaders['train'], loaders['val'], loaders['test'], conf, ModelC)
        print(f"[{args.model}] Subject {test_s} → {acc:.2f}%")
        results.append((test_s, acc))

    elapsed = time.time() - start
    avg_acc = np.mean([a for _, a in results])
    std_acc = np.std([a for _, a in results])
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"{args.model}_LOSO_{now_str}.csv"

    # Save results including elapsed time
    with open(out_csv, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['subject', 'accuracy'])
        writer.writerows(results)
        writer.writerow(['total_time_sec', f"{elapsed:.2f}"])

    print(f"LOSO average accuracy: {avg_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Total runtime: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()

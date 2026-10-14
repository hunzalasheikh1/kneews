"""Gold-free head configuration selection, then a single evaluation on the 58 expert studies.

Selection: for each outer fold, train on report-only studies outside the fold (exactly as the
registered comparison) and score on report-only studies INSIDE the held-out fold, using only
explicitly mentioned present/absent report states as targets. Expert labels are not read
during selection. The selected configuration is then evaluated once on the expert studies.
Usage: PYTHONPATH=backend .venv312/bin/python scripts/head_selection.py <experiment_id> [--device mps]
"""
import argparse
import json
import random
import time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from app.config import ROOT
from app.research.reliability import PooledReliability
from app.research.metrics import evaluate
from app.research.training import targets

CONFIGS = {
    'registered': dict(hidden=128, dropout=0.0, epochs=30, lr=1e-3, wd=1e-3),
    'short15': dict(hidden=128, dropout=0.0, epochs=15, lr=1e-3, wd=1e-3),
    'long60': dict(hidden=128, dropout=0.0, epochs=60, lr=1e-3, wd=1e-3),
    'wide_dropout': dict(hidden=256, dropout=0.3, epochs=30, lr=1e-3, wd=1e-3),
    'strong_wd': dict(hidden=128, dropout=0.2, epochs=30, lr=1e-3, wd=5e-2),
    'slow_lr': dict(hidden=128, dropout=0.2, epochs=40, lr=3e-4, wd=1e-2),
}


class Head(nn.Module):
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.slot = nn.Parameter(torch.zeros(6, hidden))
        self.project = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, hidden), nn.GELU())
        self.attention = nn.Linear(hidden, 1)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, 12))

    def forward(self, x, mask):
        h = self.project(x) + self.slot[None]
        logits = self.attention(h).squeeze(-1).masked_fill(~mask, -1e9)
        return self.classifier((h * torch.softmax(logits, 1)[..., None]).sum(1))


def fit(x, m, y, w, cfg, seed, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = Head(x.shape[-1], cfg['hidden'], cfg['dropout']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    ds = torch.utils.data.TensorDataset(*[torch.as_tensor(a) for a in (x, m, y, w)])
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed))
    for _ in range(cfg['epochs']):
        model.train()
        for a, b, c, d in loader:
            if d.sum() == 0: continue  # same as registered fit_head
            a, b, c, d = (t.to(device) for t in (a, b, c, d))
            loss = (nn.functional.binary_cross_entropy_with_logits(model(a, b), c, reduction='none') * d).sum() / d.sum()
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.inference_mode()
def predict(model, x, m, device):
    model.eval()
    return np.concatenate([model(torch.as_tensor(x[i:i+256]).to(device), torch.as_tensor(m[i:i+256]).to(device)).sigmoid().cpu().numpy()
                           for i in range(0, len(x), 256)])


def report_auc(states, p):
    aucs = []
    for f in range(12):
        sel = states[:, f] < 2  # present=0 / absent=1 only
        y = (states[sel, f] == 0).astype(int)
        if len(np.unique(y)) == 2: aucs.append(roc_auc_score(y, p[sel, f]))
    return float(np.mean(aucs))


def run_fold_predictions(z, folds, is_gold, arm, cfg, seed, device, target_rows, llm=None):
    out = np.full(z['states'].shape, np.nan)
    for fold in sorted(set(folds)):
        held = (folds == fold) & target_rows
        train = (folds != fold) & ~is_gold
        estimate = (folds != fold) & is_gold
        rel = PooledReliability().fit(z['states'][estimate], z['gold'][estimate], list(np.where(estimate)[0]), list(np.where(held & is_gold)[0])) if arm == 'pooled' else None
        y, w = targets(arm, z['states'][train], rel, llm[train] if llm is not None else None)
        model = fit(z['features'][train], z['masks'][train], y.astype(np.float32), w.astype(np.float32), cfg, seed, device)
        out[held] = predict(model, z['features'][held], z['masks'][held], device)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('experiment')
    ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    ap.add_argument('--arms', default='binary,pooled')
    a = ap.parse_args()
    E = ROOT / 'artifacts' / 'experiments' / a.experiment
    z = dict(np.load(E / 'training-inputs.npz'))
    manifest = json.loads((E / 'manifest.json').read_text())
    order = {r['id']: r['fold'] for r in manifest['folds']}
    folds = np.array([order[s] for s in z['study_ids']])
    is_gold = np.isfinite(z['gold']).all(1)
    started = time.time()
    # 1. Gold-free selection on the pooled arm (proposed method), seed 2026.
    selection = {}
    for name, cfg in CONFIGS.items():
        p = run_fold_predictions(z, folds, is_gold, 'pooled', cfg, 2026, a.device, ~is_gold)
        selection[name] = report_auc(z['states'][~is_gold], p[~is_gold])
        print(f'select {name:14s} report-only held-out AUC={selection[name]:.4f}  ({time.time()-started:.0f}s)', flush=True)
    chosen = max(selection, key=selection.get)
    print('chosen:', chosen, flush=True)
    # 2. Single expert evaluation: registered vs chosen, 3 seeds + seed ensemble, per arm.
    result = dict(experiment=a.experiment, selection_metric='report-only held-out-fold AUC on explicit present/absent states',
                  selection=selection, chosen=chosen, configs=CONFIGS, evaluation={})
    for arm in a.arms.split(','):
        for name in dict.fromkeys(['registered', chosen]):
            preds = [run_fold_predictions(z, folds, is_gold, arm, CONFIGS[name], s, a.device, is_gold)[is_gold] for s in (2026, 2027, 2028)]
            per_seed = [evaluate(z['gold'][is_gold], p, include_calibration=False) for p in preds]
            ens = evaluate(z['gold'][is_gold], np.mean(preds, 0), include_calibration=False)
            summ = lambda m: dict(macro_auc=m['macro_auc'], worst=m['worst_finding_auc'], brier=m['brier'],
                                  ece=float(np.mean([f['ece'] for f in m['per_finding']])))
            result['evaluation'][f'{arm}/{name}'] = dict(per_seed=[summ(m) for m in per_seed], seed_ensemble=summ(ens),
                                                         ensemble_per_finding=ens['per_finding'])
            print(arm, name, 'seeds', [round(m['macro_auc'], 3) for m in per_seed], 'ensemble', summ(ens), flush=True)
    out = ROOT / 'artifacts' / 'audit' / f'head-selection-{a.experiment}.json'
    out.write_text(json.dumps(result, indent=1))
    print('saved', out)


if __name__ == '__main__':
    main()

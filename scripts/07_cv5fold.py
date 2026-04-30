#!/usr/bin/env python3
"""
07_cv5fold.py — 5-fold StratifiedKFold cross-validation on all 300 GEO samples.

Replaces the 3-fold CV on 255 samples (train+val split) as the primary reported result.
Trains LiquidBiopsyNet (CNN+LSTM) with proper per-fold scaling (no leakage).
Saves results to results/cv5fold_results.json and report/figures/cv5fold_results.png.
"""

import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             balanced_accuracy_score, confusion_matrix,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'model'))
from model_architecture import LiquidBiopsyNet

SEED     = 42
N_FOLDS  = 5
N_EPOCHS = 60
PATIENCE = 10
LR       = 1e-3
BATCH    = 32
BASE     = Path(__file__).parent.parent
DPI      = 300

np.random.seed(SEED)
torch.manual_seed(SEED)


def load_data():
    df = pd.read_csv(BASE / 'data' / 'feature_matrix.csv')
    frag_cols = [c for c in df.columns if c.startswith('frag_bin_')]
    extra_cols = ['mean_frag_len', 'std_frag_len', 'short_frag_ratio',
                  'methylation_density', 'mean_gc', 'nuc_score']
    extra_cols = [c for c in extra_cols if c in df.columns]
    feat_cols = frag_cols + extra_cols

    X = df[feat_cols].values.astype(np.float32)
    y = (df['sample_type'] == 'cancer').astype(int).values

    print(f"Dataset : n={len(X)}, features={X.shape[1]}")
    print(f"Classes : cancer={y.sum()}, healthy={(y==0).sum()}, "
          f"imbalance={y.mean()*100:.1f}% cancer")
    return X, y, df['tissue_type'].values


def train_one_fold(X_tr, y_tr, X_vl, y_vl, fold_i):
    input_len = 150  # frag_bin columns
    model = LiquidBiopsyNet(input_len=input_len)

    counts = np.bincount(y_tr)
    weight = torch.FloatTensor(counts.sum() / (len(counts) * counts))
    criterion  = nn.CrossEntropyLoss(weight=weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    X_tr_t = torch.FloatTensor(X_tr)
    X_vl_t = torch.FloatTensor(X_vl)
    y_tr_t = torch.LongTensor(y_tr)
    y_vl_t = torch.LongTensor(y_vl)

    ds     = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)

    best_vl_loss  = float('inf')
    best_state    = None
    patience_ctr  = 0
    history       = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    for epoch in range(N_EPOCHS):
        model.train()
        ep_loss = 0.0
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
        ep_loss /= len(loader)

        model.eval()
        with torch.no_grad():
            vl_out  = model(X_vl_t)
            vl_loss = criterion(vl_out, y_vl_t).item()
            vl_prob = torch.softmax(vl_out, dim=1)[:, 1].numpy()
        try:
            vl_auc = roc_auc_score(y_vl, vl_prob) if len(np.unique(y_vl)) > 1 else 0.5
        except Exception:
            vl_auc = 0.5

        scheduler.step(vl_loss)
        history['train_loss'].append(ep_loss)
        history['val_loss'].append(vl_loss)
        history['val_auc'].append(vl_auc)

        if vl_loss < best_vl_loss:
            best_vl_loss = vl_loss
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        vl_out  = model(X_vl_t)
        vl_prob = torch.softmax(vl_out, dim=1)[:, 1].numpy()
        vl_pred = vl_out.argmax(dim=1).numpy()

    auc     = roc_auc_score(y_vl, vl_prob) if len(np.unique(y_vl)) > 1 else 0.5
    acc     = accuracy_score(y_vl, vl_pred)
    bal_acc = balanced_accuracy_score(y_vl, vl_pred)
    cm      = confusion_matrix(y_vl, vl_pred)
    fpr, tpr, _ = roc_curve(y_vl, vl_prob)

    stopped_epoch = len(history['train_loss'])
    print(f"  Fold {fold_i}: AUC={auc:.4f}  Acc={acc:.4f}  BalAcc={bal_acc:.4f}  "
          f"stopped_epoch={stopped_epoch}")

    return {
        'auc': float(auc), 'acc': float(acc), 'bal_acc': float(bal_acc),
        'cm': cm.tolist(), 'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
        'y_val': y_vl.tolist(), 'y_prob': vl_prob.tolist(),
        'stopped_epoch': stopped_epoch,
    }, history


def run_cv(X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []
    all_histories = []

    print(f"\nRunning {N_FOLDS}-fold StratifiedKFold CV on n={len(X)} samples...\n")
    for fold_i, (tr_idx, vl_idx) in enumerate(skf.split(X, y), 1):
        print(f"[Fold {fold_i}/{N_FOLDS}]  train={len(tr_idx)}  val={len(vl_idx)}", flush=True)
        X_tr, X_vl = X[tr_idx], X[vl_idx]
        y_tr, y_vl = y[tr_idx], y[vl_idx]

        # Scale inside fold — no leakage
        sc   = StandardScaler()
        X_tr = sc.fit_transform(X_tr).astype(np.float32)
        X_vl = sc.transform(X_vl).astype(np.float32)

        res, hist = train_one_fold(X_tr, y_tr, X_vl, y_vl, fold_i)
        fold_results.append(res)
        all_histories.append(hist)

    return fold_results, all_histories


def summarise(fold_results):
    aucs     = [r['auc']     for r in fold_results]
    accs     = [r['acc']     for r in fold_results]
    bal_accs = [r['bal_acc'] for r in fold_results]
    epochs   = [r['stopped_epoch'] for r in fold_results]

    summary = {
        'n_folds'        : N_FOLDS,
        'n_samples'      : 300,
        'seed'           : SEED,
        'cv_auc_mean'    : float(np.mean(aucs)),
        'cv_auc_std'     : float(np.std(aucs)),
        'cv_auc_per_fold': aucs,
        'cv_acc_mean'    : float(np.mean(accs)),
        'cv_acc_std'     : float(np.std(accs)),
        'cv_acc_per_fold': accs,
        'cv_bal_acc_mean': float(np.mean(bal_accs)),
        'cv_bal_acc_std' : float(np.std(bal_accs)),
        'cv_bal_acc_per_fold': bal_accs,
        'mean_stopped_epoch': float(np.mean(epochs)),
        'note': (
            '5-fold StratifiedKFold on all 300 GEO samples (GSE149438). '
            'Scaler fit inside each fold on train split only. '
            'Primary reported result replacing 3-fold CV on 255-sample subset.'
        )
    }
    return summary


def plot_results(fold_results, summary, out_path):
    n = len(fold_results)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: AUC per fold + mean±std
    ax = axes[0]
    aucs = [r['auc'] for r in fold_results]
    ax.bar(range(1, n+1), aucs, color='#3182bd', alpha=0.8, edgecolor='white')
    ax.axhline(summary['cv_auc_mean'], color='#e6550d', lw=2,
               linestyle='--', label=f"Mean={summary['cv_auc_mean']:.3f}±{summary['cv_auc_std']:.3f}")
    ax.fill_between([-0.5, n+0.5],
                    summary['cv_auc_mean'] - summary['cv_auc_std'],
                    summary['cv_auc_mean'] + summary['cv_auc_std'],
                    color='#e6550d', alpha=0.15)
    for i, v in enumerate(aucs):
        ax.text(i+1, v+0.008, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(range(1, n+1))
    ax.set_xticklabels([f'Fold {i}' for i in range(1, n+1)])
    ax.set_ylabel('AUC-ROC')
    ax.set_ylim(0.5, 1.05)
    ax.set_title('AUC-ROC per Fold', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    # Panel B: ROC curves (one per fold)
    ax2 = axes[1]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, n))
    for i, r in enumerate(fold_results):
        ax2.plot(r['fpr'], r['tpr'], color=colors[i], lw=1.5, alpha=0.8,
                 label=f"Fold {i+1} (AUC={r['auc']:.3f})")
    ax2.plot([0,1],[0,1],'k--', lw=1, alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves — All Folds', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Panel C: Bal accuracy per fold
    ax3 = axes[2]
    bal_accs = [r['bal_acc'] for r in fold_results]
    metrics  = ['AUC-ROC', 'Accuracy', 'Bal. Accuracy']
    means    = [summary['cv_auc_mean'], summary['cv_acc_mean'], summary['cv_bal_acc_mean']]
    stds     = [summary['cv_auc_std'],  summary['cv_acc_std'],  summary['cv_bal_acc_std']]
    bar_colors = ['#3182bd', '#31a354', '#e6550d']
    bars = ax3.bar(metrics, means, yerr=stds, capsize=6,
                   color=bar_colors, alpha=0.85, edgecolor='white')
    for bar, m, s in zip(bars, means, stds):
        ax3.text(bar.get_x() + bar.get_width()/2, m + s + 0.012,
                 f'{m:.3f}±{s:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax3.set_ylabel('Score')
    ax3.set_ylim(0.4, 1.1)
    ax3.set_title('Summary Metrics (mean ± std)', fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')

    fig.suptitle(
        f'LiquidBiopsyNet — {N_FOLDS}-fold StratifiedKFold CV\n'
        f'GEO Dataset GSE149438 (n=300, 84.7% cancer, {N_FOLDS} folds, seed={SEED})',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Figure saved: {out_path}")


def main():
    print('=' * 60)
    print(f'5-FOLD STRATIFIEDKFOLD CV — LiquidBiopsyNet')
    print(f'All 300 GEO samples, seed={SEED}')
    print('=' * 60)

    X, y, tissue = load_data()

    fold_results, histories = run_cv(X, y)
    summary = summarise(fold_results)

    print('\n' + '=' * 60)
    print('RESULTS SUMMARY')
    print('=' * 60)
    print(f"AUC-ROC : {summary['cv_auc_mean']:.4f} ± {summary['cv_auc_std']:.4f}")
    print(f"Accuracy: {summary['cv_acc_mean']:.4f} ± {summary['cv_acc_std']:.4f}")
    print(f"BalAcc  : {summary['cv_bal_acc_mean']:.4f} ± {summary['cv_bal_acc_std']:.4f}")
    print(f"Per-fold AUC: {[f'{v:.4f}' for v in summary['cv_auc_per_fold']]}")
    print(f"Mean stopped epoch: {summary['mean_stopped_epoch']:.1f}")

    # Save JSON
    out_json = BASE / 'results' / 'cv5fold_results.json'
    # Don't serialize fpr/tpr lists into this summary file (too large)
    summary_small = {k: v for k, v in summary.items()}
    with open(out_json, 'w') as f:
        json.dump(summary_small, f, indent=2)
    print(f"\n✓ Metrics saved: {out_json}")

    # Save fold details (with ROC curves)
    out_folds = BASE / 'results' / 'cv5fold_fold_details.json'
    fold_data = [{k: v for k, v in r.items() if k not in ('fpr','tpr')}
                 for r in fold_results]
    with open(out_folds, 'w') as f:
        json.dump(fold_data, f, indent=2)

    # Figure
    plot_results(fold_results, summary, BASE / 'report' / 'figures' / 'cv5fold_results.png')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
04_evaluate.py — Liquid Biopsy AI: Model Evaluation v2

Changes from v1:
  - Adds Logistic Regression baseline comparison
    If LR also hits 1.0: dataset is too easy / still leaky
    If LR is ~70-85% and CNN+LSTM is higher: improvement is meaningful
  - Saves corrected metrics to results/metrics_corrected.json
  - All scalers fit on train data only (leakage-free)
  - Generates updated plots
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

import random
random.seed(42)
np.random.seed(42)

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from model_architecture import LiquidBiopsyNet


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate Liquid Biopsy AI (v2 — with LR baseline)')
    p.add_argument('--data-dir',    default='../results')
    p.add_argument('--model-path',  default='../model/best_model.pt')
    p.add_argument('--results-dir', default='../results')
    p.add_argument('--figures-dir', default='../report/figures')
    p.add_argument('--class-names', nargs='+', default=['Healthy', 'Cancer'])
    return p.parse_args()


def load_model(path, device):
    if not os.path.exists(path):
        print(f"[WARN] Model not found: {path}")
        return None, None
    ckpt = torch.load(path, map_location=device)
    cfg  = ckpt.get('config', {'input_len': 150, 'num_classes': 2, 'dropout': 0.5})
    model = LiquidBiopsyNet(
        input_len   = cfg['input_len'],
        num_classes = cfg['num_classes'],
        dropout     = cfg.get('dropout', 0.5),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"[OK] Model loaded (epoch {ckpt.get('epoch','?')})")
    print(f"     CV AUC: {ckpt.get('cv_auc_mean','?'):.4f} ± {ckpt.get('cv_auc_std','?'):.4f}")
    return model, cfg


def get_nn_predictions(model, X, device, batch_size=64):
    from torch.utils.data import DataLoader, TensorDataset
    ds = DataLoader(TensorDataset(torch.FloatTensor(X)), batch_size=batch_size)
    all_probs = []
    model.eval()
    with torch.no_grad():
        for (Xb,) in ds:
            logits = model(Xb.to(device))
            all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
    probs = np.vstack(all_probs)
    return probs.argmax(1), probs


def evaluate_model(name, y_true, y_pred, y_proba, class_names):
    n_classes = len(class_names)
    out = {
        'model'    : name,
        'accuracy' : float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall'   : float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1'       : float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }
    try:
        out['auc_roc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
        out['auc_pr']  = float(average_precision_score(y_true, y_proba[:, 1]))
    except Exception:
        out['auc_roc'] = float('nan')
        out['auc_pr']  = float('nan')
    return out


def plot_confusion_matrix(y_true, y_pred, class_names, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j+0.5, i+0.5, f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)',
                    ha='center', va='center', fontsize=13, fontweight='bold',
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {out_path}")


def plot_roc_comparison(y_true, results_dict, class_names, out_path):
    """ROC curves for all models on the same axes."""
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    fig, ax = plt.subplots(figsize=(8, 6))

    for (name, proba), color in zip(results_dict.items(), colors):
        try:
            fpr, tpr, _ = roc_curve(y_true, proba[:, 1])
            auc = roc_auc_score(y_true, proba[:, 1])
            ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{name} (AUC={auc:.3f})')
        except Exception as e:
            print(f"[WARN] ROC for {name}: {e}")

    ax.plot([0,1],[0,1],'k--',lw=1.5,alpha=0.5,label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curve Comparison (Corrected)', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Liquid Biopsy AI — Evaluation v2 ===")
    print(f"  Device: {device}\n")

    # Load data
    X_train = np.load(os.path.join(args.data_dir, 'X_train.npy')).astype(np.float32)
    y_train = np.load(os.path.join(args.data_dir, 'y_train.npy')).astype(np.int64)
    X_test  = np.load(os.path.join(args.data_dir, 'X_test.npy')).astype(np.float32)
    y_test  = np.load(os.path.join(args.data_dir, 'y_test.npy')).astype(np.int64)

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    n_classes = max(len(np.unique(y_test)), len(args.class_names))
    class_names = args.class_names[:n_classes]

    all_results  = {}   # name → metrics dict
    all_probas   = {}   # name → proba array (for ROC plot)

    # ── Baseline 1: Logistic Regression ──────────────────────────────────────
    print("\n--- Baseline: Logistic Regression (5-fold CV) ---")
    # Leakage-free: fit scaler on X_train inside CV or directly on train
    lr_scaler = StandardScaler()
    X_tr_lr   = lr_scaler.fit_transform(X_train)
    X_te_lr   = lr_scaler.transform(X_test)

    lr = LogisticRegression(max_iter=2000, random_state=42, C=0.1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr_cv_aucs = cross_val_score(lr, X_tr_lr, y_train, cv=skf, scoring='roc_auc')
    lr_cv_accs = cross_val_score(lr, X_tr_lr, y_train, cv=skf, scoring='accuracy')
    print(f"  CV Accuracy: {lr_cv_accs.mean():.4f} ± {lr_cv_accs.std():.4f}")
    print(f"  CV AUC-ROC : {lr_cv_aucs.mean():.4f} ± {lr_cv_aucs.std():.4f}")

    # Fit on full train, evaluate on test
    lr.fit(X_tr_lr, y_train)
    lr_pred  = lr.predict(X_te_lr)
    lr_proba = lr.predict_proba(X_te_lr)
    lr_metrics = evaluate_model('Logistic Regression', y_test, lr_pred, lr_proba, class_names)
    lr_metrics.update({
        'cv_accuracy_mean': float(lr_cv_accs.mean()),
        'cv_accuracy_std' : float(lr_cv_accs.std()),
        'cv_auc_mean'     : float(lr_cv_aucs.mean()),
        'cv_auc_std'      : float(lr_cv_aucs.std()),
    })
    all_results['Logistic Regression'] = lr_metrics
    all_probas['Logistic Regression']  = lr_proba

    print(f"  Test Accuracy: {lr_metrics['accuracy']:.4f}")
    print(f"  Test AUC-ROC : {lr_metrics['auc_roc']:.4f}")

    # ── Diagnosis ──────────────────────────────────────────────────────────────
    if lr_metrics['auc_roc'] >= 0.99:
        print(f"\n  ⚠  LR also achieves AUC={lr_metrics['auc_roc']:.4f}.")
        print(f"     This means the dataset itself separates trivially.")
        print(f"     CNN+LSTM shows architectural advantage only if it outperforms LR")
        print(f"     on harder (real BAM) data.")
    else:
        print(f"\n  ✓ LR AUC={lr_metrics['auc_roc']:.4f} — not perfect.")
        print(f"    Any improvement by CNN+LSTM is architecturally meaningful.")

    # ── Neural network evaluation ─────────────────────────────────────────────
    print("\n--- CNN + LSTM Model ---")
    model, cfg = load_model(args.model_path, device)

    if model is not None:
        nn_pred, nn_proba = get_nn_predictions(model, X_test, device)
        nn_metrics = evaluate_model('CNN + LSTM', y_test, nn_pred, nn_proba, class_names)

        # Pull CV stats from checkpoint
        ckpt = torch.load(args.model_path, map_location='cpu')
        nn_metrics.update({
            'cv_accuracy_mean': ckpt.get('cv_acc_mean', float('nan')),
            'cv_accuracy_std' : ckpt.get('cv_acc_std',  float('nan')),
            'cv_auc_mean'     : ckpt.get('cv_auc_mean', float('nan')),
            'cv_auc_std'      : ckpt.get('cv_auc_std',  float('nan')),
        })
        all_results['CNN + LSTM'] = nn_metrics
        all_probas['CNN + LSTM']  = nn_proba

        print(f"  Test Accuracy: {nn_metrics['accuracy']:.4f}")
        print(f"  Test AUC-ROC : {nn_metrics['auc_roc']:.4f}")
        print(f"  5-fold CV AUC: {nn_metrics['cv_auc_mean']:.4f} ± {nn_metrics['cv_auc_std']:.4f}")
        y_pred_main = nn_pred
    else:
        print("[DEMO] No model found — using LR as primary model for plots")
        y_pred_main = lr_pred

    # ── Classification report ─────────────────────────────────────────────────
    print(f"\n=== Classification Reports ===")
    for name, (pred, proba) in [
        ('Logistic Regression', (lr_pred, lr_proba)),
        *([('CNN + LSTM', (nn_pred, nn_proba))] if model is not None else []),
    ]:
        print(f"\n{name}:")
        print(classification_report(y_test, pred, target_names=class_names, zero_division=0))

    # ── Save corrected metrics ────────────────────────────────────────────────
    corrected = {
        'note'   : 'v2 — leakage-free scaler, 5-fold CV, reduced model capacity',
        'models' : all_results,
        'diagnosis': {
            'lr_auc_roc'        : float(lr_metrics['auc_roc']),
            'lr_cv_auc'         : float(lr_metrics.get('cv_auc_mean', float('nan'))),
            'dataset_difficulty': 'trivial' if lr_metrics['auc_roc'] >= 0.99 else 'non-trivial',
            'nn_better_than_lr' : (
                (all_results['CNN + LSTM']['auc_roc'] > lr_metrics['auc_roc'] + 0.01)
                if 'CNN + LSTM' in all_results else 'n/a'
            ),
        },
    }

    metrics_path = os.path.join(args.results_dir, 'metrics_corrected.json')
    with open(metrics_path, 'w') as f:
        json.dump(corrected, f, indent=2)
    print(f"\n[OK] Corrected metrics: {metrics_path}")

    # Also overwrite metrics.json with CNN+LSTM result for 05_visualise.py
    primary = all_results.get('CNN + LSTM', all_results.get('Logistic Regression', {}))
    with open(os.path.join(args.results_dir, 'metrics.json'), 'w') as f:
        json.dump(primary, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\n=== Generating plots ===")
    primary_pred  = nn_pred  if model is not None else lr_pred
    primary_proba = nn_proba if model is not None else lr_proba

    plot_confusion_matrix(
        y_test, primary_pred, class_names,
        'Confusion Matrix — CNN+LSTM (Corrected)',
        os.path.join(args.figures_dir, 'confusion_matrix.png'),
    )
    plot_roc_comparison(
        y_test, all_probas,
        class_names,
        os.path.join(args.figures_dir, 'roc_curve.png'),
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n=== Corrected Evaluation Summary ===")
    for name, m in all_results.items():
        cv_str = (f" | CV AUC: {m['cv_auc_mean']:.4f}±{m['cv_auc_std']:.4f}"
                  if 'cv_auc_mean' in m else '')
        print(f"  {name:22s} Acc={m['accuracy']:.4f} AUC={m['auc_roc']:.4f}{cv_str}")

    print(f"\n  Saved: {metrics_path}")
    print(f"  Run next: python3 05_visualise.py")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
03_train_model.py — Liquid Biopsy AI: Model Training v2
NOTE: Run this on Google Colab if no GPU is available locally.

Changes from v1:
  - 5-fold StratifiedKFold cross-validation (standard in bioinformatics papers)
  - Scaler fit inside each fold on train split only (no leakage)
  - Dropout 0.5 (was 0.3)
  - L2 weight_decay = 1e-4 (unchanged, now explicit)
  - Model uses reduced architecture from model_architecture.py v2
  - Reports mean ± std across folds
  - Trains final model on full train+val for deployment
"""

import argparse
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from model_architecture import LiquidBiopsyNet

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def parse_args():
    p = argparse.ArgumentParser(description='Train Liquid Biopsy AI (5-fold CV)')
    p.add_argument('--data-dir',    default='../results')
    p.add_argument('--model-dir',   default='../model')
    p.add_argument('--results-dir', default='../results')
    p.add_argument('--epochs',      type=int,   default=50)
    p.add_argument('--batch-size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--patience',    type=int,   default=10)
    p.add_argument('--n-folds',     type=int,   default=5)
    p.add_argument('--num-classes', type=int,   default=2)
    p.add_argument('--seed',        type=int,   default=42)
    return p.parse_args()


def load_data(data_dir):
    """Load preprocessed .npy splits."""
    parts = {}
    for name in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']:
        path = os.path.join(data_dir, f'{name}.npy')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run 01_preprocess.py first.")
        parts[name] = np.load(path)
    return parts


def make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def build_model(input_len, num_classes, device):
    model = LiquidBiopsyNet(
        input_len=min(input_len, 150),
        num_classes=num_classes,
        dropout=0.5,          # regularisation
    ).to(device)
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    tot_loss = tot_acc = n = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot_loss += loss.item()
        tot_acc  += (logits.argmax(1) == yb).float().mean().item()
        n += 1
    return tot_loss / n, tot_acc / n


def eval_epoch(model, loader, criterion, device):
    model.eval()
    tot_loss = tot_acc = n = 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            loss   = criterion(logits, yb)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            tot_loss += loss.item()
            tot_acc  += (logits.argmax(1) == yb).float().mean().item()
            all_probs.append(probs)
            all_labels.append(yb.cpu().numpy())
            n += 1
    probs  = np.vstack(all_probs)
    labels = np.concatenate(all_labels)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = float('nan')
    return tot_loss / n, tot_acc / n, auc


def train_fold(X_tr, y_tr, X_vl, y_vl, input_len, num_classes,
               args, device, fold_idx):
    """Train one fold. Scaler already applied before calling this."""
    model     = build_model(input_len, num_classes, device)
    weights   = torch.FloatTensor(
        len(np.unique(y_tr)) / (len(np.unique(y_tr)) * np.bincount(y_tr))
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-4  # L2 regularisation
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    tr_loader = make_loader(X_tr, y_tr, args.batch_size, shuffle=True)
    vl_loader = make_loader(X_vl, y_vl, args.batch_size, shuffle=False)

    best_vl_loss  = float('inf')
    best_state    = None
    patience_ctr  = 0
    fold_history  = []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc             = train_epoch(model, tr_loader, criterion, optimizer, device)
        vl_loss, vl_acc, vl_auc    = eval_epoch(model, vl_loader, criterion, device)
        scheduler.step(vl_loss)

        fold_history.append({
            'fold': fold_idx, 'epoch': epoch,
            'train_loss': tr_loss, 'train_acc': tr_acc,
            'val_loss': vl_loss, 'val_acc': vl_acc, 'val_auc': vl_auc,
        })

        if vl_loss < best_vl_loss:
            best_vl_loss = vl_loss
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    _, best_acc, best_auc = eval_epoch(model, vl_loader, criterion, device)
    return model, best_acc, best_auc, fold_history


def main():
    args = parse_args()
    os.makedirs(args.model_dir,   exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Liquid Biopsy AI — Training v2 (5-fold StratifiedKFold) ===")
    print(f"  Device  : {device}")
    if device.type == 'cuda':
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
    else:
        print(f"  NOTE    : No GPU — consider Google Colab for faster training.")

    # Load splits
    data = load_data(args.data_dir)
    X_train_full = data['X_train']
    y_train_full = data['y_train']
    X_test       = data['X_test']
    y_test       = data['y_test']

    # Merge train + val for CV (val was a held-out slice, CV handles this properly)
    if 'X_val' in data:
        X_cv = np.vstack([X_train_full, data['X_val']])
        y_cv = np.concatenate([y_train_full, data['y_val']])
    else:
        X_cv, y_cv = X_train_full, y_train_full

    input_len  = min(X_cv.shape[1], 150)
    n_classes  = max(len(np.unique(y_cv)), args.num_classes)

    print(f"\n  CV data : {X_cv.shape[0]} samples, {X_cv.shape[1]} features")
    print(f"  Test    : {X_test.shape[0]} samples")
    print(f"  Classes : {n_classes}")
    print(f"\n  NOTE: Scaler is fit inside each fold on the CV-train split only.")
    print(f"        This is the correct, leakage-free procedure.\n")

    # ── 5-fold Stratified Cross-Validation ───────────────────────────────────
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)

    fold_accs = []
    fold_aucs = []
    all_history = []

    for fold_idx, (tr_idx, vl_idx) in enumerate(skf.split(X_cv, y_cv), 1):
        print(f"--- Fold {fold_idx}/{args.n_folds} ---")

        X_tr_raw, X_vl_raw = X_cv[tr_idx], X_cv[vl_idx]
        y_tr,     y_vl     = y_cv[tr_idx], y_cv[vl_idx]

        # LEAKAGE-FREE: fit scaler on this fold's training split only
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr_raw).astype(np.float32)
        X_vl   = scaler.transform(X_vl_raw).astype(np.float32)

        _, fold_acc, fold_auc, fold_hist = train_fold(
            X_tr, y_tr, X_vl, y_vl,
            input_len, n_classes, args, device, fold_idx
        )

        fold_accs.append(fold_acc)
        fold_aucs.append(fold_auc)
        all_history.extend(fold_hist)

        print(f"  Fold {fold_idx} → Acc={fold_acc:.4f}, AUC={fold_auc:.4f}")

    print(f"\n=== 5-Fold Cross-Validation Results ===")
    print(f"  Accuracy : {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    print(f"  AUC-ROC  : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"  Per-fold AUC: {[round(a,4) for a in fold_aucs]}")

    # ── Train final model on full train data ─────────────────────────────────
    print(f"\n=== Training final model on full train+val data ===")
    final_scaler = StandardScaler()
    X_cv_scaled  = final_scaler.fit_transform(X_cv).astype(np.float32)
    X_test_scaled = final_scaler.transform(X_test).astype(np.float32)

    final_model = build_model(input_len, n_classes, device)
    weights   = torch.FloatTensor(
        len(np.unique(y_cv)) / (len(np.unique(y_cv)) * np.bincount(y_cv))
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        final_model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    tr_loader   = make_loader(X_cv_scaled,   y_cv,    args.batch_size, shuffle=True)
    test_loader = make_loader(X_test_scaled, y_test,  args.batch_size, shuffle=False)

    best_loss = float('inf')
    best_state = None
    patience_ctr = 0
    final_history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc          = train_epoch(final_model, tr_loader, criterion, optimizer, device)
        te_loss, te_acc, te_auc  = eval_epoch(final_model, test_loader, criterion, device)
        scheduler.step(tr_loss)

        final_history.append({
            'fold': 'final', 'epoch': epoch,
            'train_loss': tr_loss, 'train_acc': tr_acc,
            'val_loss': te_loss,   'val_acc': te_acc, 'val_auc': te_auc,
        })

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: train_loss={tr_loss:.4f} acc={tr_acc:.4f} "
                  f"| test_acc={te_acc:.4f} auc={te_auc:.4f}")

        if tr_loss < best_loss:
            best_loss  = tr_loss
            best_state = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    final_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Final test evaluation
    _, test_acc, test_auc = eval_epoch(final_model, test_loader, criterion, device)
    print(f"\n  Final test → Acc={test_acc:.4f}, AUC={test_auc:.4f}")

    # Save model
    model_path = os.path.join(args.model_dir, 'best_model.pt')
    torch.save({
        'epoch'      : args.epochs,
        'model_state': final_model.state_dict(),
        'config'     : {'input_len': input_len, 'num_classes': n_classes, 'dropout': 0.5},
        'val_loss'   : best_loss,
        'val_acc'    : test_acc,
        'cv_acc_mean': float(np.mean(fold_accs)),
        'cv_acc_std' : float(np.std(fold_accs)),
        'cv_auc_mean': float(np.mean(fold_aucs)),
        'cv_auc_std' : float(np.std(fold_aucs)),
    }, model_path)
    print(f"  Model saved: {model_path}")

    # Save training history (all folds + final)
    all_history.extend(final_history)
    hist_df = pd.DataFrame(all_history)
    hist_path = os.path.join(args.results_dir, 'training_history.csv')
    hist_df.to_csv(hist_path, index=False)

    # Save summary
    summary = {
        'cv_accuracy_mean' : float(np.mean(fold_accs)),
        'cv_accuracy_std'  : float(np.std(fold_accs)),
        'cv_auc_mean'      : float(np.mean(fold_aucs)),
        'cv_auc_std'       : float(np.std(fold_aucs)),
        'cv_per_fold_acc'  : [float(a) for a in fold_accs],
        'cv_per_fold_auc'  : [float(a) for a in fold_aucs],
        'test_accuracy'    : float(test_acc),
        'test_auc'         : float(test_auc),
        'n_params'         : final_model.count_parameters(),
        'n_folds'          : args.n_folds,
        'leakage_fix'      : 'scaler fit inside each fold on CV-train split',
        'regularisation'   : 'Dropout=0.5, L2=1e-4, reduced capacity (GAP)',
    }
    with open(os.path.join(args.results_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== FINAL RESULTS ===")
    print(f"  5-fold CV Accuracy : {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    print(f"  5-fold CV AUC-ROC  : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"  Hold-out Test AUC  : {test_auc:.4f}")
    print(f"\n  Run next: python3 04_evaluate.py")


if __name__ == '__main__':
    main()

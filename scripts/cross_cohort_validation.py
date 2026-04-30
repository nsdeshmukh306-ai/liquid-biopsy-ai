#!/usr/bin/env python3
"""
cross_cohort_validation.py
==========================
Apply the trained CNN+LSTM model (model/best_model.pt) to the Jiang HCC
external cohort (PRJNA291063) features extracted by the Nextflow pipeline.

Computes:
  - AUC-ROC on external cohort
  - Balanced accuracy, sensitivity, specificity
  - Comparison of GEO (internal) vs PRJNA291063 (external) performance
  - Generates cross_cohort_validation.png

Also loads GEO internal test-set metrics from results/full_metrics_publication.json
for side-by-side comparison.

Requires:
  - results/validation/jiang_hcc_features.csv  (from process_validation_cohorts.py)
  - model/best_model.pt
  - results/X_train.npy  (for scaler fitting)
  - results/y_train.npy
"""

import os, sys, json
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), '..')

# ── Model architecture (must match train_model.py) ───────────────────────────
import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, input_size=150, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool  = nn.MaxPool1d(2)
        self.dropout_cnn = nn.Dropout(0.3)
        # After 2x pool: input_size//4 timesteps, but use global avg pool instead
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.lstm  = nn.LSTM(input_size=1, hidden_size=64, num_layers=2,
                             batch_first=True, dropout=0.3)
        self.dropout_lstm = nn.Dropout(0.3)
        # CNN branch: 64 features, LSTM branch: 64 features
        self.fc1 = nn.Linear(64 + 64, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, 150)
        # CNN branch
        c = x.unsqueeze(1)                         # (batch, 1, 150)
        c = torch.relu(self.bn1(self.conv1(c)))
        c = torch.relu(self.bn2(self.conv2(c)))
        c = self.dropout_cnn(c)
        c = self.global_avg_pool(c).squeeze(-1)    # (batch, 64)
        # LSTM branch
        l = x.unsqueeze(-1)                        # (batch, 150, 1)
        l, _ = self.lstm(l)
        l = self.dropout_lstm(l[:, -1, :])         # last hidden (batch, 64)
        # Combine
        out = torch.cat([c, l], dim=1)             # (batch, 128)
        out = torch.relu(self.fc1(out))
        return self.fc2(out)


def load_and_align_features(ext_df, internal_feature_cols):
    """
    Align external cohort feature columns to match training feature columns.
    Fills missing columns with 0, drops extra columns.
    """
    # Find numeric columns that look like fragment bin features
    frag_cols = [c for c in ext_df.columns if c.startswith('bin_') or
                 c.startswith('frag_') or c.startswith('f_') or
                 any(c == f'bin_{i}' for i in range(200))]

    if not frag_cols:
        # Try to use all numeric columns except metadata
        meta_cols = {'sample_id', 'label', 'cancer_type', 'cohort', 'study',
                     'data_type', 'NOTE'}
        frag_cols = [c for c in ext_df.columns
                     if c not in meta_cols and pd.api.types.is_numeric_dtype(ext_df[c])]

    print(f"  External cohort feature columns found: {len(frag_cols)}")

    # Build aligned matrix
    X_ext = np.zeros((len(ext_df), len(internal_feature_cols)))
    matched = 0
    for i, col in enumerate(internal_feature_cols):
        if col in frag_cols:
            X_ext[:, i] = ext_df[col].fillna(0).values
            matched += 1

    print(f"  Matched {matched}/{len(internal_feature_cols)} internal feature columns")
    if matched < len(internal_feature_cols) * 0.5:
        print("  WARNING: <50% feature overlap — results may not be meaningful")

    return X_ext


def run_validation():
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                                  recall_score, precision_score,
                                  average_precision_score)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("CROSS-COHORT VALIDATION")
    print("=" * 60)

    # ── Load internal training data (for scaler) ──────────────────────────
    X_train_path = os.path.join(BASE, 'results', 'X_train.npy')
    y_train_path = os.path.join(BASE, 'results', 'y_train.npy')
    X_test_path  = os.path.join(BASE, 'results', 'X_test.npy')
    y_test_path  = os.path.join(BASE, 'results', 'y_test.npy')

    if not all(os.path.exists(p) for p in [X_train_path, y_train_path]):
        print("ERROR: results/X_train.npy not found — run 01_preprocess.py first")
        sys.exit(1)

    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_test  = np.load(X_test_path)
    y_test  = np.load(y_test_path)

    # Load feature matrix to get column names
    feat_matrix_path = os.path.join(BASE, 'data', 'feature_matrix.csv')
    feat_df = pd.read_csv(feat_matrix_path)
    label_col = [c for c in feat_df.columns if c in ('label', 'Label', 'target')][0]
    feature_cols = [c for c in feat_df.columns if c != label_col]
    print(f"Internal feature columns: {len(feature_cols)}")

    # Fit scaler on train set only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Load trained model ────────────────────────────────────────────────
    model_path = os.path.join(BASE, 'model', 'best_model.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNNLSTM(input_size=X_train.shape[1])
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Loaded model from {model_path}")

    # ── Internal test-set performance ────────────────────────────────────
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test_scaled)
        logits = model(X_t)
        probs_test = torch.softmax(logits, dim=1)[:, 1].numpy()
    preds_test = (probs_test >= 0.5).astype(int)

    int_auc   = roc_auc_score(y_test, probs_test)
    int_auprc = average_precision_score(y_test, probs_test)
    int_bal   = balanced_accuracy_score(y_test, preds_test)
    int_sens  = recall_score(y_test, preds_test, pos_label=1, zero_division=0)
    int_spec  = recall_score(y_test, preds_test, pos_label=0, zero_division=0)
    print(f"\nInternal test-set (GEO GSE149438 holdout, n={len(y_test)}):")
    print(f"  AUC-ROC={int_auc:.3f}  AUPRC={int_auprc:.3f}  "
          f"BalAcc={int_bal:.3f}  Sens={int_sens:.3f}  Spec={int_spec:.3f}")

    # ── Load external cohort features ─────────────────────────────────────
    ext_path = os.path.join(BASE, 'results', 'validation', 'jiang_hcc_features.csv')
    if not os.path.exists(ext_path):
        print(f"\nERROR: {ext_path} not found.")
        print("Run scripts/process_validation_cohorts.py first (after FASTQs complete).")
        return None

    ext_df = pd.read_csv(ext_path)
    print(f"\nExternal cohort (PRJNA291063 Jiang HCC, n={len(ext_df)}):")
    print(f"  Labels: {dict(ext_df['label'].value_counts())}")

    y_ext = ext_df['label'].values

    if len(np.unique(y_ext)) < 2:
        print("  ERROR: Only one class present in external cohort — AUC undefined")
        return None

    # ── Align and scale external features ────────────────────────────────
    X_ext = load_and_align_features(ext_df, feature_cols)
    X_ext_scaled = scaler.transform(X_ext)

    # ── Predict on external cohort ────────────────────────────────────────
    with torch.no_grad():
        X_e = torch.FloatTensor(X_ext_scaled)
        logits_ext = model(X_e)
        probs_ext = torch.softmax(logits_ext, dim=1)[:, 1].numpy()
    preds_ext = (probs_ext >= 0.5).astype(int)

    ext_auc  = roc_auc_score(y_ext, probs_ext)
    ext_bal  = balanced_accuracy_score(y_ext, preds_ext)
    ext_sens = recall_score(y_ext, preds_ext, pos_label=1, zero_division=0)
    ext_spec = recall_score(y_ext, preds_ext, pos_label=0, zero_division=0)

    print(f"\nExternal validation (PRJNA291063):")
    print(f"  AUC-ROC={ext_auc:.3f}  BalAcc={ext_bal:.3f}  "
          f"Sens={ext_sens:.3f}  Spec={ext_spec:.3f}")

    # Generalisation gap
    gap = int_auc - ext_auc
    print(f"\n  Generalisation gap (internal - external AUC): {gap:+.3f}")
    if gap > 0.1:
        verdict = "POOR generalisation (gap > 0.10)"
    elif gap > 0.05:
        verdict = "MODERATE generalisation (gap 0.05–0.10)"
    else:
        verdict = "GOOD generalisation (gap ≤ 0.05)"
    print(f"  Verdict: {verdict}")

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        'internal_GEO_GSE149438': {
            'n': int(len(y_test)),
            'n_cancer': int((y_test == 1).sum()),
            'n_healthy': int((y_test == 0).sum()),
            'auc': round(float(int_auc), 4),
            'auprc': round(float(int_auprc), 4),
            'balanced_accuracy': round(float(int_bal), 4),
            'sensitivity': round(float(int_sens), 4),
            'specificity': round(float(int_spec), 4),
        },
        'external_PRJNA291063_JiangHCC': {
            'n': int(len(y_ext)),
            'n_cancer': int((y_ext == 1).sum()),
            'n_healthy': int((y_ext == 0).sum()),
            'auc': round(float(ext_auc), 4),
            'balanced_accuracy': round(float(ext_bal), 4),
            'sensitivity': round(float(ext_sens), 4),
            'specificity': round(float(ext_spec), 4),
        },
        'generalisation_gap_auc': round(float(gap), 4),
        'generalisation_verdict': verdict,
        'note': ('External features extracted from Jiang 2015 HCC cfDNA BAM files '
                 'via Nextflow pipeline, aligned to GEO GSE149438 feature columns. '
                 'Feature overlap may be partial due to pipeline differences.'),
    }

    out_dir = os.path.join(BASE, 'results', 'validation')
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, 'cross_cohort_validation.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")

    # ── Generate comparison figure ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('Cross-Cohort Generalisation\nCNN+LSTM Model (trained on GEO GSE149438)',
                 fontsize=13, fontweight='bold')

    # Bar chart: AUC comparison
    ax = axes[0]
    cohorts = ['Internal\n(GEO GSE149438\nn=%d)' % len(y_test),
               'External\n(PRJNA291063 HCC\nn=%d)' % len(y_ext)]
    aucs = [int_auc, ext_auc]
    colors = ['#2196F3', '#FF5722']
    bars = ax.bar(cohorts, aucs, color=colors, width=0.5, edgecolor='black')
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Random (0.5)')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('AUC-ROC')
    ax.set_title('AUC-ROC by Cohort')
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, auc + 0.02,
                f'{auc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax.legend()

    # Bar chart: Sens/Spec breakdown
    ax2 = axes[1]
    metrics = ['Sensitivity\n(Cancer Recall)', 'Specificity\n(Healthy Recall)']
    int_vals = [int_sens, int_spec]
    ext_vals = [ext_sens, ext_spec]
    x = np.arange(len(metrics))
    width = 0.35
    ax2.bar(x - width/2, int_vals, width, label='Internal', color='#2196F3', edgecolor='black')
    ax2.bar(x + width/2, ext_vals, width, label='External', color='#FF5722', edgecolor='black')
    ax2.set_ylim(0, 1.15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_ylabel('Value')
    ax2.set_title('Sensitivity / Specificity')
    ax2.legend()
    for xi, (iv, ev) in zip(x, zip(int_vals, ext_vals)):
        ax2.text(xi - width/2, iv + 0.03, f'{iv:.2f}', ha='center', fontsize=9)
        ax2.text(xi + width/2, ev + 0.03, f'{ev:.2f}', ha='center', fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(BASE, 'report', 'figures', 'cross_cohort_validation.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {fig_path}")

    return results


if __name__ == '__main__':
    run_validation()

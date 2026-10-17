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
  - data/feature_matrix.csv  (raw GEO train split, to fit the scaler)
  - results/X_test.npy, results/y_test.npy  (from 01_preprocess.py)
"""

import os, sys, json
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), '..')

# ── Model architecture: same class used by 03_train_model.py ─────────────────
import torch
sys.path.insert(0, os.path.join(BASE, 'model'))
from model_architecture import LiquidBiopsyNet

META_COLS = ('gsm_id', 'tissue_type', 'cancer_label', 'sample_type')


def fit_internal_scaler(seed=42):
    """Rebuild the raw GEO train split exactly as 01_preprocess.py does
    (70/15/15, stratified, seed 42) and fit the scaler on raw TRAIN rows only.
    results/X_train.npy is already standardised, so re-fitting a scaler on it
    would be an identity transform and external features would stay unscaled."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv(os.path.join(BASE, 'data', 'feature_matrix.csv'))
    feature_cols = [c for c in df.columns if c not in META_COLS]
    # label semantics: cancer_label 1 == sample_type 'cancer' (verified in data)
    y = df['cancer_label'].values.astype(np.int64)
    assert ((df['sample_type'] == 'cancer').astype(int).values == y).all(), \
        'cancer_label and sample_type disagree'
    X = df[feature_cols].values.astype(np.float32)
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.30,
                                        random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    return scaler, feature_cols


def load_and_align_features(ext_df, internal_feature_cols):
    """
    Strict alignment: every internal feature column must be present, by exact
    name, with no missing values in the external cohort. Returns None otherwise.
    (The earlier version zero-filled absent columns and only warned below 50%
    overlap, which can produce an apparently valid but meaningless evaluation.
    Matching names also does not prove the features are defined the same way.
    Check feature definitions by hand, e.g. bin edges and methylation_density.)
    """
    missing = [c for c in internal_feature_cols if c not in ext_df.columns]
    if missing:
        print(f"  ERROR: external cohort lacks {len(missing)}/{len(internal_feature_cols)} "
              f"internal feature columns (first: {missing[:5]}). Not evaluating.")
        return None
    X_ext = ext_df[internal_feature_cols]
    n_nan = int(X_ext.isna().sum().sum())
    if n_nan:
        print(f"  ERROR: {n_nan} missing values in external features. Not evaluating.")
        return None
    return X_ext.values.astype(np.float32)


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

    # ── Internal scaler (raw GEO train rows only) and held-out test split ──
    scaler, feature_cols = fit_internal_scaler()
    print(f"Internal feature columns: {len(feature_cols)}")
    # Pre-standardised hold-out arrays written by 01_preprocess.py
    X_test_scaled = np.load(os.path.join(BASE, 'results', 'X_test.npy'))
    y_test        = np.load(os.path.join(BASE, 'results', 'y_test.npy'))

    # ── Load trained model ────────────────────────────────────────────────
    model_path = os.path.join(BASE, 'model', 'best_model.pt')
    device = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device)
    cfg = checkpoint.get('config', {})
    model = LiquidBiopsyNet(input_len=cfg.get('input_len', 150),
                            num_classes=cfg.get('num_classes', 2),
                            dropout=cfg.get('dropout', 0.5))
    model.load_state_dict(checkpoint['model_state'])
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
    # Must be 1 = cancer, 0 = healthy to match the GEO labels; refuse anything else.
    if not set(np.unique(y_ext)) <= {0, 1}:
        print(f"  ERROR: external 'label' values {np.unique(y_ext)} are not 0/1 (0=healthy, 1=cancer)")
        return None

    if len(np.unique(y_ext)) < 2:
        print("  ERROR: Only one class present in external cohort — AUC undefined")
        return None

    # ── Align and scale external features ────────────────────────────────
    X_ext = load_and_align_features(ext_df, feature_cols)
    if X_ext is None:
        return None
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

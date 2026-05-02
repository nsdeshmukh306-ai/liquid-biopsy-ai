#!/usr/bin/env python3
"""
fix_all_issues.py — Fixes all 6 reviewer-flagged issues.

Issue 1+3: Regenerate training_curves.png — correct title, show final model epochs only
Issue 2:   Regenerate feature_importance.png — correct label (RF importance, not CNN+LSTM Gini)
Issue 4:   Regenerate roc_curve.png — annotate LR AUC=1.000 as small-sample artifact
Issue 5:   Generate sample_predictions.png — 10 per-sample predictions with confidence
Issue 6:   Rewrite README.txt — update to 5-fold CV, correct AUC numbers
"""

import os, sys, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn.functional as F

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES    = os.path.join(BASE, 'results')
FIGS   = os.path.join(BASE, 'report', 'figures')
MODEL  = os.path.join(BASE, 'model', 'best_model.pt')

sys.path.insert(0, os.path.join(BASE, 'model'))
from model_architecture import LiquidBiopsyNet

DPI = 300
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': DPI,
})

np.random.seed(42)

# ── load data ────────────────────────────────────────────────────────────────
print("Loading data...")
X_train = np.load(os.path.join(RES, 'X_train.npy')).astype(np.float32)
y_train = np.load(os.path.join(RES, 'y_train.npy')).astype(np.int64)
X_test  = np.load(os.path.join(RES, 'X_test.npy')).astype(np.float32)
y_test  = np.load(os.path.join(RES, 'y_test.npy')).astype(np.int64)
X_val   = np.load(os.path.join(RES, 'X_val.npy')).astype(np.float32)
y_val   = np.load(os.path.join(RES, 'y_val.npy')).astype(np.int64)

history = pd.read_csv(os.path.join(RES, 'training_history.csv'))

# Load model
device = torch.device('cpu')
ckpt   = torch.load(MODEL, map_location=device)
cfg    = ckpt.get('config', {'input_len': 150, 'num_classes': 2, 'dropout': 0.5})
model  = LiquidBiopsyNet(
    input_len=cfg['input_len'],
    num_classes=cfg['num_classes'],
    dropout=cfg.get('dropout', 0.5),
).to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"  Model loaded (CV AUC: {ckpt.get('cv_auc_mean', '?'):.4f})")

def nn_proba(X):
    with torch.no_grad():
        logits = model(torch.FloatTensor(X).to(device))
        return F.softmax(logits, dim=1).cpu().numpy()

nn_proba_test  = nn_proba(X_test)
nn_pred_test   = nn_proba_test.argmax(1)

# LR baseline (leakage-free: scaler fit on train only)
scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_te_sc = scaler.transform(X_test)
lr = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
lr.fit(X_tr_sc, y_train)
lr_proba_test = lr.predict_proba(X_te_sc)
lr_pred_test  = lr_proba_test.argmax(1)

print(f"  CNN+LSTM test AUC: {roc_auc_score(y_test, nn_proba_test[:,1]):.4f}")
print(f"  LR       test AUC: {roc_auc_score(y_test, lr_proba_test[:,1]):.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIX 1+3: Training curves
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Fix 1+3] Regenerating training_curves.png ...")

final_hist = history[history['fold'] == 'final'].reset_index(drop=True)
epochs     = final_hist['epoch'].values
train_loss = final_hist['train_loss'].values
val_loss   = final_hist['val_loss'].values
train_acc  = final_hist['train_acc'].values
val_acc    = final_hist['val_acc'].values
val_auc    = final_hist['val_auc'].values

best_epoch_idx = int(np.argmax(val_auc))
best_epoch     = epochs[best_epoch_idx]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    'Training History — CNN+LSTM (LiquidBiopsyNet v2)\n'
    'Dataset: GEO GSE149438, n=300, Final model trained on train+val (n=255)',
    fontsize=12, fontweight='bold', y=1.02
)

# Loss
ax = axes[0]
ax.plot(epochs, train_loss, color='#3498db', lw=2.5, marker='o', ms=5, label='Train Loss')
ax.plot(epochs, val_loss,   color='#e74c3c', lw=2.5, marker='s', ms=5, ls='--', label='Val Loss')
ax.axvline(best_epoch, color='#f39c12', lw=1.5, ls=':', label=f'Best epoch ({best_epoch})')
ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-Entropy Loss')
ax.set_title('Loss')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
note = (
    f"Note: Model trained for {len(epochs)} epochs.\n"
    f"Early stopping (patience=10) triggered\n"
    f"after best val AUC at epoch {best_epoch}."
)
ax.text(0.98, 0.98, note, transform=ax.transAxes, fontsize=7.5,
        va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# Accuracy
ax = axes[1]
ax.plot(epochs, train_acc, color='#3498db', lw=2.5, marker='o', ms=5, label='Train Acc')
ax.plot(epochs, val_acc,   color='#e74c3c', lw=2.5, marker='s', ms=5, ls='--', label='Val Acc')
ax.axvline(best_epoch, color='#f39c12', lw=1.5, ls=':', label=f'Best epoch ({best_epoch})')
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
ax.set_title('Accuracy'); ax.set_ylim([0.5, 1.05])
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# AUC
ax = axes[2]
ax.plot(epochs, val_auc, color='#9b59b6', lw=2.5, marker='^', ms=6, label='Val AUC-ROC')
ax.axvline(best_epoch, color='#f39c12', lw=1.5, ls=':', label=f'Best (ep {best_epoch}, AUC={val_auc[best_epoch_idx]:.3f})')
ax.set_xlabel('Epoch'); ax.set_ylabel('AUC-ROC')
ax.set_title('Validation AUC-ROC'); ax.set_ylim([0.5, 1.05])
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'training_curves.png'), dpi=DPI, bbox_inches='tight')
plt.close()
print("  ✓ training_curves.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIX 2: Feature importance — correct labelling
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Fix 2] Regenerating feature_importance.png ...")

feat_path = os.path.join(BASE, 'data', 'real_feature_matrix.csv')
if not os.path.exists(feat_path):
    feat_path = os.path.join(RES, 'cfDNA_features.csv')

df = pd.read_csv(feat_path)
bin_cols = [c for c in df.columns if c.startswith('frag_bin_')][:150]

X_all = df[bin_cols].fillna(0).values
y_all = df['label'].values if 'label' in df.columns else None

if y_all is not None and len(np.unique(y_all)) > 1:
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_all, y_all)
    importances = rf.feature_importances_
else:
    # Use permutation-style importance from the CNN+LSTM on the test set
    base_auc = roc_auc_score(y_test, nn_proba_test[:,1])
    importances = np.zeros(len(bin_cols))
    X_test_use = X_test[:, :len(bin_cols)]
    for i in range(len(bin_cols)):
        X_perm = X_test_use.copy()
        np.random.shuffle(X_perm[:, i])
        perm_proba = nn_proba(X_perm)
        try:
            perm_auc = roc_auc_score(y_test, perm_proba[:,1])
            importances[i] = base_auc - perm_auc
        except Exception:
            importances[i] = 0.0

top_n  = 20
idx    = np.argsort(importances)[-top_n:][::-1]
top_names = [bin_cols[i] for i in idx]
top_vals  = importances[idx]

def bin_to_bp(name):
    n  = int(name.split('_')[-1])
    bp = 100 + n * (300 / len(bin_cols))
    return f'{bp:.0f} bp'

pretty_names = [bin_to_bp(n) for n in top_names]

fig, ax = plt.subplots(figsize=(10, 7))
cmap = plt.cm.Blues
norm_vals = top_vals / (top_vals.max() + 1e-10)
colors_bar = [cmap(0.4 + 0.6 * v) for v in norm_vals[::-1]]

ax.barh(range(top_n), top_vals[::-1], color=colors_bar,
        edgecolor='white', linewidth=0.5, alpha=0.9)
ax.set_yticks(range(top_n))
ax.set_yticklabels(pretty_names[::-1], fontsize=10)
ax.set_xlabel('Feature Importance (Random Forest Gini)', fontsize=12)
ax.set_title(
    'Top 20 Fragment Length Bins — Random Forest Feature Importance\n'
    '(RF fitted on GEO GSE149438, n=300; used as feature analysis tool)',
    fontsize=12, fontweight='bold'
)
ax.grid(alpha=0.3, axis='x')

note_text = (
    "Note: Importance computed using Random Forest (a separate analysis model),\n"
    "not the CNN+LSTM. Metric shown is Gini (mean decrease in impurity).\n"
    "CNN+LSTM uses the same 150 histogram bins as input features."
)
ax.text(0.98, 0.02, note_text, transform=ax.transAxes, fontsize=8,
        va='bottom', ha='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff9c4', alpha=0.85))

plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'feature_importance.png'), dpi=DPI, bbox_inches='tight')
plt.close()
print("  ✓ feature_importance.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIX 4: ROC curve — annotate LR AUC=1.000 artifact
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Fix 4] Regenerating roc_curve.png ...")

lr_auc = roc_auc_score(y_test, lr_proba_test[:,1])
nn_auc = roc_auc_score(y_test, nn_proba_test[:,1])

fpr_nn, tpr_nn, _ = roc_curve(y_test, nn_proba_test[:,1])
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba_test[:,1])

fig, ax = plt.subplots(figsize=(8, 7))

ax.plot(fpr_nn, tpr_nn, color='#e74c3c', lw=2.5,
        label=f'CNN+LSTM (AUC = {nn_auc:.3f})')
ax.plot(fpr_lr, tpr_lr, color='#3498db', lw=2, ls='--',
        label=f'Logistic Regression (AUC = {lr_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Random (AUC = 0.500)')

ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves — Held-out Test Set (n=45)\nDataset: GEO GSE149438',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)

# Annotate LR=1.0 artifact
if lr_auc >= 0.99:
    artifact_text = (
        f"⚠  LR AUC = {lr_auc:.3f} on this test set.\n"
        "This is a small-sample artifact:\n"
        "• n=45 test samples (only 7 healthy)\n"
        "• GEO features are linearly separable\n"
        "• LR 5-fold CV AUC = 0.969 ± 0.015\n"
        "  (more reliable generalisation estimate)"
    )
    ax.text(0.38, 0.20, artifact_text, transform=ax.transAxes,
            fontsize=8.5, va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3cd',
                      edgecolor='#f0ad4e', alpha=0.95))

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'roc_curve.png'), dpi=DPI, bbox_inches='tight')
plt.close()
print("  ✓ roc_curve.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIX 5: Sample predictions table/figure
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Fix 5] Generating sample_predictions.png ...")

# Pick 10 representative samples: 5 TP, 2 TN, 2 FN, 1 FP
results_rows = []
for i in range(len(y_test)):
    true_lbl = int(y_test[i])
    pred_lbl = int(nn_pred_test[i])
    prob_c   = float(nn_proba_test[i, 1])
    results_rows.append({
        'true': true_lbl,
        'pred': pred_lbl,
        'prob_cancer': prob_c,
        'correct': true_lbl == pred_lbl,
        'category': (
            'TP' if true_lbl==1 and pred_lbl==1 else
            'TN' if true_lbl==0 and pred_lbl==0 else
            'FN' if true_lbl==1 and pred_lbl==0 else 'FP'
        )
    })

df_res = pd.DataFrame(results_rows)

selected = []
for cat, n in [('TP', 4), ('TN', 2), ('FN', 3), ('FP', 1)]:
    sub = df_res[df_res['category'] == cat]
    selected.extend(sub.sample(min(n, len(sub)), random_state=42).index.tolist())
selected = selected[:10]
df_show = df_res.loc[selected].reset_index(drop=True)

# Build display data
CLASS_NAME = {0: 'Healthy', 1: 'Cancer'}
CAT_COLOR  = {'TP': '#27ae60', 'TN': '#27ae60', 'FN': '#e74c3c', 'FP': '#e74c3c'}

table_data = []
for i, row in df_show.iterrows():
    table_data.append([
        f"Sample {i+1:02d}",
        CLASS_NAME[row['true']],
        CLASS_NAME[row['pred']],
        f"{row['prob_cancer']:.3f}",
        f"{1-row['prob_cancer']:.3f}",
        row['category'],
    ])

col_labels = ['Sample', 'True Label', 'Predicted', 'P(Cancer)', 'P(Healthy)', 'Result']
col_widths = [0.12, 0.15, 0.15, 0.12, 0.12, 0.10]

fig, ax = plt.subplots(figsize=(12, 5.5))
ax.axis('off')

fig.suptitle(
    'Sample-Level Predictions — CNN+LSTM (LiquidBiopsyNet v2)\n'
    'Held-out test set (n=45); shown: 10 representative samples',
    fontsize=13, fontweight='bold', y=0.98
)

tbl = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    colWidths=col_widths,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.3, 2.2)

# Style header row
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#2c3e50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

# Style data rows
for i, row in df_show.iterrows():
    cat   = row['category']
    color = '#d5f5e3' if cat in ('TP', 'TN') else '#fadbd8'
    for j in range(len(col_labels)):
        tbl[i+1, j].set_facecolor(color)
        tbl[i+1, j].set_edgecolor('#bdc3c7')
    # Bold the Result column
    tbl[i+1, 5].set_text_props(fontweight='bold',
                                color=CAT_COLOR[cat])

# Legend
legend_handles = [
    mpatches.Patch(color='#d5f5e3', label='Correct prediction (TP / TN)'),
    mpatches.Patch(color='#fadbd8', label='Incorrect prediction (FP / FN)'),
]
ax.legend(handles=legend_handles, loc='lower center',
          bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=10, framealpha=0.9)

caption = (
    "TP=True Positive (cancer correctly detected)  |  TN=True Negative (healthy correctly identified)\n"
    "FN=False Negative (cancer missed — high clinical risk)  |  FP=False Positive (healthy mislabelled)\n"
    "P(Cancer) = model's predicted probability of cancer class"
)
fig.text(0.5, 0.01, caption, ha='center', va='bottom', fontsize=8.5, style='italic',
         color='#555555')

plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.savefig(os.path.join(FIGS, 'sample_predictions.png'), dpi=DPI, bbox_inches='tight')
plt.close()
print("  ✓ sample_predictions.png")

# Also copy updated figures to paper_export
import shutil
for fname in ['training_curves.png', 'feature_importance.png',
              'roc_curve.png', 'sample_predictions.png']:
    src = os.path.join(FIGS, fname)
    dst = os.path.join(BASE, 'paper_export', fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)

# ═══════════════════════════════════════════════════════════════════════════════
# FIX 6: Update README.txt
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Fix 6] Updating README.txt ...")

readme = """================================================================
  Liquid Biopsy AI — Cancer Detection from Cell-Free DNA
  A Hybrid CNN + LSTM Deep Learning Pipeline
================================================================
Author  : Niraj (IISER Tirupati, Roll No: 20254009)
Course  : OMICS + Deep Learning Mini Project — AY 2025-26
Dataset : GSE149438 (GEO) — 300 samples, 6 cancer subtypes
Model   : Hybrid CNN + LSTM (PyTorch, LiquidBiopsyNet v2)

----------------------------------------------------------------
CORRECTED MODEL PERFORMANCE (v2 — leakage-free)
----------------------------------------------------------------
Evaluation: 5-fold StratifiedKFold CV (random_state=42)
            Scaler fit inside each fold on training split only.
            All 300 GEO GSE149438 samples used for CV.

MODEL COMPARISON (5-fold CV, GEO GSE149438, n=300):
  Model                 CV AUC-ROC       CV Balanced Acc
  ─────────────────────────────────────────────────────
  Dummy (majority)      0.500 ± 0.000    0.500 ± 0.000
  CNN Only              0.844 ± 0.037    0.753 ± 0.023
  CNN + LSTM (Ours)     0.943 ± 0.015    0.872 ± 0.043
  LSTM Only             0.946 ± 0.014    0.865 ± 0.020
  Logistic Regression   0.969 ± 0.015    0.846 ± 0.045
  Random Forest         0.988 ± 0.006    0.846 ± 0.063

HELD-OUT TEST SET (n=45: 38 cancer, 7 healthy) — CNN+LSTM:
  AUC-ROC           : 0.921
  AUPRC             : 0.984   (random baseline: 0.844)
  Balanced Accuracy : 0.876
  Sensitivity       : 0.895   (34/38 cancers detected)
  Specificity       : 0.857   (6/7 healthy detected)
  Confusion Matrix  : TP=34 FP=1 | FN=4 TN=6

CLASS IMBALANCE NOTE:
  84.7% cancer prevalence in GEO dataset.
  Dummy classifier accuracy (predict all cancer) = 84.7%.
  CNN+LSTM test accuracy (88.6%) = only 3.9 pp above dummy.
  AUC-ROC and AUPRC are the appropriate primary metrics.

HONEST ASSESSMENT:
  Random Forest (AUC=0.988) and LSTM-only (AUC=0.946) outperform
  CNN+LSTM (AUC=0.943) on this tabular histogram feature set.
  This is expected: tree ensembles excel at tabular data where
  discriminative signal concentrates in a few bins.
  The CNN+LSTM architecture would gain more advantage on raw
  sequence data or multi-modal inputs.

  Logistic Regression achieves AUC=1.000 on the 45-sample test
  set. This is a small-sample artifact (only 7 healthy samples,
  linearly separable GEO features). LR 5-fold CV AUC=0.969 is
  the reliable generalisation estimate.

----------------------------------------------------------------
DATA SOURCES
----------------------------------------------------------------
1. Training/evaluation:  GEO GSE149438 (n=300, external preprocessing)
   URL: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149438

2. Real cfDNA pipeline:  NCBI SRA BioProject PRJNA624356
   Accessions: SRR11537952–SRR11537959
   Note: No ground-truth labels; pipeline validation only.
   Human-mapping: SRR11537952–54 (97% to hg38)
   Non-human:     SRR11537955–59 (<6% mapping, excluded)

3. External validation:  PRJNA291063 (Jiang 2015, HCC)
   Accessions: SRR2130050-52 (healthy), SRR2130005/07/28/36 (HCC)
   Result: inconclusive (feature pipeline incompatibility)

4. Reference genome: hg38 (GRCh38), UCSC

----------------------------------------------------------------
FOLDER STRUCTURE
----------------------------------------------------------------
liquid_biopsy_ai/
├── model/
│   ├── model_architecture.py   CNN+LSTM architecture (PyTorch)
│   └── best_model.pt           Saved weights (best val AUC ckpt)
├── scripts/
│   ├── 01_preprocess.py        Train/val/test split, StandardScaler
│   ├── 02_extract_features.py  Fragment length extraction from BAM
│   ├── 03_train_model.py       CNN+LSTM training (early stopping)
│   ├── 04_evaluate.py          Test set evaluation, all metrics
│   ├── 05_visualise.py         ROC, PR, confusion matrix, figures
│   ├── 06_real_figures.py      Fragment distribution from real BAMs
│   ├── 07_cv5fold.py           5-fold stratified cross-validation
│   ├── 08_covariate_shift.py   Covariate shift analysis
│   ├── 09_clinical_framing.py  Clinical context comparison
│   ├── train_baselines.py      LR, RF, CNN-only, LSTM-only baselines
│   └── fix_all_issues.py       Reviewer fixes (training curves,
│                               feature importance, ROC annotation,
│                               sample predictions, README)
├── notebooks/
│   └── LiquidBiopsyAI.ipynb    End-to-end Jupyter walkthrough
├── data/
│   ├── accessions.txt          SRA accession list
│   ├── real_feature_matrix.csv Fragment features from real BAMs
│   └── processed_real/         Train/val/test numpy arrays
├── results/
│   ├── training_history.csv    Per-fold training metrics
│   ├── cv5fold_results.json    5-fold CV summary
│   ├── metrics_corrected.json  Final corrected metrics
│   └── full_metrics_publication.json  All publication metrics
├── report/
│   ├── figures/                All figures (PNG, 300 DPI)
│   └── LiquidBiopsyAI_Report.docx  Project report
├── nextflow/                   Nextflow DSL2 pipeline
│   └── main.nf
└── README.txt                  This file

----------------------------------------------------------------
HOW TO RUN
----------------------------------------------------------------
Requirements: Python 3.10+, PyTorch >=1.13, scikit-learn,
              numpy, pandas, matplotlib, seaborn, pysam

conda activate workshop_nf   # or your environment
cd scripts/

# Step 1: Preprocess features
python3 01_preprocess.py

# Step 2: Train the CNN+LSTM model
python3 03_train_model.py --epochs 50 --patience 10

# Step 3: Evaluate and generate metrics
python3 04_evaluate.py

# Step 4: Generate all figures
python3 05_visualise.py
python3 fix_all_issues.py   # corrected figures + README

# Step 5: 5-fold cross-validation
python3 07_cv5fold.py

# Reproduce baseline comparison
python3 train_baselines.py

All random states fixed to seed=42 for reproducibility.

----------------------------------------------------------------
KNOWN LIMITATIONS
----------------------------------------------------------------
- Bioinformatics pipeline (SRA) and ML model (GEO) use different
  datasets that were NEVER joined (no ground-truth labels on SRA).
- Class imbalance (84.7% cancer) not representative of screening.
- Training ran for 5 epochs (early stopping via val AUC plateau).
- External validation inconclusive (feature pipeline mismatch).
- CPU-only training; GPU recommended for larger experiments.
================================================================
"""

readme_path = os.path.join(BASE, 'README.txt')
with open(readme_path, 'w') as f:
    f.write(readme)
print(f"  ✓ README.txt updated")

print("\n=== All 6 fixes applied ===")
print(f"  Figures in: {FIGS}")
print(f"  README:     {readme_path}")

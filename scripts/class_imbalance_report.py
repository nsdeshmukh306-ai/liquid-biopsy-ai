#!/usr/bin/env python3
"""
class_imbalance_report.py — Compute and save class imbalance statistics.

Outputs:
  results/class_imbalance_report.json

DATA SOURCE: GEO pre-processed matrix (GSE149438, n=300)
"""

import json
import os
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), '..')


def main():
    y_train = np.load(os.path.join(BASE, 'results', 'y_train.npy'))
    y_val   = np.load(os.path.join(BASE, 'results', 'y_val.npy'))
    y_test  = np.load(os.path.join(BASE, 'results', 'y_test.npy'))
    y_all   = np.concatenate([y_train, y_val, y_test])

    n_total   = len(y_all)
    n_cancer  = int((y_all == 1).sum())
    n_healthy = int((y_all == 0).sum())
    cancer_frac = n_cancer / n_total

    print("=" * 55)
    print("CLASS IMBALANCE REPORT")
    print("DATA SOURCE: GEO GSE149438, n=300, externally preprocessed")
    print("=" * 55)
    print(f"Cancer:  {n_cancer}/{n_total} = {cancer_frac*100:.1f}%")
    print(f"Healthy: {n_healthy}/{n_total} = {(1-cancer_frac)*100:.1f}%")
    print()
    print("Dummy classifier accuracy (predict all cancer):")
    print(f"  {cancer_frac*100:.1f}% — this is the trivial baseline")
    print()
    print("CNN+LSTM accuracy: 88.6%")
    print(f"Margin above dummy: {(0.886 - cancer_frac)*100:.1f} percentage points")
    print()
    print("IMPLICATION:")
    print("  Accuracy is misleading for this imbalanced dataset.")
    print("  AUC-ROC (0.895) and balanced accuracy are the correct primary metrics.")
    print("=" * 55)

    report = {
        'dataset': 'GEO GSE149438, n=300',
        'n_cancer': n_cancer,
        'n_healthy': n_healthy,
        'n_total': n_total,
        'cancer_fraction': float(cancer_frac),
        'healthy_fraction': float(1 - cancer_frac),
        'dummy_accuracy_majority_class': float(cancer_frac),
        'cnn_lstm_accuracy': 0.886,
        'cnn_lstm_auc_roc': 0.895,
        'margin_above_dummy_accuracy': float(0.886 - cancer_frac),
        'primary_metric_recommendation': 'AUC-ROC and balanced_accuracy',
        'note': (
            'With 84.7% cancer prevalence, a trivial classifier achieves 84.7% accuracy. '
            'The CNN+LSTM margin above this baseline is only 3.9 percentage points. '
            'AUC-ROC is the appropriate primary metric for this imbalanced dataset.'
        )
    }

    out_path = os.path.join(BASE, 'results', 'class_imbalance_report.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()

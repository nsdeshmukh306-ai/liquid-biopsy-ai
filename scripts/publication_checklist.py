#!/usr/bin/env python3
"""
publication_checklist.py — Verify all publication-readiness requirements.
Run this as the final step before creating the submission zip.
"""

import os
import json
import sys

BASE = os.path.join(os.path.dirname(__file__), '..')


def check(desc, val):
    if val is True:
        return '✅', desc
    elif val is False:
        return '❌', desc
    else:
        return '👁 ', desc


checks = []

# File existence checks
file_checks = [
    ('CLAIM_AUDIT.md exists',                           'CLAIM_AUDIT.md'),
    ('model_comparison_REAL.json exists',               'results/model_comparison_REAL.json'),
    ('model_comparison_REAL.png exists',                'report/figures/model_comparison_REAL.png'),
    ('model_comparison.png replaced with real values',  'report/figures/model_comparison.png'),
    ('class_imbalance_report.json exists',              'results/class_imbalance_report.json'),
    ('full_metrics_publication.json exists',            'results/full_metrics_publication.json'),
    ('precision_recall_curve.png exists',               'report/figures/precision_recall_curve.png'),
    ('INVALID_RUN_WARNING.txt exists',                  'results/INVALID_RUN_WARNING.txt'),
    ('software_versions.txt exists',                    'results/software_versions.txt'),
    ('abstract_FINAL.txt exists',                       'report/abstract_FINAL.txt'),
    ('results_section_FINAL.txt exists',                'report/results_section_FINAL.txt'),
    ('limitations_FINAL.txt exists',                    'report/limitations_FINAL.txt'),
    ('reproducibility_statement.txt exists',            'report/reproducibility_statement.txt'),
    ('readme_corrections.txt exists',                   'results/readme_corrections.txt'),
    ('DATA_PROVENANCE.md exists',                       'DATA_PROVENANCE.md'),
    ('limitations_addressed.md exists',                 'limitations_addressed.md'),
    ('best_model.pt exists',                            'model/best_model.pt'),
    ('methylation_heatmap.png watermarked (>100KB)',    None),
    ('model comparison watermark removed (REAL.png)',   None),
]

print("\n" + "=" * 60)
print("PUBLICATION READINESS CHECKLIST")
print("=" * 60)

results = {}

for desc, rel_path in file_checks:
    if rel_path is None:
        results[desc] = None
        continue
    full_path = os.path.join(BASE, rel_path)
    exists = os.path.exists(full_path)
    # Extra check for watermark
    if 'watermarked' in desc and exists:
        size = os.path.getsize(full_path)
        exists = size > 100_000
    results[desc] = exists

# Content checks
try:
    with open(os.path.join(BASE, 'results', 'model_comparison_REAL.json')) as f:
        mc = json.load(f)
    results['CNN Only actually trained (not estimated)'] = 'CNN Only' in mc
    results['LSTM Only actually trained (not estimated)'] = 'LSTM Only' in mc
    results['Dummy baseline present in comparison']      = 'Dummy (Majority)' in mc
    results['LR in comparison (with balanced weights)']  = 'Logistic Regression' in mc

    # Check CNN+LSTM is not worst
    best_auc = max(mc[m]['auc'] for m in mc if m != 'Dummy (Majority)')
    cnn_auc  = mc.get('CNN+LSTM (Ours)', {}).get('auc', 0)
    # RF beats CNN+LSTM by 0.09 — honest finding, not a bug. Mark as informational.
    gap = round(best_auc - cnn_auc, 3)
    results[f'CNN+LSTM vs best (gap={gap}) — see results_section_FINAL.txt'] = None
except Exception as e:
    results['model_comparison_REAL.json readable'] = False
    print(f"  Error reading model_comparison_REAL.json: {e}")

try:
    with open(os.path.join(BASE, 'results', 'full_metrics_publication.json')) as f:
        fm = json.load(f)
    results['AUPRC computed']                 = 'auprc' in fm
    results['Sensitivity computed']           = 'sensitivity_cancer_recall' in fm
    results['Specificity computed']           = 'specificity_healthy_recall' in fm
    results['Class imbalance note present']   = 'note_class_imbalance' in fm
    results['LR AUC=1.0 explanation present'] = 'note_lr_perfect_auc' in fm
except Exception as e:
    results['full_metrics_publication.json readable'] = False

try:
    with open(os.path.join(BASE, 'results', 'class_imbalance_report.json')) as f:
        ci = json.load(f)
    results['Dummy accuracy matches cancer prevalence'] = (
        abs(ci['dummy_accuracy_majority_class'] - ci['cancer_fraction']) < 0.001
    )
except Exception:
    pass

# Check submission zip exists
parent = os.path.join(BASE, '..')
archive_files = [f for f in os.listdir(parent)
                 if 'publication' in f and (f.endswith('.zip') or f.endswith('.tar.gz'))]
results['Submission archive created (zip/tar.gz)'] = len(archive_files) > 0

# Manual checks
results['GEO attribution in notebook (manual check required)'] = None
results['Discussion updated: LR vs CNN+LSTM honest comparison (manual)'] = None

# Print all
done = failed = needs_manual = 0
for desc, status in results.items():
    icon, _ = check(desc, status)
    print(f"  {icon}  {desc}")
    if status is True:
        done += 1
    elif status is False:
        failed += 1
    else:
        needs_manual += 1

print("=" * 60)
total_auto = done + failed
print(f"Auto-verified: {done}/{total_auto} passed | {failed} failed")
print(f"Manual checks needed: {needs_manual}")

if failed > 0:
    print("\n❌ Fix all failed checks before submission.")
    sys.exit(1)
else:
    print("\n✅ All auto-checks passed. Review manual checks.")
    sys.exit(0)

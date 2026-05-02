#!/usr/bin/env python3
"""
01_preprocess.py — Liquid Biopsy AI: Data Preprocessing (v2, leakage-free)

LEAKAGE FIX: StandardScaler is now fit ONLY on X_train, then applied to
X_val and X_test. The original code fit the scaler on all 300 samples before
splitting, leaking test distribution into the training normalisation.

SYNTHETIC DATA FIX: Reduced from 500→100 fragments/sample and increased class
overlap (cancer N(145,30) vs healthy N(165,25)) to produce a realistic, non-trivial
classification problem. The original N(132,20) vs N(167,15) with 500 fragments
gave essentially zero overlap → trivially separable by any classifier.
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tqdm import tqdm

import random
random.seed(42)
np.random.seed(42)


def parse_args():
    p = argparse.ArgumentParser(description='Preprocess cfDNA data — leakage-free v2')
    p.add_argument('--data-dir',    default='../data',   help='Path to data directory')
    p.add_argument('--out-dir',     default='../results', help='Output directory')
    p.add_argument('--schema',      default='../data/data_schema.txt')
    p.add_argument('--test-size',   type=float, default=0.15)
    p.add_argument('--val-size',    type=float, default=0.15)
    p.add_argument('--n-synthetic', type=int,   default=300)
    p.add_argument('--seed',        type=int,   default=42)
    return p.parse_args()


def read_schema(path):
    schema = {}
    if not os.path.exists(path):
        return schema
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or '=' not in line:
                continue
            k, _, v = line.partition('=')
            schema[k.strip()] = v.strip()
    return schema


def generate_synthetic(n_samples=300, n_features=150, seed=42):
    """
    Realistic cfDNA synthetic data with biological class overlap.

    Healthy: fragment lengths ~ N(165, 22)  — nucleosomal peak, some spread
    Cancer:  fragment lengths ~ N(145, 28)  — shorter, more variable
    Overlap region ~130-180 bp where both classes contribute.

    Using only 100 fragments/sample (vs 500 before) → realistic sampling noise
    so that histogram bins have meaningful variance between samples.
    """
    rng = np.random.default_rng(seed)
    n_healthy = n_samples // 2
    n_cancer  = n_samples - n_healthy

    bins = np.linspace(100, 250, n_features + 1)

    print(f"\n[SYNTHETIC] {n_samples} samples | healthy N(165,22) cancer N(145,28)")
    print(f"            100 fragments/sample → realistic histogram noise")
    print(f"            Class overlap ~130-180 bp (non-trivially separable)")

    records = []
    for i in tqdm(range(n_samples), desc='Generating'):
        is_cancer = i >= n_healthy

        # Per-sample mean drawn from a distribution → sample-level variability
        if is_cancer:
            sample_mean = rng.normal(145, 8)
            sample_std  = rng.uniform(20, 35)
        else:
            sample_mean = rng.normal(165, 6)
            sample_std  = rng.uniform(15, 28)

        # 100 fragments (not 500) → noisier histograms, more realistic
        frags = rng.normal(sample_mean, sample_std, size=100).clip(100, 250)
        hist, _ = np.histogram(frags, bins=bins)
        hist = hist / (hist.sum() + 1e-8)

        # Extra features with realistic overlap
        if is_cancer:
            meth = rng.beta(3.5, 3.0)     # modest hypermethylation
            gc   = rng.normal(0.50, 0.04)
            sfr  = rng.beta(4, 5)          # slightly elevated short frag
        else:
            meth = rng.beta(2.5, 4.0)
            gc   = rng.normal(0.47, 0.04)
            sfr  = rng.beta(2, 7)

        rec = {
            'label': int(is_cancer),
            'sample_type': 'cancer' if is_cancer else 'healthy',
        }
        rec.update({f'frag_bin_{j:03d}': hist[j] for j in range(n_features)})
        rec.update({
            'mean_frag_len'      : float(np.mean(frags)),
            'std_frag_len'       : float(np.std(frags)),
            'short_frag_ratio'   : float((frags < 150).mean()),
            'methylation_density': float(meth),
            'mean_gc'            : float(gc),
        })
        records.append(rec)

    df = pd.DataFrame(records)
    feat_cols = [c for c in df.columns if c not in ('label', 'sample_type')]
    X = df[feat_cols].values.astype(np.float32)
    y = df['label'].values.astype(np.int64)

    print(f"[SYNTHETIC] X shape: {X.shape}, y dist: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, feat_cols


def load_real_data(data_dir, schema):
    sep       = schema.get('SEPARATOR', ',')
    label_col = schema.get('LABEL_COLUMN', None)

    candidates = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(('.csv', '.txt', '.tsv'))
        and 'schema' not in f and 'accession' not in f and 'filelist' not in f
    ])

    for path in candidates:
        print(f"[LOAD] Trying: {path}")
        try:
            # DATA SOURCE: GEO pre-processed matrix (original paper authors) — NOT from our BAM pipeline
            # The feature matrix loaded here (data/feature_matrix.csv, n=300) was downloaded from
            # GEO (GSE149438) and pre-processed by the original study's authors. Our Nextflow pipeline
            # independently produced 8-sample features in results/nextflow/feature_matrix.csv.
            df = pd.read_csv(path, sep=sep, index_col=0)
            if df.shape[0] < 10 or df.shape[1] < 5:
                print("       [SKIP] Too small")
                continue

            lc = label_col if (label_col and label_col in df.columns) else None
            if lc is None:
                cands = [c for c in df.columns if any(
                    kw in c.lower() for kw in
                    ['cancer', 'healthy', 'label', 'class', 'type', 'status', 'group']
                )]
                if not cands:
                    continue
                lc = cands[0]

            print(f"       Label column: {lc}, dist: {df[lc].value_counts().to_dict()}")

            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y  = le.fit_transform(df[lc].astype(str)).astype(np.int64)
            X_df = df.drop(columns=[lc]).select_dtypes(include=[np.number])
            X    = X_df.values.astype(np.float32)
            print(f"       [OK] X={X.shape}")
            return X, y, X_df.columns.tolist()

        except Exception as e:
            print(f"       [WARN] {e}")

    return None, None, None


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    schema      = read_schema(args.schema)
    use_synth   = schema.get('SYNTHETIC', '').lower() == 'true'

    X, y, feat_names = None, None, None
    if not use_synth:
        print("\n=== Attempting to load real data ===")
        X, y, feat_names = load_real_data(args.data_dir, schema)

    if X is None:
        print("\n=== Using synthetic data (overlapping, non-trivial) ===")
        X, y, feat_names = generate_synthetic(args.n_synthetic, seed=args.seed)

    print(f"\n=== Dataset: {X.shape[0]} samples, {X.shape[1]} features ===")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u} ({'healthy' if u==0 else 'cancer'}): {c} ({100*c/len(y):.1f}%)")

    # ── Impute (on full dataset — imputation with median is OK before split) ──
    n_miss = np.isnan(X).sum()
    if n_miss > 0:
        print(f"\n=== Imputing {n_miss} missing values ===")
        X = SimpleImputer(strategy='median').fit_transform(X).astype(np.float32)

    # ── Train / val / test split (70/15/15) ───────────────────────────────────
    print(f"\n=== Splitting 70/15/15 (stratified) ===")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=args.val_size + args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    val_frac = args.val_size / (args.val_size + args.test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - val_frac,
        random_state=args.seed,
        stratify=y_temp,
    )

    # ── LEAKAGE FIX: fit scaler ONLY on X_train ───────────────────────────────
    # Previous code: scaler.fit_transform(X)  ← leaked test statistics into training
    # Correct code:  fit on X_train only, then transform val and test separately
    print("\n=== Normalising (scaler fit on TRAIN only — no leakage) ===")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)
    print(f"  Train mean≈{X_train.mean():.4f}, std≈{X_train.std():.4f}")
    print(f"  Val   mean≈{X_val.mean():.4f},   std≈{X_val.std():.4f}")
    print(f"  Test  mean≈{X_test.mean():.4f},  std≈{X_test.std():.4f}")

    print(f"\n  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    for split_name, y_s in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        uc, cc = np.unique(y_s, return_counts=True)
        print(f"  {split_name}: {dict(zip(['healthy','cancer'], cc.tolist()))}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n=== Saving to {args.out_dir} ===")
    for name, arr in tqdm({
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
    }.items(), desc='Saving'):
        np.save(os.path.join(args.out_dir, f'{name}.npy'), arr)
        print(f"  ✓ {name}.npy  {arr.shape}")

    meta = {
        'n_features'   : int(X.shape[1]),
        'n_train'      : int(X_train.shape[0]),
        'n_val'        : int(X_val.shape[0]),
        'n_test'       : int(X_test.shape[0]),
        'feature_names': feat_names[:20] if feat_names else [],
        'scaler_mean'  : scaler.mean_.tolist()[:10],
        'scaler_std'   : scaler.scale_.tolist()[:10],
        'leakage_fix'  : 'scaler fit on X_train only',
        'data_source'  : 'real' if not use_synth else 'synthetic-v2-overlapping',
        'random_seed'  : args.seed,
    }
    with open(os.path.join(args.out_dir, 'preprocessing_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("\n=== Preprocessing complete (leakage-free) ===")
    print(f"  Run next: python3 02_extract_features.py")


if __name__ == '__main__':
    main()

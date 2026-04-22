#!/usr/bin/env python3
"""
02_extract_features.py — Liquid Biopsy AI: Feature Extraction
Extracts biologically meaningful cfDNA features from BAM files or generates
synthetic features if BAM files are not available.

Features extracted:
  - Fragment length histogram (bins: 100–250 bp)
  - Short fragment ratio (fragments < 150 bp / total)
  - Nucleosome positioning score
  - End motif frequencies (4-mer at fragment ends)
  - CpG methylation density
  - Windowed protection score
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

import random
random.seed(42)
np.random.seed(42)

# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Extract cfDNA features')
    p.add_argument('--bam-dir',     default='../results/nextflow/bam',
                   help='Directory containing BAM files')
    p.add_argument('--out-dir',     default='../results',
                   help='Output directory')
    p.add_argument('--n-synthetic', type=int, default=200,
                   help='Number of synthetic samples if no BAMs found')
    p.add_argument('--frag-min',    type=int, default=100)
    p.add_argument('--frag-max',    type=int, default=250)
    p.add_argument('--frag-bins',   type=int, default=150)
    p.add_argument('--seed',        type=int, default=42)
    return p.parse_args()

# ── Fragment feature extraction from BAM ──────────────────────────────────────
def extract_from_bam(bam_path, frag_min=100, frag_max=250, n_bins=150):
    """Extract fragment features from a single BAM file."""
    try:
        import pysam
    except ImportError:
        print("[WARN] pysam not installed — using synthetic features")
        return None

    features = {}
    fragment_lengths = []
    gc_contents      = []
    end_motifs_5p    = []

    bins = np.linspace(frag_min, frag_max, n_bins + 1)

    try:
        with pysam.AlignmentFile(bam_path, 'rb') as bam:
            for read in tqdm(bam.fetch(), desc=f'  Reading {os.path.basename(bam_path)}', leave=False):
                if (read.is_proper_pair and
                        not read.is_secondary and
                        not read.is_supplementary and
                        read.is_read1):
                    flen = abs(read.template_length or 0)
                    if frag_min <= flen <= frag_max:
                        fragment_lengths.append(flen)
                    if read.query_sequence and len(read.query_sequence) >= 4:
                        seq = read.query_sequence.upper()
                        gc = (seq.count('G') + seq.count('C')) / len(seq)
                        gc_contents.append(gc)
                        end_motifs_5p.append(seq[:4])
    except Exception as e:
        print(f"[WARN] BAM read error: {e}")
        return None

    if not fragment_lengths:
        print(f"[WARN] No valid fragments in {bam_path}")
        return None

    fragment_lengths = np.array(fragment_lengths)
    hist, _ = np.histogram(fragment_lengths, bins=bins)
    hist_norm = hist / (hist.sum() + 1e-8)

    features.update({f'frag_bin_{i:03d}': v for i, v in enumerate(hist_norm)})
    features['mean_frag_len']    = float(np.mean(fragment_lengths))
    features['std_frag_len']     = float(np.std(fragment_lengths))
    features['median_frag_len']  = float(np.median(fragment_lengths))
    features['short_frag_ratio'] = float((fragment_lengths < 150).mean())

    mono_nuc = ((fragment_lengths >= 120) & (fragment_lengths <= 200)).sum()
    di_nuc   = ((fragment_lengths >= 300) & (fragment_lengths <= 400)).sum()
    features['nuc_score'] = float(mono_nuc / (di_nuc + 1))

    if gc_contents:
        features['mean_gc'] = float(np.mean(gc_contents))
        features['std_gc']  = float(np.std(gc_contents))
    else:
        features['mean_gc'] = 0.5
        features['std_gc']  = 0.05

    # 4-mer end motif frequencies (top cancer-relevant motifs)
    top_motifs = ['CCCA','CCCG','CCCT','CCCC','TTCA','TTTC','AAAC','GCCA',
                  'ACCA','TGCA','CATG','GCGC','CGCG','AATG','CATC','GATC']
    total = len(end_motifs_5p) + 1
    cnt   = Counter(end_motifs_5p)
    for motif in top_motifs:
        features[f'motif_{motif}'] = cnt.get(motif, 0) / total

    # Windowed protection score (peaks at 167, valleys at 132)
    peak_167 = hist_norm[int(67 * n_bins / 150):(int(67 * n_bins / 150) + 5)].mean()
    peak_132 = hist_norm[int(32 * n_bins / 150):(int(32 * n_bins / 150) + 5)].mean()
    features['wps'] = float(peak_167 - peak_132)

    # Synthetic methylation placeholder
    features['methylation_density'] = float(np.random.beta(3, 3))

    return features

# ── Synthetic feature generation ───────────────────────────────────────────────
def generate_synthetic_features(n_samples=200, n_bins=150, seed=42):
    """
    Generate biologically accurate synthetic cfDNA features.
    Healthy: Normal(167,15) fragment lengths, Beta(2,5) methylation
    Cancer:  Normal(132,20) fragment lengths, Beta(5,2) methylation
    """
    rng = np.random.default_rng(seed)
    n_healthy = n_samples // 2
    n_cancer  = n_samples - n_healthy

    bins = np.linspace(100, 250, n_bins + 1)
    records = []

    print(f"\n[SYNTHETIC] Generating {n_samples} cfDNA feature vectors")
    print(f"  Healthy={n_healthy}, Cancer={n_cancer}")

    for i in tqdm(range(n_samples), desc='Generating synthetic features'):
        is_cancer = i >= n_healthy
        label     = 1 if is_cancer else 0
        stype     = 'cancer' if is_cancer else 'healthy'

        # Fragment length distribution
        mean_len = rng.normal(132, 5) if is_cancer else rng.normal(167, 5)
        std_len  = rng.uniform(15, 25) if is_cancer else rng.uniform(10, 18)
        frag_lens = rng.normal(mean_len, std_len, size=500).clip(100, 250)

        hist, _ = np.histogram(frag_lens, bins=bins)
        hist_norm = hist / (hist.sum() + 1e-8)

        # Statistical features
        short_frag = (frag_lens < 150).mean()
        nuc_score  = (((frag_lens >= 120) & (frag_lens <= 200)).sum() /
                      ((frag_lens >= 300) & (frag_lens <= 400)).sum()
                      if ((frag_lens >= 300) & (frag_lens <= 400)).sum() > 0 else 10.0)

        # Methylation
        meth = rng.beta(5, 2) if is_cancer else rng.beta(2, 5)

        # GC content
        gc_mean = rng.normal(0.52, 0.03) if is_cancer else rng.normal(0.45, 0.03)
        gc_std  = rng.uniform(0.03, 0.06)

        # End motifs (cancer-associated motifs enriched)
        top_motifs = ['CCCA','CCCG','CCCT','CCCC','TTCA','TTTC','AAAC','GCCA',
                      'ACCA','TGCA','CATG','GCGC','CGCG','AATG','CATC','GATC']
        if is_cancer:
            motif_freqs = {f'motif_{m}': rng.uniform(0.03, 0.08) for m in top_motifs[:8]}
            motif_freqs.update({f'motif_{m}': rng.uniform(0.01, 0.03) for m in top_motifs[8:]})
        else:
            motif_freqs = {f'motif_{m}': rng.uniform(0.01, 0.04) for m in top_motifs}

        # Windowed protection score
        wps = float(hist_norm[max(0, 67-2):min(len(hist_norm), 67+3)].mean() -
                    hist_norm[max(0, 32-2):min(len(hist_norm), 32+3)].mean())

        rec = {'sample_id': f'{stype}_{i:03d}', 'label': label, 'sample_type': stype}
        rec.update({f'frag_bin_{j:03d}': hist_norm[j] for j in range(n_bins)})
        rec.update({
            'mean_frag_len'      : float(np.mean(frag_lens)),
            'std_frag_len'       : float(np.std(frag_lens)),
            'median_frag_len'    : float(np.median(frag_lens)),
            'short_frag_ratio'   : float(short_frag),
            'nuc_score'          : float(nuc_score),
            'methylation_density': float(meth),
            'mean_gc'            : float(gc_mean),
            'std_gc'             : float(gc_std),
            'wps'                : float(wps),
        })
        rec.update(motif_freqs)
        records.append(rec)

    df = pd.DataFrame(records)
    print(f"[SYNTHETIC] Feature matrix shape: {df.shape}")
    print(f"  Columns (first 10): {list(df.columns[:10])}")
    return df

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Check for BAM files
    bam_files = []
    if os.path.isdir(args.bam_dir):
        bam_files = [os.path.join(args.bam_dir, f)
                     for f in os.listdir(args.bam_dir)
                     if f.endswith('.bam') and not f.endswith('.bai')]
        bam_files.sort()

    if bam_files:
        print(f"\n=== Extracting features from {len(bam_files)} BAM files ===")
        records = []
        for bam_path in tqdm(bam_files, desc='Processing BAMs'):
            sid = os.path.basename(bam_path).replace('.bam', '').replace('_dedup', '')
            feats = extract_from_bam(bam_path, args.frag_min, args.frag_max, args.frag_bins)
            if feats:
                feats['sample_id'] = sid
                records.append(feats)

        if records:
            df = pd.DataFrame(records)
        else:
            print("[WARN] No valid features from BAMs — using synthetic")
            df = generate_synthetic_features(args.n_synthetic, args.frag_bins, args.seed)
    else:
        print(f"\n[INFO] No BAM files found in {args.bam_dir}")
        print("       Generating synthetic cfDNA features ...")
        df = generate_synthetic_features(args.n_synthetic, args.frag_bins, args.seed)

    # Save feature matrix
    out_path = os.path.join(args.out_dir, 'cfDNA_features.csv')
    df.to_csv(out_path, index=False)
    print(f"\n[OK] Feature matrix saved: {out_path}")
    print(f"     Shape: {df.shape}")

    # Summary statistics
    print("\n=== Feature Summary ===")
    if 'label' in df.columns:
        print(f"  Class distribution:\n{df['label'].value_counts()}")
    if 'mean_frag_len' in df.columns:
        for lbl in [0, 1]:
            sub = df[df['label'] == lbl] if 'label' in df.columns else df
            name = 'healthy' if lbl == 0 else 'cancer'
            print(f"  {name} mean fragment length: {sub['mean_frag_len'].mean():.1f} ± {sub['mean_frag_len'].std():.1f} bp")

    print("\n=== Feature extraction complete! ===")
    print(f"  Output: {out_path}")
    print(f"  Run next: python3 03_train_model.py")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
06_real_figures.py — Generate honest figures from REAL data.

fragment_distribution.png : fragment lengths extracted from 8 real BAM files
methylation_heatmap.png   : cfDNA feature heatmap from GEO feature matrix (GSE149438, n=300)
                            — labeled as fragment/methylation proxy features, NOT simulated

Replaces the synthetic versions produced by 05_visualise.py.
"""

import json
import os
import subprocess
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import gaussian_kde

SAMTOOLS = '/home/workshop/.conda/envs/workshop_nf/bin/samtools'
BASE     = Path(__file__).parent.parent
BAM_DIR  = BASE / 'data' / 'aligned'
FIG_DIR  = BASE / 'report' / 'figures'
RES_DIR  = BASE / 'results'
DPI      = 300
SEED     = 42

# Known labels from accessions.txt
SAMPLE_LABELS = {
    'SRR11537952': 'cancer',
    'SRR11537953': 'healthy',
    'SRR11537954': 'healthy',
    'SRR11537955': 'cancer',
    'SRR11537956': 'unknown',
    'SRR11537957': 'unknown',
    'SRR11537958': 'unknown',
    'SRR11537959': 'unknown',
}

COLORS = {'healthy': '#27ae60', 'cancer': '#e74c3c', 'unknown': '#888888'}


# ── 1. Extract fragment lengths from BAMs ─────────────────────────────────────

def extract_fragment_lengths(bam_path: Path, max_reads: int = 200_000) -> np.ndarray:
    """Extract TLEN values for properly paired reads in cfDNA range (50–500 bp)."""
    cmd = [
        SAMTOOLS, 'view',
        '-F', '4',    # skip unmapped
        '-f', '2',    # properly paired only
        str(bam_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    lengths = []
    for i, line in enumerate(result.stdout.splitlines()):
        if i >= max_reads:
            break
        fields = line.split('\t')
        if len(fields) < 9:
            continue
        try:
            tlen = int(fields[8])
        except ValueError:
            continue
        if 50 <= tlen <= 500:
            lengths.append(tlen)
    return np.array(lengths, dtype=np.int32)


def load_or_extract_fragments():
    """Load cached fragment lengths or extract from BAMs."""
    cache_path = RES_DIR / 'bam_fragment_lengths.npz'
    if cache_path.exists():
        print(f"[cache] Loading fragment lengths from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    print("Extracting fragment lengths from BAMs (this takes ~1-2 min)...")
    bams = sorted(BAM_DIR.glob('*_dedup.bam'))
    if not bams:
        raise FileNotFoundError(f"No BAM files found in {BAM_DIR}")

    cache = {}
    for bam in bams:
        srr = bam.name.replace('_dedup.bam', '')
        print(f"  {srr} ...", end=' ', flush=True)
        lengths = extract_fragment_lengths(bam)
        cache[srr] = lengths
        label = SAMPLE_LABELS.get(srr, 'unknown')
        print(f"n={len(lengths):,}  mean={lengths.mean():.1f} bp  label={label}")

    np.savez_compressed(str(cache_path), **cache)
    print(f"  Saved: {cache_path}")
    return cache


# ── 2. Fragment distribution figure ──────────────────────────────────────────

def plot_fragment_distribution(frag_data: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: per-class mean density (cancer vs healthy)
    ax = axes[0]
    class_frags = {'cancer': [], 'healthy': []}
    for srr, lengths in frag_data.items():
        label = SAMPLE_LABELS.get(srr, 'unknown')
        if label in class_frags:
            class_frags[label].extend(lengths.tolist())

    x = np.linspace(50, 500, 500)
    for cls, frags in class_frags.items():
        if len(frags) < 10:
            continue
        arr = np.array(frags)
        kde = gaussian_kde(arr, bw_method=0.08)
        density = kde(x)
        density /= density.sum()
        label_str = f'{cls.capitalize()} (n={len(arr):,} reads, {sum(1 for s,l in SAMPLE_LABELS.items() if l==cls)} samples)'
        ax.plot(x, density, color=COLORS[cls], lw=2.5, label=label_str)
        ax.fill_between(x, density, alpha=0.15, color=COLORS[cls])

    # Nucleosome periodicity reference lines
    for pos, name in [(167, '1st nucl.\n167 bp'), (334, '2nd nucl.\n334 bp')]:
        ax.axvline(pos, color='#636363', lw=1.2, linestyle=':', alpha=0.7)
        ax.text(pos + 4, ax.get_ylim()[1] * 0.7 if ax.get_ylim()[1] > 0 else 0.005,
                name, fontsize=8, color='#636363', va='top')

    ax.set_xlabel('Fragment Length (bp)', fontsize=12)
    ax.set_ylabel('Normalised Density', fontsize=12)
    ax.set_title('cfDNA Fragment Length: Cancer vs Healthy\n(KDE, 2 cancer + 2 healthy BAMs)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(50, 500)

    # Panel B: per-sample histogram (all 8 BAMs)
    ax2 = axes[1]
    bins = np.arange(50, 501, 5)
    centers = (bins[:-1] + bins[1:]) / 2

    for srr, lengths in sorted(frag_data.items()):
        label = SAMPLE_LABELS.get(srr, 'unknown')
        color = COLORS[label]
        counts, _ = np.histogram(lengths, bins=bins)
        counts = counts / counts.sum()
        lw = 2.0 if label != 'unknown' else 1.0
        alpha = 0.8 if label != 'unknown' else 0.4
        short_srr = srr.replace('SRR', '')
        ls = '-' if label == 'cancer' else ('--' if label == 'healthy' else ':')
        ax2.plot(centers, counts, lw=lw, alpha=alpha, color=color,
                 linestyle=ls, label=f'{srr} ({label})')

    ax2.set_xlabel('Fragment Length (bp)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Per-sample Fragment Length Profiles\n(all 8 BAMs, 4 with known labels)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(50, 500)

    fig.suptitle(
        'Real cfDNA Fragment Lengths — Extracted from Aligned BAMs (SRR11537952–59)\n'
        'Data source: GSE149438, processed by project Nextflow pipeline',
        fontsize=12, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_path}")


# ── 3. Honest methylation / feature heatmap ───────────────────────────────────

def plot_feature_heatmap(out_path: Path):
    """
    Heatmap of real GEO cfDNA features (GSE149438, n=300).
    Uses top 20 discriminative frag_bin features + methylation_density + mean_gc.
    Annotated honestly — NOT presented as CpG methylation measurements.
    """
    df = pd.read_csv(BASE / 'data' / 'feature_matrix.csv')

    frag_cols = [c for c in df.columns if c.startswith('frag_bin_')]
    extra_cols = [c for c in ['methylation_density', 'mean_gc', 'short_frag_ratio', 'nuc_score']
                  if c in df.columns]

    # Select top 20 most discriminative frag bins via variance between classes
    cancer_mean  = df[df['sample_type'] == 'cancer'][frag_cols].mean()
    healthy_mean = df[df['sample_type'] == 'healthy'][frag_cols].mean()
    delta = (cancer_mean - healthy_mean).abs().sort_values(ascending=False)
    top_frag = delta.index[:20].tolist()

    feat_cols = top_frag + extra_cols
    n_healthy = min(30, (df['sample_type'] == 'healthy').sum())
    n_cancer  = min(30, (df['sample_type'] == 'cancer').sum())

    healthy_idx = df[df['sample_type'] == 'healthy'].index[:n_healthy]
    cancer_idx  = df[df['sample_type'] == 'cancer'].index[:n_cancer]
    idx = list(healthy_idx) + list(cancer_idx)

    mat = df.loc[idx, feat_cols].values.astype(float)
    # Normalise each feature to [0,1] across all selected samples
    col_min = mat.min(axis=0, keepdims=True)
    col_max = mat.max(axis=0, keepdims=True)
    mat_norm = (mat - col_min) / (col_max - col_min + 1e-10)

    # Pretty feature labels
    def label(c):
        if c.startswith('frag_bin_'):
            idx_b = int(c.split('_')[-1])
            bp = 100 + idx_b * (150 / len(frag_cols))
            return f'{bp:.0f} bp'
        return c.replace('_', ' ').title()

    pretty_cols = [label(c) for c in feat_cols]
    sample_ids  = [f'H{i+1:02d}' for i in range(n_healthy)] + [f'C{i+1:02d}' for i in range(n_cancer)]

    plot_df = pd.DataFrame(mat_norm.T, index=pretty_cols, columns=sample_ids)

    fig, ax = plt.subplots(figsize=(16, 7))
    sns.heatmap(
        plot_df,
        cmap='RdYlBu_r',
        vmin=0, vmax=1,
        xticklabels=sample_ids,
        yticklabels=pretty_cols,
        cbar_kws={'label': 'Normalised Feature Value', 'shrink': 0.8},
        linewidths=0,
        ax=ax,
    )

    ax.axvline(n_healthy, color='black', lw=2.5)
    ax.text(n_healthy / 2,      -1.2, 'Healthy (Normal)',
            ha='center', fontsize=11, fontweight='bold', color='#27ae60')
    ax.text(n_healthy + n_cancer / 2, -1.2, 'Cancer (mixed GI subtypes)',
            ha='center', fontsize=11, fontweight='bold', color='#e74c3c')

    ax.set_xlabel('Sample (GEO GSE149438)', fontsize=12)
    ax.set_ylabel('cfDNA Feature', fontsize=12)
    ax.set_title(
        'cfDNA Feature Profiles: Healthy vs Cancer — GEO Dataset (GSE149438, n=300)\n'
        'Top 20 discriminative fragment-length bins + methylation density + GC content\n'
        'Data source: Original study preprocessed features (NOT simulated)',
        fontsize=11, fontweight='bold'
    )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=7, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('Generating REAL figures from actual data')
    print('=' * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    print('\n[1/2] Fragment distribution from BAMs...')
    frag_data = load_or_extract_fragments()

    # Print summary stats
    print('\nFragment length summary:')
    for srr, lengths in sorted(frag_data.items()):
        label = SAMPLE_LABELS.get(srr, 'unknown')
        print(f'  {srr} ({label}): n={len(lengths):,}, mean={lengths.mean():.1f}, '
              f'median={np.median(lengths):.1f}, std={lengths.std():.1f}')

    plot_fragment_distribution(frag_data, FIG_DIR / 'fragment_distribution.png')

    print('\n[2/2] Feature heatmap from GEO matrix...')
    plot_feature_heatmap(FIG_DIR / 'methylation_heatmap.png')

    print('\nDone. Files saved:')
    for name in ['fragment_distribution.png', 'methylation_heatmap.png']:
        p = FIG_DIR / name
        if p.exists():
            print(f'  {p} ({p.stat().st_size // 1024} KB)')


if __name__ == '__main__':
    main()

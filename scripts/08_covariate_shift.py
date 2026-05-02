#!/usr/bin/env python3
"""
08_covariate_shift.py — Quantify and visualise the covariate shift between
GEO (GSE149438, n=300, methylation-seq preprocessed features) and
Nextflow BAM pipeline features (n=3 valid WGS-aligned samples).

Produces:
  results/covariate_shift_metrics.json
  report/figures/covariate_shift.png

This is a scientific FINDING, not just a limitation: the two pipelines produce
fundamentally incompatible feature spaces due to differences in:
  1. Fragment length bin ranges (methylation-seq vs WGS)
  2. Normalisation schemes (nuc_score: 0-1 vs raw read counts)
  3. Short-fragment enrichment (library prep artefact)
"""

import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE    = Path(__file__).parent.parent
DPI     = 300
SEED    = 42

# Known labels for NF samples (from accessions.txt)
NF_LABELS = {'SRR11537952': 'cancer', 'SRR11537953': 'healthy', 'SRR11537954': 'healthy'}
NF_COLORS = {'cancer': '#e74c3c', 'healthy': '#27ae60'}
GEO_COLOR = '#3182bd'
NF_COLOR  = '#e6550d'


def load_data():
    geo = pd.read_csv(BASE / 'data' / 'feature_matrix.csv')
    nf  = pd.read_csv(BASE / 'results' / 'nextflow' / 'feature_matrix.csv')

    # Keep only valid NF samples (n_fragments > 1000)
    nf_clean = nf[nf['n_fragments'] > 1000].copy()
    print(f"GEO samples : {len(geo)} (cancer={( geo['sample_type']=='cancer').sum()}, "
          f"healthy={(geo['sample_type']=='healthy').sum()})")
    print(f"NF samples  : {len(nf_clean)} valid of {len(nf)} total "
          f"(excluded {len(nf)-len(nf_clean)} degenerate samples with <1000 fragments)")
    return geo, nf_clean


def kl_divergence_hist(a, b, n_bins=50):
    """Symmetric KL divergence (Jensen-Shannon distance) from two 1D arrays."""
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    bins = np.linspace(lo, hi, n_bins + 1)
    p, _ = np.histogram(a, bins=bins, density=True)
    q, _ = np.histogram(b, bins=bins, density=True)
    eps = 1e-10
    p = p + eps;  p /= p.sum()
    q = q + eps;  q /= q.sum()
    kl_pq = float(entropy(p, q))
    kl_qp = float(entropy(q, p))
    js    = float(jensenshannon(p, q))
    return kl_pq, kl_qp, js


def compute_shift_metrics(geo, nf_clean):
    frag_cols   = [c for c in geo.columns if c.startswith('frag_bin_')]
    scalar_feats = ['mean_frag_len', 'std_frag_len', 'short_frag_ratio', 'mean_gc']

    metrics = {}

    # 1. Scalar feature statistics
    for feat in scalar_feats:
        if feat not in geo.columns or feat not in nf_clean.columns:
            continue
        geo_vals = geo[feat].dropna().values
        nf_vals  = nf_clean[feat].dropna().values
        kl_pq, kl_qp, js = kl_divergence_hist(geo_vals, nf_vals)
        metrics[feat] = {
            'geo_mean': float(geo_vals.mean()),
            'geo_std' : float(geo_vals.std()),
            'nf_mean' : float(nf_vals.mean()),
            'nf_std'  : float(nf_vals.std()),
            'abs_mean_diff': float(abs(geo_vals.mean() - nf_vals.mean())),
            'kl_geo_to_nf' : kl_pq,
            'kl_nf_to_geo' : kl_qp,
            'js_distance'  : js,
        }
        print(f"  {feat:<22}: GEO={geo_vals.mean():.4f}±{geo_vals.std():.4f}  "
              f"NF={nf_vals.mean():.4f}±{nf_vals.std():.4f}  JS={js:.4f}")

    # 2. nuc_score — different units entirely
    geo_ns = geo['nuc_score'].values
    nf_ns  = nf_clean['nuc_score'].values
    metrics['nuc_score'] = {
        'geo_mean': float(geo_ns.mean()), 'geo_std': float(geo_ns.std()),
        'nf_mean' : float(nf_ns.mean()),  'nf_std' : float(nf_ns.std()),
        'unit_note': 'GEO: normalised 0-1 score; NF: raw read counts (~1.5M). '
                     'Direct comparison is invalid — different normalisation.',
        'scale_ratio': float(nf_ns.mean() / (geo_ns.mean() + 1e-10)),
    }
    print(f"  {'nuc_score':<22}: GEO={geo_ns.mean():.4f} (0-1)  "
          f"NF={nf_ns.mean():.0f} (raw counts)  scale_ratio={metrics['nuc_score']['scale_ratio']:.0f}x")

    # 3. Fragment bin profile shift
    geo_frag = geo[frag_cols].mean().values
    nf_frag  = nf_clean[frag_cols].mean().values
    geo_peak = int(np.argmax(geo_frag))
    nf_peak  = int(np.argmax(nf_frag))
    # JS divergence on mean profiles
    eps = 1e-10
    p = geo_frag + eps;  p /= p.sum()
    q = nf_frag  + eps;  q /= q.sum()
    frag_js = float(jensenshannon(p, q))
    metrics['frag_bins'] = {
        'geo_peak_bin': geo_peak,
        'nf_peak_bin' : nf_peak,
        'peak_bin_shift': abs(geo_peak - nf_peak),
        'js_distance_mean_profiles': frag_js,
        'note': (
            f'GEO bins span ~100-250 bp (methylation-seq library prep). '
            f'NF bins span ~50-500 bp (WGS). '
            f'Same bin index = different bp ranges. '
            f'Profile JS-divergence = {frag_js:.4f}.'
        )
    }
    print(f"  {'frag_bin profile':<22}: GEO peak=bin{geo_peak}  NF peak=bin{nf_peak}  "
          f"peak shift={abs(geo_peak-nf_peak)} bins  JS={frag_js:.4f}")

    return metrics


def plot_covariate_shift(geo, nf_clean, metrics):
    frag_cols = [c for c in geo.columns if c.startswith('frag_bin_')]

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # ── Panel A: Mean frag_bin profiles ──────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, :2])
    bins = np.arange(len(frag_cols))

    geo_cancer  = geo[geo['sample_type'] == 'cancer'][frag_cols].mean().values
    geo_healthy = geo[geo['sample_type'] == 'healthy'][frag_cols].mean().values
    nf_mean     = nf_clean[frag_cols].mean().values
    nf_std      = nf_clean[frag_cols].std().values

    ax_a.plot(bins, geo_cancer,  color='#e74c3c', lw=2,   alpha=0.9, label='GEO — cancer (n=254)')
    ax_a.plot(bins, geo_healthy, color='#27ae60', lw=2,   alpha=0.9, label='GEO — healthy (n=46)')
    ax_a.plot(bins, nf_mean,     color=NF_COLOR,  lw=2.5, alpha=1.0, label='Nextflow BAM — mean (n=3 WGS)')
    ax_a.fill_between(bins, nf_mean - nf_std, nf_mean + nf_std,
                      color=NF_COLOR, alpha=0.15)

    ax_a.axvline(np.argmax(geo_cancer),  color='#e74c3c', lw=1.2, linestyle='--', alpha=0.6)
    ax_a.axvline(np.argmax(nf_mean),     color=NF_COLOR,  lw=1.2, linestyle='--', alpha=0.6)

    ax_a.set_xlabel('frag_bin index (0–149)', fontsize=12)
    ax_a.set_ylabel('Mean bin frequency', fontsize=12)
    ax_a.set_title(
        'Fragment Length Profile: GEO vs Nextflow\n'
        'CRITICAL: Same bin index encodes different bp ranges in each pipeline',
        fontsize=11, fontweight='bold'
    )
    ax_a.legend(fontsize=10)
    ax_a.grid(alpha=0.3)

    # Annotation boxes
    js_frag = metrics['frag_bins']['js_distance_mean_profiles']
    ax_a.text(0.02, 0.95,
              f"JS-divergence (GEO cancer vs NF) = {js_frag:.3f}\n"
              f"GEO peak: bin {metrics['frag_bins']['geo_peak_bin']}  "
              f"NF peak: bin {metrics['frag_bins']['nf_peak_bin']}  "
              f"(shift = {metrics['frag_bins']['peak_bin_shift']} bins)",
              transform=ax_a.transAxes, fontsize=9,
              verticalalignment='top',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3cd', alpha=0.9))

    # ── Panel B: JS divergence per scalar feature ─────────────────────────────
    ax_b = fig.add_subplot(gs[0, 2])
    feats_for_bar = [k for k in metrics if k not in ('nuc_score', 'frag_bins')]
    js_vals  = [metrics[k]['js_distance'] for k in feats_for_bar]
    colors_b = ['#e6550d' if v > 0.3 else '#3182bd' for v in js_vals]
    pretty   = {'mean_frag_len': 'Mean frag len',
                'std_frag_len':  'Std frag len',
                'short_frag_ratio': 'Short frag ratio',
                'mean_gc':       'GC content'}
    labels_b = [pretty.get(k, k) for k in feats_for_bar]

    bars = ax_b.barh(labels_b, js_vals, color=colors_b, alpha=0.85, edgecolor='white')
    ax_b.axvline(0.3, color='#636363', lw=1.5, linestyle='--', alpha=0.7,
                 label='Moderate shift threshold')
    for bar, v in zip(bars, js_vals):
        ax_b.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                  f'{v:.3f}', va='center', fontsize=10, fontweight='bold')
    ax_b.set_xlabel('Jensen-Shannon Distance', fontsize=11)
    ax_b.set_title('Feature Shift\n(JS Distance, 0=identical)', fontsize=11, fontweight='bold')
    ax_b.set_xlim(0, max(js_vals) * 1.25)
    ax_b.legend(fontsize=8)
    ax_b.grid(alpha=0.3, axis='x')

    # ── Panel C: Mean fragment length distribution ────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    geo_mfl = geo['mean_frag_len'].values
    nf_mfl  = nf_clean['mean_frag_len'].values

    ax_c.hist(geo_mfl[geo['sample_type'] == 'cancer'],
              bins=25, color='#e74c3c', alpha=0.6, density=True, label='GEO cancer')
    ax_c.hist(geo_mfl[geo['sample_type'] == 'healthy'],
              bins=25, color='#27ae60', alpha=0.6, density=True, label='GEO healthy')
    for srr, mfl in zip(nf_clean['sample_id'], nf_mfl):
        lbl = NF_LABELS.get(srr, 'unknown')
        ax_c.axvline(mfl, color=NF_COLORS.get(lbl, '#888888'), lw=2.5,
                     linestyle=':', label=f'NF {srr} ({lbl})')
    ax_c.set_xlabel('Mean fragment length (bp)', fontsize=11)
    ax_c.set_ylabel('Density', fontsize=11)
    ax_c.set_title('Mean Fragment Length\nGEO vs Nextflow BAM', fontsize=11, fontweight='bold')
    ax_c.legend(fontsize=8)
    ax_c.grid(alpha=0.3)
    diff = metrics['mean_frag_len']['abs_mean_diff']
    js_v = metrics['mean_frag_len']['js_distance']
    ax_c.text(0.02, 0.97, f'Δmean={diff:.1f} bp\nJS={js_v:.3f}',
              transform=ax_c.transAxes, fontsize=9, va='top',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3cd', alpha=0.9))

    # ── Panel D: short_frag_ratio ─────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    geo_sfr = geo['short_frag_ratio'].values
    nf_sfr  = nf_clean['short_frag_ratio'].values
    ax_d.hist(geo_sfr[geo['sample_type'] == 'cancer'],
              bins=20, color='#e74c3c', alpha=0.6, density=True, label='GEO cancer')
    ax_d.hist(geo_sfr[geo['sample_type'] == 'healthy'],
              bins=20, color='#27ae60', alpha=0.6, density=True, label='GEO healthy')
    for srr, sfr in zip(nf_clean['sample_id'], nf_sfr):
        lbl = NF_LABELS.get(srr, 'unknown')
        ax_d.axvline(sfr, color=NF_COLORS.get(lbl, '#888'), lw=2.5,
                     linestyle=':', label=f'NF {srr} ({lbl})')
    ax_d.set_xlabel('Short fragment ratio (<150 bp)', fontsize=11)
    ax_d.set_ylabel('Density', fontsize=11)
    ax_d.set_title('Short Fragment Ratio\nGEO vs Nextflow BAM', fontsize=11, fontweight='bold')
    ax_d.legend(fontsize=8)
    ax_d.grid(alpha=0.3)
    diff2 = metrics['short_frag_ratio']['abs_mean_diff']
    js_v2 = metrics['short_frag_ratio']['js_distance']
    ax_d.text(0.02, 0.97, f'Δmean={diff2:.3f}\nJS={js_v2:.3f}',
              transform=ax_d.transAxes, fontsize=9, va='top',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3cd', alpha=0.9))

    # ── Panel E: nuc_score scale incompatibility ──────────────────────────────
    ax_e = fig.add_subplot(gs[1, 2])
    ns_data  = {
        'GEO\n(n=300)\n[0–1 norm.]': geo['nuc_score'].values,
    }
    nf_ns_scaled = nf_clean['nuc_score'].values / 1e6  # scale to millions for display

    positions = [1, 2]
    bp1 = ax_e.boxplot(geo['nuc_score'].values, positions=[1], widths=0.4,
                       patch_artist=True,
                       boxprops=dict(facecolor=GEO_COLOR, alpha=0.7),
                       medianprops=dict(color='black', lw=2))
    ax_e2 = ax_e.twinx()
    bp2 = ax_e2.boxplot(nf_clean['nuc_score'].values, positions=[2], widths=0.4,
                         patch_artist=True,
                         boxprops=dict(facecolor=NF_COLOR, alpha=0.7),
                         medianprops=dict(color='black', lw=2))

    ax_e.set_xticks([1, 2])
    ax_e.set_xticklabels(['GEO\n(normalised\n0–1)', 'Nextflow BAM\n(raw counts\n×10⁶)'], fontsize=10)
    ax_e.set_ylabel('GEO nuc_score (0–1)', color=GEO_COLOR, fontsize=10)
    ax_e2.set_ylabel('NF nuc_score (raw counts)', color=NF_COLOR, fontsize=10)
    ax_e.set_title('nuc_score: Incompatible Units\n(6 orders of magnitude difference)',
                   fontsize=11, fontweight='bold')
    scale = metrics['nuc_score']['scale_ratio']
    ax_e.text(0.5, 0.5, f'Scale ratio:\n~{scale/1e6:.1f}M×',
              transform=ax_e.transAxes, fontsize=14, ha='center', va='center',
              color='#e6550d', fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3cd', alpha=0.9))
    ax_e.grid(alpha=0.3)

    fig.suptitle(
        'Covariate Shift: GEO Preprocessed Features vs Nextflow BAM Pipeline Features\n'
        'Finding: Feature spaces are fundamentally incompatible — different bin ranges, '
        'normalisation schemes, and library preparation artefacts',
        fontsize=12, fontweight='bold'
    )

    out_path = BASE / 'report' / 'figures' / 'covariate_shift.png'
    plt.savefig(str(out_path), dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f'\n✓ Figure saved: {out_path}')


def main():
    print('=' * 60)
    print('COVARIATE SHIFT ANALYSIS')
    print('GEO (GSE149438, n=300) vs Nextflow BAM pipeline (n=3 valid)')
    print('=' * 60)

    geo, nf_clean = load_data()

    print('\nComputing shift metrics...')
    metrics = compute_shift_metrics(geo, nf_clean)

    # Save JSON
    out_json = BASE / 'results' / 'covariate_shift_metrics.json'
    with open(out_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\n✓ Metrics saved: {out_json}')

    print('\nGenerating figure...')
    plot_covariate_shift(geo, nf_clean, metrics)

    print('\n' + '=' * 60)
    print('SUMMARY OF FINDINGS')
    print('=' * 60)
    print(f"mean_frag_len   : GEO {metrics['mean_frag_len']['geo_mean']:.1f} bp  vs  "
          f"NF {metrics['mean_frag_len']['nf_mean']:.1f} bp  "
          f"(Δ={metrics['mean_frag_len']['abs_mean_diff']:.1f} bp, JS={metrics['mean_frag_len']['js_distance']:.3f})")
    print(f"short_frag_ratio: GEO {metrics['short_frag_ratio']['geo_mean']:.3f}  vs  "
          f"NF {metrics['short_frag_ratio']['nf_mean']:.3f}  "
          f"(Δ={metrics['short_frag_ratio']['abs_mean_diff']:.3f}, JS={metrics['short_frag_ratio']['js_distance']:.3f})")
    print(f"nuc_score       : GEO normalised 0–1 vs NF raw counts ~{metrics['nuc_score']['nf_mean']:.0f}  "
          f"(scale ratio ~{metrics['nuc_score']['scale_ratio']:.0f}×)")
    print(f"frag_bin profile: JS-divergence = {metrics['frag_bins']['js_distance_mean_profiles']:.3f}  "
          f"peak shift = {metrics['frag_bins']['peak_bin_shift']} bins")
    print('\nConclusion: The covariate shift is not primarily biological but methodological:')
    print('  1. GEO used targeted methylation-seq (short, enriched fragments)')
    print('  2. Nextflow pipeline used WGS-aligned reads (broader fragment range)')
    print('  3. nuc_score normalisation is incompatible between pipelines')
    print('  This explains external validation AUC=0.0 — the model sees OOD inputs.')


if __name__ == '__main__':
    main()

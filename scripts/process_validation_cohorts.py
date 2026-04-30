#!/usr/bin/env python3
"""
process_validation_cohorts.py
==============================
Step 1: Wait for downloaded FASTQs, then run Nextflow feature extraction.
Step 2: Load BAM-derived features from Jiang HCC cohort (PRJNA291063).
Step 3: Combine with GSE71378 WPS-footprint statistics.
Step 4: Save unified external cohort feature matrix.

PRJNA291063 sample labels (Jiang 2015 HCC plasma cfDNA study):
  IH* = Individual Healthy (plasma from healthy donors)
  IC* = Individual Cancer  (plasma from HCC patients)
  IA* = Individual Adjacent (adjacent non-tumour tissue — EXCLUDED)
  BH01 = Bulk Healthy reference (deep coverage, same as GSE71378 BH01)

Run after FASTQs have downloaded and Nextflow has processed them.
"""

import os, sys, json, glob
import numpy as np
import pandas as pd
import subprocess

BASE = os.path.join(os.path.dirname(__file__), '..')
JIANG_DIR   = os.path.join(BASE, 'data', 'validation', 'jiang_hcc')
SNYDER_DIR  = os.path.join(BASE, 'data', 'validation', 'snyder_healthy')
OUT_DIR     = os.path.join(BASE, 'results', 'validation')
NF_OUTDIR   = os.path.join(BASE, 'results', 'validation', 'jiang_hcc_nextflow')

os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(NF_OUTDIR, exist_ok=True)

FRAG_MIN, FRAG_MAX, N_BINS = 100, 400, 150

JIANG_LABELS = {
    # healthy controls
    'SRR2130052': (0, 'healthy',  'IH03', 'PRJNA291063'),
    'SRR2130051': (0, 'healthy',  'IH02', 'PRJNA291063'),
    'SRR2130050': (0, 'healthy',  'IH01', 'PRJNA291063'),
    # HCC cancer patients
    'SRR2130005': (1, 'HCC',      'IC05', 'PRJNA291063'),
    'SRR2130007': (1, 'HCC',      'IC08', 'PRJNA291063'),
    'SRR2130028': (1, 'HCC',      'IC29', 'PRJNA291063'),
    'SRR2130036': (1, 'HCC',      'IC38', 'PRJNA291063'),
}


# ── Step 1: Check downloaded FASTQs ─────────────────────────────────────────
def check_downloads():
    fastqs = glob.glob(os.path.join(JIANG_DIR, '*.fastq.gz'))
    print(f"Downloaded FASTQs ({len(fastqs)} files):")
    by_sample = {}
    for f in sorted(fastqs):
        bn = os.path.basename(f)
        srr = bn.split('_')[0]
        by_sample.setdefault(srr, []).append(f)
        mb = os.path.getsize(f) / 1e6
        print(f"  {bn}: {mb:.0f} MB")

    complete = {}  # SRR → (R1, R2) pairs
    for srr, files in by_sample.items():
        r1 = [f for f in files if '_1.fastq.gz' in f]
        r2 = [f for f in files if '_2.fastq.gz' in f]
        if r1 and r2:
            r1_mb = os.path.getsize(r1[0]) / 1e6
            r2_mb = os.path.getsize(r2[0]) / 1e6
            # Consider complete if both >100 MB (at least partially downloaded)
            if r1_mb > 100 and r2_mb > 100:
                complete[srr] = (r1[0], r2[0])
                print(f"  ✓ {srr}: paired complete ({r1_mb:.0f} MB + {r2_mb:.0f} MB)")
            else:
                print(f"  ⏳ {srr}: still downloading ({r1_mb:.0f} MB + {r2_mb:.0f} MB)")
    return complete


# ── Step 2: Run Nextflow on downloaded samples ───────────────────────────────
def run_nextflow_on_samples(complete_samples):
    """Submit each sample pair through the existing Nextflow pipeline."""
    if not complete_samples:
        print("No complete sample pairs available yet.")
        return {}

    results = {}
    for srr, (r1, r2) in complete_samples.items():
        sample_outdir = os.path.join(NF_OUTDIR, srr)
        os.makedirs(sample_outdir, exist_ok=True)

        feature_csv = os.path.join(sample_outdir, 'features', f'{srr}_features.csv')
        if os.path.exists(feature_csv):
            print(f"  {srr}: features already extracted, skipping")
            results[srr] = feature_csv
            continue

        print(f"\nRunning Nextflow for {srr}...")
        # Use existing Nextflow pipeline
        nf_main = os.path.join(BASE, 'nextflow', 'main.nf')
        cmd = [
            'nextflow', 'run', nf_main,
            '--reads', f'{os.path.dirname(r1)}/{srr}_{{1,2}}.fastq.gz',
            '--genome', '/home/workshop/niraj/reference/hg38.fa',
            '--bwa_index', '/home/workshop/niraj/reference/hg38',
            '--outdir', sample_outdir,
            '-resume',
        ]
        print(f"  CMD: {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=7200,
                cwd=os.path.join(BASE, 'nextflow')
            )
            if proc.returncode == 0:
                print(f"  ✓ Nextflow succeeded for {srr}")
                results[srr] = feature_csv
            else:
                print(f"  ✗ Nextflow failed: {proc.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            print(f"  ✗ Nextflow timeout for {srr}")

    return results


# ── Step 3: Load Nextflow feature CSVs ───────────────────────────────────────
def load_nextflow_features(results):
    records = []
    for srr, csv_path in results.items():
        if not os.path.exists(csv_path):
            # Try merged matrix
            alt = os.path.join(os.path.dirname(os.path.dirname(csv_path)),
                               'feature_matrix.csv')
            if os.path.exists(alt):
                csv_path = alt
            else:
                print(f"  WARN: no features for {srr}")
                continue
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                sid = row.get('sample_id', srr)
                label, ctype, title, cohort = JIANG_LABELS.get(srr, (None, None, None, None))
                if label is None:
                    # Try to infer from sample title
                    label = 1 if 'IC' in srr else 0
                    ctype = 'HCC' if label else 'healthy'
                    cohort = 'PRJNA291063'
                row_dict = row.to_dict()
                row_dict.update({
                    'sample_id':   f'PRJNA291063_{title or srr}',
                    'label':       label,
                    'cancer_type': ctype,
                    'cohort':      cohort,
                    'study':       'Jiang2015_HCC',
                })
                records.append(row_dict)
        except Exception as e:
            print(f"  Error loading {csv_path}: {e}")

    return pd.DataFrame(records) if records else None


# ── Step 4: GSE71378 WPS footprint statistics ────────────────────────────────
def extract_wps_statistics():
    """
    Extract summary statistics from GSE71378 WPS BigBed files.
    Note: BigBed entries are WPS nucleosome footprints (7-150bp peaks),
    NOT cfDNA fragment coordinates. Cannot be directly input to our model.
    These statistics are used for supplementary analysis only.
    """
    try:
        import pyBigWig
    except ImportError:
        print("pyBigWig not available — skipping WPS analysis")
        return None

    SNYDER_LABELS = {
        'BH01': (0, 'healthy',   'GSE71378'),
        'IH01': (0, 'healthy',   'GSE71378'),
        'IH02': (0, 'healthy',   'GSE71378'),
        'CA01': (1, 'colorectal','GSE71378'),
        'CH01': (1, 'colorectal','GSE71378'),
    }

    records = []
    print("\nExtracting WPS statistics from GSE71378 BigBed files...")

    for fname in sorted(os.listdir(SNYDER_DIR)):
        if not fname.endswith('.bb'):
            continue
        sample_id = fname.replace('GSE71378_', '').replace('.bb', '')
        label_info = SNYDER_LABELS.get(sample_id)
        if label_info is None:
            continue
        label, ctype, cohort = label_info

        bb_path = os.path.join(SNYDER_DIR, fname)
        print(f"  {sample_id} ({'healthy' if label==0 else 'cancer'})...", end=' ', flush=True)

        try:
            bb = pyBigWig.open(bb_path)
            chroms = bb.chroms()
            sizes, scores = [], []

            for chrom in list(chroms.keys())[:5]:  # sample first 5 chroms
                entries = bb.entries(chrom, 0, chroms[chrom], withString=True)
                if not entries:
                    continue
                for start, end, string in entries[:200_000]:
                    sizes.append(end - start)
                    try:
                        parts = string.split('\t')
                        if len(parts) > 1:
                            scores.append(float(parts[1]))
                    except Exception:
                        pass
            bb.close()

            sizes  = np.array(sizes)
            scores = np.array(scores) if scores else np.array([0])

            rec = {
                'sample_id':           f'GSE71378_{sample_id}',
                'label':               label,
                'cancer_type':         ctype,
                'cohort':              cohort,
                'study':               'Snyder2016',
                'data_type':           'WPS_footprints',
                'n_footprints_sampled': len(sizes),
                'wps_median_length':   float(np.median(sizes)),
                'wps_mean_length':     float(sizes.mean()),
                'wps_pct_short_lt50':  float((sizes < 50).mean()),
                'wps_mean_score':      float(scores.mean()),
                'wps_median_score':    float(np.median(scores)),
                'NOTE': 'WPS footprints, NOT fragment lengths. Cannot be used with fragment-length model.',
            }
            records.append(rec)
            print(f"{len(sizes):,} footprints, median={np.median(sizes):.0f}bp, mean_score={scores.mean():.1f}")
        except Exception as e:
            print(f"ERROR: {e}")

    if not records:
        return None
    df = pd.DataFrame(records)
    out = os.path.join(OUT_DIR, 'snyder2016_wps_statistics.csv')
    df.to_csv(out, index=False)
    print(f"Saved WPS statistics: {out}")
    return df


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("VALIDATION COHORT PROCESSING")
    print("=" * 60)

    complete = check_downloads()

    if complete:
        print(f"\n{len(complete)} sample pair(s) ready for Nextflow processing")
        nf_results = run_nextflow_on_samples(complete)
        ext_df = load_nextflow_features(nf_results)
        if ext_df is not None and not ext_df.empty:
            out_path = os.path.join(OUT_DIR, 'jiang_hcc_features.csv')
            ext_df.to_csv(out_path, index=False)
            print(f"\nSaved Jiang HCC features: {out_path}")
            print(f"  {len(ext_df)} samples: {(ext_df['label']==1).sum()} cancer, {(ext_df['label']==0).sum()} healthy")
    else:
        print("\nNo complete FASTQ pairs yet — run again after downloads finish.")

    # Extract GSE71378 WPS stats (supplementary)
    wps_df = extract_wps_statistics()

    # Report
    avail = {}
    if complete:
        avail['PRJNA291063_downloaded'] = len(complete)
    jiang_feat = os.path.join(OUT_DIR, 'jiang_hcc_features.csv')
    if os.path.exists(jiang_feat):
        avail['PRJNA291063_features'] = len(pd.read_csv(jiang_feat))
    if wps_df is not None:
        avail['GSE71378_wps_samples'] = len(wps_df)
    avail['note_bigbed'] = ('GSE71378 BigBed files contain WPS nucleosome footprints '
                            '(7-150 bp peaks), NOT cfDNA fragment lengths. '
                            'Not compatible with fragment-length-based model. '
                            'WPS statistics extracted for supplementary analysis only.')

    with open(os.path.join(OUT_DIR, 'cohort_processing_status.json'), 'w') as f:
        json.dump(avail, f, indent=2)
    print(f"\nStatus saved: {os.path.join(OUT_DIR, 'cohort_processing_status.json')}")


if __name__ == '__main__':
    main()

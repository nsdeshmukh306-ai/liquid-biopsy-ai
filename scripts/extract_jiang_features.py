#!/usr/bin/env python3
"""
extract_jiang_features.py
=========================
Direct fragment length extraction from Jiang HCC paired-end FASTQs.
This script handles the full pipeline:
  FASTQs → BWA align → samtools sort → fragment length histogram

Bypasses Nextflow for simplicity; uses the same BWA/samtools installed
in the workshop_nf conda environment.

Run after FASTQs have downloaded:
  python3 scripts/extract_jiang_features.py

Requires:
  - data/validation/jiang_hcc/{SRR}_1.fastq.gz  (R1)
  - data/validation/jiang_hcc/{SRR}_2.fastq.gz  (R2)
  - /home/workshop/niraj/reference/hg38.fa (BWA index)
"""

import os, sys, subprocess, json, tempfile
import numpy as np
import pandas as pd

BASE   = os.path.join(os.path.dirname(__file__), '..')
INDIR  = os.path.join(BASE, 'data', 'validation', 'jiang_hcc')
OUTDIR = os.path.join(BASE, 'results', 'validation')
os.makedirs(OUTDIR, exist_ok=True)

BWA      = '/home/workshop/.conda/envs/workshop_nf/bin/bwa'
SAMTOOLS = '/home/workshop/.conda/envs/workshop_nf/bin/samtools'
REF      = '/home/workshop/niraj/reference/hg38.fa'
THREADS  = 4

FRAG_MIN, FRAG_MAX, N_BINS = 100, 400, 150

JIANG_LABELS = {
    'SRR2130052': (0, 'healthy', 'IH03', 'PRJNA291063'),
    'SRR2130051': (0, 'healthy', 'IH02', 'PRJNA291063'),
    'SRR2130050': (0, 'healthy', 'IH01', 'PRJNA291063'),
    'SRR2130005': (1, 'HCC',    'IC05', 'PRJNA291063'),
    'SRR2130007': (1, 'HCC',    'IC08', 'PRJNA291063'),
    'SRR2130028': (1, 'HCC',    'IC29', 'PRJNA291063'),
    'SRR2130036': (1, 'HCC',    'IC38', 'PRJNA291063'),
}


def check_available_samples():
    """Return SRR IDs where both R1 and R2 are >50 MB (usable)."""
    available = []
    for srr in JIANG_LABELS:
        r1 = os.path.join(INDIR, f'{srr}_1.fastq.gz')
        r2 = os.path.join(INDIR, f'{srr}_2.fastq.gz')
        if os.path.exists(r1) and os.path.exists(r2):
            r1_mb = os.path.getsize(r1) / 1e6
            r2_mb = os.path.getsize(r2) / 1e6
            if r1_mb > 50 and r2_mb > 50:
                available.append(srr)
                print(f"  ✓ {srr}: R1={r1_mb:.0f}MB, R2={r2_mb:.0f}MB")
            else:
                print(f"  ⏳ {srr}: R1={r1_mb:.0f}MB, R2={r2_mb:.0f}MB (incomplete)")
        else:
            print(f"  ✗ {srr}: missing FASTQ files")
    return available


def align_and_extract(srr, r1, r2, work_dir):
    """
    Run BWA-MEM paired-end alignment and extract fragment lengths.
    Returns numpy array of fragment lengths in [100, 400] bp range.
    """
    bam_path = os.path.join(work_dir, f'{srr}.bam')

    print(f"  Aligning {srr}...")
    # BWA-MEM | samtools view (filter mapped pairs) | samtools sort
    bwa_cmd = [BWA, 'mem', '-t', str(THREADS), REF, r1, r2]
    sam_view = [SAMTOOLS, 'view', '-b', '-f', '2', '-F', '2048', '-']
    sam_sort = [SAMTOOLS, 'sort', '-o', bam_path, '-']

    p1 = subprocess.Popen(bwa_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p2 = subprocess.Popen(sam_view, stdin=p1.stdout, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL)
    p1.stdout.close()
    p3 = subprocess.Popen(sam_sort, stdin=p2.stdout, stderr=subprocess.DEVNULL)
    p2.stdout.close()
    p3.wait()

    if not os.path.exists(bam_path) or os.path.getsize(bam_path) == 0:
        raise RuntimeError(f"BWA alignment produced empty BAM for {srr}")

    # Index BAM
    subprocess.run([SAMTOOLS, 'index', bam_path], check=True, stderr=subprocess.DEVNULL)
    print(f"  ✓ Alignment done: {os.path.getsize(bam_path)/1e6:.0f} MB BAM")

    # Extract TLEN (template length = fragment size)
    # Use samtools view -f 3 (properly paired) on first read of pair (-f 64)
    view_cmd = [SAMTOOLS, 'view', '-f', '67',   # flags: paired + mapped + read1
                '-F', '1284',                    # not unmapped, not supplementary
                bam_path]
    p = subprocess.Popen(view_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True)

    lengths = []
    for line in p.stdout:
        parts = line.split('\t')
        if len(parts) < 9:
            continue
        tlen = int(parts[8])   # TLEN field
        if tlen > 0:           # positive TLEN = inferred insert size
            lengths.append(tlen)
    p.wait()

    print(f"  Extracted {len(lengths):,} fragment lengths")
    return np.array(lengths)


def lengths_to_histogram(lengths, frag_min=100, frag_max=400, n_bins=150):
    """Convert fragment lengths to normalised histogram (150 bins)."""
    mask = (lengths >= frag_min) & (lengths <= frag_max)
    filtered = lengths[mask]
    bins = np.linspace(frag_min, frag_max, n_bins + 1)
    hist, _ = np.histogram(filtered, bins=bins)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    frac_in_range = mask.mean()
    return hist, bins, frac_in_range


def main():
    print("=" * 60)
    print("JIANG HCC FEATURE EXTRACTION")
    print("=" * 60)

    # Check available samples
    print("\nChecking available FASTQ pairs:")
    available = check_available_samples()

    if not available:
        print("\nNo complete FASTQ pairs yet. Check download status:")
        print("  ls -lh data/validation/jiang_hcc/*.fastq.gz")
        sys.exit(0)

    print(f"\n{len(available)} sample(s) ready: {available}")

    records = []
    with tempfile.TemporaryDirectory(prefix='jiang_bwa_') as tmpdir:
        for srr in available:
            label, ctype, title, cohort = JIANG_LABELS[srr]
            r1 = os.path.join(INDIR, f'{srr}_1.fastq.gz')
            r2 = os.path.join(INDIR, f'{srr}_2.fastq.gz')

            print(f"\nProcessing {srr} ({title}, {'HCC' if label else 'healthy'})...")

            try:
                lengths = align_and_extract(srr, r1, r2, tmpdir)
                hist, bins, frac = lengths_to_histogram(lengths)

                # Build feature row (bin_0 to bin_149)
                row = {'sample_id': f'PRJNA291063_{title}',
                       'label': label,
                       'cancer_type': ctype,
                       'cohort': cohort,
                       'study': 'Jiang2015_HCC',
                       'srr_accession': srr,
                       'n_fragments_total': len(lengths),
                       'frac_in_100_400bp': round(float(frac), 4)}

                bin_centers = (bins[:-1] + bins[1:]) / 2
                for i, (bc, h) in enumerate(zip(bin_centers, hist)):
                    row[f'bin_{i}'] = round(float(h), 8)

                records.append(row)
                print(f"  ✓ {srr}: {len(lengths):,} fragments, "
                      f"{frac*100:.1f}% in 100-400bp range, "
                      f"mode={bin_centers[hist.argmax()]:.0f} bp")

            except Exception as e:
                print(f"  ✗ Error processing {srr}: {e}")

    if not records:
        print("\nNo features extracted.")
        sys.exit(1)

    df = pd.DataFrame(records)
    out = os.path.join(OUTDIR, 'jiang_hcc_features.csv')
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(f"  {len(df)} samples: "
          f"{(df['label']==1).sum()} HCC, {(df['label']==0).sum()} healthy")

    # Also print fragment mode summary
    print("\nFragment length mode per sample:")
    bin_cols = [c for c in df.columns if c.startswith('bin_')]
    bins_centers = np.linspace(100, 400, 150)
    for _, row in df.iterrows():
        hist = row[bin_cols].values
        mode = bins_centers[hist.argmax()]
        print(f"  {row['sample_id']}: mode={mode:.0f}bp "
              f"({'HCC' if row['label'] else 'healthy'})")

    return df


if __name__ == '__main__':
    main()

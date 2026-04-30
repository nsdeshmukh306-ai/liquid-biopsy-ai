"""
extract_fragment_lengths.py
============================
Extract real fragment lengths from BAM files produced by our Nextflow pipeline
and generate a fragment length distribution plot.

Usage:
    python3 extract_fragment_lengths.py [--bam-dir PATH] [--out-csv PATH] [--out-png PATH]

Data Source: BAM Pipeline (n=8, our pipeline) — NOT GEO pre-processed data
"""

import pysam
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import argparse


def parse_args():
    p = argparse.ArgumentParser(description='Extract real fragment lengths from BAM files')
    p.add_argument('--bam-dir',  default='results/nextflow/bam', help='Directory containing *_dedup.bam files')
    p.add_argument('--out-csv',  default='real_fragment_lengths.csv')
    p.add_argument('--out-png',  default='real_fragment_distribution.png')
    p.add_argument('--max-per-sample', type=int, default=50000, help='Max fragments per BAM (for speed)')
    return p.parse_args()


def extract_fragments(bam_path, max_per_sample=50000):
    """Extract proper-pair fragment lengths from a BAM file (no index required)."""
    records = []
    sample = os.path.basename(bam_path).replace('_dedup.bam', '')
    try:
        bam = pysam.AlignmentFile(bam_path, "rb", check_sq=False)
        for read in bam.fetch(until_eof=True):
            if (read.is_proper_pair and
                    not read.is_secondary and
                    not read.is_supplementary and
                    read.template_length > 0):
                tlen = abs(read.template_length)
                if 50 < tlen < 500:
                    records.append({'sample': sample, 'fragment_length': tlen})
                    if len(records) >= max_per_sample:
                        break
        bam.close()
    except Exception as e:
        print(f"  ERROR reading {bam_path}: {e}")
    return records


def main():
    args = parse_args()
    bam_files = sorted(glob.glob(os.path.join(args.bam_dir, '*_dedup.bam')))

    if not bam_files:
        print(f"No BAM files found in {args.bam_dir}")
        return

    print(f"Found {len(bam_files)} BAM files")
    all_records = []

    for bam_path in bam_files:
        records = extract_fragments(bam_path, args.max_per_sample)
        all_records.extend(records)
        print(f"  {os.path.basename(bam_path)}: {len(records):,} fragments")

    print(f"\nTotal fragments extracted: {len(all_records):,}")

    if len(all_records) < 100:
        print("INSUFFICIENT_READS: < 100 fragments found. Check BAM files and alignment stats.")
        print("Tip: Low mapping rates (~3-5%) were observed for some samples (non-human reads).")
        return

    df = pd.DataFrame(all_records)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(df['fragment_length'], bins=100, color='steelblue', alpha=0.75, edgecolor='none')
    ax.axvline(132, color='red',    linestyle='--', lw=1.8, label='132 bp (nucleosome-free / cancer enriched)')
    ax.axvline(167, color='orange', linestyle='--', lw=1.8, label='167 bp (mono-nucleosome / healthy)')
    ax.set_xlabel("Fragment Length (bp)", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(
        f"Real cfDNA Fragment Length Distribution — BAM Pipeline (n={len(bam_files)} samples, "
        f"{len(all_records):,} fragments)",
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    plt.close()
    print(f"Saved: {args.out_png}")

    # Summary stats
    print(f"\nFragment length stats:")
    print(f"  Mean:   {df['fragment_length'].mean():.1f} bp")
    print(f"  Median: {df['fragment_length'].median():.1f} bp")
    print(f"  <150bp: {(df['fragment_length'] < 150).mean()*100:.1f}%")
    print(f"  150-200bp: {((df['fragment_length'] >= 150) & (df['fragment_length'] <= 200)).mean()*100:.1f}%")


if __name__ == '__main__':
    main()

# Limitations Addressed — Liquid Biopsy AI Project

This document summarises what integrity issues were identified, what was fixed,
what remains partially addressed, and what requires future work.

---

## ✅ FIXED

### 1. Data Attribution (FIX 1)
**Issue**: The 300-sample GEO matrix and the 8-sample BAM pipeline output were not
clearly distinguished in the notebook and scripts.

**Fix**:
- Added `DATA_PROVENANCE.md` with full figure/result source table
- Added data attribution disclaimer table in notebook (Cell 4 — Data section)
- Added `# DATA SOURCE: GEO pre-processed matrix...` comments in `scripts/01_preprocess.py`
  and `scripts/05_visualise.py` at every data loading point
- Added ⚠ disclaimer before Results section in notebook
- Updated all figure titles in `05_visualise.py` to append data source labels

### 2. Methylation Heatmap — Watermarked (FIX 2)
**Issue**: `report/figures/methylation_heatmap.png` was generated entirely from
`rng.beta()` synthetic values, not from real BAM or GEO methylation data.

**Fix**:
- Added "⚠ SIMULATED DATA — Not measured from BAM files" watermark to existing PNG
- Updated `05_visualise.py` figure title to state "SIMULATED DATA"
- Added warning markdown cell in notebook before the inline methylation plot
- Added caption note explaining this is a proxy visualisation

**Status**: Watermark applied. Real methylation data would require bisulfite sequencing
(not standard WGS). This is an acknowledged limitation of the dataset.

### 3. Model Comparison — Estimated Values Disclosed (FIX 3 Option B)
**Issue**: `model_comparison.png` showed CNN-only and LSTM-only bars with values
computed as `rf_score + 0.07` offset — not from actual separate training runs.

**Fix**:
- Added "⚠ CNN-only / LSTM-only: Estimated offsets..." watermark to existing PNG
- The notebook (Cell 11) already had a `* Simulated CNN/LSTM-only scores` footnote —
  this is now more prominent via the figure title update
- Added note in DATA_PROVENANCE.md table

**Not done**: Actually training CNN-only and LSTM-only branches (Option A) — this would
require ~10-20 minutes of additional training. The existing watermark approach is used instead.

### 4. Data Leakage Fix Documented (FIX 4)
**Issue**: v1 of the pipeline fitted StandardScaler on the full dataset before splitting,
producing artificially perfect AUC = 1.0.

**Fix**:
- Added markdown cell in notebook explicitly documenting the v1 leakage and v2 correction
- `scripts/01_preprocess.py` header already documented this (line 6)
- `results/metrics_corrected.json` contains v2 (leakage-free) results
- v1 metrics preserved in `results/metrics.json` for comparison

### 5. QC Summary Created (FIX 5)
**Issue**: No consolidated QC report.

**Fix**:
- Created `qc_summary.txt` (207 lines) with:
  - Full `samtools flagstat` output for all 8 deduped BAMs
  - Trimmomatic adapter trimming statistics (pairs surviving)
  - Known data quality issues (low mapping rates for 5 samples)
  - FastQC HTML report listing
- Individual FastQC HTML reports available in `results/nextflow/fastqc/`

### 6. Real Fragment Extraction Script (FIX 2 Attempt A)
**Status**: PARTIAL SUCCESS
- `extract_fragment_lengths.py` written and functional
- Successfully extracted 150,000 fragments from 3/8 samples (SRR11537952–54, which
  have 97% mapping rate to hg38)
- `real_fragment_lengths.csv` and `real_fragment_distribution.png` generated
- 5/8 samples had <6% mapping rate (non-human reads) → 0 fragments extracted from those
- The real fragment distribution (from 3 human samples) is now available

---

## ⚠ PARTIAL

### Model Comparison — Real Training
CNN-only and LSTM-only baselines were NOT separately trained (Option A not executed).
The estimated values remain in the figure with a watermark disclosure.
To fully fix: train separate CNN-only and LSTM-only models on the GEO dataset.

### Fragment Distribution — Real vs Simulated
The figure in `report/figures/fragment_distribution.png` correctly uses GEO pre-processed
data (not synthetic). The notebook inline plot is synthetic. Both are now clearly labelled.
The new `real_fragment_distribution.png` from BAM pipeline exists but shows only the 3
human samples (SRR11537952–54) due to non-human reads in the other 5 samples.

---

## ❌ NOT ADDRESSED (Future Work)

### Real Methylation Data
The methylation heatmap cannot be corrected without bisulfite sequencing data.
The existing proxy visualisation is now clearly watermarked as simulated.

### Sample Identity for Low-Mapping FASTQs
SRR11537955–59 showed <6% mapping to hg38. The true identity of these reads
(which organism/experiment) was not investigated. Future work: BLAST a subset of
reads to confirm organism.

### MultiQC Integration
Nextflow pipeline does not include a MultiQC step. Individual FastQC HTMLs exist.
Future: add `nf-core/multiqc` module to the Nextflow pipeline.

### Subtype Classification
The current model is binary (cancer vs healthy). Multi-class cancer subtype
classification (lung, breast, colorectal, etc.) was not implemented.

---

*Generated: 2026-04-24*

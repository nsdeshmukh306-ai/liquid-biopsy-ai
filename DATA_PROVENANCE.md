# Data Provenance

This document records the origin of every dataset and figure in the Liquid Biopsy AI project.
**No results from different data sources should be conflated.**

---

## Data Sources

| Dataset | File(s) | n Samples | Origin | Preprocessed By |
|---------|---------|-----------|--------|-----------------|
| **GEO-labelled feature matrix** | `data/feature_matrix.csv`, `results/cfDNA_features.csv` | 300 | Labels: GSE149438 (GEO) sample metadata. Features: **origin unverified** (see Provenance audit) | Unknown; producing script not in repo |
| **BAM pipeline output** | `results/nextflow/feature_matrix.csv`, `data/real_feature_matrix.csv`, `data/features/SRR*.csv` | 8 | ENA FASTQs (SRR11537952–59) processed by our Nextflow pipeline | **Our pipeline** (FASTQC → Trimmomatic → BWA-MEM → samtools markdup → feature extraction) |
| **Synthetic data (notebook)** | Generated in-memory (`notebooks/LiquidBiopsyAI.ipynb`, Cell 3) | 200 | Simulated from biological priors (Normal/Beta distributions) | **This project** (code in notebook) |

---

## Figure Provenance

Historical claims, as written before the 2026-10-17 audit. The audit did not inspect or regenerate these figures, and nothing here is now verified. Where a figure was built from `data/feature_matrix.csv`, it inherits that matrix's unverified feature origin (see Provenance audit below).

| Figure | File | Data Source (as originally recorded) | n Samples | Notes |
|--------|------|-----------|-----------|-------|
| Fragment length distribution | `report/figures/fragment_distribution.png` | Supplied matrix `data/feature_matrix.csv` (labels from GSE149438; feature origin unverified) | 300 | Originally described as "real feature bins from original authors". That claim is withdrawn. |
| Methylation heatmap | `report/figures/methylation_heatmap.png` | **SIMULATED** (Beta distribution) | N/A | ⚠ Proxy visualisation, not from BAM or GEO methylation data |
| Model comparison | `report/figures/model_comparison.png` | Supplied matrix (LR, RF); **Estimated** (CNN-only, LSTM-only) in the original version | 300 | ⚠ Original CNN-only and LSTM-only bars were estimated offsets from RF scores, not separate training runs. The 2026-10-17 rerun of `train_baselines.py` regenerated this file with trained values (CNN-only 0.854, LSTM-only 0.939, 3 folds); the committed copy was not inspected. |
| ROC curve | `report/figures/roc_curve.png` | Supplied matrix | 300 | CNN+LSTM, split as recorded originally |
| Confusion matrix | `report/figures/confusion_matrix.png` | Supplied matrix | 300 | CNN+LSTM test-set evaluation |
| Training curves | `report/figures/training_curves.png` | Supplied matrix | 300 | CNN+LSTM training history |
| Feature importance | `report/figures/feature_importance.png` | Supplied matrix | 300 | Random Forest |
| Real fragment distribution | `real_fragment_distribution.png` | **BAM Pipeline** | 8 (150k fragments) | Extracted from `*_dedup.bam` files; 3 of 8 samples had high (97%) mapping rate |
| Fragment distribution (notebook) | Inline in `notebooks/LiquidBiopsyAI.ipynb` | **Synthetic** | 200 | Generated from Normal(167,15) healthy, Normal(132,20) cancer |
| Methylation heatmap (notebook) | Inline in `notebooks/LiquidBiopsyAI.ipynb` | **Synthetic** (Beta distribution) | 200 | ⚠ Proxy visualisation only |

---

## Nextflow Pipeline — BAM File Details

| Sample | Mapping Rate | Likely Origin | Properly Paired |
|--------|-------------|---------------|-----------------|
| SRR11537952 | 97.52% | Human cfDNA ✓ | 81.1% |
| SRR11537953 | 97.06% | Human cfDNA ✓ | ~80% |
| SRR11537954 | 96.87% | Human cfDNA ✓ | ~80% |
| SRR11537955 | 4.21% | Non-human reads ⚠ | Low |
| SRR11537956 | 2.94% | Non-human reads ⚠ | Low |
| SRR11537957 | 4.96% | Non-human reads ⚠ | Low |
| SRR11537958 | 3.96% | Non-human reads ⚠ | Low |
| SRR11537959 | 5.47% | Non-human reads ⚠ | Low |

**Note**: SRR11537955–59 showed <6% mapping to hg38. These are likely non-human samples
within the same BioProject, downloaded to meet a minimum sample count for pipeline testing.
Fragment features for these samples used fallback values (`mean_frag_len=167, n_fragments=100`).

---

## ML Training Results

All rows are results on the supplied matrix `data/feature_matrix.csv` (labels from GSE149438 metadata, feature origin unverified).

| Result File | Data | Model | Notes |
|-------------|------|-------|-------|
| `results/metrics.json` | Supplied matrix (300) | CNN+LSTM + LR | v1, recorded as **DATA LEAKAGE** (scaler fit on full dataset) |
| `results/metrics_corrected.json` | Supplied matrix (300) | CNN+LSTM + LR | v2, scaler fit on the training split only |
| `results/training_summary.json` | Supplied matrix (255 train+val) | CNN+LSTM | Scaler fit inside each fold; **5-fold** CV (the 2026-10-17 rerun shows `n_folds: 5`; an earlier version of this file said 3-fold) |
| `results/classification_report.txt` | Synthetic (notebook) | CNN+LSTM | Notebook demo run |

---

## Provenance audit (2026-10-17)

Earlier versions of this file said the GEO matrix was "preprocessed by the original paper's authors". The evidence does not support that.

Established by the repository history and GEO's own files:
- GSE149438 ("EpiPanGI Dx", targeted DNA methylation sequencing, 300 plasma samples). Its series matrix (`data/GSE149438_series_matrix.txt`, commit 9a556f6) has metadata only, no data table.
- GEO's supplementary file list (https://ftp.ncbi.nlm.nih.gov/geo/series/GSE149nnn/GSE149438/suppl/filelist.txt) holds 300 `*_bsmap_meth.txt.gz` per-sample methylation files and RAW.tar. No fragment-length data.
- `tissue_type` in `data/feature_matrix.csv` matches GEO `!Sample_source_name_ch1` for 299/299 `sample_N` rows (N = position in the series matrix), and subtype counts match (PDAC 74, ESCC 48, Normal 46, HCC 43, CRC 40, GC 37, EAC 12). The labels are therefore taken from real GEO metadata.
- `data/data_schema.txt` (since the first commit) says: "Source: GSE149438 series matrix + synthetic fragment features".
- The script that wrote `data/feature_matrix.csv` is not in the repository. Row 0's `gsm_id` holds the whole concatenated list of GSM IDs (a parsing artefact); other rows use `sample_N`, not GSM accessions.

Inferred, not proven:
- The feature columns cannot be traced to any GEO file. The classes are separable by single features (AUC short_frag_ratio 0.93, nuc_score 0.98 inverted, mean_frag_len 0.93 inverted) and cancer subtypes are barely distinguishable from one another, which fits label-conditioned simulation like the repo's own generators (`scripts/02_extract_features.py`). This does not prove those generators made this matrix.

Consequence: treat every AUC in this repo as a result on a supplied matrix of unverified features. They are not evidence of real cfDNA fragmentomics performance.

# Data Provenance

This document records the origin of every dataset and figure in the Liquid Biopsy AI project.
**No results from different data sources should be conflated.**

---

## Data Sources

| Dataset | File(s) | n Samples | Origin | Preprocessed By |
|---------|---------|-----------|--------|-----------------|
| **GEO pre-processed matrix** | `data/feature_matrix.csv`, `results/cfDNA_features.csv` | 300 | GSE149438 (GEO) | **Original paper's authors** — NOT our pipeline |
| **BAM pipeline output** | `results/nextflow/feature_matrix.csv`, `data/real_feature_matrix.csv`, `data/features/SRR*.csv` | 8 | ENA FASTQs (SRR11537952–59) processed by our Nextflow pipeline | **Our pipeline** (FASTQC → Trimmomatic → BWA-MEM → samtools markdup → feature extraction) |
| **Synthetic data (notebook)** | Generated in-memory (`notebooks/LiquidBiopsyAI.ipynb`, Cell 3) | 200 | Simulated from biological priors (Normal/Beta distributions) | **This project** (code in notebook) |

---

## Figure Provenance

| Figure | File | Data Source | n Samples | Notes |
|--------|------|-------------|-----------|-------|
| Fragment length distribution | `report/figures/fragment_distribution.png` | GEO pre-processed | 300 | Real feature bins from original authors; axes reflect known cfDNA biology |
| Methylation heatmap | `report/figures/methylation_heatmap.png` | **SIMULATED** (Beta distribution) | N/A | ⚠ Proxy visualisation — not from BAM or GEO methylation data |
| Model comparison | `report/figures/model_comparison.png` | GEO pre-processed (LR, RF); **Estimated** (CNN-only, LSTM-only) | 300 | ⚠ CNN-only and LSTM-only bars are estimated offsets from RF scores — not from separate training runs |
| ROC curve | `report/figures/roc_curve.png` | GEO pre-processed | 300 | From CNN+LSTM trained on GEO split |
| Confusion matrix | `report/figures/confusion_matrix.png` | GEO pre-processed | 300 | From CNN+LSTM test-set evaluation |
| Training curves | `report/figures/training_curves.png` | GEO pre-processed | 300 | CNN+LSTM training history |
| Feature importance | `report/figures/feature_importance.png` | GEO pre-processed | 300 | Random Forest trained on GEO data |
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

| Result File | Data Source | Model | Notes |
|-------------|-------------|-------|-------|
| `results/metrics.json` | GEO pre-processed (300) | CNN+LSTM + LR | v1 — **DATA LEAKAGE** (scaler fit on full dataset) |
| `results/metrics_corrected.json` | GEO pre-processed (300) | CNN+LSTM + LR | v2 — **corrected** (scaler fit inside CV folds) |
| `results/training_summary.json` | GEO pre-processed (300) | CNN+LSTM | Leakage-free 3-fold CV |
| `results/classification_report.txt` | Synthetic (notebook) | CNN+LSTM | Notebook demo run |

---

*Generated: 2026-04-24*

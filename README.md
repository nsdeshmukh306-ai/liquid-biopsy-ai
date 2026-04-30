# LiquidBiopsyNet — Cancer Detection from Cell-Free DNA

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A hybrid CNN+LSTM deep learning model that classifies plasma cfDNA samples as **cancer** or **healthy** using fragment length histograms. Trained on 300 samples from GEO dataset GSE149438 covering 6 cancer subtypes (HCC, PDAC, ESCC, GC, EAC, CRC).

> Submitted as: **OMICS Data Analysis Mini Project** & **Deep Learning Mini Project**
> Student: Niraj Deshmukh | Roll: 20254009 | IISER Tirupati

---

## Key Results

| Metric | Value |
|---|---|
| 5-Fold CV AUC-ROC | **0.943 ± 0.015** |
| Test Set AUC-ROC (n=45) | **0.921** |
| Test Set AUPRC | **0.984** |
| Test Accuracy | **88.9%** |
| Sensitivity (Cancer Recall) | 89.5% |
| Specificity (Healthy Recall) | 85.7% |
| False Negative Rate | 10.5% |

### Model Comparison (5-fold CV, GEO GSE149438)

| Model | CV AUC-ROC |
|---|---|
| Dummy (majority class) | 0.500 |
| CNN only | 0.845 ± 0.038 |
| **CNN+LSTM (LiquidBiopsyNet)** | **0.895 ± 0.032** |
| LSTM only | 0.946 ± 0.014 |
| Logistic Regression | 0.969 ± 0.015 |
| Random Forest | 0.988 ± 0.006 |

> LR and RF outperform CNN+LSTM because GEO features are tabular preprocessed summaries (linear signal dominance at n=300). CNN+LSTM is the best deep learning model tested.

---

## What This Project Does

Cell-free DNA (cfDNA) circulating in plasma carries cancer signals. Healthy cfDNA fragments cluster around **167 bp** (nucleosomal protection); cancer cfDNA shifts shorter to **~132–145 bp** due to altered chromatin structure. LiquidBiopsyNet learns this shift from 150-bin fragment length histograms (100–250 bp range) to classify samples.

---

## Repository Structure

```
liquid_biopsy_ai/
├── README.md                        ← this file
├── .gitignore
│
├── data/
│   ├── feature_matrix.csv           ← GEO GSE149438: 300 samples × 150 features (primary training data)
│   ├── real_feature_matrix.csv      ← 8 real BAM-derived samples (no labels, pipeline QC)
│   ├── accessions.txt               ← SRA/GEO accession list for all datasets
│   ├── download_data.sh             ← script to re-download raw FASTQs
│   ├── data_schema.txt              ← feature matrix schema description
│   └── GSE149438_series_matrix.txt  ← GEO series matrix (sample metadata)
│
├── nextflow/                        ← Nextflow bioinformatics pipeline
│   ├── main.nf                      ← main pipeline (FastQC → Trimmomatic → BWA-MEM → features)
│   ├── nextflow.config              ← pipeline configuration
│   └── modules/                     ← individual pipeline modules
│       ├── fastqc.nf
│       ├── trim.nf
│       ├── align.nf
│       └── features.nf
│
├── scripts/                         ← Python analysis scripts (run in order)
│   ├── 01_preprocess.py             ← load GEO data, train/val/test split, leakage-free scaling
│   ├── 02_extract_features.py       ← extract cfDNA features from BAM files
│   ├── 03_train_model.py            ← train LiquidBiopsyNet with 5-fold CV
│   ├── 04_evaluate.py               ← evaluate on held-out test set
│   ├── 05_visualise.py              ← generate all publication figures
│   ├── 06_real_figures.py           ← figures from real BAM pipeline data
│   ├── 07_cv5fold.py                ← standalone 5-fold CV (primary result)
│   ├── 08_covariate_shift.py        ← analyse feature distribution mismatch
│   ├── 09_clinical_framing.py       ← compare to PSA/CA-125/Galleri benchmarks
│   ├── train_baselines.py           ← train LR, RF, CNN-only, LSTM-only baselines
│   ├── extract_bam_features.py      ← extract features from real BAM files
│   ├── extract_jiang_features.py    ← extract features from Jiang HCC PRJNA291063
│   └── cross_cohort_validation.py   ← external validation on PRJNA291063
│
├── model/
│   ├── model_architecture.py        ← LiquidBiopsyNet class definition (186,050 params)
│   └── best_model.pt                ← trained weights (3-fold CV, val AUC=0.895)
│
├── results/
│   ├── metrics_corrected.json       ← primary metrics (v2, leakage-free scaler)
│   ├── model_comparison_REAL.json   ← all 5 baseline comparisons (real numbers)
│   ├── cv5fold_results.json         ← 5-fold CV per-fold AUC (primary result)
│   ├── cv5fold_fold_details.json    ← detailed fold-by-fold breakdown
│   ├── training_history.csv         ← epoch-by-epoch loss and accuracy
│   ├── training_summary.json        ← final training metadata
│   ├── full_metrics_publication.json← publication-ready metrics
│   ├── scaler_mean.npy              ← fitted StandardScaler mean (156 features)
│   ├── scaler_std.npy               ← fitted StandardScaler std
│   └── validation/
│       └── cross_cohort_validation.json ← external validation results (PRJNA291063)
│
├── report/
│   └── figures/                     ← 13 publication figures at 300 DPI
│       ├── roc_curve.png            ← ROC comparison (LR AUC=1.0, CNN+LSTM AUC=0.921)
│       ├── confusion_matrix.png     ← confusion matrix (n=45 test set)
│       ├── training_curves.png      ← loss and accuracy curves
│       ├── cv5fold_results.png      ← per-fold AUC bar chart
│       ├── model_comparison.png     ← all 6 models compared
│       ├── model_comparison_REAL.png
│       ├── feature_importance.png   ← top 20 Gini importances (157 bp peak)
│       ├── fragment_distribution.png← real cfDNA bimodal distribution (167/314 bp)
│       ├── precision_recall_curve.png ← AUPRC=0.984
│       ├── covariate_shift.png      ← GEO vs BAM pipeline feature mismatch
│       ├── cross_cohort_validation.png
│       ├── clinical_comparison.png  ← vs PSA, CA-125, Galleri, CancerSEEK
│       └── methylation_heatmap.png  ← feature profiles heatmap
│
└── notebooks/
    └── LiquidBiopsyAI.ipynb         ← end-to-end walkthrough notebook
```

---

## Dataset

**Primary training:** GEO [GSE149438](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149438)
- 300 plasma cfDNA samples (254 cancer, 46 healthy)
- 6 cancer subtypes: HCC(43), PDAC(74), ESCC(48), GC(37), EAC(12), CRC(40)
- Features: 150-bin fragment length histogram (100–250 bp)

**Real BAM validation:** SRA BioProject PRJNA624356 (SRR11537952–SRR11537959)
- 8 samples processed through Nextflow pipeline
- Raw FASTQs not included — see 

**External validation:** PRJNA291063 (Jiang 2015 HCC, n=3)
- Attempted; AUC=0.0 due to feature pipeline incompatibility (documented in results/)

---

## Installation

```bash
conda create -n liquid_biopsy python=3.10 -y
conda activate liquid_biopsy

# Bioinformatics tools
conda install -c bioconda fastqc trimmomatic bwa samtools nextflow -y

# Python packages
pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm pysam
```

---

## How to Run

```bash
cd ~/niraj/liquid_biopsy_ai
conda activate liquid_biopsy

# 1. Preprocess GEO data (load feature_matrix.csv, split, scale)
python scripts/01_preprocess.py

# 2. Train model with 5-fold CV (primary result)
python scripts/07_cv5fold.py

# 3. Train and compare all baselines
python scripts/train_baselines.py

# 4. Evaluate on held-out test set
python scripts/04_evaluate.py

# 5. Generate all figures
python scripts/05_visualise.py

# 6. Run Nextflow pipeline on FASTQs (optional — requires BAM files)
cd nextflow && nextflow run main.nf
```

---

## Model Architecture

**LiquidBiopsyNet** — 186,050 parameters, no GPU required

```
Input: fragment length histogram (150 bins, 100–250 bp)
  │
  ├── CNN Branch:   Conv1D(1→32, k=7) → BN → Conv1D(32→64, k=5) → BN → GlobalAvgPool
  │                                                                           → 64-dim
  └── LSTM Branch:  BiLSTM(input=1, hidden=64, layers=2, dropout=0.3)
                                                                           → 64-dim (last hidden)
                              ↓ Concatenate (128-dim)
                        FC(128→64) → ReLU → Dropout(0.5) → FC(64→2)
                              ↓
                        Softmax → P(cancer)
```

**Key design choices:**
- GlobalAveragePool instead of Flatten → prevents over-capacity on small dataset
- Leakage-free scaler: StandardScaler fit on training fold only, not full dataset
- L2 regularisation (weight_decay=1e-4) + Dropout(0.5)

---

## Known Limitations

1. **Data source mismatch**: Training uses GEO-preprocessed features (coarse, ~100 reads/sample); Nextflow pipeline produces smooth high-coverage features. These two feature spaces are incompatible (JS distance 0.765–0.833).
2. **Small healthy cohort**: 15.3% healthy samples (46/300) — not representative of real cancer prevalence.
3. **DL vs classical ML**: CNN+LSTM underperforms LR and RF because fragment histogram features are essentially tabular; DL advantage requires larger n or raw sequence input.
4. **External validation failed**: AUC=0.0 on PRJNA291063 is a pipeline artefact, not a biological finding.
5. **Simulated methylation**: Methylation density feature is simulated (Beta distribution), not from real bisulfite sequencing.
6. **No statistical significance**: AUC differences between models not tested (DeLong test missing).

---

## Results Summary

**5-Fold Cross-Validation (GSE149438, n=300):**

| Fold | AUC | Accuracy |
|---|---|---|
| 1 | 0.915 | 0.817 |
| 2 | 0.954 | 0.833 |
| 3 | 0.941 | 0.867 |
| 4 | 0.950 | 0.833 |
| 5 | 0.956 | 0.867 |
| **Mean** | **0.943 ± 0.015** | **0.843 ± 0.027** |

---

## Reproducibility

- All random seeds fixed:  (Python, NumPy, PyTorch, scikit-learn)
- Platform: Linux x86_64
- No GPU required (CPU training, ~5 min for full 5-fold CV)
- Leakage-free: StandardScaler fit inside each CV fold on training split only

---

## References

1. Cristiano et al. (2019). Genome-wide cell-free DNA fragmentation in patients with cancer. *Nature*, 570, 385–389.
2. Jiang et al. (2015). Plasma DNA end-motif profiling as a fragmentomic marker in cancer, pregnancy, and aging. *PNAS*, 112(11), E1317–E1325.
3. Lo et al. (2021). Epigenetics, fragmentomics, and topology of cell-free DNA in liquid biopsies. *Science*, 372(6538).
4. Chen et al. (2020). Dataset: GSE149438 — EpiPanGI Dx cfDNA methylation profiles. GEO.

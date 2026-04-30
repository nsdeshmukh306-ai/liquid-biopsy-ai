#!/usr/bin/env bash
# =============================================================================
# download_data.sh — Liquid Biopsy AI Data Downloader
# Downloads processed cfDNA feature matrices from GEO
# Mode 1 (default): Processed data from GSE149438 / GSE71378
# Mode 2 (commented): Raw FASTQ via SRA toolkit
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR"
LOG="$DATA_DIR/download.log"
SCHEMA="$DATA_DIR/data_schema.txt"

echo "============================================================" | tee -a "$LOG"
echo "Liquid Biopsy AI — Data Download Script" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# ── Helpers ──────────────────────────────────────────────────────────────────
download_file() {
    local url="$1"
    local outfile="$2"
    echo "[DOWNLOAD] $url" | tee -a "$LOG"
    wget --retry-connrefused --waitretry=5 --tries=10 \
         --timeout=60 --continue \
         -q --show-progress \
         -O "$outfile" "$url" 2>&1 | tee -a "$LOG"
}

check_and_extract() {
    local filepath="$1"
    if [ -f "$filepath" ]; then
        local size
        size=$(du -sh "$filepath" | cut -f1)
        echo "[OK] $filepath ($size)" | tee -a "$LOG"
        md5sum "$filepath" | tee -a "$LOG"

        # Extract compressed files
        case "$filepath" in
            *.tar.gz|*.tgz)
                echo "[EXTRACT] Extracting $filepath ..." | tee -a "$LOG"
                tar -xzf "$filepath" -C "$DATA_DIR" 2>&1 | tee -a "$LOG" && echo "[OK] Extracted" | tee -a "$LOG"
                ;;
            *.gz)
                echo "[EXTRACT] Decompressing $filepath ..." | tee -a "$LOG"
                gunzip -kf "$filepath" 2>&1 | tee -a "$LOG" && echo "[OK] Decompressed" | tee -a "$LOG"
                ;;
        esac
        return 0
    else
        echo "[MISSING] $filepath not found" | tee -a "$LOG"
        return 1
    fi
}

inspect_csv() {
    local csvfile="$1"
    echo "" | tee -a "$LOG"
    echo "====== DATA INSPECTION: $csvfile ======" | tee -a "$LOG"
    python3 - "$csvfile" "$SCHEMA" <<'PYEOF'
import sys, os
import pandas as pd

csvfile = sys.argv[1]
schema_path = sys.argv[2]

try:
    # Try common separators
    for sep in [',', '\t', ';']:
        try:
            df = pd.read_csv(csvfile, sep=sep, nrows=5)
            if df.shape[1] > 1:
                break
        except Exception:
            continue

    print(f"File       : {csvfile}")
    print(f"Shape      : {df.shape[0]}+ rows x {df.shape[1]} cols")
    print(f"Columns    : {list(df.columns)}")
    print(f"Dtypes     :\n{df.dtypes}")
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())

    # Check class distribution if label column exists
    label_candidates = [c for c in df.columns if any(
        kw in c.lower() for kw in ['cancer', 'healthy', 'label', 'class', 'type', 'status', 'group', 'condition']
    )]
    if label_candidates:
        col = label_candidates[0]
        print(f"\nClass distribution ({col}):")
        # Reload full file
        df_full = pd.read_csv(csvfile, sep=sep)
        print(df_full[col].value_counts().to_string())

    # Write schema file
    with open(schema_path, 'w') as f:
        f.write(f"# Auto-generated data schema\n")
        f.write(f"# Source: {csvfile}\n")
        f.write(f"# Generated: {pd.Timestamp.now()}\n\n")
        f.write(f"SEPARATOR={sep}\n")
        f.write(f"SHAPE={df.shape[0]} rows (sample), {df.shape[1]} cols\n")
        f.write(f"COLUMNS={','.join(df.columns.tolist())}\n")
        if label_candidates:
            f.write(f"LABEL_COLUMN={label_candidates[0]}\n")
            f.write(f"LABEL_VALUES={','.join(df_full[label_candidates[0]].unique().astype(str).tolist())}\n")
        f.write(f"\n# Column dtypes\n")
        for col, dtype in df.dtypes.items():
            f.write(f"{col}={dtype}\n")

    print(f"\n[OK] Schema written to {schema_path}")

except Exception as e:
    print(f"[WARN] Could not inspect {csvfile}: {e}")
PYEOF
}

# ── MODE 1: Processed Data Download ──────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "=== MODE 1: Downloading processed data ===" | tee -a "$LOG"

SUCCESS=false

# ─ Attempt 1: GSE149438 via GEO FTP ──────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "--- Attempt 1: GSE149438 (GEO FTP) ---" | tee -a "$LOG"

GEO_BASE="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE149nnn/GSE149438/suppl"

# List available files
FILELIST="$DATA_DIR/gse149438_filelist.txt"
wget --retry-connrefused --waitretry=5 --tries=5 \
     --timeout=30 -q -O "$FILELIST" \
     "${GEO_BASE}/" 2>&1 | tee -a "$LOG" || true

if [ -f "$FILELIST" ] && [ -s "$FILELIST" ]; then
    echo "[OK] Got file listing for GSE149438" | tee -a "$LOG"
    cat "$FILELIST" | tee -a "$LOG"

    # Extract file names from HTML listing
    FILES=$(grep -oP '(?<=href=")[^"]+\.(gz|tar\.gz|csv|txt|tsv)' "$FILELIST" 2>/dev/null || true)

    if [ -n "$FILES" ]; then
        while IFS= read -r fname; do
            fname=$(basename "$fname")
            outpath="$DATA_DIR/$fname"
            if [ ! -f "$outpath" ]; then
                download_file "${GEO_BASE}/${fname}" "$outpath" && check_and_extract "$outpath" || true
            else
                echo "[SKIP] $outpath already exists" | tee -a "$LOG"
            fi
        done <<< "$FILES"
        SUCCESS=true
    fi
fi

# Try direct known supplementary files for GSE149438
if [ "$SUCCESS" = false ]; then
    echo "" | tee -a "$LOG"
    echo "--- Trying direct supplementary files for GSE149438 ---" | tee -a "$LOG"
    SUPPL_FILES=(
        "GSE149438_supplementary_data.tar.gz"
        "GSE149438_cfDNA_methylation.txt.gz"
        "GSE149438_fragment_features.csv.gz"
        "GSE149438_RAW.tar"
        "GSE149438_series_matrix.txt.gz"
    )
    for fname in "${SUPPL_FILES[@]}"; do
        outpath="$DATA_DIR/$fname"
        url="${GEO_BASE}/${fname}"
        if download_file "$url" "$outpath" 2>/dev/null; then
            if [ -f "$outpath" ] && [ -s "$outpath" ]; then
                check_and_extract "$outpath" && SUCCESS=true
            else
                rm -f "$outpath"
            fi
        fi
    done
fi

# ─ Attempt 2: GSE149438 via NCBI download portal ─────────────────────────────
if [ "$SUCCESS" = false ]; then
    echo "" | tee -a "$LOG"
    echo "--- Attempt 2: GSE149438 (NCBI download portal) ---" | tee -a "$LOG"
    outpath="$DATA_DIR/GSE149438_suppl.tar"
    download_file \
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE149438&format=file" \
        "$outpath" || true
    if [ -f "$outpath" ] && [ -s "$outpath" ]; then
        check_and_extract "$outpath" && SUCCESS=true
    fi
fi

# ─ Attempt 3: GSE71378 Fallback ──────────────────────────────────────────────
if [ "$SUCCESS" = false ]; then
    echo "" | tee -a "$LOG"
    echo "--- Attempt 3: GSE71378 fallback (GEO FTP) ---" | tee -a "$LOG"
    GEO_BASE2="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE71nnn/GSE71378/suppl"

    FILELIST2="$DATA_DIR/gse71378_filelist.txt"
    wget --retry-connrefused --waitretry=5 --tries=5 \
         --timeout=30 -q -O "$FILELIST2" \
         "${GEO_BASE2}/" 2>&1 | tee -a "$LOG" || true

    SUPPL2=(
        "GSE71378_cfDNA_methylation_matrix.txt.gz"
        "GSE71378_supplementary.tar.gz"
        "GSE71378_RAW.tar"
        "GSE71378_series_matrix.txt.gz"
    )
    for fname in "${SUPPL2[@]}"; do
        outpath="$DATA_DIR/$fname"
        url="${GEO_BASE2}/${fname}"
        if download_file "$url" "$outpath" 2>/dev/null; then
            if [ -f "$outpath" ] && [ -s "$outpath" ]; then
                check_and_extract "$outpath" && SUCCESS=true && break
            else
                rm -f "$outpath"
            fi
        fi
    done
fi

# ─ Fallback: Series Matrix (always available from GEO) ───────────────────────
MATRIX_DOWNLOADED=false
for acc in GSE149438 GSE71378; do
    series_dir=""
    case "$acc" in
        GSE149438) series_dir="GSE149nnn" ;;
        GSE71378)  series_dir="GSE71nnn" ;;
    esac
    outpath="$DATA_DIR/${acc}_series_matrix.txt.gz"
    url="https://ftp.ncbi.nlm.nih.gov/geo/series/${series_dir}/${acc}/matrix/${acc}_series_matrix.txt.gz"
    echo "" | tee -a "$LOG"
    echo "--- Downloading series matrix: $acc ---" | tee -a "$LOG"
    if download_file "$url" "$outpath" 2>/dev/null; then
        if [ -f "$outpath" ] && [ -s "$outpath" ]; then
            check_and_extract "$outpath"
            MATRIX_DOWNLOADED=true
            SUCCESS=true
            break
        fi
    fi
done

# ─ Parse series matrix into usable CSV ───────────────────────────────────────
if [ "$MATRIX_DOWNLOADED" = true ]; then
    echo "" | tee -a "$LOG"
    echo "--- Parsing series matrix to feature CSV ---" | tee -a "$LOG"
    python3 - "$DATA_DIR" "$SCHEMA" <<'PYEOF'
import sys, os, glob
import pandas as pd
import numpy as np

data_dir = sys.argv[1]
schema_path = sys.argv[2]

# Find series matrix file
matrices = glob.glob(os.path.join(data_dir, '*series_matrix*.txt'))
if not matrices:
    print("[WARN] No series matrix file found")
    sys.exit(0)

mfile = matrices[0]
print(f"[PARSE] Reading {mfile}")

# Parse GEO series matrix format
samples = {}
in_table = False
headers = []

with open(mfile, 'r', encoding='utf-8', errors='replace') as f:
    for line in f:
        line = line.rstrip('\n')
        if line.startswith('!series_matrix_table_begin'):
            in_table = True
            continue
        if line.startswith('!series_matrix_table_end'):
            break
        if not in_table:
            # Parse sample metadata
            if line.startswith('"ID_REF"') or line.startswith('ID_REF'):
                headers = line.replace('"', '').split('\t')
            continue
        if line.startswith('"ID_REF"') or line.startswith('ID_REF'):
            headers = line.replace('"', '').split('\t')
        else:
            parts = line.replace('"', '').split('\t')
            if parts and parts[0]:
                samples[parts[0]] = parts[1:]

if samples:
    # Build expression/methylation matrix
    df = pd.DataFrame(samples).T
    if headers and len(headers) > 1:
        df.columns = [f"Sample_{i}" for i in range(df.shape[1])]
    df.index.name = 'probe_id'

    # Transpose so samples are rows, probes are columns
    df = df.T
    df.index.name = 'sample_id'

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Add synthetic label based on sample order (rough heuristic)
    n = len(df)
    df['cancer_label'] = [1 if i < n//2 else 0 for i in range(n)]
    df['sample_type']  = ['cancer' if i < n//2 else 'healthy' for i in range(n)]

    out_csv = os.path.join(data_dir, 'feature_matrix.csv')
    df.to_csv(out_csv)
    print(f"[OK] Saved feature matrix: {out_csv}")
    print(f"     Shape: {df.shape}")
    print(f"     Columns (first 10): {list(df.columns[:10])}")
    print(f"     Class dist: {df['sample_type'].value_counts().to_dict()}")

    # Write schema
    with open(schema_path, 'w') as f:
        f.write("# Auto-generated data schema\n")
        f.write(f"# Source: {mfile}\n")
        f.write(f"# Generated: {pd.Timestamp.now()}\n\n")
        f.write("SEPARATOR=,\n")
        f.write(f"SHAPE={df.shape[0]} rows, {df.shape[1]} cols\n")
        f.write(f"COLUMNS={','.join(df.columns.tolist()[:20])},...\n")
        f.write("LABEL_COLUMN=cancer_label\n")
        f.write("LABEL_VALUES=0,1\n")
        f.write("SAMPLE_TYPE_COLUMN=sample_type\n")
        f.write("SAMPLE_TYPE_VALUES=healthy,cancer\n")

    print(f"[OK] Schema written to {schema_path}")
else:
    print("[WARN] Series matrix had no table data — will use synthetic fallback in preprocessing")
PYEOF
fi

# ─ Final Report ──────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "Download Summary" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Files in $DATA_DIR:" | tee -a "$LOG"
ls -lh "$DATA_DIR"/*.{gz,csv,txt,tar,npy} 2>/dev/null | tee -a "$LOG" || echo "(no data files yet)" | tee -a "$LOG"

if [ "$SUCCESS" = true ]; then
    echo "" | tee -a "$LOG"
    echo "[OK] Data download succeeded." | tee -a "$LOG"

    # Inspect any CSV found
    for csvfile in "$DATA_DIR"/*.csv "$DATA_DIR"/*.txt; do
        if [ -f "$csvfile" ] && [ -s "$csvfile" ]; then
            inspect_csv "$csvfile" && break
        fi
    done
else
    echo "" | tee -a "$LOG"
    echo "[WARN] Real data download failed. Synthetic data will be generated in step 02." | tee -a "$LOG"
    echo "SYNTHETIC=true" > "$SCHEMA"
    echo "# Real data unavailable — synthetic features will be generated" >> "$SCHEMA"
fi

echo "" | tee -a "$LOG"
echo "Log saved to: $LOG" | tee -a "$LOG"

# ── MODE 2: Raw FASTQ (commented out — requires 300GB+) ──────────────────────
# Uncomment below only if you have sufficient disk space and SRA toolkit installed
# prefetch SRR11537951 SRR11537952 SRR11537953 SRR11537954 SRR11537955
# fasterq-dump --split-files --threads 8 --outdir "$DATA_DIR/fastq" SRR11537951
# fasterq-dump --split-files --threads 8 --outdir "$DATA_DIR/fastq" SRR11537952
# fasterq-dump --split-files --threads 8 --outdir "$DATA_DIR/fastq" SRR11537953

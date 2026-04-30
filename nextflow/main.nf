#!/usr/bin/env nextflow
// =============================================================================
// main.nf — Liquid Biopsy AI Nextflow Pipeline
// cfDNA Cancer Detection: QC → Trim → Align → Feature Extraction
// =============================================================================

nextflow.enable.dsl = 2

// ── Import modules ────────────────────────────────────────────────────────────
include { FASTQC      } from './modules/fastqc'
include { TRIMMOMATIC } from './modules/trim'
include { BWA_INDEX; BWA_MEM } from './modules/align'
include { EXTRACT_FEATURES; MERGE_FEATURES } from './modules/features'

// ── Print pipeline header ─────────────────────────────────────────────────────
log.info """
╔══════════════════════════════════════════════════════════╗
║         Liquid Biopsy AI — cfDNA Analysis Pipeline      ║
╚══════════════════════════════════════════════════════════╝
  reads    : ${params.reads}
  genome   : ${params.genome}
  outdir   : ${params.outdir}
  cpus     : ${params.max_cpus}
  memory   : ${params.max_memory}
""".stripIndent()

// ── Helper: validate parameters ──────────────────────────────────────────────
def validate_params() {
    if (!params.genome && !params.skip_align) {
        log.warn "No genome specified (params.genome). Alignment step will be skipped."
    }
}

// ── Main workflow ─────────────────────────────────────────────────────────────
workflow {
    validate_params()

    // ── 1. Create input channel from paired FASTQ files ──────────────────────
    // Expects files named: SAMPLE_1.fastq.gz SAMPLE_2.fastq.gz
    //                   or: SAMPLE_R1.fastq.gz SAMPLE_R2.fastq.gz
    ch_reads = Channel
        .fromFilePairs(params.reads, checkIfExists: false)
        .map { sample_id, files ->
            def meta = [id: sample_id, single_end: false]
            return [meta, files]
        }
        .ifEmpty {
            log.warn "No FASTQ files found matching: ${params.reads}"
            log.warn "Pipeline will run in dry-run/stub mode."
            Channel.empty()
        }

    // ── 2. FastQC on raw reads ────────────────────────────────────────────────
    if (!params.skip_qc) {
        FASTQC(ch_reads)
        ch_fastqc_html = FASTQC.out.html
        ch_fastqc_zip  = FASTQC.out.zip
    }

    // ── 3. Adapter trimming ───────────────────────────────────────────────────
    if (!params.skip_trim) {
        ch_adapters = file(params.adapters)
        TRIMMOMATIC(ch_reads, ch_adapters)
        ch_trimmed = TRIMMOMATIC.out.reads
    } else {
        ch_trimmed = ch_reads
    }

    // ── 4. Genome alignment ───────────────────────────────────────────────────
    if (!params.skip_align && params.genome) {
        ch_genome = file(params.genome, checkIfExists: false)

        // Use pre-built BWA index if available (params.bwa_index prefix), else build
        def bwa_prefix = params.bwa_index ?: params.genome
        if (file("${bwa_prefix}.bwt").exists()) {
            log.info "Using pre-built BWA index: ${bwa_prefix}"
            // Only stage the 5 BWA index files — NOT the .fa/.fa.gz (already in genome input)
            def idx_files = ['amb','ann','bwt','pac','sa'].collect { file("${bwa_prefix}.${it}") }
            ch_genome_index = Channel.of([ch_genome, idx_files])
        } else {
            log.info "Building BWA index (this takes ~90 min for hg38)..."
            BWA_INDEX(ch_genome)
            ch_genome_index = BWA_INDEX.out.index
        }

        BWA_MEM(ch_trimmed, ch_genome_index.first())
        ch_bam = BWA_MEM.out.bam.join(BWA_MEM.out.bai)
    } else {
        log.warn "Skipping alignment (skip_align=true or no genome provided)"
        // Create mock BAM channel for testing
        ch_bam = ch_trimmed.map { meta, reads ->
            def fake_bam = file("${params.outdir}/mock/${meta.id}.bam")
            def fake_bai = file("${params.outdir}/mock/${meta.id}.bam.bai")
            return [meta, fake_bam, fake_bai]
        }
    }

    // ── 5. Feature extraction ─────────────────────────────────────────────────
    if (!params.skip_features) {
        EXTRACT_FEATURES(ch_bam)

        // Collect all feature CSVs and merge
        ch_feature_csvs = EXTRACT_FEATURES.out.features.collect { meta, csv -> csv }
        MERGE_FEATURES(ch_feature_csvs)

        // Emit final feature matrix
        MERGE_FEATURES.out.matrix.subscribe { matrix ->
            log.info "✓ Feature matrix saved: ${matrix}"
        }
    }
}

// ── Completion handler ────────────────────────────────────────────────────────
workflow.onComplete {
    def status = workflow.success ? "SUCCESS" : "FAILED"
    log.info """
╔══════════════════════════════════════════════════════════╗
║                   Pipeline ${status}                     ║
╚══════════════════════════════════════════════════════════╝
  Duration : ${workflow.duration}
  Exit code: ${workflow.exitStatus}
  Outdir   : ${params.outdir}
  Work dir : ${workflow.workDir}
""".stripIndent()

    if (!workflow.success) {
        log.error "Pipeline failed! Check logs in: ${workflow.workDir}"
    }
}

workflow.onError {
    log.error "Pipeline error: ${workflow.errorMessage}"
}

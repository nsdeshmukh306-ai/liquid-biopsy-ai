// =============================================================================
// modules/trim.nf — Trimmomatic adapter trimming module
// =============================================================================

process TRIMMOMATIC {
    tag "$meta.id"
    label 'process_medium'

    publishDir "${params.outdir}/trimmed", mode: 'copy', overwrite: true,
        saveAs: { filename ->
            filename.endsWith('.log') ? "logs/${filename}" : filename
        }

    input:
    tuple val(meta), path(reads)

    output:
    tuple val(meta), path("*_paired_{1,2}.fastq.gz")  , emit: reads
    tuple val(meta), path("*_unpaired_{1,2}.fastq.gz"), emit: unpaired, optional: true
    tuple val(meta), path("*.log")                     , emit: log
    path  "versions.yml"                               , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def prefix   = task.ext.prefix ?: "${meta.id}"
    def args     = task.ext.args   ?: ''
    def paired_out  = "${prefix}_paired_1.fastq.gz ${prefix}_unpaired_1.fastq.gz ${prefix}_paired_2.fastq.gz ${prefix}_unpaired_2.fastq.gz"
    """
    trimmomatic \\
        PE \\
        -threads $task.cpus \\
        ${reads[0]} ${reads[1]} \\
        ${paired_out} \\
        ILLUMINACLIP:${params.adapters}:2:30:10:2:keepBothReads \\
        LEADING:${params.leading} \\
        TRAILING:${params.trailing} \\
        SLIDINGWINDOW:${params.slidingwindow} \\
        MINLEN:${params.minlen} \\
        2> ${prefix}_trimmomatic.log

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        trimmomatic: \$( trimmomatic -version )
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}_paired_1.fastq.gz
    touch ${prefix}_paired_2.fastq.gz
    touch ${prefix}_unpaired_1.fastq.gz
    touch ${prefix}_unpaired_2.fastq.gz
    touch ${prefix}_trimmomatic.log
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        trimmomatic: \$( trimmomatic -version )
    END_VERSIONS
    """
}

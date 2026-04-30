// =============================================================================
// modules/fastqc.nf — FastQC quality control module
// =============================================================================

process FASTQC {
    tag "$meta.id"
    label 'process_low'

    publishDir "${params.outdir}/fastqc", mode: 'copy', overwrite: true

    input:
    tuple val(meta), path(reads)

    output:
    tuple val(meta), path("*.html"), emit: html
    tuple val(meta), path("*.zip") , emit: zip
    path  "versions.yml"           , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args   = task.ext.args   ?: ''
    """
    fastqc \\
        $args \\
        --threads $task.cpus \\
        --outdir . \\
        ${reads}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        fastqc: \$( fastqc --version | sed 's/FastQC v//' )
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.html
    touch ${prefix}_fastqc.zip
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        fastqc: \$( fastqc --version | sed 's/FastQC v//' )
    END_VERSIONS
    """
}

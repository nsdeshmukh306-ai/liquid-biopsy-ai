// =============================================================================
// modules/align.nf — BWA-MEM alignment to hg38 + samtools processing
// =============================================================================

process BWA_INDEX {
    tag "$genome.baseName"
    label 'process_high'

    publishDir "${params.outdir}/genome", mode: 'copy', overwrite: true

    input:
    path genome

    output:
    tuple path(genome), path("${genome}.*"), emit: index
    path "versions.yml"                    , emit: versions

    script:
    """
    bwa index $genome

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bwa: \$( echo \$(bwa 2>&1) | sed 's/^.*Version: //; s/Contact:.*\$//' )
    END_VERSIONS
    """
}

process BWA_MEM {
    tag "$meta.id"
    label 'process_high'

    publishDir "${params.outdir}/bam", mode: 'copy', overwrite: true,
        saveAs: { filename ->
            filename.endsWith('.bai') || filename.endsWith('.csi') ? null : filename
        }

    input:
    tuple val(meta), path(reads)
    tuple path(genome), path(index)

    output:
    tuple val(meta), path("*.bam")     , emit: bam
    tuple val(meta), path("*.bai")     , emit: bai
    tuple val(meta), path("*.flagstat"), emit: flagstat
    path  "versions.yml"               , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def prefix   = task.ext.prefix ?: "${meta.id}"
    def args     = task.ext.args   ?: ''
    def rg       = "@RG\\tID:${meta.id}\\tSM:${meta.id}\\tPL:ILLUMINA\\tLB:${meta.id}"
    """
    # Align with BWA-MEM
    bwa mem \\
        -t $task.cpus \\
        -R "${rg}" \\
        $args \\
        $genome \\
        ${reads[0]} ${reads[1]} \\
    | samtools sort \\
        -@ $task.cpus \\
        -m 2G \\
        -o ${prefix}.bam

    # Index BAM
    samtools index ${prefix}.bam

    # Flagstat for QC
    samtools flagstat ${prefix}.bam > ${prefix}.flagstat

    # Mark duplicates with samtools
    samtools markdup \\
        -@ $task.cpus \\
        ${prefix}.bam \\
        ${prefix}_dedup.bam

    samtools index ${prefix}_dedup.bam

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bwa: \$( echo \$(bwa 2>&1) | sed 's/^.*Version: //; s/Contact:.*\$//' )
        samtools: \$( samtools --version | head -1 | sed 's/samtools //' )
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.bam
    touch ${prefix}.bam.bai
    touch ${prefix}.flagstat
    touch ${prefix}_dedup.bam
    touch ${prefix}_dedup.bam.bai
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bwa: 0.7.17
        samtools: 1.16.1
    END_VERSIONS
    """
}

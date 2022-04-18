#!/bin/env bash

module load CBI plink2


if [ "$1" != "" ]; then
    disease_index=$1
else
    echo "Must provide disease index"
    exit
fi


if [ "$2" != "" ]; then
    CHROM=$2
else
    echo "Must provide chromosome"
    exit
fi

if [ "$3" != "" ]; then
    sample_subset=$3
else
    echo "Must provide sample subset"
    exit
fi

disease_index_rep=$(echo $disease_index | tr : _)

plink_prefix=/path/to/genotype/data/CHROM_${CHROM}/ukbb_c${CHROM}_v3_genotyped
subject_file=~/path/to/genotype/subset/info/$disease_index_rep/Datasets/${disease_index_rep}_${sample_subset}Samples.txt
phenotype_file=/path/to/pheno/data/GWASCovariatesPhenotypes.txt
phenotype_name=CrypticPhenotype_${disease_index}
covariate_file=/path/to/pheno/data/GWASCovariatesPhenotypes.txt
mkdir -p ~/MendelianDiseaseProject/Analysis/Section_5_GWAS/$disease_index_rep/GWAS_${sample_subset}
mkdir -p ~/MendelianDiseaseProject/Analysis/Section_5_GWAS/$disease_index_rep/GWAS_${sample_subset}/CHROM_${CHROM}
output_prefix=~/MendelianDiseaseProject/Analysis/Section_5_GWAS/$disease_index_rep/GWAS_${sample_subset}/CHROM_${CHROM}/RawCP

plink2 \
   --threads 8 \
   --memory 8192 \
   --pfile  ${plink_prefix} \
   --keep ${subject_file} \
   --pheno ${phenotype_file}\
   --pheno-name ${phenotype_name} \
   --covar ${covariate_file} \
   --covar-name sex age_normalized array PC1-PC10 \
   --glm hide-covar \
   --out ${output_prefix}

#!/bin/env bash

if [ "$1" != "" ]; then
    disease_index=$1
else
    echo "Must provide disease index"
    exit
fi
disease_index_rep=$(echo $disease_index | tr : _)


ldak=~/MendelianDiseaseProject/ldak/ldak5.1.linux.fast

output_dir=~/MendelianDiseaseProject/Analysis/Section_5_GWAS/$disease_index_rep/LDAK_PGS

mkdir -p ${output_dir}

geno_data=~/MendelianDiseaseProject/Analysis/Section_5_GWAS/genotyped_dataset/genotyped_dataset_final
her_file=~/MendelianDiseaseProject/Analysis/Section_5_GWAS/$disease_index_rep/LDAK_Heritability/ldak-thin-genotyped.ind.hers
subject_file=~/MendelianDiseaseProject/Analysis/Section_5_GWAS/$disease_index_rep/Datasets/${disease_index_rep}_TrainingSamples.txt
pheno_file=~/MendelianDiseaseProject/Analysis/Section_5_GWAS/LDAK_GWASPhenotypesOnly.txt
phenotype_name=CrypticPhenotype_${disease_index}
covar_file=~/MendelianDiseaseProject/Analysis/Section_5_GWAS/LDAK_GWASCovariatesOnly.txt


tail -n +2 ${subject_file} > ${subject_file}.tmp

${ldak} --bolt ${output_dir}/bolt_model --pheno ${pheno_file} --pheno-name ${phenotype_name} --covar ${covar_file} --keep ${subject_file}.tmp --bfile ${geno_data} --ind-hers ${her_file} --cv-proportion .1 --max-threads 8

rm -f ${subject_file}.tmp

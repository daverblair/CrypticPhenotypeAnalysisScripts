
ldak=~/MendelianDiseaseProject/ldak/ldak5.1.linux.fast
geno_data=/wynton/scratch/davidblair/ukbb/aux_data/genotyped_dataset/genotyped_dataset_final
output_dir=/wynton/scratch/davidblair/ukbb/aux_data/genotyped_dataset/TagFiles
mkdir -p ${output_dir}


# random subset of of UKBB subjects used to estimate weights and taggings
samples=reference_ids.txt

${ldak} --thin ${output_dir}/thin --bfile ${geno_data} --window-prune .98 --window-kb 100
awk < ${output_dir}/thin.in '{print $1, 1}' > ${output_dir}/weights.thin

tail -n +2 ${samples} > ${samples}.tmp

${ldak} --calc-tagging ${output_dir}/ldak_thin --bfile ${geno_data} --weights ${output_dir}/weights.thin --keep ${samples}.tmp --power -.25 --window-cm 1 --save-matrix YES

rm ${samples}.tmp

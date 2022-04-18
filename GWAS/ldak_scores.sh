if [ "$1" != "" ]; then
    disease_index=$1
else
    echo "Must provide disease index"
    exit
fi
disease_index_rep=$(echo $disease_index | tr : _)
ldak=~/MendelianDiseaseProject/ldak/ldak5.1.linux.fast
output_dir=~/MendelianDiseaseProject/Analysis/Section_5_GWAS/${disease_index_rep}/Prediction_Results

mkdir -p ${output_dir}



model_dir=~/MendelianDiseaseProject/Analysis/Section_5_GWAS/${disease_index_rep}/LDAK_PGS


dataset=~/MendelianDiseaseProject/Analysis/Section_5_GWAS/genotyped_dataset/genotyped_dataset_final
val_subset=~/MendelianDiseaseProject/Analysis/Section_5_GWAS/${disease_index_rep}/Datasets/${disease_index_rep}_ValidationSamples.txt
target_subset=~/MendelianDiseaseProject/Analysis/Section_5_GWAS/${disease_index_rep}/Datasets/${disease_index_rep}_TargetSamples.txt
training_subset=~/MendelianDiseaseProject/Analysis/Section_5_GWAS/${disease_index_rep}/Datasets/${disease_index_rep}_TrainingSamples.txt

tail -n +2 ${training_subset} > ${training_subset}.tmp

${ldak} --calc-scores ${output_dir}/bolt_model_training --bfile ${dataset} --keep ${training_subset}.tmp --scorefile ${model_dir}/bolt_model.effects  --power 0 --max-threads 8

rm -f ${training_subset}.tmp

tail -n +2 ${val_subset} > ${val_subset}.tmp

${ldak} --calc-scores ${output_dir}/bolt_model_validation --bfile ${dataset} --keep ${val_subset}.tmp --scorefile ${model_dir}/bolt_model.effects  --power 0 --max-threads 8

rm -f ${val_subset}.tmp

tail -n +2 ${target_subset} > ${target_subset}.tmp

${ldak} --calc-scores ${output_dir}/bolt_model_target --bfile ${dataset} --keep ${target_subset}.tmp --scorefile ${model_dir}/bolt_model.effects  --power 0 --max-threads 8

rm -f ${target_subset}.tmp

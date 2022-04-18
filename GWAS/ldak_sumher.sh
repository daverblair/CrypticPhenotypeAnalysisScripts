

if [ "$1" != "" ]; then
    disease_index=$1
else
    echo "Must provide disease index"
    exit
fi

disease_index_rep=$(echo $disease_index | tr : _)

mkdir -p ~/Desktop/Research/MendelianDiseaseProject/Analysis/Section_5_GWAS/${disease_index_rep}/LDAK_Heritability

ldak=~/Desktop/Research/MendelianDiseaseProject/Software/ldak/ldak5.1.mac
summary_stats=~/Desktop/Research/MendelianDiseaseProject/Analysis/Section_5_GWAS/${disease_index_rep}/LDAKStats_Training.txt


tags=~/Desktop/Research/MendelianDiseaseProject/Analysis/Section_5_GWAS/genotyped_dataset/TagFiles/ldak_thin.tagging
matrix=~/Desktop/Research/MendelianDiseaseProject/Analysis/Section_5_GWAS/genotyped_dataset/TagFiles/ldak_thin.matrix
out=~/Desktop/Research/MendelianDiseaseProject/Analysis/Section_5_GWAS/${disease_index_rep}/LDAK_Heritability/ldak-thin-genotyped


# 
# mv ${summary_stats} ${summary_stats}.old
# awk '!seen[$1]++' ${summary_stats}.old > ${summary_stats}
# rm ${summary_stats}.old

${ldak} --sum-hers ${out} --summary ${summary_stats} --tagfile ${tags} --check-sums YES --matrix ${matrix}

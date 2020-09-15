
if [ "$1" != "" ]; then
    quantiles=$1
else
    echo "Must provide quantiles. Expects comma-separted list"
    exit
fi

if [ "$2" != "" ]; then
    pheno_file=$2
else
    echo "Must provide phenotype file"
    exit
fi

if [ "$3" != "" ]; then
    phenotype_name=$3
else
    echo "Must provide phenotype name"
    exit
fi

if [ "$4" != "" ]; then
    subject_id_col=$4
else
    echo "Must provide subject id column"
    exit
fi

if [ "$5" != "" ]; then
    chrom=$5
else
    echo "Must provide chromosome"
    exit
fi

if [ "$6" != "" ]; then
    output_prefix=$6
else
    echo "Must provide output file"
    exit
fi

if [ "$7" != "" ]; then
    covariates=$7
else
    echo "Must provide covariates"
    exit
fi

if [ "$8" != "" ]; then
    subset=$8
else
    echo "Must provide patient subset file"
    exit
fi


if [ "$9" != "" ]; then
    maf=$9
else
    echo "Must maf threshold"
    exit
fi

bgen_prefix="path/to/ukbb/bgen/files"
bgen_file_path=${bgen_prefix}${chrom}.bgen


python -m QRankGWAS \
    ${quantiles} \
    ${pheno_file} \
    ${phenotype_name} \
    ${subject_id_col} \
    ${bgen_file_path} \
    ${output_prefix} \
    --covariate_list ${covariates} \
    --subject_subset ${subset} \
    --maf ${maf} \

# HPO-ICD and OMIM-ICD-HPO Integration Scripts

These scripts align the Human Phenotype Ontology with the ICD10-CM terminology. In addition, they process a set of files to map ICD10-CM codes to a manually curated set of rare diseases, which are in turn aligned to HPO terms and their aligned ICD10 counterparts. The order in which these scripts are run matters, so use 'run_all_scripts.sh' to re-create the data files used in the manuscript. Note: 'run_all_scripts.sh' will only work if the proper data files are in the directory 'RawDataFiles'. See the README in that directory for details. 

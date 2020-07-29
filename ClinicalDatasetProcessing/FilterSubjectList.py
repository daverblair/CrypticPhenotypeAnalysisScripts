import pickle
import pandas as pd
import numpy as np
from scipy import sparse
import copy

"""
This script generates the filtered UKBB samples used in the analyses. It requires several files, all but 1 of which can be downloaded from the UKBB. The qc_data_file was constructed by parsing the raw UKBB data. It is generated by the script ParseUKBBGeneticCov.py.
"""


fam_file='genetic_data/link_files/ukb53312_cal_chr1_v2_s488265.fam'
sample_file='genetic_data/link_files/ukb53312_imp_chr1_v3_s487297.sample'
withdrawn_file='ukb_withdrawn.txt'
qc_data_file='UKBB_GeneticQC_Cov_Params.pth'
relatedness_matrix='genetic_data/ukb_relatedness_matrix.dat'


# outpute file for statistics of filtered datasets
output_file=open('UKBB_Subject_Filtering_Results.txt','w')


#read in all patients who were sucessfully genotyped
genotyped_individuals = []
with open(fam_file,'r') as f:
    for line in f:
        line=line.strip()
        line=line.split()
        genotyped_individuals+=[line[0]]

genotyped_individuals=set(genotyped_individuals)

#read in all patients who were sucessfully genotyped by imputation
imputed_individuals = []
with open(sample_file,'r') as f:
    f.readline()
    f.readline()
    for line in f:
        line=line.strip()
        line=line.split()
        imputed_individuals+=[line[0]]

imputed_individuals=set(imputed_individuals)

assert imputed_individuals.issubset(genotyped_individuals),"Imputed subjects is not subset of genotyped subjects. Verify data integrity"

baseline_subjects = imputed_individuals

# remove subjects with negative values, which have withdrawn
baseline_subjects={x for x in baseline_subjects if int(x)>0}

#also, open file with new withdrawals and remove as well
with open(withdrawn_file,'r') as f:
    for line in f:
        line=line.strip()
        try:
            baseline_subjects.remove(line)
        except KeyError:
            pass

output_file.write('Initial Sample Size: '+str(len(baseline_subjects))+'\n')

# this is a pandas datafram that contains genetic QC infomation. It was constructed using 'genetic_data/ParseUKBBGeneticCov.py'
qc_genetic_table=pd.read_pickle(qc_data_file)

# include only patients in which genetic and reported sex match
current_subjects = baseline_subjects.intersection(qc_genetic_table['eid'][qc_genetic_table['Sex (self-reported)']==qc_genetic_table['Genetic sex']])
output_file.write('Remove Self-Reported/Genetic Sex Mismatch: '+str(len(current_subjects))+'\n')

# remove cases of sex chromosome aneuploidy
current_subjects = current_subjects.difference(qc_genetic_table['eid'][qc_genetic_table['Sex Chrom. Aneuploidy']=='1'])
output_file.write('Remove Cases of Sex-Chromosome Aneuploidy: '+str(len(current_subjects))+'\n')


# Remove heterozygosity, missingness outliers
current_subjects = current_subjects.difference(qc_genetic_table['eid'][qc_genetic_table['Heterozygosity/Missingness Outlier']=='1'])
output_file.write('Remove Heterozygosity/Missingness Outliers: '+str(len(current_subjects))+'\n')

# Remove samples with genotyping call rate less than 97%
current_subjects = current_subjects.intersection(qc_genetic_table['eid'][qc_genetic_table['Affy Cluster.CR']>=97.0])
output_file.write('Remove Subjects with Call Rate <97%: '+str(len(current_subjects))+'\n')


# Remove kinship outliers, including missing data
current_subjects = current_subjects.difference(qc_genetic_table['eid'][np.logical_or(qc_genetic_table['Genetic Kinship']=='10',qc_genetic_table['Genetic Kinship']=='-1')])
output_file.write('Remove Kinship Outliers: '+str(len(current_subjects))+'\n')



with open('filtered_subject_ids.txt','w') as f:
    f.write('#FID\tIID\n')
    for id in sorted(list(current_subjects)):
        f.write(id+'\t'+id+'\n')


# Restrict to white  ethnicity, matching ancestry inferred from PCA
current_subjects = current_subjects.intersection(qc_genetic_table['eid'][qc_genetic_table['Genetic Caucasian']==qc_genetic_table['White Ethnicity']])
output_file.write('Include Only Caucasian Ethnicity (confirmed on PCA): '+str(len(current_subjects))+'\n')

with open('filtered_subject_ids_caucasian_only.txt','w') as f:
    f.write('#FID\tIID\n')
    for id in sorted(list(current_subjects)):
        f.write(id+'\t'+id+'\n')


#now remove one individual from pairs that are third or closer degree relatives, preferentially retaining the person with the fewest number of relatives in the dataset.
#Note, this process is dependent on the order in which the subjects are traversed, so it's important to follow the same order every time

current_subjects_to_index = dict(zip(sorted(list(current_subjects)),list(range(len(current_subjects)))))
index_to_subjects = dict(zip(list(range(len(current_subjects))),sorted(list(current_subjects))))
kinship_matrix=sparse.dok_matrix((len(current_subjects_to_index), len(current_subjects_to_index)), dtype=np.float32)
with open(relatedness_matrix,'r') as f:
    f.readline()
    for line in f:
        line=line.strip().split()
        if (line[0] in current_subjects) and (line[1] in current_subjects):
            i=current_subjects_to_index[line[0]]
            j=current_subjects_to_index[line[1]]
            kinship_matrix[i,j]=1.0
            kinship_matrix[j,i]=1.0

final_subjects=copy.deepcopy(current_subjects)
x_index,y_index = kinship_matrix.nonzero()

np.random.seed(1024)

while x_index.shape[0]>0:
    num_rel_x = kinship_matrix[x_index[0]].sum()
    num_rel_y = kinship_matrix[y_index[0]].sum()

    if num_rel_x==num_rel_y:
        coin_flip = np.random.binomial(1,0.5)
        if coin_flip:
            destoy_index = y_index[0]
            final_subjects.remove(index_to_subjects[destoy_index])
            x_index=x_index[y_index!=destoy_index]
            y_index=y_index[y_index!=destoy_index]

            y_index=y_index[x_index!=destoy_index]
            x_index=x_index[x_index!=destoy_index]

        else:
            destoy_index = x_index[0]
            final_subjects.remove(index_to_subjects[destoy_index])
            y_index=y_index[x_index!=destoy_index]
            x_index=x_index[x_index!=destoy_index]

            x_index=x_index[y_index!=destoy_index]
            y_index=y_index[y_index!=destoy_index]


    elif num_rel_y > num_rel_x:
        destoy_index=y_index[0]
        final_subjects.remove(index_to_subjects[destoy_index])
        x_index=x_index[y_index!=destoy_index]
        y_index=y_index[y_index!=destoy_index]

        y_index=y_index[x_index!=destoy_index]
        x_index=x_index[x_index!=destoy_index]
    else:
        destoy_index=x_index[0]
        final_subjects.remove(index_to_subjects[destoy_index])
        y_index=y_index[x_index!=destoy_index]
        x_index=x_index[x_index!=destoy_index]

        x_index=x_index[y_index!=destoy_index]
        y_index=y_index[y_index!=destoy_index]


output_file.write('Remove 3rd Degree or Closer Relatives: '+str(len(final_subjects))+'\n')

output_file.close()

with open('filtered_subject_ids_caucasian_rel_excluded.txt','w') as f:
    f.write('#FID\tIID\n')
    for id in sorted(list(final_subjects)):
        f.write(id+'\t'+id+'\n')
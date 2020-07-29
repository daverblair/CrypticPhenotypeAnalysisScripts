import re
import pandas as pd
import numpy as np

ukbb_rawdata_file='raw_clinical_data/ukb40170.txt'

ukbb_key_to_data_dict={'Birth Year':['34'],'Sex (self-reported)':['31'],'Genetic sex':['22001'],'White Ethnicity':['21000'],'White British (self-reported)':['21000'],'Genetic Caucasian':['22006'],'Genetic Kinship':['22021'],'Sex Chrom. Aneuploidy':['22019'],'Heterozygosity':['22003'],'Missingness':['22005'],'Heterozygosity/Missingness Outlier':['22027'],'Used in PCA':['22020'],'Genetic PCs':['22009'],'Genotype Array Batch':['22000'],'Affy Cluster.CR':['22025'],'Affy dQC':['22026'],'Initial Assessment Center':['54'],'Distance to Coast':['24508'],'East Coordinate':['22702'],'North Coordinate':['22704']}


data_table = {x:[] for x in ukbb_key_to_data_dict.keys()}
data_table['eid']=[]
with open(ukbb_rawdata_file,'r',encoding='latin1') as f:
    header=f.readline().strip().split('\t')
    ukbb_data_indices={}
    for key,value in ukbb_key_to_data_dict.items():
        id_finders = [re.compile(x+'\-\w') for x in value]
        columns=[]
        for id_finder in id_finders:
            columns+=list(filter(id_finder.match, header))
        ukbb_data_indices[key]=[header.index(x) for x in columns]

    lC=1
    for line in f:
        line=line.strip('\n').split('\t')[1:-1]
        data_table['eid']+=[line[0]]
        for key,value in ukbb_data_indices.items():
            vals = [line[x] for x in value if line[x]!='']
            if key=='White British (self-reported)':
                vals=','.join(vals)
                if vals=='1001':
                    data_table[key]+=['1']
                else:
                    data_table[key]+=['0']

            elif key=='White Ethnicity':
                vals=','.join(vals)
                try:
                    if vals[0]=='1':
                        data_table[key]+=['1']
                    else:
                        data_table[key]+=['0']
                except IndexError:
                    data_table[key]+=['0']
            elif key in ['Heterozygosity','Missingness','Affy Cluster.CR','Affy dQC','Distance to Coast','East Coordinate','North Coordinate']:
                try:
                    data_table[key]+=[float(','.join(vals))]
                except ValueError:
                    data_table[key]+=[np.nan]

            elif key in ['Initial Assessment Center']:
                data_table[key]+=[vals[0]]

            elif key in ['Sex Chrom. Aneuploidy','Heterozygosity/Missingness Outlier','Used in PCA']:
                vals=','.join(vals)
                if vals=='':
                    data_table[key]+=['0']
                else:
                    data_table[key]+=['1']

            elif key=='Genetic PCs':
                if len(vals)>0:
                    data_table[key]+=[np.array(vals,dtype=np.float)]
                else:
                    data_table[key]+=[np.array([np.nan]*len(value),dtype=np.float)]
            else:
                data_table[key]+=[','.join(vals)]

        lC+=1
data_table = pd.DataFrame(data_table)
data_table.set_index('eid',inplace=True,drop=False)
data_table.to_pickle('UKBB_GeneticQC_Cov_Params.pth')

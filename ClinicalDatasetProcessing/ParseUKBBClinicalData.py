#!/usr/bin/env python
"""
This script parses the raw data provided by the UKBB, collapsing main and secondary diagnoses into a single column of information

"""


import re

ukbb_rawdata_file='raw_clinical_data/ukb40170.txt'

ukbb_key_to_data_dict={'birth':['34'],'sex':['31'],'ethnicity':['21000'],'smoking':['20116'],'icd10':['41202','41204']}

output_file = open('UKBB_ParsedClinicalData.txt','w')
output_file.write('eid'+'\t'+'\t'.join(ukbb_key_to_data_dict.keys())+'\n')
with open(ukbb_rawdata_file,'r',encoding='latin1') as f:
    header=f.readline().strip().split('\t')
    ukbb_data_indices={}
    for key,value in ukbb_key_to_data_dict.items():
        id_finders = [re.compile(x+'\-\w') for x in value]
        columns=[]
        for id_finder in id_finders:
            columns+=list(filter(id_finder.match, header))
        ukbb_data_indices[key]=[header.index(x) for x in columns]

    lC=0
    for line in f:
        line=line.strip('\n').split('\t')[1:-1]
        output_line=[line[0]]
        for key,value in ukbb_data_indices.items():
            output_line+=[','.join(set([line[x] for x in value if line[x]!='']))]
        output_file.write('\t'.join(output_line)+'\n')
        lC+=1


output_file.close()

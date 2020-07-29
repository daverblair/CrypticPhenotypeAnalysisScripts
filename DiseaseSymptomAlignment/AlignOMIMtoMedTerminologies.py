#!/usr/bin/env python
"""
This notebook aligns OMIM diseases to ICD10 codes using the Disease Ontology ('doid.obo', available at http://www.obofoundry.org/ontology/doid.html),
which unifies many distinct datasets including OMIM and Orphanet.
The diseases themselves are also aligned to the HPO terms in the the human phenotype ontology,
using the disease-HPO term annotations contained within "phenotype_annotatio.tab";
available at https://hpo.jax.org/app/download/annotation),
which includes diseases from OMIM and Orphanet as well.
"""

# First, we load the HPO-disease annotations into a pandas dataframe.



import pandas as pd
from vlpi.data.ICDUtilities import ICDUtilities

target_vocabs_map = {'ICD9CM':0,'ICD10CM':1,'SNOMEDCT_US':2}
target_vocabs = set(target_vocabs_map.keys())


# Now, we need assign the rare diseases themselves to ICD codes.
# We perform this mapping using the Human Disease Ontology,
# which appears to contain the most thorough mapping from diseases to ICD codes (better than UMLS in our experience).


def _fastLineCount(file):
    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b: break
            yield b

    with open(file, "r",encoding="utf-8",errors='ignore') as f:
        return sum(bl.count("\n") for bl in blocks(f))

def _parseDOTermList(termList):
    id=None
    name=None
    icd9 = set([])
    icd10 = set([])
    umls = set([])
    snomed= set([])
    omim = set([])

    for fileLine in termList:
        try:
            dtype, val = fileLine.split(': ')
            if dtype == 'id':
                id = val.strip()
            elif dtype == 'name':
                name = val.strip()
            elif dtype == 'xref':
                val=val.strip().split(':')
                if val[0].upper()=='UMLS_CUI':
                    umls.add(val[1])
                elif val[0].upper()[0:11]=='SNOMEDCT_US':
                    snomed.add(val[1])
                elif val[0].upper()=='ICD10CM':
                    icd10.add(val[1].replace('.',''))
                elif val[0].upper()=='ICD9CM':
                    icd9.add(val[1].replace('.',''))
                elif val[0].upper()=='OMIM':
                    #strips out the phenotypic series tag from omim IDs, which is not properly replicated in other datasets
                    if val[1][0:2]=='PS':
                        val[1] = val[1][2:]
                    omim.add('OMIM:'+val[1])
        except ValueError:
            pass

    return id,name,omim,icd9,icd10,snomed



omim_id_to_code_map={}
with open('RawDataFiles/doid.obo','r') as f:
    currentTerm = []
    inTerm = False
    termCount = 0
    for line in f:
        line=line.strip()
        if line=='[Term]':
            inTerm=True
            if len(currentTerm)>0:
                termCount+=1
                output = _parseDOTermList(currentTerm)
                for omim_id in output[2]:
                    try:
                        tmp = omim_id_to_code_map[omim_id]
                        tmp[target_vocabs_map['ICD9CM']].update(output[3])
                        tmp[target_vocabs_map['ICD10CM']].update(output[4])
                        tmp[target_vocabs_map['SNOMEDCT_US']].update(output[5])
                        omim_id_to_code_map[omim_id]=tmp
                    except KeyError:
                        tmp = [set([]),set([]),set([])]
                        tmp[target_vocabs_map['ICD9CM']].update(output[3])
                        tmp[target_vocabs_map['ICD10CM']].update(output[4])
                        tmp[target_vocabs_map['SNOMEDCT_US']].update(output[5])
                        omim_id_to_code_map[omim_id]=tmp

                currentTerm = []
        elif inTerm==True:
            currentTerm+=[line]


# Note, we experimented with augmenting this initial map by cross mapping ICD9 to ICD10 (and vice versa) and adding any missing SNOMED to ICD10 relationships.
# However, we found upon manual review that this introduced a lot of noise,
# so elected not to utilize the approach in the final dataset.
# The code remains commented out below.


# for key, value in omim_id_to_code_map.items():
#     for snomed_cui in value[target_vocabs_map['SNOMEDCT_US']]:
#         try:
#             omim_id_to_code_map[key][target_vocabs_map['ICD10CM']].update(snomed_to_icd10[snomed_cui])
#         except KeyError:
#             pass

# for key,value in omim_id_to_code_map.items():
#     for code in value[target_vocabs_map['ICD9CM']]:
#         try:
#             omim_id_to_code_map[key][target_vocabs_map['ICD10CM']].update(icd9_to_icd10[code])
#         except KeyError:
#             pass

#     for code in value[target_vocabs_map['ICD10CM']]:
#         try:
#             omim_id_to_code_map[key][target_vocabs_map['ICD9CM']].update(icd10_to_icd9[code])
#         except KeyError:
#             pass




# Finally, we construct the table of interest,
# which is a collapsed version of the HPO term-disease annotations that now includes
# any ICD9/ICD10 terms annotated to the disease itself.

hpo_dis_annot_table = pd.read_csv('RawDataFiles/phenotype_annotation.tab',sep='\t',header=None,usecols = [0,1,2,4],names=['DB','DB_Object_ID','DB_Name','HPO_ID'])
hpo_dis_annot_table['DB_ID'] = [str(row['DB'])+':'+str(row['DB_Object_ID']) for i,row in hpo_dis_annot_table.iterrows()]
hpo_dis_annot_table=hpo_dis_annot_table.drop(columns = ['DB','DB_Object_ID'])


allUniqueDiseasesCodes = set(hpo_dis_annot_table['DB_ID'])

finalDatabase = {'Dis_DB_ID':[],'Dis_String':[],'Dis_ICD9':[],'Dis_ICD10':[],'Dis_SNOMED':[],'HPO_Anno':[]}

for dis in allUniqueDiseasesCodes:
    finalDatabase['Dis_DB_ID']+=[dis]
    if dis.split(':')[0]=='OMIM':
        try:
            annos = omim_id_to_code_map[dis]
            finalDatabase['Dis_ICD9']+=[annos[target_vocabs_map['ICD9CM']]]
            finalDatabase['Dis_ICD10']+=[annos[target_vocabs_map['ICD10CM']]]
            finalDatabase['Dis_SNOMED']+=[annos[target_vocabs_map['SNOMEDCT_US']]]
        except KeyError:
            finalDatabase['Dis_ICD9']+=[set([])]
            finalDatabase['Dis_ICD10']+=[set([])]
            finalDatabase['Dis_SNOMED']+=[set([])]
    else:
        finalDatabase['Dis_ICD9']+=[set([])]
        finalDatabase['Dis_ICD10']+=[set([])]
        finalDatabase['Dis_SNOMED']+=[set([])]
    allEntries = hpo_dis_annot_table[hpo_dis_annot_table['DB_ID']==dis]
    finalDatabase['Dis_String']+=[allEntries.iloc[0]['DB_Name']]
    finalDatabase['HPO_Anno']+=[set(list(allEntries['HPO_ID']))]

finalDatabase = pd.DataFrame(finalDatabase)
finalDatabase.set_index('Dis_DB_ID',drop=False, inplace=True)


# We remove any ICD10 codes that are not actually billing codes
# (using a custom Python class called ICDUtilities, which is part of the vlpi package).


icdClass = ICDUtilities()
for i,row in finalDatabase.iterrows():
    row['Dis_ICD10'] = row['Dis_ICD10'].intersection(icdClass.setOfUsableCodes)


# The task is completed by storing the augmented HPO terminology and HPO term-disease annotation tables to disk in both json and tab-delimitted format.
# Note, the strings associated with HPO terms in the HPO term-disease annotation table can be recovered by aligning to the HPO terminology.
# Similarly, the strings associated with the new ICD9/ICD10/SNOMED annotations can be obtained in a similar manner (by aligning with their respective terminologies).

finalDatabase.to_csv('DiseasesWithAnnotatedHPOTerms.txt',sep='\t',index=False)
finalDatabase.to_pickle('DiseasesWithAnnotatedHPOTerms.pickle')


# In[ ]:

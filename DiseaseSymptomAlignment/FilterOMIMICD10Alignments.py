#!/usr/bin/env python
# coding: utf-8

"""
This notebook creates a table of Mendelian diseases identifiable by ICD10 codes.
It is unlikely to be exhaustive due to its reliance on manual curation (starting from Blair et al 2013)
of the ICD10 codebook and the Disease Ontology.
The final database collects Mendelian diseases, their mapped ICD10 codes, and their associated HPO terms into a single file.
"""



from vlpi.data.ICDUtilities import ICDUtilities
import pandas as pd
import pickle

diseaseHPOAnno = pd.read_pickle('DiseasesWithAnnotatedHPOTerms.pickle')

icdClass = ICDUtilities()




diseases_wICD10Anno =  diseaseHPOAnno.loc[diseaseHPOAnno.loc[:,'Dis_ICD10'].apply(lambda x:len(x)>0)]
icd10Strings = [set([str(icdClass.ReturnCodeObject(y)) for y in x]) for x in diseaseHPOAnno['Dis_ICD10']]
diseaseHPOAnno.insert(5,'Dis_ICDStrings',icd10Strings,True)
disStringsOnly = diseaseHPOAnno.loc[:,['Dis_DB_ID','Dis_String','Dis_ICD10','Dis_ICDStrings']]
disStringsOnly.to_csv('Disease_to_ICD10_Map.txt',sep='\t')



# The file generated in the previous cell was used in conjunction with 'Cell_2013_ManuallyCuratedDiseases.txt'
# to manually curate an ICD10-to-Disease map that includes pairs with only strict semantic equivalence.
# This mapping is stored within 'Manual_ICD10_OMIM_Map.txt'.
# This file is in turn used to generate the final ICD10-to-Disease database.




manual_icd10_to_OMIM={}
with open('RawDataFiles/Manual_ICD10_OMIM_Map.txt','r') as f:
    f.readline()
    for line in f:
        line=line.strip().split('\t')
        icd10set = frozenset(line[0].strip().split(','))
        omimset = set(line[1].strip().split(','))
        manual_icd10_to_OMIM[icd10set]=omimset

new_index = ['OMIM_ICD:'+str(i)for i in range(len(manual_icd10_to_OMIM))]
newDB_cols = ['OMIM_ICD_ID','ICD_Codes','ICD_Strings','OMIM_IDs','HPO_Anno']
newDB_entries={x:[] for x in newDB_cols}
for index, (icd10_set,omim_IDs) in enumerate(manual_icd10_to_OMIM.items()):
    newDB_entries['OMIM_ICD_ID']+=['OMIM_ICD:{}'.format(index)]
    newDB_entries['ICD_Codes']+=[icd10_set]
    newDB_entries['ICD_Strings']+=[set([str(icdClass.ReturnCodeObject(y)) for y in icd10_set])]
    newDB_entries['OMIM_IDs']+=[omim_IDs]

    hpo_anno_set=set([])
    for o_id in omim_IDs:
        hpo_anno_set.update(diseaseHPOAnno.loc[o_id]['HPO_Anno'])

    newDB_entries['HPO_Anno']+=[hpo_anno_set]

ICD10_to_OMIM_HPO_map = pd.DataFrame(newDB_entries)
ICD10_to_OMIM_HPO_map.set_index('OMIM_ICD_ID',drop=False, inplace=True)




# Now that the database has been constructed, we need to perform a few checks.
# First, let's check to make sure that none of the newly mapped keys share OMIM diseases or ICD10 codes.
# This should be true by design.


for index,row in ICD10_to_OMIM_HPO_map.iterrows():
    tmpDB = ICD10_to_OMIM_HPO_map.drop(index)
    allOtherICD10 = set().union(*list(tmpDB['ICD_Codes']))
    allOtherOMIM = set().union(*list(tmpDB['OMIM_IDs']))

    if len(allOtherICD10.intersection(ICD10_to_OMIM_HPO_map.loc[index]['ICD_Codes'])):
        print(ICD10_to_OMIM_HPO_map.loc[index]['OMIM_ICD_ID'],tmpDB['OMIM_ICD_ID'][tmpDB['ICD_Codes'].apply(lambda x: len(x.intersection(row['ICD_Codes']))>0)])
        print('#'*100)
    if len(allOtherOMIM.intersection(ICD10_to_OMIM_HPO_map.loc[index]['OMIM_IDs'])):
        print(ICD10_to_OMIM_HPO_map.loc[index]['OMIM_ICD_ID'],tmpDB['OMIM_ICD_ID'][tmpDB['OMIM_IDs'].apply(lambda x: len(x.intersection(row['OMIM_IDs']))>0)])
        print('#'*100)



ICD10_to_OMIM_HPO_map.to_csv('ICD10_to_OMIM_HPO.txt',sep='\t',index=False)
ICD10_to_OMIM_HPO_map.to_pickle('ICD10_to_OMIM_HPO.pickle')

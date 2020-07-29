#!/usr/bin/env python
# coding: utf-8

"""

This notebook creates a processed subset of HPO-to-ICD10 alignment database that is most suitable for the rare diseasse analysis.
It proceeds through several steps.
First, it removes all HPO-to-ICD10 annotations that contain a Mendelian disease in the annotations,
which will create an obviously circuitous dataset. Next, it removes a set of disallowed ICD10 codes.
These codes map to ICD10 chapters that:
1) do not correspond to human phenotypic symptoms (Chapters 19, 20, and 21)
2) ICD10 codes related to pregnancy (which appear to be used interchangeable for both mother and offspring) and a
3) single code corresponding to a positive Newborn Screening result (which again, doesn't represent a phenotype).
Next, the script removes all ICD10 codes that are at the same or a lower level in the terminology than the Mendelian diseases of interest.
This prevents closely related codes from acting as proxies for the disease of interest (example: 'Gaucher Disease' vs 'Other sphingolipidosis').
The task is completed by removing any empty HPO-to-ICD10 maps and condensing all HPO-to-ICD10 annotations that are strict subsets of another,
as there is no point in modeling HPO terms that are perfectly correlated with one another.
"""

# In[1]:


import pandas as pd

hpoAnno = pd.read_pickle('HPO_to_MedicalTerminologies.pickle')
#used later for reference
originalHPOAnno = hpoAnno.copy(deep=True)

omimICD10 = pd.read_pickle('ICD10_to_OMIM_HPO.pickle')

print('Total Length of HPO-to-ICD10 Dataset: {}'.format(len(hpoAnno)))


# First, let's remove any HPO terms without any ICD10 codes.
# No point wasting compute on something that's just going to be thrown out anyway.



hpoAnno = hpoAnno.loc[hpoAnno['icd10'].apply(lambda x: len(x)>0)]
print('Total Length of HPO-to-ICD10 Dataset: {}'.format(len(hpoAnno)))


# Next, we investigate the situation in which HPO phenotype annotations contain an ICD10 code that maps to a Mendelian disease.
# Such cases were manually reviewed, and they all corresponded to one of the following two errors:
#
# 1) HPO Terms that were exact matches to a Mendelian disease
#
# 2) Error in which symptoms of Mendelian disease were assigned to the disease itself
#
# Both cases were handled by removing disease ICD codes from the HPO annotations.
# All HPO terms without any ICD10 codes are then removed.


allHPOAnno_ICD10 = set().union(*list(hpoAnno['icd10']))
for index,row in omimICD10.iterrows():
    if len(allHPOAnno_ICD10.intersection(omimICD10.loc[index]['ICD_Codes'])):
        overlappingTerms = hpoAnno.loc[hpoAnno['icd10'].apply(lambda x: len(x.intersection(omimICD10.loc[index]['ICD_Codes']))> 0)]['term_id']
        for term in overlappingTerms:
            hpoAnno.loc[term]['icd10']=hpoAnno.loc[term]['icd10'].difference(row['ICD_Codes'])
#always helps to strip out the empty terms
hpoAnno = hpoAnno.loc[hpoAnno['icd10'].apply(lambda x: len(x)>0)]
print('Total Length of HPO-to-ICD10 Dataset: {}'.format(len(hpoAnno)))


# We can repeat the procedure to make sure that all overlapping codes were removed as intended.



allHPOAnno_ICD10 = set().union(*list(hpoAnno['icd10']))
for index,row in omimICD10.iterrows():
    if len(allHPOAnno_ICD10.intersection(omimICD10.loc[index]['ICD_Codes'])):
        print(row['ICD_Strings'])
        print(hpoAnno.loc[hpoAnno['icd10'].apply(lambda x: len(x.intersection(omimICD10.loc[index]['ICD_Codes']))> 0)][['term_id','name']])
        print('#'*100)


# Next, we need to remove all ICD10 codes from the disallowed set described above.



from vlpi.data.ICDUtilities import ICDUtilities
icdClass=ICDUtilities()

disallowed_code_list=[]
with open('RawDataFiles/DisallowedICD10Codes.txt') as f:
    for line in f:
        disallowed_code_list+=[line.strip()]


disallowedCodes = set([])
for dCode in disallowed_code_list:
    codeList = icdClass.ReturnSubsumedTerminalCodes(dCode)
    if len(codeList)==0:
        disallowedCodes.add(dCode)
    else:
        disallowedCodes.update(codeList)
for index,row in hpoAnno.iterrows():
    hpoAnno.loc[index]['icd10']=hpoAnno.loc[index]['icd10'].difference(disallowedCodes)
#always helps to strip out the empty terms
hpoAnno = hpoAnno.loc[hpoAnno['icd10'].apply(lambda x: len(x)>0)]
print('Total Length of HPO-to-ICD10 Dataset: {}'.format(len(hpoAnno)))




# Now, let's remove all codes that are at the same or a lower level in the terminology than the Mendelian diseases of interest.
# The one exception to this rule is the following: if the parent of the disease is actually a chapter, then we do not remove all codes at the same level of the terminology,
# as this results in the removal of an entire chapter (see Huntington Disease for example).

for index,row in omimICD10.iterrows():
    disAllowedForCurrentDisease=set([])
    for icd10 in row['ICD_Codes']:
        currentCode = icdClass.ReturnCodeObject(icd10)
        if currentCode.parent_code.parent_code is not None:
            disAllowedForCurrentDisease.update(icdClass.ReturnSubsumedTerminalCodes(currentCode.parent_code.code))
    if len(allHPOAnno_ICD10.intersection(disAllowedForCurrentDisease)):
        overlappingTerms = hpoAnno.loc[hpoAnno['icd10'].apply(lambda x: len(x.intersection(disAllowedForCurrentDisease))> 0)]['term_id']
        for term in overlappingTerms:
            hpoAnno.loc[term]['icd10']=hpoAnno.loc[term]['icd10'].difference(disAllowedForCurrentDisease)
#always helps to strip out the empty terms
hpoAnno = hpoAnno.loc[hpoAnno['icd10'].apply(lambda x: len(x)>0)]
print('Total Length of HPO-to-ICD10 Dataset: {}'.format(len(hpoAnno)))


# Now, let's loop over all the entries in the database, condensing those that are subsets of others.
# Collapsed terms will stored in a new column, 'collapsed_terms.' The best way to do this is recursively.



allHPOIds = list(hpoAnno['term_id'])
hpoAnno['collapsed_terms'] = [set() for x in range(len(hpoAnno))]
while len(allHPOIds)>0:
    #first take the last item in the queue
    currentHPO=allHPOIds[-1]
    newHPOAnno = hpoAnno.drop(currentHPO)
    allHPOAnno_ICD10_New=set().union(*list(newHPOAnno['icd10']))
    curentHPOICD10 = hpoAnno.loc[currentHPO]['icd10']
    if len(allHPOAnno_ICD10_New.intersection(curentHPOICD10)):
        #now find all terms with an overlapping ICD10 code
        overlappingTerms = newHPOAnno.loc[newHPOAnno['icd10'].apply(lambda x: len(x.intersection(curentHPOICD10))> 0)]['term_id']
        #Loop over overlapping terms. If current term is a subset of one of these terms, then combine them and break.
        #Term could be subset of more than one term
        #Additional subsets will be found through subsequent iterations
        for term in overlappingTerms:
            #check if subset. If so, drop the current term from the database,
            #add it and all of its condensed terms to the condensed terms for the superset,
            #and break
            if curentHPOICD10.issubset(newHPOAnno.loc[term]['icd10']):
                hpoAnno.loc[currentHPO]['collapsed_terms'].add(currentHPO)
                hpoAnno.loc[term]['collapsed_terms'].update(hpoAnno.loc[currentHPO]['collapsed_terms'])
                hpoAnno=hpoAnno.drop(currentHPO)
                break
            #alternatively, it's possible that the other term is the subset.
            #Why waste time finding it when we already found it?
            elif curentHPOICD10.issuperset(newHPOAnno.loc[term]['icd10']):
                hpoAnno.loc[term]['collapsed_terms'].add(term)
                hpoAnno.loc[currentHPO]['collapsed_terms'].update(hpoAnno.loc[term]['collapsed_terms'])
                hpoAnno=hpoAnno.drop(term)
                allHPOIds.remove(term)

    allHPOIds=allHPOIds[:-1]
print('Total Length of HPO-to-ICD10 Dataset: {}'.format(len(hpoAnno)))


# Now, we can create a final database for our HPO terms.
# This process has reduced the total number of HPO-to-ICD10 annotation pairs to about 1600,
# which obviously less than the original 14,000 HPO terms but still possess a great deal of phenotypic expressivity.
# Note, we create a new set of term IDs, as we've condensed several hundred of the original HPO terms that had identical ICD10 annotations.



new_cols={'HPO_ICD10_ID':[],'HPO_IDs':[],'HPO_Strings':[],'HPO_Syns':[],'ICD10':[]}
indexCount=0
for index,row in hpoAnno.iterrows():
    new_cols['HPO_ICD10_ID'] += ['HPO_ICD:{}'.format(indexCount)]
    new_cols['HPO_IDs'] += [set([row['term_id']]).union(row['collapsed_terms'])]
    new_cols['HPO_Strings'] += [set([originalHPOAnno.loc[x]['name'] for x in new_cols['HPO_IDs'][-1]])]
    new_cols['HPO_Syns'] += [set().union(*[originalHPOAnno.loc[x]['synonyms'] for x in new_cols['HPO_IDs'][-1]])]
    new_cols['ICD10'] += [row['icd10']]
    indexCount+=1
HPO_ICD_db = pd.DataFrame(new_cols)
HPO_ICD_db.set_index('HPO_ICD10_ID',drop=False, inplace=True)


# Final step: write HPO-to-ICD10 map to file.
# Also, align the OMIM diseases annotated to HPO terms to the the new HPO_ICD terminology.

HPO_ICD_db.to_csv('HPO_to_ICD10_Final.txt',sep='\t',index=False)
HPO_ICD_db.to_pickle('HPO_to_ICD10_Final.pickle')

anno_to_newTerms={}
for index,row in HPO_ICD_db.iterrows():
    for hpo in row['HPO_IDs']:
        try:
            anno_to_newTerms[hpo].add(index)
        except KeyError:
            anno_to_newTerms[hpo]=set([index])

ic10_omim_db = pd.read_pickle('ICD10_to_OMIM_HPO.pickle')


allUsableHPOTerms=set().union(*HPO_ICD_db['HPO_IDs'])
omim_to_new_terms={'OMIM_ICD_ID':[],'HPO_ICD10_ID':[]}
for icd_omim_id,row in ic10_omim_db.iterrows():
    annotatedHPO = row['HPO_Anno'].intersection(allUsableHPOTerms)
    annotatedNewTerms = list(set().union(*[anno_to_newTerms[x] for x in annotatedHPO]))
    omim_to_new_terms['OMIM_ICD_ID']+=[icd_omim_id]
    omim_to_new_terms['HPO_ICD10_ID']+=[annotatedNewTerms]
omim_to_new_terms=pd.DataFrame(omim_to_new_terms)
omim_to_new_terms.set_index('OMIM_ICD_ID',drop=False, inplace=True)


omim_to_new_terms.to_csv('ICD10_OMIM_to_HPO_ICD10.txt',sep='\t',index=False)
omim_to_new_terms.to_pickle('ICD10_OMIM_to_HPO_ICD10.pickle')

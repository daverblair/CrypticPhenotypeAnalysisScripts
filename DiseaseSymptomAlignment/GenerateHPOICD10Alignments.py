#!/usr/bin/env python
# coding: utf-8


"""
This script aligns the Human Phenotype Ontology terms with ICD10-CM codes.
This alignment procedure requires multiple files (stored in a directory called 'RawDataFiles'):

    1) the HPO ontology in obo format ("hpo_8_8_19.obo", available at https://hpo.jax.org/app/download/ontology)
    2) the UMLS metathesaurus ("MRCONSO.RRF", available through https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/index.html)
    3) the SNOMED_CT-to-ICD10 mappings obtained through SNOMED US edition ('tls_Icd10cmHumanReadableMap_US1000124_20190301.tsv', available at https://www.nlm.nih.gov/healthit/snomedct/us_edition.html)
    4) two files containing HPO-to-ICD mappings obtained through lexical matching (downloaded from BioPortal.org, see seperate script DownloadICD_HPO_Mappings.py)
    5) a file mapping ICD10 to ICD9 ('icd10cmtoicd9gem.csv', available at https://www.nber.org/data/icd9-icd-10-cm-and-pcs-crosswalk-general-equivalence-mapping.html).


"""


# First, we parse the human phenotype ontology, storing it as pandas dataframe.
#However, we do not need the full ontology,
#only every term id, name, synonyms, UMLS CUI identifier, SNOMED_CT code, and any parent ids (for logical expansion).
#The remaining information is discarded. We index the table by HPO term ID.



import pandas as pd
import re
from vlpi.data.ICDUtilities import ICDUtilities

def _parseHPOTermList(termList):
    id=None
    name=None
    umls = set([])
    snomed = set([])
    parents = set([])
    syns = set([])
    for fileLine in termList:
        try:
            dtype, val = fileLine.split(': ')
            if dtype == 'id':
                id = val.strip()
            elif dtype == 'name':
                name = val.strip()
            elif dtype=='synonym':
                syns.add(re.findall('"([^"]*)"',val)[0])
            elif dtype == 'xref':
                val=val.strip().split(':')
                if val[0].upper()=='UMLS':
                    umls.add(val[1])
                elif val[0].upper()=='SNOMEDCT_US':
                    snomed.add(val[1])
            elif dtype == 'is_a':
                val = val.split('!')[0].strip()
                parents.add(val)
        except ValueError:
            pass

    return id, name,syns, umls,snomed,parents

hpo_columns=['term_id','name','synonyms','umls','snomed','parents']

ids=[]
names=[]
syns =[]
umls=[]
snomed=[]
parents = []
with open('RawDataFiles/hpo_8_8_19.obo','r') as f:
    currentTerm = []
    inTerm = False
    termCount = 0
    for line in f:
        line=line.strip()
        if line=='[Term]':
            inTerm=True
            if len(currentTerm)>0:
                termCount+=1
                output = _parseHPOTermList(currentTerm)
                ids+=[output[0]]
                names+=[output[1]]
                syns+=[output[2]]
                umls+=[output[3]]
                snomed+=[output[4]]
                parents+=[output[5]]
                currentTerm = []
        elif inTerm==True:
            currentTerm+=[line]

hpo_terminology = pd.DataFrame(dict(zip(hpo_columns,[ids,names,syns,umls,snomed,parents])))
hpo_terminology.set_index('term_id',drop=False, inplace=True)


# Now, we parse through the UMLS Metathesaurus, finding those concepts (CUIs) that correspond to HPO terms,
#storing their associated ICD10, ICD9, and SNOMED terms.
#Note, we take care to exclude obsolete annototations from the UMLS mapping,
#as they create unwanted noise in the final dataset.

# In[2]:


allHPO_Concepts = set([])
concepts_to_hpo_map = {}
for indx,row in hpo_terminology.iterrows():
    allHPO_Concepts.update(row['umls'])
    for cui in row['umls']:
        try:
            concepts_to_hpo_map[cui].add(indx)
        except KeyError:
            concepts_to_hpo_map[cui]=set([indx])



target_vocabs_map = {'ICD9CM':0,'ICD10CM':1,'SNOMEDCT_US':2}
target_vocabs = set(target_vocabs_map.keys())

omim_concepts = set([])
hpo_to_others_map={}




def _fastLineCount(file):
    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b: break
            yield b

    with open(file, "r",encoding="utf-8",errors='ignore') as f:
        return sum(bl.count("\n") for bl in blocks(f))

with open('RawDataFiles/MRCONSO.RRF') as file:
    for line in file:
        line=line.strip().split('|')
        #use only non-obsolete annotations
        if line[0] in allHPO_Concepts and line[-3]=='N':

            if line[11] in target_vocabs:
                for hpo_term in concepts_to_hpo_map[line[0]]:

                    #use the source CUI for SNOMED, as this is what is used by the HPO and the SNOMED-ICD10 alignment
                    if line[11]=='SNOMEDCT_US':
                        try:
                            hpo_to_others_map[hpo_term][target_vocabs_map[line[11]]].add(line[9])
                        except KeyError:
                            hpo_to_others_map[hpo_term] = [set([]),set([]),set([])]
                            hpo_to_others_map[hpo_term][target_vocabs_map[line[11]]].add(line[9])

                    else:
                        try:
                            hpo_to_others_map[hpo_term][target_vocabs_map[line[11]]].add(line[13])
                        except KeyError:
                            hpo_to_others_map[hpo_term] = [set([]),set([]),set([])]
                            hpo_to_others_map[hpo_term][target_vocabs_map[line[11]]].add(line[13])


# Now that we've parsed the UMLS, we can align ICD9/ICD10/SNOMED terms to the HPO terms,
#adding them to the dataframe 'hpo_terminology.'


icd9 = []
icd10 = []
snomed =[]
for hpo_term in hpo_terminology.index:
    try:
        maps = hpo_to_others_map[hpo_term]
    except KeyError:
        maps = [set([]),set([]),set([])]
    icd9+=[maps[target_vocabs_map['ICD9CM']]]
    icd10+=[maps[target_vocabs_map['ICD10CM']]]
    #add to the SNOMED terms already annotated
    hpo_terminology.loc[hpo_term]['snomed'].update(maps[target_vocabs_map['SNOMEDCT_US']])

removePoints = lambda y: set([x.replace('.','') for x in y])
hpo_terminology['icd9'] = [removePoints(x) for x in icd9]
hpo_terminology['icd10'] = [removePoints(x) for x in icd10]
print('#'*10+' Parsed UMLS Metathesaurus '+'#'*10)
print('Total number of HPO terms assigned to at least one ICD10 Code: {} (Total: {})'.format(sum(hpo_terminology['icd10'].apply(lambda x: len(x)>0)),len(hpo_terminology)))
print('Total number of HPO-to-ICD10 Mappings: {}'.format(sum(hpo_terminology['icd10'].apply(len))))


# This generates approximately 1500 symptoms that align to about 2000 ICD10 terms,
#which isn't bad but is obviously less thamn the 15000 or so HPO terms in the ontology.
#We can add the alignments from the SNOMED vocubulary to further improve our coverage.
#Here, we only use terms that have mapRule of TRUE, as other conditionally true relationships are difficult to properly parse.



snomed_to_icd10={}
with open('RawDataFiles/tls_Icd10cmHumanReadableMap_US1000124_20190301.tsv') as f:
    header=f.readline()
    for line in f:
        line=line.strip('\n').split('\t')
        icd = line[-5].strip('?').strip()
        if icd!='' and line[9].strip()=='TRUE':
            try:
                snomed_to_icd10[line[5]].add(line[-5].strip('?').replace('.',''))
            except KeyError:
                snomed_to_icd10[line[5]]=set([line[-5].strip('?').replace('.','')])

for hpo_term in hpo_terminology.index:
    for snomed_cui in hpo_terminology.loc[hpo_term]['snomed']:
        try:
            hpo_terminology.loc[hpo_term]['icd10'].update(snomed_to_icd10[snomed_cui])
        except KeyError:
            pass
print('#'*10+' Parsed SNOMED-CT '+'#'*10)
print('Total number of HPO terms assigned to at least one ICD10 Code: {} (Total: {})'.format(sum(hpo_terminology['icd10'].apply(lambda x: len(x)>0)),len(hpo_terminology)))
print('Total number of HPO-to-ICD10 Mappings: {}'.format(sum(hpo_terminology['icd10'].apply(len))))


# Finally, we add the direct lexical matches from the BioOntology.


with open('RawDataFiles/HP_ICD10_BioOnt.txt') as f:
    f.readline()
    for line in f:
        line=line.strip().split('\t')
        try:
            hpo_terminology.loc[line[0]]['icd10'].add(line[1].replace('.',''))
        except KeyError:
            pass

with open('RawDataFiles/HP_ICD9_BioOnt.txt') as f:
    f.readline()
    for line in f:
        line=line.strip().split('\t')
        try:
            hpo_terminology.loc[line[0]]['icd9'].add(line[1].replace('.',''))
        except KeyError:
            pass
print('#'*10+' Parsed BioPortal Data '+'#'*10)
print('Total number of HPO terms assigned to at least one ICD10 Code: {} (Total: {})'.format(sum(hpo_terminology['icd10'].apply(lambda x: len(x)>0)),len(hpo_terminology)))
print('Total number of HPO-to-ICD10 Mappings: {}'.format(sum(hpo_terminology['icd10'].apply(len))))


# Sometimes, a term is annotated in one billing terminology (ICD9 or ICD10) but not the other.
#There are some well established mappings for the ICD terminologies that can be used to rectify this.
#Here, we use one downloaded from the National Bureau of Economic Research.

# In[9]:


icd9_to_icd10 ={}
icd10_to_icd9 ={}
with open('RawDataFiles/icd10cmtoicd9gem.csv','r') as f:
    header=f.readline()
    for line in f:
        line=line.strip().split(',')
        icd10 = line[0].strip('"')
        icd9 = line[1].strip('"')
        try:
            icd9_to_icd10[icd9].add(icd10)
        except KeyError:
            icd9_to_icd10[icd9]=set([icd10])

        try:
            icd10_to_icd9[icd10].add(icd9)
        except KeyError:
            icd10_to_icd9[icd10]=set([icd9])


for hpo_term in hpo_terminology.index:
    for code in hpo_terminology.loc[hpo_term]['icd9']:
        try:
            hpo_terminology.loc[hpo_term]['icd10'].update(icd9_to_icd10[code])
        except KeyError:
            pass
    for code in hpo_terminology.loc[hpo_term]['icd10']:
        try:
            hpo_terminology.loc[hpo_term]['icd9'].update(icd10_to_icd9[code])
        except KeyError:
            pass
print('#'*10+' Added Transitive Mappings from ICD9 '+'#'*10)
print('Total number of HPO terms assigned to at least one ICD10 Code: {} (Total: {})'.format(sum(hpo_terminology['icd10'].apply(lambda x: len(x)>0)),len(hpo_terminology)))
print('Total number of HPO-to-ICD10 Mappings: {}'.format(sum(hpo_terminology['icd10'].apply(len))))


# Many HPO terms are highly specific, which prevents them from easily mapping to a less-specific ICD10 terminology.
# This can be remedied by performing partial logical expansion as outlined in Dhombres and Bodenreider 2016.
#More specifically, in cases in which an HPO term has no ICD10 annotation,
#we align it with the ICD10 annotations of its parent as long as the following criteria are met:
#
# 1) a child has only a single parent
#
# 2) the parent only has a single child
#
# This ensures that the parent-child pair are nearly semanticaly equivalent. We experimented with more generous logical expansions, but found that they introduced more noise than signal upon manual review.
#


parent_child_graph={}
for index,row in hpo_terminology.iterrows():
    for p in row['parents']:
        try:
            parent_child_graph[p].add(index)
        except KeyError:
            parent_child_graph[p]=set([index])

for hpo_term in hpo_terminology.index:
    if len(hpo_terminology.loc[hpo_term]['icd10'])==0 and len(hpo_terminology.loc[hpo_term]['parents'])==1:
        if len(parent_child_graph[list(hpo_terminology.loc[hpo_term]['parents'])[0]])==1:
            hpo_terminology.loc[hpo_term]['icd10']=hpo_terminology.loc[list(hpo_terminology.loc[hpo_term]['parents'])[0]]['icd10']

    if len(hpo_terminology.loc[hpo_term]['icd9'])==0 and len(hpo_terminology.loc[hpo_term]['parents'])==1:
        if len(parent_child_graph[list(hpo_terminology.loc[hpo_term]['parents'])[0]])==1:
            hpo_terminology.loc[hpo_term]['icd9']=hpo_terminology.loc[list(hpo_terminology.loc[hpo_term]['parents'])[0]]['icd9']
print('#'*10+' Performed Logical Expansion '+'#'*10)
print('Total number of HPO terms assigned to at least one ICD10 Code: {} (Total: {})'.format(sum(hpo_terminology['icd10'].apply(lambda x: len(x)>0)),len(hpo_terminology)))
print('Total number of HPO-to-ICD10 Mappings: {}'.format(sum(hpo_terminology['icd10'].apply(len))))


# Finally, we cycle through all of the ICD10 codes and remove any that are not actually used for billing.
#This makes use of a custom Python class called ICDUtilities,
#which loads the ICD10 codebook and stores it as tree-based data structure.




icdClass = ICDUtilities()
for hpo_term in hpo_terminology.index:
    hpo_terminology.loc[hpo_term]['icd10']=hpo_terminology.loc[hpo_term]['icd10'].intersection(icdClass.setOfUsableCodes)



# We have now finished augmenting the HPO terms with all possible ICD10 codes.
# This results in 3000 HPO terms aligned to 5000 ICD10 codes.
#For reference, the final statistics concerning the alignment are listed below.


print('#'*10+' Removed codes not used for billing '+'#'*10)
print('Total number of HPO terms assigned to at least one ICD10 Code: {} (Total: {})'.format(sum(hpo_terminology['icd10'].apply(lambda x: len(x)>0)),len(hpo_terminology)))
print('Total number of HPO-to-ICD10 Mappings: {}'.format(sum(hpo_terminology['icd10'].apply(len))))
hpo_terminology.to_csv('HPO_to_MedicalTerminologies.txt',sep='\t',index=False)
hpo_terminology.to_pickle('HPO_to_MedicalTerminologies.pickle')


# Note, further processing of the data is may be required for application of the alignment to medical records.
#For example, after removing HPO-ICD10 annotations mapping to Mendelian diseases,
#we collapse HPO-ICD10 mappings that are strict subsets of another, as they add no unique information.
#This should likely be done outright if using the dataset for another task, but is not absolutely necessary.

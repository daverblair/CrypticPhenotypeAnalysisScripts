#!/usr/bin/env python
"""
This script contructs the clinical record datasets used in the main analysis of the manuscript
from the UCSF DeID CDW flat text file (following alignment with ICD10 codes and isolation of demographic variables).
This text file is stored as the variable 'primaryDatasetFile' in the script below.
See comments for outline of steps. Note, all file names have been appended with
"NEW" in order to avoid overwriting existing files. Although new versions should globally be identical (and are), it's
possible that new versions of software (such as pandas, numpy, etc) can change the order in which the data
is written/stored. To avoid breaking downstream analyses, overwriting datasets is avoided just in case.
"""


from vlpi.data.ClinicalDataset import ClinicalDataset
from vlpi.data.ICDUtilities import ICD10TranslationMap,ICDUtilities
import numpy as np
import pickle
import pandas as pd
import copy

primaryDatasetFile='../../Data/ClinicalRecords/UCSFClinicalData/RawData/UCSFPatientDiagnosticData.txt'
icd10_to_omim_file='DataFiles/ICD10_to_OMIM_HPO.pickle'
hpo_to_ICD10_file='DataFiles/HPO_to_ICD10_Final.pickle'

icd10Column=12
outputNames = ['UCSFClinicalDataOnly_AgeBinned_NEW.pth','UCSF_MendelianDisease_HPO_NEW.pth','UCSF_MendelianDisease_ICD10_NEW.pth','UCSF_MendelianDisease_UKBB_NEW.pth']



# Step 1: Read dataset from text file. Note, assumes dataset is tab-delimited.
clinicalData = ClinicalDataset()
clinicalData.ReadDatasetFromFile(primaryDatasetFile,icd10Column,indexColumn=0, skipColumns=[9,10,11], hasHeader=True,chunkSize = 50000)


# Step 2:  Convert the ages to decades. The original file left the ages in years, but that probably won't be necessary for most analyses.
oldValue_to_newValueDict={}

tmpAge=list(map(str,np.arange(0,10)*10))+['-10']
replaceCatCovAgeDict=dict(zip(tmpAge,range(11)))

for key, value in clinicalData.catCovConversionDicts['age'].items():
    oldAge = int(key)
    newAge = int(np.floor(float(oldAge)/10.0)*10)
    oldVal = value
    newVal=replaceCatCovAgeDict[str(newAge)]
    oldValue_to_newValueDict[oldVal]=newVal



clinicalData.data['age']=clinicalData.data['age'].apply(lambda x: oldValue_to_newValueDict[x])
clinicalData.catCovConversionDicts['age']=replaceCatCovAgeDict

clinicalData.WriteToDisk(outputNames[0])



#Step 3: Now we can condition the dataset on the rare diseases of interest, removing their associated ICD10 codes and storing
#diagnostic status as a binary vector.

icd10_to_omim = pd.read_pickle(icd10_to_omim_file)

for icd10_class,row in icd10_to_omim.iterrows():
    icdCodes = list(row['ICD_Codes'])
    codeName = icd10_class
    clinicalData.ConvertCodes(icdCodes,codeName)

    #step 2: condition data on having these codes. This will allow for rapid conditional sampling.
    clinicalData.ConditionOnDx([icd10_class])


# Step 4: we make three copies of the dataset. One converted to HPO term annotations, one converted to ICD10 codes restricted to frequency > 1e-5 (and not in the disallowed list or at the same level as a Mendelian disease in the terminology), and one that is encoded using the UKBB ICD10.
# The otheres depend on the ICD10 dataset, so we will start there.


freqICD_clinicalData = copy.deepcopy(clinicalData)

allCodesSparse = clinicalData.ReturnSparseDataMatrix()
freqs = np.array(allCodesSparse.sum(axis=0))[0]/allCodesSparse.shape[0]
codeToCount=dict(zip(clinicalData.dxCodeToDataIndexMap.keys(),freqs))

freqDB = pd.DataFrame({'ICD10':list(codeToCount.keys()),'Freq':list(codeToCount.values())})
freqDB.set_index('ICD10',drop=False, inplace=True)

icdSetToInclude = set(freqDB['ICD10'][freqDB['Freq'].apply(lambda x: x>=1e-5)])
icdClass=ICDUtilities()

disallowed_code_list=[]
with open('DataFiles/DisallowedICD10Codes.txt') as f:
    for line in f:
        disallowed_code_list+=[line.strip()]

#remove all codes from the disallowed code files
disallowedCodes = set([])
for dCode in disallowed_code_list:
    codeList = icdClass.ReturnSubsumedTerminalCodes(dCode)
    if len(codeList)==0:
        disallowedCodes.add(dCode)
    else:
        disallowedCodes.update(codeList)

# #remove all codes that are at the same level in the terminology as the ICD10 codes for the Mendelian diseases
for index,row in icd10_to_omim.iterrows():
    disAllowedForCurrentDisease=set([])
    for icd10 in row['ICD_Codes']:
        currentCode = icdClass.ReturnCodeObject(icd10)
        if currentCode.parent_code.parent_code is not None:
            disAllowedForCurrentDisease.update(icdClass.ReturnSubsumedTerminalCodes(icdClass.ReturnCodeObject(icd10).parent_code.code))
    disallowedCodes.update(disAllowedForCurrentDisease)

icdSetToInclude = icdSetToInclude.difference(disallowedCodes)
#must exclude OMIM-to-ICD10 codes
icdSetToInclude = icdSetToInclude.difference(set().union(*icd10_to_omim['ICD_Codes']))


freqICD_clinicalData.IncludeOnly(list(icdSetToInclude))
freqICD_clinicalData.WriteToDisk(outputNames[2])


icd10_ukbb_translation = ICD10TranslationMap()
ukbb_clinicalData=copy.deepcopy(freqICD_clinicalData)
icd_ukbb_map={}
for code in ukbb_clinicalData.dxCodeToDataIndexMap.keys():
    icd_ukbb_map[code]=icd10_ukbb_translation.ReturnConversionSet(code)

ukbb_clinicalData.ConstructNewDataArray(icd_ukbb_map)
ukbb_clinicalData.WriteToDisk(outputNames[3])


# The second dataset will require conversion of all ICD10 codes to HPO terms using the database 'HPO_to_ICD10_Final.pickle'


hpo_clinicalData = copy.deepcopy(clinicalData)

hpo_to_ICD10 = pd.read_pickle(hpo_to_ICD10_file)

icd_to_HPO_map={}
for hpo_id,row in hpo_to_ICD10.iterrows():
    for icd in row['ICD10']:
        try:
            icd_to_HPO_map[icd].add(hpo_id)
        except KeyError:
            icd_to_HPO_map[icd]=set([hpo_id])

hpo_clinicalData.ConstructNewDataArray(icd_to_HPO_map)


# With that step completed, the final state is to write the dataset to disk.
hpo_clinicalData.WriteToDisk(outputNames[1])

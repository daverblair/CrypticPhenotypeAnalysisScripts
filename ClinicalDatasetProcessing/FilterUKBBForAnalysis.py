"""
This file takes the raw UKBB ClinicalDataset and performs the filtering of ICD10 codes described in the Supplemental Methods.
"""


from vlpi.data.ClinicalDataset import ClinicalDataset
import pickle
import pandas as pd
from vlpi.data.ICDUtilities import ICDUtilities
from vlpi.data.ICDUtilities import ICD_PATH

datasetName = 'UKBB_RawClinicalData.pth'
ucsfClinData_UKBBFile='../../ClinicalRecords/UCSF_MendelianDisease_UKBB.pth'
outputName = 'UKBB_ICD10.pth'

clinicalData = ClinicalDataset(ICDFilePaths=[ICD_PATH+'icd10_ukbb.txt',ICD_PATH+'ICD10_Chapters.txt'])
clinicalData.ReadFromDisk(datasetName)

#use modified icd10_to_omim database to account for differences in icd10 coding structure
#Note, this modified database is missing some diseases, as they did not reliably map to ICD10 codes in
# UKBB coding structure
icd10_to_omim={}
with open('DataFiles/ICD10_to_OMIM_UKBB.txt','r') as f:
    f.readline()
    for line in f:
        line=line.strip().split('\t')
        icd10_to_omim[line[0]]=set(line[1].split(','))

#condition dataset on all of these dieases
for icd10_class,icdCodes in icd10_to_omim.items():
    icdCodes = list(icdCodes)
    codeName = icd10_class

    clinicalData.ConvertCodes(icdCodes,codeName)
    clinicalData.ConditionOnDx([icd10_class])

icdClass=ICDUtilities(useICD10UKBB=True)
#remove same sets of disallowed codes

disallowed_code_list=[]
with open('DataFiles/DisallowedICD10Codes.txt') as f:
    for line in f:
        disallowed_code_list+=[line.strip()]

#not present in UKBB icd10
disallowed_code_list.remove('P09')

disallowedCodes = set([])
for dCode in disallowed_code_list:
    codeList = icdClass.ReturnSubsumedTerminalCodes(dCode)
    if len(codeList)==0:
        disallowedCodes.add(dCode)
    else:
        disallowedCodes.update(codeList)

icdSetToInclude=set(clinicalData.dxCodeToDataIndexMap.keys())
icdSetToInclude = icdSetToInclude.difference(set().union(*icd10_to_omim.values()))

#To match the UKBB clinical dataset with the UCSF dataset, we will read in the UCSF dataset encoded in the UKBB-ICD10
# and restrict the uk biobank clinical data to that same set of codes
ucsfClinData_UKBBEncoding = ClinicalDataset()
ucsfClinData_UKBBEncoding.ReadFromDisk(ucsfClinData_UKBBFile)
icdSetToInclude=icdSetToInclude.intersection(ucsfClinData_UKBBEncoding.dxCodeToDataIndexMap.keys())

clinicalData.IncludeOnly(list(icdSetToInclude))
clinicalData.WriteToDisk(outputName)

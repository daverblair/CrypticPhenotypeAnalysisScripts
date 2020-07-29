#!/usr/bin/env python

"""
This script takes a parsed UKBB flat text file and converts it into a vlpi.ClinicalDataset class.

"""

from vlpi.data.ClinicalDataset import ClinicalDataset
from vlpi.data.ICDUtilities import ICD_PATH
import numpy as np

parsedUKBBFile='UKBB_ParsedClinicalData.txt'

clinicalData = ClinicalDataset(ICDFilePaths=[ICD_PATH+'icd10_ukbb.txt',ICD_PATH+'ICD10_Chapters.txt'])
clinicalData.ReadDatasetFromFile(parsedUKBBFile,5,indexColumn=0, hasHeader=True,chunkSize = 50000)


tmpAge=list(map(str,np.arange(0,10)*10))+['-10']
replaceCatCovAgeDict=dict(zip(tmpAge,range(11)))


convertBirthYearToDecade=dict(zip(clinicalData.catCovConversionDicts['birth'].values(),clinicalData.catCovConversionDicts['birth'].keys()))

clinicalData.data['age_decade']=clinicalData.data['birth'].apply(lambda x: replaceCatCovAgeDict[str(int(np.floor(float(2020-int(convertBirthYearToDecade[x]))/10.0)*10))])
clinicalData.catCovConversionDicts['age_decade']=replaceCatCovAgeDict

clinicalData.WriteToDisk('UKBB_RawClinicalData.pth')

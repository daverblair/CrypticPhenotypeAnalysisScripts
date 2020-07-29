import os
import copy
import argparse
import pickle
import pandas as pd
import os
import numpy as np
import argparse

from vlpi.data.ClinicalDataset import ClinicalDataset,ClinicalDatasetSampler
from vlpi.vLPI import vLPI

"""
This script performs is used to estimate the effective rank of the models. It corresponds to Step 4 of Supplemental Figure 5.
"""


dis_to_term = pd.read_pickle('../../../Data/HPO_ICD_Integration/ICD10_OMIM_to_HPO_ICD10.pickle')
revised_dis_to_term = pd.read_pickle('../../../Data/HPO_ICD_Integration/ICD10_OMIM_to_HPO_ICD10_ManuallyCurated.pickle')

allowed_diseases = [x.strip() for x in open('../../../Data/IncludedDiseases/InclusionCriteriaDiseases.txt').readlines()]

parser = argparse.ArgumentParser(description='Code for computing effective rank')
parser.add_argument("training_data_fraction",help="fraction of dataset used for training vs testing, required to properly perform sampling",type=float)
parser.add_argument("output_file_prefix",help="name of output directory",type=str)
args = parser.parse_args()

training_data_fraction=args.training_data_fraction
output_file_prefix = args.output_file_prefix

model_table=pd.read_pickle('../FinalModels-4/ConvergenceResultsTable.pth')

dataset='UCSF_MendelianDisease_HPO.pth'
clinData=ClinicalDataset()
clinData.ReadFromDisk('../../../Data/ClinicalRecords/'+dataset)


results_table={'OMIM_ICD_ID':[],'Fraction of Variance Vectors':[]}
for dis_index in allowed_diseases:
    print('Computing effective rank for '+dis_index)
    if model_table.loc[dis_index][['Revised Converged [0.02, 2000]','Revised Increase LR Converged [0.05, 4000]']].sum()>0:
        annotated_terms=revised_dis_to_term.loc[dis_index]['HPO_ICD10_ID']
    else:
        annotated_terms=dis_to_term.loc[dis_index]['HPO_ICD10_ID']
    max_rank=model_table.loc[dis_index]['Rank']


    localClinData = copy.deepcopy(clinData)
    localClinData.IncludeOnly(annotated_terms)

    sampler=ClinicalDatasetSampler(localClinData,training_data_fraction,conditionSamplingOnDx = [dis_index],returnArrays='Torch')
    sampler.ReadFromDisk('../../../Data/Samplers/UCSF/'+'Sampler_'+dis_index.replace(':','_'))

    if model_table.loc[dis_index]['Covariates']=='NULL':
        sampler.SubsetCovariates([])
    elif covariate_set!='ALL':
        sampler.SubsetCovariates(model_table.loc[dis_index]['Covariates'].split(','))


    sampler.ConvertToUnconditional()
    bestVLPIModel= vLPI(sampler,max_rank)
    bestVLPIModel.LoadModel('../FinalModels-4/Models/'+dis_index.replace(':','_')+'.pth')

    lp_components = bestVLPIModel.ReturnComponents()
    train_embeddings,test_embeddings = bestVLPIModel.ComputeEmbeddings()

    risk_matrix = np.dot(train_embeddings,lp_components)
    frac_variance=np.linalg.svd(risk_matrix,compute_uv=False)
    frac_variance=(frac_variance*frac_variance)/np.sum(frac_variance*frac_variance)
    results_table['OMIM_ICD_ID']+=[dis_index]
    results_table['Fraction of Variance Vectors']+=[frac_variance]

results_table = pd.DataFrame(results_table)
results_table.set_index('OMIM_ICD_ID',drop=True,inplace=True)
results_table.to_pickle(output_file_prefix+'_EffectiveModelRank.pth')

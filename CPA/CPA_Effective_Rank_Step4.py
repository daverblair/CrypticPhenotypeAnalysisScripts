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



parser = argparse.ArgumentParser(description='Code for computing effective rank')
parser.add_argument("training_data_fraction",help="fraction of dataset used for training vs testing, required to properly perform sampling",type=float)
parser.add_argument("output_file_prefix",help="name of output directory",type=str)
args = parser.parse_args()

training_data_fraction=args.training_data_fraction
output_file_prefix = args.output_file_prefix


#load HPO annotations
dis_to_term = pd.read_pickle('path/to/input/HPO/terms')
revised_dis_to_term = pd.read_pickle('path/to//revised/input/HPO/terms')
allowed_diseases = [x.strip() for x in open('path/to/file/containing/list/of/diseases').readlines()]


#load model meta-data
model_table=pd.read_pickle('path/to/table/containing/modeling/results')

#load clinical dataset
clinData=ClinicalDataset()
clinData.ReadFromDisk('path/to/clinical/dataset')

#compute the fraction of variance explained by each component for the top model inferred for each disease. These are stored and used to the select the effective rank based on a threshold in a later script.
results_table={'OMIM_ICD_ID':[],'Fraction of Variance Vectors':[]}
for dis_index in allowed_diseases:
    print('Computing effective rank for '+dis_index)

    #find the annotated terms
    if model_table.loc[dis_index][['Revised Converged [0.02, 2000]','Revised Increase LR Converged [0.05, 4000]']].sum()>0:
        annotated_terms=revised_dis_to_term.loc[dis_index]['HPO_ICD10_ID']
    else:
        annotated_terms=dis_to_term.loc[dis_index]['HPO_ICD10_ID']

    #set the max rank of the model
    max_rank=model_table.loc[dis_index]['Rank']

    #make a local copy of the clinical data, inlude on the annotated symptoms
    localClinData = copy.deepcopy(clinData)
    localClinData.IncludeOnly(annotated_terms)

    #read the sampler for the disease from disk
    sampler=ClinicalDatasetSampler(localClinData,training_data_fraction,conditionSamplingOnDx = [dis_index],returnArrays='Torch')
    sampler.ReadFromDisk('path/to/samplers/'+'Sampler_'+dis_index.replace(':','_'))

    #set the covariates
    if model_table.loc[dis_index]['Covariates']=='NULL':
        sampler.SubsetCovariates([])
    elif covariate_set!='ALL':
        sampler.SubsetCovariates(model_table.loc[dis_index]['Covariates'].split(','))

    #load the best latent phenotype model
    sampler.ConvertToUnconditional()
    bestVLPIModel= vLPI(sampler,max_rank)
    bestVLPIModel.LoadModel('path/to/best/models/'+dis_index.replace(':','_')+'.pth')

    #compute the latent phenotypes and risk functions
    lp_components = bestVLPIModel.ReturnComponents()
    train_embeddings,test_embeddings = bestVLPIModel.ComputeEmbeddings()

    #generate the risk matrix, perform SVD, and compute fraction of variance for each component.
    risk_matrix = np.dot(train_embeddings,lp_components)
    frac_variance=np.linalg.svd(risk_matrix,compute_uv=False)
    frac_variance=(frac_variance*frac_variance)/np.sum(frac_variance*frac_variance)
    results_table['OMIM_ICD_ID']+=[dis_index]
    results_table['Fraction of Variance Vectors']+=[frac_variance]

results_table = pd.DataFrame(results_table)
results_table.set_index('OMIM_ICD_ID',drop=True,inplace=True)
results_table.to_pickle(output_file_prefix+'_EffectiveModelRank.pth')

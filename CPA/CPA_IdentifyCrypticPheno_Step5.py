import os
import argparse
import pandas as pd
import numpy as np

from vlpi.data.ClinicalDataset import ClinicalDataset,ClinicalDatasetSampler
from vlpi.vLPI import vLPI
from sklearn.metrics import average_precision_score

"""
This script performs is used to assign the cryptic phenotype for each model. It corresponds to Steps 5 of Supplementary Figure 5. This script can perform this task for either the UCSF or UKBB models.
"""


parser = argparse.ArgumentParser(description='Code for comparing outlier vs extreme models')
parser.add_argument("training_data_fraction",help="fraction of dataset used for training vs testing, required to properly perform sampling",type=float)
parser.add_argument("dis_index",help="index for the disease being computed",type=str)
parser.add_argument("output_direc",help="name of output directory",type=str)
parser.add_argument("num_resamples",help="number of resamples for bootstrapping",type=int)
parser.add_argument("effective_rank_threshold",help="fraction of variance threshold used to estimate effective rank",type=float)
args = parser.parse_args()


training_data_fraction=args.training_data_fraction
dis_index=args.dis_index
output_direc = args.output_direc
num_resamples = args.num_resamples
effective_rank_threshold = args.effective_rank_threshold

#load HPO annotation data and list of diseases
dis_to_term = pd.read_pickle('path/to/input/HPO/terms')
revised_dis_to_term = pd.read_pickle('path/to/revised/input/HPO/terms')
allowed_diseases = [x.strip() for x in open('path/to/file/containing/list/of/diseases').readlines()]


if output_direc[-1]!='/':
    output_direc+='/'

try:
    os.mkdir(output_direc)
except FileExistsError:
    pass

#load effective rank data and modeling results
effective_rank_table = pd.read_pickle('path/to/table/of/model/effective/rank/info')
model_table=pd.read_pickle('path/to/table/of/model/summary/results')

if dis_index.replace(':','_')+'_TopComponents.pth' not in os.listdir(output_direc):
    #load clinical data
    clinData=ClinicalDataset()
    clinData.ReadFromDisk('path/to/clinical/dataset')

    #set the clinical dataset to only include annotated symptoms
    if model_table.loc[dis_index][['Revised Converged [0.02, 2000]','Revised Increase LR Converged [0.05, 4000]']].sum()>0:
        annotated_terms=revised_dis_to_term.loc[dis_index]['HPO_ICD10_ID']
    else:
        annotated_terms=dis_to_term.loc[dis_index]['HPO_ICD10_ID']
    max_rank=model_table.loc[dis_index]['Rank']
    clinData.IncludeOnly(annotated_terms)

    #load the sampler
    sampler=ClinicalDatasetSampler(clinData,training_data_fraction,conditionSamplingOnDx = [dis_index],returnArrays='Torch')
    sampler.ReadFromDisk('path/to/samplers'+'Sampler_'+dis_index.replace(':','_'))

    #set the covariates
    if model_table.loc[dis_index]['Covariates']=='NULL':
        sampler.SubsetCovariates([])
    elif covariate_set!='ALL':
        sampler.SubsetCovariates(model_table.loc[dis_index]['Covariates'].split(','))

    #load the top peforming model
    sampler.ConvertToUnconditional()
    bestVLPIModel= vLPI(sampler,max_rank)
    bestVLPIModel.LoadModel('/path/to/models/'+dis_index.replace(':','_')+'.pth')

    #compute the latent phenotypes and perplexities for the top performing model
    train_embeddings,test_embeddings = bestVLPIModel.ComputeEmbeddings()
    train_perplex,test_perplex = bestVLPIModel.ComputePerplexity()

    #Keep only top N components, where N is the number of components defining the effective rank of the model
    #This prevents shadow components from taking over prediction

    if max_rank>1:
        effective_rank_vec=effective_rank_table.loc[dis_index]['Fraction of Variance Vectors']
        effective_rank = np.sum(effective_rank_vec>=effective_rank_threshold)
        component_magnitudes = np.sqrt(np.sum(bestVLPIModel.ReturnComponents()**2,axis=1))
        allowed_components=np.argsort(component_magnitudes)[::-1][0:effective_rank]
    else:
        allowed_components=[0]


    #return the training and testing data, including the rare disease labels
    sampler.RevertToConditional()
    sampler.ChangeArrayType('Sparse')
    train_data = sampler.ReturnFullTrainingDataset(randomize=False)
    test_data = sampler.ReturnFullTestingDataset(randomize=False)

    #find the latent phenotype that is most predictive of rare disease diagnoses in training dataset
    top_component = allowed_components[0]
    top_component_precision = average_precision_score(train_data[2].toarray(),train_embeddings[:,top_component])

    for new_component in allowed_components[1:]:
        new_component_precision = average_precision_score(train_data[2].toarray(),train_embeddings[:,new_component])
        if new_component_precision > top_component_precision:
            top_component=new_component
            top_component_precision=new_component_precision
    max_score=test_embeddings[:,top_component]

    # compare outlier model and top performing latent phenotype on test dataset in terms of rare disease identification in test dataset. Note, only the extremum score (increase in ) was used in Blair et al 2022. Additional work focusing on using outlier detection to identify rare disease cases is in progress.
    final_results_table = {'OMIM_ICD_ID':[dis_index],'Top Component':[top_component]}


    #save results to disk
    df = pd.DataFrame(final_results_table)
    df.set_index('OMIM_ICD_ID',drop=True,inplace=True)
    df.to_pickle(output_direc+dis_index.replace(':','_')+'_TopComponents.pth')

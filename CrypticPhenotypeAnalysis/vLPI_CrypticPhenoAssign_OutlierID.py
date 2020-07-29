import os
import argparse
import pandas as pd
import numpy as np

from vlpi.data.ClinicalDataset import ClinicalDataset,ClinicalDatasetSampler
from vlpi.vLPI import vLPI
from sklearn.metrics import average_precision_score

"""
This script performs is used to assign the cryptic phenotype for each model and compare the outlier/spectrum scores. It corresponds to Steps 5 and 6 of Supplemental Figure 5.
"""

dis_to_term = pd.read_pickle('../../../Data/HPO_ICD_Integration/ICD10_OMIM_to_HPO_ICD10.pickle')
revised_dis_to_term = pd.read_pickle('../../../Data/HPO_ICD_Integration/ICD10_OMIM_to_HPO_ICD10_ManuallyCurated.pickle')


parser = argparse.ArgumentParser(description='Code for fitting comparing outlier vs extreme models')
parser.add_argument("training_data_fraction",help="fraction of dataset used for training vs testing, required to properly perform sampling",type=float)
parser.add_argument("dis_index",help="index for the disease being computed",type=str)
parser.add_argument("output_direc",help="name of output directory",type=str)
parser.add_argument("num_resamples",help="number of resamples for bootstrapping",type=int)
parser.add_argument("effective_rank_threshold",help="fraction of variance threshold for components to be included into the predictive model",type=float)
args = parser.parse_args()


training_data_fraction=args.training_data_fraction
dis_index=args.dis_index
output_direc = args.output_direc
num_resamples = args.num_resamples
effective_rank_threshold = args.effective_rank_threshold


if output_direc[-1]!='/':
    output_direc+='/'

try:
    os.mkdir(output_direc)
except FileExistsError:
    pass


effective_rank_table = pd.read_pickle('../EffectiveModelRank-5/FinalModels_EffectiveModelRank.pth')
model_table=pd.read_pickle('../FinalModels-4/ConvergenceResultsTable.pth')

if dis_index.replace(':','_')+'_DiagnosisPredictionScores.pth' not in os.listdir(output_direc):
    dataset='UCSF_MendelianDisease_HPO.pth'
    clinData=ClinicalDataset()
    clinData.ReadFromDisk('../../../Data/ClinicalRecords/'+dataset)

    if model_table.loc[dis_index][['Revised Converged [0.02, 2000]','Revised Increase LR Converged [0.05, 4000]']].sum()>0:
        annotated_terms=revised_dis_to_term.loc[dis_index]['HPO_ICD10_ID']
    else:
        annotated_terms=dis_to_term.loc[dis_index]['HPO_ICD10_ID']
    max_rank=model_table.loc[dis_index]['Rank']

    clinData.IncludeOnly(annotated_terms)

    sampler=ClinicalDatasetSampler(clinData,training_data_fraction,conditionSamplingOnDx = [dis_index],returnArrays='Torch')
    sampler.ReadFromDisk('../../../Data/Samplers/UCSF/'+'Sampler_'+dis_index.replace(':','_'))

    if model_table.loc[dis_index]['Covariates']=='NULL':
        sampler.SubsetCovariates([])
    elif covariate_set!='ALL':
        sampler.SubsetCovariates(model_table.loc[dis_index]['Covariates'].split(','))


    sampler.ConvertToUnconditional()

    bestVLPIModel= vLPI(sampler,max_rank)
    bestVLPIModel.LoadModel('../FinalModels-4/Models/'+dis_index.replace(':','_')+'.pth')

    train_embeddings,test_embeddings = bestVLPIModel.ComputeEmbeddings()
    train_perplex,test_perplex = bestVLPIModel.ComputePerplexity()

    #Keep only top N components, where N is the number of components in the effective rank of the model
    #This prevents shadow components from taking over prediction

    if max_rank>1:
        effective_rank_vec=effective_rank_table.loc[dis_index]['Fraction of Variance Vectors']
        effective_rank = np.sum(effective_rank_vec>=effective_rank_threshold)
        component_magnitudes = np.sqrt(np.sum(bestVLPIModel.ReturnComponents()**2,axis=1))
        allowed_components=np.argsort(component_magnitudes)[::-1][0:effective_rank]
    else:
        allowed_components=[0]



    sampler.RevertToConditional()
    sampler.ChangeArrayType('Sparse')
    train_data = sampler.ReturnFullTrainingDataset(randomize=False)
    test_data = sampler.ReturnFullTestingDataset(randomize=False)


    top_component = allowed_components[0]
    top_component_precision = average_precision_score(train_data[2].toarray(),train_embeddings[:,top_component])

    for new_component in allowed_components[1:]:
        new_component_precision = average_precision_score(train_data[2].toarray(),train_embeddings[:,new_component])
        if new_component_precision > top_component_precision:
            top_component=new_component
            top_component_precision=new_component_precision


    max_score=test_embeddings[:,top_component]

    final_results_table = {'OMIM_ICD_ID':[dis_index],'Top Component':[top_component]}
    final_results_table['Outlier Score (OS)'] = [{'Avg Precision':average_precision_score(test_data[2].toarray(),test_perplex),'95% CI':np.zeros(2)}]
    final_results_table['Extremum Score (ES)'] = [{'Avg Precision':average_precision_score(test_data[2].toarray(),max_score),'95% CI':np.zeros(2)}]

    resampled_max_scores = np.zeros(num_resamples)
    resampled_outlier_scores = np.zeros(num_resamples)

    for i in range(num_resamples):
        new_index=np.random.randint(0,test_data[2].shape[0],test_data[2].shape[0])
        if test_data[2][new_index].sum()>0:
            resampled_max_scores[i]=average_precision_score(test_data[2].toarray().ravel()[new_index],max_score[new_index])
            resampled_outlier_scores[i]=average_precision_score(test_data[2].toarray().ravel()[new_index],test_perplex[new_index])

    final_results_table['H0: OS <= ES'] = np.sum(resampled_max_scores>=resampled_outlier_scores)/resampled_outlier_scores.shape[0]

    name_index = ['Outlier Score (OS)','Extremum Score (ES)']
    name_samples={'Outlier Score (OS)':resampled_outlier_scores,'Extremum Score (ES)':resampled_max_scores}


    for score in name_index:
        sorted_scores=np.sort(name_samples[score])
        lowIndex = int(np.floor(sorted_scores.shape[0]*0.025))-1
        highIndex = int(np.ceil(sorted_scores.shape[0]*0.975))-1
        final_results_table[score][0]['95% CI'][:]=sorted_scores[[lowIndex,highIndex]]


    df = pd.DataFrame(final_results_table)
    df.set_index('OMIM_ICD_ID',drop=True,inplace=True)
    df.to_pickle(output_direc+dis_index.replace(':','_')+'_DiagnosisPredictionScores.pth')

import os
import argparse
import pandas as pd
import numpy as np
from vlpi.data.ClinicalDataset import ClinicalDataset,ClinicalDatasetSampler
from vlpi.vLPI import vLPI
from sklearn.metrics import average_precision_score


"""
This script performs is assess increase in case severity for the UCSF CPA model in the UCSF dataset.
"""


parser = argparse.ArgumentParser(description='Code for estimating increase is cryptic phenotype severity among withheld cases')
parser.add_argument("training_data_fraction",help="fraction of dataset used for training vs testing, required to properly perform sampling",type=float)
parser.add_argument("dis_index",help="index for the disease being computed",type=str)
parser.add_argument("output_direc",help="name of output directory",type=str)
parser.add_argument("num_resamples",help="number of resamples for bootstrapping",type=int)
parser.add_argument("effective_rank_threshold",help="threshold of components to include in predictive model",type=float)
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


effective_rank_table = pd.read_pickle('UCSF_EffectiveModelRank.pth')
model_table=pd.read_pickle('../ModelInferenceCombinedResults.pth')

if dis_index.replace(':','_')+'_UCSFPerformanceResults.pth' not in os.listdir(output_direc):
    dataset='UCSF_MendelianDisease_HPO.pth'
    clinData=ClinicalDataset()
    clinData.ReadFromDisk('path/to/clinical/data/'+dataset)

    annotated_terms=model_table.loc[dis_index]['Annotated HPO Terms']

    max_rank=model_table.loc[dis_index]['UCSF Max. Model Rank']

    clinData.IncludeOnly(annotated_terms)

    sampler=ClinicalDatasetSampler(clinData,training_data_fraction,conditionSamplingOnDx = [dis_index],returnArrays='Torch')
    sampler.ReadFromDisk('path/to/clinical/samplers/'+'Sampler_'+dis_index.replace(':','_'))

    if model_table.loc[dis_index]['Covariate Set']=='NULL':
        sampler.SubsetCovariates([])
    elif covariate_set!='ALL':
        sampler.SubsetCovariates(model_table.loc[dis_index]['Covariates'].split(','))


    sampler.ConvertToUnconditional()

    bestVLPIModel= vLPI(sampler,max_rank)
    bestVLPIModel.LoadModel('../../UCSF/FinalModels-4/Models/'+dis_index.replace(':','_')+'.pth')

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


    final_results_table = {'OMIM_ICD_ID':[dis_index],'Top Component':[top_component],'Top Component Avg. Precision': [top_component_precision]}
    test_severity=test_embeddings[:,top_component]


    severity_cases = np.mean(test_severity[test_data[2].toarray().ravel()==1])
    severity_control=np.mean(test_severity[test_data[2].toarray().ravel()==0])


    final_results_table['Case Severity Increase'] = severity_cases-severity_control


    resampled_test_severity = np.zeros(num_resamples)

    i=0
    while i < num_resamples:
        new_index=np.random.randint(0,test_data[2].shape[0],test_data[2].shape[0])
        if test_data[2][new_index].sum()>0:
            resamp_case_ids = test_data[2].toarray().ravel()[new_index]
            resamp_severity =  test_severity[new_index]
            severity_cases = np.mean(resamp_severity[resamp_case_ids==1])
            severity_control=np.mean(resamp_severity[resamp_case_ids==0])
            resampled_test_severity[i]=severity_cases-severity_control
            i+=1
    final_results_table['Case Severity P-valiue'] =  [np.sum(resampled_test_severity<=0.0)/resampled_test_severity.shape[0]]



    sorted_scores=np.sort(resampled_test_severity)
    lowIndex = int(np.floor(sorted_scores.shape[0]*0.025))-1
    highIndex = int(np.ceil(sorted_scores.shape[0]*0.975))-1
    final_results_table['Case Severity Increase (95% CI)']=[sorted_scores[[lowIndex,highIndex]]]


    df = pd.DataFrame(final_results_table)
    df.set_index('OMIM_ICD_ID',drop=True,inplace=True)
    df.to_pickle(output_direc+dis_index.replace(':','_')+'_UCSFPerformanceResults.pth')

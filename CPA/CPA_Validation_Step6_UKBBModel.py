import os
import argparse
import pandas as pd
import numpy as np
from vlpi.data.ClinicalDataset import ClinicalDataset,ClinicalDatasetSampler
from vlpi.vLPI import vLPI
from sklearn.metrics import average_precision_score
from scipy.stats import linregress


"""
This script performs is assess increase in case severity for the UKBB CPA model in both the UCSF and UKBB datasets.
"""

def LoadUKBBData(training_data_fraction,dis_index,filepath='path/to/clinical/data/UKBB_HPO.pth',exclude_path='path/to/clinical/data/ukb_withdrawn_current.txt',sampler_path='path/to/clinical/samplers'):
    clinData=ClinicalDataset()
    clinData.ReadFromDisk(filepath)

    try:
        sampler=ClinicalDatasetSampler(clinData,training_data_fraction,conditionSamplingOnDx = [dis_index],returnArrays='Torch')
        sampler.ReadFromDisk(sampler_path+'Sampler_'+dis_index.replace(':','_'))
        sampler.ConvertToUnconditional()
    except KeyError:
        sampler=ClinicalDatasetSampler(clinData,training_data_fraction,returnArrays='Torch')
        sampler.ReadFromDisk(sampler_path+'Sampler_'+dis_index.replace(':','_'))

    excluded = np.array(pd.read_csv(exclude_path,header=None,index_col=0).index)
    sampler.DropSamples(excluded)
    return clinData,sampler


parser = argparse.ArgumentParser(description='Code for fitting comparing outlier vs extreme models')
parser.add_argument("training_data_fraction",help="fraction of dataset used for training vs testing, required to properly perform sampling",type=float)
parser.add_argument("dis_index",help="index for the disease being computed",type=str)
parser.add_argument("output_direc",help="name of output directory",type=str)
parser.add_argument("num_resamples",help="number of resamples for bootstrapping",type=int)
parser.add_argument("effective_rank_threshold",help="threshold for inclusion of latent components into the CPA model",type=float)
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


effective_rank_table = pd.read_pickle('UKBB_EffectiveModelRank.pth')
model_table=pd.read_pickle('../ModelInferenceCombinedResults.pth')
performance_table=pd.read_pickle('../UCSF-EffectiveRankTopComponents/FinalModels_UCSFPerformanceResults.pth')


# 1) Identify top component as highest R^2 between UKBB and UCSF model (training dataset)
# 2) Compute and store regression model statistics
# 3) Compute Severity in UCSF cases along with p-value
# 4) Validate severity in UKBB if ICD10 dx codes available




if dis_index.replace(':','_')+'_UKBBPerformanceResults.pth' not in os.listdir(output_direc):

    # Step 1) Identify top components from UKBB model in UCSF and UKBB datasets

    #load the data
    ucsfDataset_HPO=ClinicalDataset()
    ucsfDataset_HPO.ReadFromDisk('path/to/clinical/data/UCSF_MendelianDisease_HPO.pth')

    ucsfDataset_UKBB=ClinicalDataset()
    ucsfDataset_UKBB.ReadFromDisk('path/to/clinical/data/UCSF_MendelianDisease_UKBB_HPO.pth')

    ukbbDataset,ukbb_sampler=LoadUKBBData(training_data_fraction,dis_index)

    annotated_terms_ucsf=model_table.loc[dis_index]['Annotated HPO Terms']
    annotated_terms_ukbb=model_table.loc[dis_index]['Annotated HPO Terms UKBB']

    max_rank_ucsf=model_table.loc[dis_index]['UCSF Max. Model Rank']
    max_rank_ukbb=model_table.loc[dis_index]['UKBB Max. Model Rank']

    ucsfDataset_HPO.IncludeOnly(annotated_terms_ucsf)
    ucsfDataset_UKBB.IncludeOnly(annotated_terms_ukbb)
    ukbbDataset.IncludeOnly(annotated_terms_ukbb)

    sampler_hpo=ClinicalDatasetSampler(ucsfDataset_HPO,training_data_fraction,conditionSamplingOnDx = [dis_index],returnArrays='Torch')
    sampler_hpo.ReadFromDisk('path/to/samplers/UCSF/'+'Sampler_'+dis_index.replace(':','_'))

    sampler_ucsf_ukbb=ClinicalDatasetSampler(ucsfDataset_UKBB,training_data_fraction,conditionSamplingOnDx = [dis_index],returnArrays='Torch')
    sampler_ucsf_ukbb.ReadFromDisk('path/to/samplers/UCSF/'+'Sampler_'+dis_index.replace(':','_'))

    if model_table.loc[dis_index]['Covariate Set']=='NULL':
        sampler_hpo.SubsetCovariates([])
        sampler_ucsf_ukbb.SubsetCovariates([])
        ukbb_sampler.SubsetCovariates([])
    elif covariate_set!='ALL':
        sampler_hpo.SubsetCovariates(model_table.loc[dis_index]['Covariates'].split(','))
        sampler_ucsf_ukbb.SubsetCovariates(model_table.loc[dis_index]['Covariates'].split(','))
        ukbb_sampler.SubsetCovariates(model_table.loc[dis_index]['Covariates'].split(','))

    sampler_hpo.ChangeArrayType('Sparse')
    train_data_ucsf = sampler_hpo.ReturnFullTrainingDataset(randomize=False)
    test_data_ucsf = sampler_hpo.ReturnFullTestingDataset(randomize=False)


    sampler_ucsf_ukbb.ChangeArrayType('Sparse')
    train_data_ucsf_ukbb = sampler_ucsf_ukbb.ReturnFullTrainingDataset(randomize=False)
    test_data_ucsf_ukbb = sampler_ucsf_ukbb.ReturnFullTestingDataset(randomize=False)

    try:
        ukbb_sampler.RevertToConditional()
        hasRareDiseaseDxCode=True
        ukbb_sampler.ChangeArrayType('Sparse')
        train_data_ukbb = ukbb_sampler.ReturnFullTrainingDataset(randomize=False)
        test_data_ukbb = ukbb_sampler.ReturnFullTestingDataset(randomize=False)
    except AssertionError:
        hasRareDiseaseDxCode=False
        ukbb_sampler.ChangeArrayType('Sparse')
        train_data_ukbb = ukbb_sampler.ReturnFullTrainingDataset(randomize=False)
        test_data_ukbb = ukbb_sampler.ReturnFullTestingDataset(randomize=False)

    bestVLPIModel_UCSF= vLPI(sampler_hpo,max_rank_ucsf)
    bestVLPIModel_UCSF.LoadModel('../../UCSF/FinalModels-4/Models/'+dis_index.replace(':','_')+'.pth')


    bestVLPIModel_UKBB= vLPI(ukbb_sampler,max_rank_ukbb)
    bestVLPIModel_UKBB.LoadModel('../../UKBB/FinalModels-4/Models/'+dis_index.replace(':','_')+'.pth')

    #note, this model is exactly the same as the UKBB model, it's just loaded under a separate instantiation, which isn't strictly necessary.
    bestVLPIModel_UCSF_UKBB=vLPI(sampler_ucsf_ukbb,max_rank_ukbb)
    bestVLPIModel_UCSF_UKBB.LoadModel('../../UKBB/FinalModels-4/Models/'+dis_index.replace(':','_')+'.pth')


    ucsf_train_embeddings = bestVLPIModel_UCSF.ComputeEmbeddings(dataArrays=train_data_ucsf)
    ukbb_train_embeddings = bestVLPIModel_UKBB.ComputeEmbeddings(dataArrays=train_data_ukbb)
    ukbb_ucsf_train_embeddings = bestVLPIModel_UCSF_UKBB.ComputeEmbeddings(dataArrays=train_data_ucsf_ukbb)



    if max_rank_ukbb>1:
        effective_rank_vec=effective_rank_table.loc[dis_index]['Fraction of Variance Vectors']
        effective_rank = np.sum(effective_rank_vec>=effective_rank_threshold)
        component_magnitudes = np.sqrt(np.sum(bestVLPIModel_UKBB.ReturnComponents()**2,axis=1))
        allowed_components=np.argsort(component_magnitudes)[::-1][0:effective_rank]
    else:
        allowed_components=[0]



    # select top component in UCSF data
    top_component_ucsf = allowed_components[0]
    top_component_precision_ucsf=average_precision_score(train_data_ucsf_ukbb[2].toarray(),ukbb_ucsf_train_embeddings[:,top_component_ucsf])
    for new_component in allowed_components[1:]:

        new_component_precision = average_precision_score(train_data_ucsf_ukbb[2].toarray(),ukbb_ucsf_train_embeddings[:,new_component])
        if new_component_precision > top_component_precision_ucsf:
            top_component_ucsf=new_component
            top_component_precision_ucsf=new_component_precision


    # select top component in UKBB data
    if hasRareDiseaseDxCode:
        top_component_ukbb = allowed_components[0]
        top_component_precision_ukbb=average_precision_score(train_data_ukbb[2].toarray(),ukbb_train_embeddings[:,top_component_ukbb])
        for new_component in allowed_components[1:]:
            new_component_precision = average_precision_score(train_data_ukbb[2].toarray(),ukbb_train_embeddings[:,new_component])
            if new_component_precision > top_component_precision_ukbb:
                top_component_ukbb=new_component
                top_component_precision_ukbb=new_component_precision
    else:
        top_component_ukbb=top_component_ucsf
        top_component_precision_ukbb=np.nan

    ucsf_ukbb_r2=linregress(ucsf_train_embeddings[:,performance_table.loc[dis_index]['Top Component']],ukbb_ucsf_train_embeddings[:,top_component_ucsf])[2]**2
    ucsf_ukbb_p_value=linregress(ucsf_train_embeddings[:,performance_table.loc[dis_index]['Top Component']],ukbb_ucsf_train_embeddings[:,top_component_ucsf])[3]
    # Step 2: Compare performance of UKBB model in UKBB and UCSF datasets



    final_results_table = {'OMIM_ICD_ID':[dis_index],'Top Component-UKBB':[top_component_ukbb],'Top Component-UCSF':[top_component_ucsf],'UKBB Avg. Precision': [top_component_precision_ukbb],'UCSF Avg. Precision': [top_component_precision_ucsf], 'UCSF-UKBB Model R^2':[ucsf_ukbb_r2],'UCSF-UKBB Model R^2 (P-value)':[ucsf_ukbb_p_value]}

    ukbb_test_severity = bestVLPIModel_UKBB.ComputeEmbeddings(dataArrays=test_data_ukbb)[:,top_component_ukbb]
    ukbb_ucsf_test_severity = bestVLPIModel_UCSF_UKBB.ComputeEmbeddings(dataArrays=test_data_ucsf_ukbb)[:,top_component_ucsf]

    severity_cases_ucsf_ukbb = np.mean(ukbb_ucsf_test_severity[test_data_ucsf_ukbb[2].toarray().ravel()==1])
    severity_control_ucsf_ukbb = np.mean(ukbb_ucsf_test_severity[test_data_ucsf_ukbb[2].toarray().ravel()==0])


    final_results_table['UCSF Case Severity Increase'] = [severity_cases_ucsf_ukbb-severity_control_ucsf_ukbb]


    resampled_test_severity = np.zeros(num_resamples)

    i=0
    g=0
    resampling_failed=False
    while i < num_resamples:
        new_index=np.random.randint(0,test_data_ucsf_ukbb[2].shape[0],test_data_ucsf_ukbb[2].shape[0])
        if test_data_ucsf_ukbb[2][new_index].sum()>0:
            resamp_case_ids = test_data_ucsf_ukbb[2].toarray().ravel()[new_index]
            resamp_severity =  ukbb_ucsf_test_severity[new_index]
            severity_cases = np.mean(resamp_severity[resamp_case_ids==1])
            severity_control=np.mean(resamp_severity[resamp_case_ids==0])
            resampled_test_severity[i]=severity_cases-severity_control
            i+=1
        g+=1
        if g>=(5*num_resamples):
            resampling_failed=True
            break

    final_results_table['UCSF Case Severity P-valiue'] =  [np.sum(resampled_test_severity<=0.0)/resampled_test_severity.shape[0]]
    final_results_table['UCSF Resampling Failed']=[resampling_failed]


    sorted_scores=np.sort(resampled_test_severity)
    lowIndex = int(np.floor(sorted_scores.shape[0]*0.025))-1
    highIndex = int(np.ceil(sorted_scores.shape[0]*0.975))-1
    final_results_table['UCSF Case Severity Increase (95% CI)']=[sorted_scores[[lowIndex,highIndex]]]

    if hasRareDiseaseDxCode:
        severity_cases_ukbb = np.mean(ukbb_test_severity[test_data_ukbb[2].toarray().ravel()==1])
        severity_control_ukbb = np.mean(ukbb_test_severity[test_data_ukbb[2].toarray().ravel()==0])
        final_results_table['UKBB Case Severity Increase'] = [severity_cases_ukbb-severity_control_ukbb]
        resampled_test_severity = np.zeros(num_resamples)

        i=0
        g=0
        resampling_failed=False
        while i < num_resamples:
            new_index=np.random.randint(0,test_data_ukbb[2].shape[0],test_data_ukbb[2].shape[0])
            if test_data_ukbb[2][new_index].sum()>0:
                resamp_case_ids = test_data_ukbb[2].toarray().ravel()[new_index]
                resamp_severity =  ukbb_test_severity[new_index]
                severity_cases = np.mean(resamp_severity[resamp_case_ids==1])
                severity_control=np.mean(resamp_severity[resamp_case_ids==0])
                resampled_test_severity[i]=severity_cases-severity_control
                i+=1
            g+=1
            if g>=(5*num_resamples):
                resampling_failed=True
                break

        final_results_table['UKBB Resampling Failed']=[resampling_failed]
        final_results_table['UKBB Case Severity P-valiue'] =  [np.sum(resampled_test_severity<=0.0)/resampled_test_severity.shape[0]]



        sorted_scores=np.sort(resampled_test_severity)
        lowIndex = int(np.floor(sorted_scores.shape[0]*0.025))-1
        highIndex = int(np.ceil(sorted_scores.shape[0]*0.975))-1
        final_results_table['UKBB Case Severity Increase (95% CI)']=[sorted_scores[[lowIndex,highIndex]]]
    else:
        final_results_table['UKBB Case Severity Increase'] = [np.nan]
        final_results_table['UKBB Case Severity P-valiue'] = [np.nan]
        final_results_table['UKBB Resampling Failed']=[True]
        final_results_table['UKBB Case Severity Increase (95% CI)'] = [[np.nan,np.nan]]


    df = pd.DataFrame(final_results_table)
    df.set_index('OMIM_ICD_ID',drop=True,inplace=True)
    df.to_pickle(output_direc+dis_index.replace(':','_')+'_UKBBPerformanceResults.pth')

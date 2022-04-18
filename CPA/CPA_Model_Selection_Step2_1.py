import sys
import argparse
import pickle
import pandas as pd
import os
import numpy as np
from vlpi.data.ClinicalDataset import ClinicalDataset,ClinicalDatasetSampler
from vlpi.vLPI import vLPI
from sklearn.utils import resample
from sklearn.metrics import precision_recall_curve,average_precision_score

"""
This script selects the top perfroming model from some set contained within the input directory. It is included in Step 1 of Supplementary Figure 5. Note, model selection is performed in the training dataset to preserve the testing dataset for final validation.
"""

parser = argparse.ArgumentParser(description='Code for model selection')
parser.add_argument("training_data_fraction",help="fraction of dataset used for training vs testing, required to properly perform sampling",type=float)
parser.add_argument("dis_index",help="index for the disease being computed",type=str)
parser.add_argument("input_hpo_file",help="file containing hpo annotations for disease of interest",type=str)
parser.add_argument("num_trials",help="number of trials used during fitting",type=int)
parser.add_argument("max_rank",help="maximum rank of model",type=int)
parser.add_argument('covariate_set',help="str that indicates which covariates to include into the analysis. Expects: 'NULL' (none), 'ALL', or comma-sep list",type=str)
parser.add_argument("input_direc",help="name of input directory containing models",type=str)
parser.add_argument("output_direc",help="name of output directory",type=str)
parser.add_argument("num_resamples",help="number of resamples for bootstrapping",type=int)
args = parser.parse_args()


training_data_fraction=args.training_data_fraction
dis_index=args.dis_index
input_hpo_file=args.input_hpo_file
num_trials=args.num_trials
max_rank=args.max_rank
covariate_set=args.covariate_set
input_direc = args.input_direc
output_direc = args.output_direc
num_resamples = args.num_resamples



if input_direc[-1]!='/':
    input_direc+='/'

if output_direc[-1]!='/':
    output_direc+='/'

dis_to_term = pd.read_pickle(input_hpo_file)

try:
    os.mkdir(output_direc)
except FileExistsError:
    pass

#read the clinical dataset from disk, and include only the annotated symptoms
clinData=ClinicalDataset()
clinData.ReadFromDisk('path/to/clinical/dataset')
annotated_terms=dis_to_term.loc[dis_index]['HPO_ICD10_ID']

if (len(annotated_terms)-1)<max_rank:
    max_rank = len(annotated_terms)-1
clinData.IncludeOnly(annotated_terms)

#load the sampler
sampler=ClinicalDatasetSampler(clinData,training_data_fraction,conditionSamplingOnDx = [dis_index],returnArrays='Torch')
sampler.ReadFromDisk('path/to/clinicaldataset/samplers/'+'Sampler_'+dis_index.replace(':','_'))

#set the covariates
if covariate_set=='NULL':
    sampler.SubsetCovariates([])
elif covariate_set!='ALL':
    sampler.SubsetCovariates(covariate_set.split(','))


#load the different models and store in memory
all_models ={}
sampler.ConvertToUnconditional()
for trial in range(1,num_trials+1):
    vlpiModel= vLPI(sampler,max_rank)
    vlpiModel.LoadModel(input_direc+'MendelianDiseaseIndex_'+dis_index.replace(':','_')+'/Models/trialNum_'+str(trial)+'.pth')
    all_models[trial] = vlpiModel


#### selection criteria based on perplexity on training data.
# cycle through each model, compute pairwise perplexity statistic, and keep the top performing model.
try:
    os.mkdir(output_direc+'MendelianDiseaseIndex_'+dis_index.replace(':','_'))

    current_perplex_best = all_models[1].ComputePerplexity()[0]
    best_fitting_trial=1
    for trial in range(2,num_trials+1):
        new_perplex = all_models[trial].ComputePerplexity()[0]
        diff = current_perplex_best-new_perplex
        resampled_avg_perplex_diff = []
        for i in range(num_resamples):
            resampled_avg_perplex_diff+=[np.mean(resample(diff))]
        #check if lower end of of 95% CI excludes 0, after bonferroni correction for total number of comparisons (numTrials - 1). Indicates the previous model performs systematically worse.
        resampled_avg_perplex_diff=np.array(resampled_avg_perplex_diff)
        resampled_avg_perplex_diff=np.sort(resampled_avg_perplex_diff)
        if resampled_avg_perplex_diff[int(np.floor(num_resamples*(0.025/(num_trials-1))))-1] > 0.0:
            #new model is systematically better than old
            current_perplex_best = new_perplex
            best_fitting_trial=trial
    all_models[best_fitting_trial].PackageModel(output_direc+'MendelianDiseaseIndex_'+dis_index.replace(':','_')+'/BestModelTrial_'+str(best_fitting_trial)+'.pth')
except FileExistsError:
    pass

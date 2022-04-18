#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from sklearn.metrics import precision_recall_curve,average_precision_score
import os
import numpy as np
import argparse

from vlpi.data.ClinicalDataset import ClinicalDataset,ClinicalDatasetSampler
from vlpi.vLPI import vLPI

"""
This script fits the latent phenotype model to the symptom sets. It corresponds to Step 1 of Supplementary Figure 5.
"""


#parse command line arguments
parser = argparse.ArgumentParser(description='Script to automate fit of vlpi model with option to use GPU')
parser.add_argument("training_data_fraction",help="fraction of dataset used for training vs testing, required to properly perform sampling",type=float)
parser.add_argument("validation_fraction",help="fraction of training dataset used for early stopping",type=float)
parser.add_argument("dis_index",help="index for the disease being computed",type=str)
parser.add_argument("input_hpo_file",help="file containing hpo annotations for disease of interest",type=str)
parser.add_argument("model_rank",help="rank for model being fit",type=int)
parser.add_argument("trial",help="trial number",type=str)
parser.add_argument('covariate_set',help="str that indicates which covariates to include into the analysis. Expects: 'NULL' (none), 'ALL', or comma-sep list. Blair et al. 2022 used NULL but included covariates in downstream analyses.",type=str)
parser.add_argument("final_error_tol",help="error tolerance on parameters for model fitting",type=float)
parser.add_argument("learning_rate",help="max learning rate for SGD inference",type=float)
parser.add_argument("batch_size",help="size of batches for learning",type=int)
parser.add_argument("max_epochs",help="maximum number of epochs for training",type=int)
parser.add_argument("output_direc",help="name of output directory",type=str)
parser.add_argument("--useGPU",help="whether or not to use gpu for inference",action="store_true")
args = parser.parse_args()

training_data_fraction=args.training_data_fraction
validation_fraction=args.validation_fraction
dis_index=args.dis_index
input_hpo_file=args.input_hpo_file
rank=args.model_rank
trial=args.trial
covariate_set=args.covariate_set
final_error_tol=args.final_error_tol
learning_rate=args.learning_rate
batch_size = args.batch_size
max_epochs = args.max_epochs
direcPrefix = args.output_direc





if args.useGPU:
    try:
        gpu_device = int(os.environ['SGE_GPU'].split(',')[0])
    except KeyError:
        gpu_device = 0
    num_workers_dataloader = 3
else:
    gpu_device=None
    num_workers_dataloader=0


## set up output directory
if direcPrefix[-1]!='/':
    direcPrefix+='/'

outputFileDirec = 'MendelianDiseaseIndex_'+args.dis_index.replace(':','_')
try:
    os.mkdir(direcPrefix+outputFileDirec)
except FileExistsError:
    pass

try:
    os.mkdir(direcPrefix+outputFileDirec+'/Models')
except FileExistsError:
    pass

# read the hpo terms from disk
dis_to_term = pd.read_pickle(input_hpo_file)

#load the dataset from disk, include only the HPO terms annotated to the the disease
clinData=ClinicalDataset()
clinData.ReadFromDisk('path/to/clinical/record/dataset')
annotated_terms=dis_to_term.loc[dis_index]['HPO_ICD10_ID']
clinData.IncludeOnly(annotated_terms)

#make sure the maximum rank of the model is less than the number of annotated HPO terms
if (len(annotated_terms)-1)<rank:
    rank=len(annotated_terms)-1

## load the stored dataset sampler
sampler=ClinicalDatasetSampler(clinData,training_data_fraction,conditionSamplingOnDx = [dis_index],returnArrays='Torch')
sampler.ReadFromDisk('path/to/clinical/dataset/samplers/'+'Sampler_'+dis_index.replace(':','_'))

#set the covariates
if covariate_set=='NULL':
    sampler.SubsetCovariates([])
elif covariate_set!='ALL':
    sampler.SubsetCovariates(covariate_set.split(','))

#make sure the model hasn't been fit before. If not, then fit it and write to disk.
if 'trialNum_'+trial+'.pth' not in os.listdir(direcPrefix+outputFileDirec+'/Models/'):

    sampler.ConvertToUnconditional()
    validation_sampler=sampler.GenerateValidationSampler(validation_fraction)

    vlpiModel= vLPI(validation_sampler,rank)

    output = vlpiModel.FitModel(batch_size=batch_size,maxLearningRate =learning_rate,finalErrorTol=final_error_tol,computeDevice=gpu_device,numDataLoaders=num_workers_dataloader,maxEpochs=max_epochs)

    fitted_model_dict=vlpiModel.PackageModel(direcPrefix+outputFileDirec+'/Models/trialNum_'+str(trial)+'.pth')

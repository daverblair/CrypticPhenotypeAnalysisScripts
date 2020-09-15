#!/usr/bin/env python
# coding: utf-8
import pickle
import pandas as pd
import os
import numpy as np
import torch
import argparse
from sklearn.metrics import precision_recall_curve,average_precision_score
from vlpi.data.ClinicalDataset import ClinicalDataset,ClinicalDatasetSampler
from vlpi.data.ClinicalDataSimulator import ClinicalDataSimulator
from vlpi.vLPI import vLPI



#parse command line arguments
parser = argparse.ArgumentParser(description='Script to automate simulation and inference of vLPI model for testing.')
parser.add_argument("num_samples",help="number of samples to simulate",type=int)
parser.add_argument("num_symptoms",help="number of observed symptoms to simulate",type=int)
parser.add_argument("rare_disease_freq",help="frequency of rare disease in the population",type=float)
parser.add_argument("training_data_fraction",help="fraction of dataset used for training vs testing",type=float)
parser.add_argument("validation_fraction",help="fraction of training dataset used to monitor for early stopping",type=float)
parser.add_argument("simulation_model_rank",help="rank for model being simulated",type=int)
parser.add_argument("inference_model_rank",help="rank for model being fit",type=int)
parser.add_argument("output_prefix",help="prefix to write table of results",type=str)
parser.add_argument("--isOutlier",help="whether to include the disease as an outlier or not. Default is False,",action="store_true")
parser.add_argument("--useGPU",help="whether or not to use gpu for inference. If using gpu, expects 3 additional cpu cores available to assist in data loading.",action="store_true")
parser.add_argument("--effectiveRankThresh",help="threshold to determine effective rank. Default is 1.0/num_samples",type=float)
args = parser.parse_args()


num_samples=args.num_samples
num_symptoms=args.num_symptoms
rare_disease_freq=args.rare_disease_freq
training_data_fraction=args.training_data_fraction
validation_fraction=args.validation_fraction
sim_rank=args.simulation_model_rank
inf_rank=args.inference_model_rank
output_prefix = args.output_prefix
if args.isOutlier:
    isOutlier=True
else:
    isOutlier=False
sim_rank+=(int(isOutlier))

if args.useGPU:
    try:
        gpu_device = int(os.environ['SGE_GPU'].split(',')[0])
    except KeyError:
        gpu_device = 0
    num_workers_dataloader = 3
else:
    gpu_device=None
    num_workers_dataloader=0

if args.effectiveRankThresh is not None:
    effectiveRankThresh=args.effectiveRankThresh
else:
    effectiveRankThresh=1.0/num_samples



simulator = ClinicalDataSimulator(num_symptoms,sim_rank,rare_disease_freq,isOutlier=isOutlier)
simData=simulator.GenerateClinicalData(num_samples)

clinData = ClinicalDataset()
disList =list(clinData.dxCodeToDataIndexMap.keys())[0:num_symptoms+1]

#load data into clinical dataset
clinData.IncludeOnly(disList)
clinData.LoadFromArrays(torch.cat([simData['incidence_data'],simData['target_dis_dx'].reshape(-1,1)],axis=1),simData['covariate_data'],[],catCovDicts=None, arrayType = 'Torch')
clinData.ConditionOnDx([disList[-1]])
base_sampler = ClinicalDatasetSampler(clinData,training_data_fraction,returnArrays='Torch',conditionSamplingOnDx = [disList[-1]])
base_sampler.ConvertToUnconditional()

validation_sampler=base_sampler.GenerateValidationSampler(validation_fraction)

#inference
vlpiModel= vLPI(validation_sampler,inf_rank)

inference_output = vlpiModel.FitModel(batch_size=200,computeDevice=gpu_device,numDataLoaders=num_workers_dataloader,errorTol=(1.0/num_samples))
fitted_model_dict=vlpiModel.PackageModel(output_prefix+'_InfModel.pth')

lp_components=vlpiModel.ReturnComponents()

base_sampler.RevertToConditional()
base_sampler.ChangeArrayType('Sparse')

full_training_dataset=base_sampler.ReturnFullTrainingDataset(randomize=False)
full_testing_dataset=base_sampler.ReturnFullTestingDataset(randomize=False)

full_training_embeddings=vlpiModel.ComputeEmbeddings(full_training_dataset[0:2])
full_testing_embeddings=vlpiModel.ComputeEmbeddings(full_testing_dataset[0:2])
perplexity_score = vlpiModel.ComputePerplexity(full_testing_dataset[0:2])

#effective rank
risk_matrix = np.dot(full_training_embeddings,lp_components)
frac_variance=np.linalg.svd(risk_matrix,compute_uv=False)
frac_variance=(frac_variance*frac_variance)/np.sum(frac_variance*frac_variance)

if inf_rank>1:
    effective_rank = np.sum(frac_variance>=effectiveRankThresh)
    component_magnitudes = np.sqrt(np.sum(lp_components**2,axis=1))
    allowed_components=np.argsort(component_magnitudes)[::-1][0:effective_rank]
else:
    effective_rank=1
    allowed_components=[0]


#identify best component for rare disease spectrum
top_component = allowed_components[0]
top_component_precision = average_precision_score(full_training_dataset[2].toarray(),full_training_embeddings[:,top_component])

for new_component in allowed_components[1:]:
    new_component_precision = average_precision_score(full_training_dataset[2].toarray(),full_training_embeddings[:,new_component])
    if new_component_precision > top_component_precision:
        top_component=new_component
        top_component_precision=new_component_precision


top_component_score=full_testing_embeddings[:,top_component]

precision_spectrum_model=average_precision_score(full_testing_dataset[2].toarray(),top_component_score)
precision_outlier_model=average_precision_score(full_testing_dataset[2].toarray(),perplexity_score)



output_table=pd.DataFrame({'Num Samples':[num_samples],'Num Symptoms':[num_symptoms],'Rare Disease Prev':[rare_disease_freq],'Simulated Rank':[sim_rank-int(isOutlier)],'Is Outlier':[isOutlier],'Inferred Rank':[effective_rank],'Outlier Score':[precision_outlier_model],'Spectrum Score':[precision_spectrum_model]})
output_table.to_csv(output_prefix+'_ResultsTable.txt',sep='\t',index=False)

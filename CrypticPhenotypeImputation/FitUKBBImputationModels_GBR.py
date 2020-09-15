import argparse
import os
import re
import numpy as np
import pickle
import copy
import pandas as pd

from vlpi.data.ClinicalDataset import ClinicalDataset,ClinicalDatasetSampler
from vlpi.vLPI import vLPI
from vlpi.utils.FeatureSelection import FeatureSelection


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import average_precision_score
from sklearn.utils import resample




parser = argparse.ArgumentParser(description='Code for ridge model selection')
parser.add_argument("training_data_fraction",help="fraction of dataset used for training vs testing, required to properly perform sampling",type=float)
parser.add_argument("validation_fraction",help="fraction of dataset used for validation during model inference and evaluation",type=float)
parser.add_argument("dis_index",help="index for the disease being computed",type=str)
parser.add_argument("output_direc",help="name of output directory",type=str)
parser.add_argument("comorbid_fdr",help="False Discovery Rate for feature selection. Not critical in this context, as it is only a preliminary filter and the number of included features is treated as a hyper-parameter.",type=float)
parser.add_argument("num_features_vec",help="comma-sep list, number of features to consider",type=str)
parser.add_argument("learning_rates",help="comma-sep list of possible learning rates",type=str)
parser.add_argument("n_estimators",help="comma-sep list, number of boosting stages",type=str)
parser.add_argument("min_samples_per_leaf",help="comma-sep list, min samples per leaf node",type=str)
parser.add_argument("max_depths",help="comma-sep list, maximum tree depth",type=str)
parser.add_argument("num_resamples",help="number of samples for bootstrapping confidence intervals",type=int)
args = parser.parse_args()


training_data_fraction=args.training_data_fraction
validation_fraction=args.validation_fraction
dis_index = args.dis_index
output_direc = args.output_direc
comorbid_fdr = args.comorbid_fdr
num_features_vec = list(map(int,args.num_features_vec.split(',')))
learning_rates=list(map(float,args.learning_rates.split(',')))
n_estimators=list(map(int,args.n_estimators.split(',')))
min_samples_per_leaf = list(map(int,args.min_samples_per_leaf.split(',')))
max_depths=list(map(int,args.max_depths.split(',')))
num_resamples = args.num_resamples

if output_direc[-1]!='/':
    output_direc+='/'
try:
    os.mkdir(output_direc)
except FileExistsError:
    pass

disease_direc = dis_index.replace(':','_')+'/'
try:
    os.mkdir(output_direc+disease_direc)
except FileExistsError:
    pass

try:
    os.mkdir(output_direc+disease_direc+'Models')
except FileExistsError:
    pass

#load the modeling results in pickle format
model_results_table=pd.read_pickle('path/to/model/results/table/i.e./SupplementalDataFile3.txt/in/pickle/format')

#load the clinical dataset in HPO term format
clinDataHPO=ClinicalDataset()
clinDataHPO.ReadFromDisk('path/to/clinical/record/dataset/HPO/format')
clinDataHPO.IncludeOnly(model_results_table.loc[dis_index]['Annotated HPO Terms'])

#load the clinical dataset in ICD10-UKBB term format
clinDataICD10_UKEncoding=ClinicalDataset()
clinDataICD10_UKEncoding.ReadFromDisk('path/to/clinical/record/dataset/ICD10_UKBB/format')

#load the ClinicalDatasetSampler classes, which is saved to disk to ensure replicability
sampler=ClinicalDatasetSampler(clinDataHPO,training_data_fraction,conditionSamplingOnDx = [dis_index],returnArrays='Torch')
sampler.ReadFromDisk('path/to/directory/with/stored/samplers/'+'Sampler_'+dis_index.replace(':','_'))

#note: there are two versions of the samplers, one for each dataset, but they are exactly the same.
sampler_ukbb_encoding = ClinicalDatasetSampler(clinDataICD10_UKEncoding,training_data_fraction,conditionSamplingOnDx = [dis_index],returnArrays='Torch')
sampler_ukbb_encoding.ReadFromDisk('path/to/directory/with/stored/samplers/'+'Sampler_'+dis_index.replace(':','_'))

#set the covariates used in the analysis
if model_results_table.loc[dis_index]['Covariate Set']=='NULL':
    sampler.SubsetCovariates([])
    sampler_ukbb_encoding.SubsetCovariates([])
elif covariate_set!='ALL':
    sampler.SubsetCovariates(covariate_set.split(','))
    sampler_ukbb_encoding.SubsetCovariates([])

#load the stored vlpi model
sampler.ConvertToUnconditional()
bestModel= vLPI(sampler,model_results_table.loc[dis_index]['Max. Model Rank'])
bestModel.LoadModel('path/to/top/models/'+dis_index.replace(':','_')+'.pth')

#load the features if they've been generated before. If not, find the feature list using FeatureSelection that is included with vlpi
try:
    with open(output_direc+disease_direc+'FeatureSelectionTable.pth','rb') as f:
        feature_select_table=pickle.load(f)
except FileNotFoundError:
    fullTrainingSetEmbeddings,fullTestingSetEmbeddings = bestModel.ComputeEmbeddings()
    feature_select=FeatureSelection(sampler_ukbb_encoding)
    index,scores,pvals = feature_select.SelectComorbidTraits_ContinuousFeature(fullTrainingSetEmbeddings[:,model_results_table.loc[dis_index]['Top Component']],comorbid_fdr)
    sorted_index = np.argsort(scores)[::-1]

    feature_select_table = {'Index':[x for x in sorted_index],'ICD10':[clinDataICD10_UKEncoding.dataIndexToDxCodeMap[index[x]] for x in sorted_index],'Score':[scores[x] for x in sorted_index],'P-value':[pvals[x] for x in sorted_index]}

    feature_select_table = pd.DataFrame(feature_select_table)
    feature_select_table.set_index('Index',drop=False,inplace=True)
    feature_select_table.to_pickle(output_direc+disease_direc+'FeatureSelectionTable.pth')
    feature_select_table.to_csv(output_direc+disease_direc+'FeatureSelectionTable.txt',sep='\t')

#build grid of hyper-parameters
p_grid_values = {'num_features':num_features_vec,'learning_rate':learning_rates,'n_estimators':n_estimators,'min_samples_per_leaf':min_samples_per_leaf,'max_depth':max_depths}
p_grid = ParameterGrid(p_grid_values)

#fit the GBR model for all possible settings of hyper-parameters. Write the models to disk, and their performance to a table.
results_summary_table = {'Num Features':[],'Learning Rate':[],'Num Estimators':[],'Min Samples Per Leaf':[],'Max Depth':[],'R^2':[],'Avg Precision':[]}
for p_set in p_grid:
    local_clin_data = copy.deepcopy(clinDataICD10_UKEncoding)
    if p_set['num_features']<len(feature_select_table['Index']):
        feature_vec = feature_select_table['Index'].iloc[0:p_set['num_features']]
        feature_icd10 = feature_select_table['ICD10'].iloc[0:p_set['num_features']]
        local_clin_data.IncludeOnly(feature_icd10)
    else:
        feature_icd10 = feature_select_table['ICD10']
        feature_vec=np.arange(clinDataICD10_UKEncoding.numDxCodes)



    model_string='_'.join([str(feature_vec.shape[0]),str(p_set['learning_rate']).replace('.',','),str(p_set['n_estimators']),str(p_set['min_samples_per_leaf']),str(p_set['max_depth'])])+'.pth'

    #if the parameter setting hasn't been fit yet, then fit the full model
    if model_string not in os.listdir(output_direc+disease_direc+'Models/'):
        val_sampler = sampler.GenerateValidationSampler(validation_fraction)
        val_sampler.RevertToConditional()
        bestModel.sampler=val_sampler
        train_embeddings,val_embeddings = bestModel.ComputeEmbeddings()

        val_sampler.AddAuxillaryDataset(local_clin_data)
        val_sampler.ChangeArrayType('Sparse')

        training_data = val_sampler.ReturnFullTrainingDataset(randomize=False)
        validation_data = val_sampler.ReturnFullTestingDataset(randomize=False)
        gbr_model = GradientBoostingRegressor(learning_rate=p_set['learning_rate'],n_estimators=p_set['n_estimators'],min_samples_leaf=p_set['min_samples_per_leaf'],max_depth=p_set['max_depth'])

        if len(training_data[1])>0:
            gbr_model=gbr_model.fit(val_sampler.CollapseDataArrays(training_data[3],training_data[1]),train_embeddings[:,model_results_table.loc[dis_index]['Top Component']].ravel())
            pred_val_embeddings = gbr_model.predict(val_sampler.CollapseDataArrays(validation_data[3],validation_data[1]))
            try:
                val_r2 = gbr_model.score(val_sampler.CollapseDataArrays(validation_data[3],validation_data[1]),val_embeddings[:,model_results_table.loc[dis_index]['Top Component']].ravel())
            except ValueError:
                val_r2=np.nan
        else:
            gbr_model=gbr_model.fit(training_data[3],train_embeddings[:,model_results_table.loc[dis_index]['Top Component']].ravel())
            pred_val_embeddings = gbr_model.predict(validation_data[3])
            try:
                val_r2 = gbr_model.score(validation_data[3],val_embeddings[:,model_results_table.loc[dis_index]['Top Component']].ravel())
            except ValueError:
                val_r2=np.nan
        if np.isfinite(val_r2):
            model_dict={}

            model_dict['Feature ICD10']=copy.deepcopy(feature_icd10)
            model_dict['Learning Rate']=p_set['learning_rate']
            model_dict['Num Estimators']=p_set['n_estimators']
            model_dict['Min Samples Per Leaf']=p_set['min_samples_per_leaf']
            model_dict['Max Depth']=p_set['max_depth']
            model_dict['Model'] = copy.deepcopy(gbr_model)

            with open(output_direc+disease_direc+'Models/'+model_string,'wb') as f:
                pickle.dump(model_dict,f)
    #the model has already been fit once, just re-compute it's performance after loading from disk
    else:
        with open(output_direc+disease_direc+'Models/'+model_string,'rb') as f:
            model_dict = pickle.load(f)
        gbr_model=model_dict['Model']

        if len(training_data[1])>0:
            pred_val_embeddings = gbr_model.predict(val_sampler.CollapseDataArrays(validation_data[3],validation_data[1]))
            val_r2 = gbr_model.score(val_sampler.CollapseDataArrays(validation_data[3],validation_data[1]),val_embeddings[:,model_results_table.loc[dis_index]['Top Component']].ravel())
        else:
            pred_val_embeddings = gbr_model.predict(validation_data[3])
            val_r2 = gbr_model.score(validation_data[3],val_embeddings[:,model_results_table.loc[dis_index]['Top Component']].ravel())
    #if the GBR model generates a finite r2 between validation data embeddings and predictions, then store the model for use
    if np.isfinite(val_r2):
        current_avg_prec_score = average_precision_score(validation_data[2].toarray().ravel(),pred_val_embeddings)

        resample_iter=0
        reseampled_prec_scores = np.zeros(num_resamples)

        #produce bootstrapped estimate of performance variability
        while resample_iter<num_resamples:
            new_inds = resample(np.arange(validation_data[2].shape[0]))
            if validation_data[2][new_inds].sum()>0:
                reseampled_prec_scores[resample_iter]=average_precision_score(validation_data[2][new_inds].toarray().ravel(),pred_val_embeddings[new_inds].ravel())
                resample_iter+=1

        reseampled_prec_scores=np.sort(reseampled_prec_scores)
        ci=reseampled_prec_scores[[int(np.floor(0.025*num_resamples))-1,int(np.ceil(0.975*num_resamples))-1]]

        results_summary_table['Num Features']+=[feature_vec.shape[0]]
        results_summary_table['Learning Rate']+=[p_set['learning_rate']]
        results_summary_table['Num Estimators']+=[p_set['n_estimators']]
        results_summary_table['Min Samples Per Leaf']+=[p_set['min_samples_per_leaf']]
        results_summary_table['Max Depth']+=[p_set['max_depth']]
        results_summary_table['R^2']+=[val_r2]
        results_summary_table['Avg Precision']+=[(current_avg_prec_score,ci)]



        output_str = 'Number of features: {0:4d}; Learning Rate: {1:.3f}; Num Estimators: {2:4d}; Min Samples Per Leaf: {3:2d}; Max Depth: {4:1d}; R^2: {5:.2f}; Avg Precision {6:.2} ({7:.2}, {8:.2})'.format(results_summary_table['Num Features'][-1],results_summary_table['Learning Rate'][-1],results_summary_table['Num Estimators'][-1],results_summary_table['Min Samples Per Leaf'][-1],results_summary_table['Max Depth'][-1],results_summary_table['R^2'][-1],results_summary_table['Avg Precision'][-1][0],results_summary_table['Avg Precision'][-1][1][0],results_summary_table['Avg Precision'][-1][1][1])
        print(output_str,flush=True)

results_summary_table = pd.DataFrame(results_summary_table)
results_summary_table.to_pickle(output_direc+disease_direc+'ModelSelectionResultsTable.pth')

import argparse
import torch
import pyro
import pickle
import pandas as pd
import os
import numpy as np
from vlpi.data.ClinicalDataset import ClinicalDataset,ClinicalDatasetSampler
from vlpi.vLPI import vLPI

from scipy.linalg import orthogonal_procrustes
from sklearn.utils import resample
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import r2_score
import itertools
import re


import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

"""
This script performs the consistency analysis. It corresponds to Step 2 of Supplemental Figure 5.
"""


sns.set(context='talk',color_codes=True,style='ticks',font='Arial',font_scale=1,rc={'axes.linewidth':5,"font.weight":"bold",'axes.labelweight':"bold",'xtick.major.width':4,'xtick.minor.width': 2})
cmap = cm.get_cmap('viridis', 12)
color_list=[cmap(x) for x in [0.0,0.1,0.25,0.5,0.75,0.9,1.0]]
grey_color=(0.5490196078431373, 0.5490196078431373, 0.5490196078431373)
red_color = '#b42f2f'

parser = argparse.ArgumentParser(description='Code for performing consistency analysis')
parser.add_argument("R2_threshold",help="R^2 threshold to call replication",type=float)
parser.add_argument("num_replications",help="number of replications required to establish convergence",type=float)
parser.add_argument("training_data_fraction",help="fraction of dataset used for training vs testing, required to properly perform sampling",type=float)
parser.add_argument("input_hpo_file",help="file containing hpo annotations for disease of interest",type=str)
parser.add_argument("max_rank",help="maximum rank of model",type=int)
parser.add_argument("num_trials",help="number of trials",type=int)
parser.add_argument('covariate_set',help="str that indicates which covariates to include into the analysis. Expects: 'NULL' (none), 'ALL', or comma-sep list",type=str)
parser.add_argument("input_direc",help="name of input directory containing model",type=str)
parser.add_argument("output_file_prefix",help="name of output directory",type=str)
args = parser.parse_args()

R2_threshold = args.R2_threshold
num_replications=args.num_replications
training_data_fraction=args.training_data_fraction
input_hpo_file=args.input_hpo_file
max_rank=args.max_rank
num_trials=args.num_trials
covariate_set=args.covariate_set
input_direc = args.input_direc
output_file_prefix = args.output_file_prefix



def ProcrustesCompare(mat1,mat2):
    """
    Compares similarity of two matrices according to weighted R^2 among their
    individual components.
    R^2 of inidividual components weighted by their magnitude to generate composite score.
    Matrcies aligned prior to comparison using orthogonal procrustes (rotation only).
    """
    aligned_mat2=np.copy(mat2)

    alignment_matrix=orthogonal_procrustes(mat1.T,aligned_mat2.T)[0]
    aligned_mat1=np.dot(mat1.T,alignment_matrix).T

    weights = np.sqrt(np.sqrt(np.sum(aligned_mat1*aligned_mat1,axis=1))*np.sqrt(np.sum(aligned_mat2*aligned_mat2,axis=1)))
    score_vec=np.zeros(weights.shape)
    for i in range(aligned_mat1.shape[0]):
        score_vec[i]=r2_score(aligned_mat1[i],aligned_mat2[i])

    score_vec[score_vec<0.0]=0.0
    return np.sum(score_vec*weights/np.sum(weights))

def recursive_sort(htree,N,current_ind):
    if current_ind<N:
        return [current_ind]
    else:
        left = int(htree[current_ind-N,0])
        right = int(htree[current_ind-N,1])
        return (recursive_sort(htree,N,left) + recursive_sort(htree,N,right))


def SortDistanceMatrix(distMat,aggClustIntance):
    tree=aggClustIntance.children_
    new_order=recursive_sort(tree,distMat.shape[0],2*distMat.shape[0]-2)
    sorted_distMat = distMat[new_order]
    sorted_distMat=sorted_distMat[:,new_order]
    return sorted_distMat,new_order




dis_to_term = pd.read_pickle(input_hpo_file)


allowed_diseases = [x.strip() for x in open('../../../Data/IncludedDiseases/InclusionCriteriaDiseases.txt').readlines()]
dis_names=pd.read_csv('../../../Data/IncludedDiseases/IncludedDiseases_NamesInheritance.txt',sep='\t')
dis_names.set_index('Disease ID', drop=True, inplace=True)

dataset='UCSF_MendelianDisease_HPO.pth'
clinData=ClinicalDataset()
clinData.ReadFromDisk('../../../Data/ClinicalRecords/'+dataset)



results_table={'OMIM_ICD_ID':[],'Avg Component Weighted R^2':[],'Component Weighted R^2 Matrix':[],'Cluster Labels':[],'Num Replicates, Top Model':[],'Meets Criteria':[]}
try:
    os.mkdir(output_file_prefix+'_Figures')
except FileExistsError:
    pass

for dis_index in set(allowed_diseases).intersection(dis_to_term.index):
    try:
        print('Computing matrix similarities for '+dis_index)

        sampler=ClinicalDatasetSampler(clinData,training_data_fraction,conditionSamplingOnDx = [dis_index],returnArrays='Torch')
        sampler.ReadFromDisk('../../../Data/Samplers/UCSF/'+'Sampler_'+dis_index.replace(':','_'))
        sampler.ConvertToUnconditional()
        all_procrustes_scores=[]
        procrustes_score_matrix = np.ones((num_trials,num_trials))


        for trial_pair in itertools.combinations(range(1,num_trials+1), 2):
            vlpi_1=vLPI(sampler,max_rank)
            vlpi_1.LoadModel('../vLPI_Inference-1/'+input_direc+'MendelianDiseaseIndex_'+dis_index.replace(':','_')+'/Models/trialNum_'+str(trial_pair[0])+'.pth')

            vlpi_2=vLPI(sampler,max_rank)
            vlpi_2.LoadModel('../vLPI_Inference-1/'+input_direc+'MendelianDiseaseIndex_'+dis_index.replace(':','_')+'/Models/trialNum_'+str(trial_pair[1])+'.pth')

            risk_matrix_1=vlpi_1.ReturnComponents()
            risk_matrix_2=vlpi_2.ReturnComponents()
            all_procrustes_scores+=[ProcrustesCompare(risk_matrix_1,risk_matrix_2)]
            procrustes_score_matrix[trial_pair[0]-1,trial_pair[1]-1]=all_procrustes_scores[-1]
            procrustes_score_matrix[trial_pair[1]-1,trial_pair[0]-1]=all_procrustes_scores[-1]


        all_procrustes_scores=np.array(all_procrustes_scores)
        avg_procrustes_score = np.mean(all_procrustes_scores)


        results_table['OMIM_ICD_ID']+=[dis_index]
        results_table['Avg Component Weighted R^2']+=[avg_procrustes_score]
        results_table['Component Weighted R^2 Matrix']+=[procrustes_score_matrix]




        ag =AgglomerativeClustering(affinity='precomputed',linkage='average',distance_threshold=1.0-R2_threshold,n_clusters=None)
        ag.fit(1.0-procrustes_score_matrix)
        function_clusters = ag.labels_
        results_table['Cluster Labels']+=[function_clusters]

        all_summary_files = os.listdir('../ModelSelection-2/'+input_direc+'MendelianDiseaseIndex_'+dis_index.replace(':','_'))
        r=re.compile('BestModelTrial_*')
        best_model_file = list(filter(r.match, all_summary_files))[0]
        top_trial=int(best_model_file.split('_')[1].strip('.pth'))
        top_trial_cluster=function_clusters[top_trial-1]
        num_replicates=np.sum(function_clusters==top_trial_cluster)


        results_table['Num Replicates, Top Model']+=[num_replicates]
        results_table['Meets Criteria']+=[num_replicates>=num_replications]


        f, axes = plt.subplots(1, 2,figsize=(20,6))
        axes[1].spines['right'].set_visible(False)
        axes[1].spines['top'].set_visible(False)
        bins=np.linspace(0.0, 1.0,51)
        hist_output=axes[1].hist(all_procrustes_scores,bins=bins,color=color_list[0],lw=0.0)
        axes[1].set_xlabel(r'All Pairwise $R^{2}$ Measurments'+'\nAmong Symptom Risk Functions',fontsize=18,fontweight='bold')
        axes[1].set_ylabel('# of Function Pairs',fontsize=18,fontweight='bold')


        axes[0].spines['right'].set_visible(False)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['left'].set_visible(False)
        axes[0].spines['bottom'].set_visible(False)

        sorted_dist_mat,sort_order=SortDistanceMatrix(procrustes_score_matrix,ag)
        sorted_dist_mat=np.ma.array(sorted_dist_mat,mask=np.tri(sorted_dist_mat.shape[0]).T)
        im=axes[0].matshow(sorted_dist_mat,cmap=cmap,vmin=0.0,vmax=1.0)
        axes[0].axis('off')
        y_axis_labels=np.arange(1,num_trials+1)[sort_order]
        for i in range(num_trials):
            if y_axis_labels[i]!=top_trial:
                axes[0].text(-1,i,'Trial {0:d}'.format(y_axis_labels[i]),horizontalalignment='right',color=grey_color,fontsize=12,verticalalignment='center')
            else:
                axes[0].text(-1,i,'Trial {0:d}'.format(y_axis_labels[i]),horizontalalignment='right',color=red_color,fontsize=12,verticalalignment='center')
            cluster=function_clusters[sort_order][i]
            if cluster==top_trial_cluster:
                axes[0].text(i+0.5,i,'Cluster {0:d}'.format(cluster+1),horizontalalignment='left',color=red_color,fontsize=12,verticalalignment='center')
            else:
                axes[0].text(i+0.5,i,'Cluster {0:d}'.format(cluster+1),horizontalalignment='left',color=grey_color,fontsize=12,verticalalignment='center')

        ax2_divider = make_axes_locatable(axes[0])
        cax = ax2_divider.append_axes("top", size="5%", pad="8%")

        cb=f.colorbar(im, cax=cax,orientation='horizontal',ticks=[0.0,0.5,1.0],drawedges=False,shrink=0.65)
        cb.outline.set_linewidth(3.0)
        cb.ax.set_title(r'$R^{2}$ Measurements'+'\nBetween Symptom Risk Functions',fontsize=14)
        cb.ax.tick_params(labelsize=10)
        if num_replicates>=num_replications:
            f.suptitle(dis_names.loc[dis_index]['Name']+'\nOptimization Status: Success',fontweight='bold',fontsize=22)
        else:
            f.suptitle(dis_names.loc[dis_index]['Name']+'\nOptimization Status: Failed',fontweight='bold',fontsize=22)
        plt.tight_layout(pad=2.0, w_pad=2.0, h_pad=4.0)
        plt.savefig(output_file_prefix+'_Figures/MendelianDiseaseIndex_'+dis_index.replace(':','_')+'.svg')
        plt.close()
    except FileNotFoundError:
        print('Script failed to process '+dis_index+'due to missing file error. If expecting this to work, please check to make sure all files in right place.')

results_table=pd.DataFrame(results_table)
results_table.set_index('OMIM_ICD_ID',drop=True,inplace=True)
results_table.to_pickle(output_file_prefix+'_RiskFunctionAnalysis.pth')

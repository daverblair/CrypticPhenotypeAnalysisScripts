import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import cm
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats
from sklearn.metrics import precision_recall_curve,average_precision_score

import torch
from vlpi.data.ClinicalDataset import ClinicalDataset,ClinicalDatasetSampler
from vlpi.data.ClinicalDataSimulator import ClinicalDataSimulator
from vlpi.vLPI import vLPI

sns.set(context='talk',color_codes=True,style='ticks',font='Arial',font_scale=2.5,rc={'axes.linewidth':5,"font.weight":"bold",'axes.labelweight':"bold",'xtick.major.width':4,'xtick.minor.width': 2})
cmap = cm.get_cmap('viridis', 12)
color_list=[cmap(x) for x in [0.0,0.1,0.25,0.5,0.75,0.9,1.0]]
grey_color=(0.75, 0.75, 0.75)
red_color = '#FF5D5D'
blue_color='#5DA4FF'

torch.manual_seed(1096)

num_samples=100000
num_symptoms=20
rare_disease_freq=0.001
training_data_fraction=0.75
validation_fraction=0.2
sim_rank=1
inf_rank=1
isOutlier=False


simulator = ClinicalDataSimulator(num_symptoms,sim_rank,rare_disease_freq,isOutlier=isOutlier)
simData=simulator.GenerateClinicalData(num_samples)

f,axes = plt.subplots(1, 1,figsize=(4,8))
axes.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
axes.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
axes.imshow(simData['incidence_data'][0:40],cmap=cmap,vmin=0.0,vmax=1.0,aspect='auto')

axes.set_ylabel(r'Observed Patients ($N$)',fontsize=40,fontweight='bold')
axes.set_title('Diagnosed Symptom\nMatrix '+r'($\mathbf{S}$)',fontsize=40,fontweight='bold')

legend_elements = [Patch(facecolor=color_list[0], edgecolor=None,
      label='Undiagnosed'),
                   Patch(facecolor=color_list[-1], edgecolor=None,
                         label='Diagnosed')]
axes.legend(handles=legend_elements, loc='best',fontsize=20,frameon=False)
plt.savefig('obs_symptom_matrix.svg')
plt.close()


fig, ax = plt.subplots(figsize=(10, 8))
fig.tight_layout(pad=2)
cryptic_phenotypes=simData['latent_phenotypes'].numpy().ravel()
has_rare_disease=np.array(simData['target_dis_dx'].numpy().ravel(),np.int32)

xmin=np.floor(cryptic_phenotypes.min())
xmax=np.ceil(cryptic_phenotypes.max())
ax.set_xlim(xmin,xmax)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.hist([cryptic_phenotypes[has_rare_disease==1],cryptic_phenotypes[has_rare_disease==0]],log=True,stacked=True,density=True,color=[color_list[0],grey_color],label=['Pathogenic Genotype Carriers','Control Population'])
ax.legend(loc='best',frameon=False, fontsize=20)
ax.set_xlabel('Cryptic\n'+r'Phenotype ($\mathbf{Z}$)',fontsize=40,fontweight='bold')
ax.set_ylabel('Density\n'+r'$\log_{10}$-Scale',fontsize=40,fontweight='bold')
plt.savefig('cryptic_pheno_sim.svg')
plt.close()

# #
clinData = ClinicalDataset()
#build arbitrary list of disease codes
disList =list(clinData.dxCodeToDataIndexMap.keys())[0:num_symptoms+1]

# load data into clinical dataset
clinData.IncludeOnly(disList)
clinData.LoadFromArrays(torch.cat([simData['incidence_data'],simData['target_dis_dx'].reshape(-1,1)],axis=1),simData['covariate_data'],[],catCovDicts=None, arrayType = 'Torch')
clinData.ConditionOnDx([disList[-1]])
sampler = ClinicalDatasetSampler(clinData,training_data_fraction,returnArrays='Torch',conditionSamplingOnDx = [disList[-1]])
sampler.ConvertToUnconditional()

vlpiModel= vLPI(sampler,inf_rank)
try:
    vlpiModel.LoadModel('IllustrativeExample.pth')
except FileNotFoundError:
    inference_output = vlpiModel.FitModel(batch_size=200,errorTol=(1.0/num_samples))
    vlpiModel.PackageModel('IllustrativeExample.pth')

inferredCrypticPhenotypes=vlpiModel.ComputeEmbeddings((simData['incidence_data'],simData['covariate_data']))
riskFunction=vlpiModel.ReturnComponents().ravel()
latentPhenos=vlpiModel.ComputeEmbeddings(dataArrays=(clinData.ReturnSparseDataMatrix(clinData.data.index.sort_values()),[])).ravel()

fig, ax = plt.subplots(figsize=(10, 8))
fig.tight_layout(pad=2)
xmin=np.floor(cryptic_phenotypes.min())
xmax=np.ceil(cryptic_phenotypes.max())

ymin=np.floor(latentPhenos.min())
ymax=np.ceil(latentPhenos.max())
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
o=ax.hist2d(cryptic_phenotypes[has_rare_disease==0],latentPhenos[has_rare_disease==0], norm=mpl.colors.LogNorm(), cmap=cmap,bins=[np.linspace(xmin,xmax,21),np.linspace(ymin,ymax,21)],density=True)
ax.plot(cryptic_phenotypes[has_rare_disease==1],latentPhenos[has_rare_disease==1],color=red_color,marker='^',markersize=10.0,lw=0.0,label='Pathogenic Genotype Carriers')
ax.set_xlabel('Cryptic\n'+r'Phenotype ($\mathbf{Z}$)',fontsize=40,fontweight='bold')
ax.set_ylabel('Inferred Cryptic\n'+r'Phenotype ($\mathbf{\hat{Z}}$)',fontsize=40,fontweight='bold')
ax.legend(loc='best',fontsize=20,frameon=False)
fig.colorbar(o[3], ax=ax)
plt.savefig('inferred_v_sim_cp.svg')
plt.close()

fig, ax = plt.subplots(figsize=(10, 8))
fig.tight_layout(pad=2)
trueRiskFunc=simData['model_params']['latentPhenotypeEffects'].numpy().ravel()
xmin=0.0
xmax=np.ceil(max(trueRiskFunc.max(),riskFunction.max())*10)/10
ymin=xmin
ymax=xmax
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.plot(trueRiskFunc,riskFunction,'o',ms=30.0,color=color_list[0])
ax.set_xlabel('Latent Phenotype\n'+r'Effects ($\mathbf{\theta}$)',fontsize=40,fontweight='bold')
ax.set_ylabel('Inferred Latent\n'+r'Phenotype Effects ($\mathbf{\hat{\theta}}$)',fontsize=40,fontweight='bold')
plt.plot([xmin,xmax],[xmin,xmax],'--',color=red_color,lw=5.0)
plt.savefig('risk_func_compare.svg')
plt.close()

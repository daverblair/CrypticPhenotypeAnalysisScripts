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
grey_color=(0.5490196078431373, 0.5490196078431373, 0.5490196078431373,1.0)
grey_color_dark=(0.2745098, 0.2745098, 0.2745098,1.0)
red_color = (0.70588235, 0.18431373, 0.18431373,1.0)
color_list=[cmap(x) for x in [0.0,0.1,0.25,0.5,0.75,0.9,1.0]]


#set simulation parameters
torch.manual_seed(1023)
num_samples=100000
num_symptoms=20
rare_disease_freq=0.001
training_data_fraction=0.75
validation_fraction=0.2
sim_rank=2
inf_rank=10
isOutlier=True
sim_rank+=(int(isOutlier))


#simulate the data
simulator = ClinicalDataSimulator(num_symptoms,sim_rank,rare_disease_freq,isOutlier=isOutlier)
simData=simulator.GenerateClinicalData(num_samples)

clinData = ClinicalDataset()
#build arbitrary list of disease codes
disList =list(clinData.dxCodeToDataIndexMap.keys())[0:num_symptoms+1]

#load data into clinical dataset
clinData.IncludeOnly(disList)
clinData.LoadFromArrays(torch.cat([simData['incidence_data'],simData['target_dis_dx'].reshape(-1,1)],axis=1),simData['covariate_data'],[],catCovDicts=None, arrayType = 'Torch')
clinData.ConditionOnDx([disList[-1]])
sampler = ClinicalDatasetSampler(clinData,training_data_fraction,returnArrays='Torch',conditionSamplingOnDx = [disList[-1]])
sampler.ConvertToUnconditional()

#fit the model, unless it's alreadt been fit
vlpiModel= vLPI(sampler,inf_rank)
try:
    vlpiModel.LoadModel('IllustrativeExample.pth')
except FileNotFoundError:
    inference_output = vlpiModel.FitModel(batch_size=200,errorTol=(1.0/num_samples))
    vlpiModel.PackageModel('IllustrativeExample.pth')

#compute cryptic phenotypes and outlier scores
inferredCrypticPhenotypes=vlpiModel.ComputeEmbeddings((simData['incidence_data'],simData['covariate_data']))
riskFunction=vlpiModel.ReturnComponents()
perplexityTraining, perplexityTest=vlpiModel.ComputePerplexity(randomize=False)
latentTraining,latentTest=vlpiModel.ComputeEmbeddings(randomize=False)

#estimate the effective rank
risk_matrix = np.dot(latentTraining,riskFunction)
frac_variance=np.linalg.svd(risk_matrix,compute_uv=False)
frac_variance=(frac_variance*frac_variance)/np.sum(frac_variance*frac_variance)
effective_rank = np.sum(frac_variance>=1e-5)
component_magnitudes = np.sqrt(np.sum(riskFunction**2,axis=1))
allowed_components=np.argsort(component_magnitudes)[::-1][0:effective_rank]


# isolate the top performing cryptic phenotype
sampler.RevertToConditional()
train_data=sampler.ReturnFullTrainingDataset(randomize=False)
test_data=sampler.ReturnFullTestingDataset(randomize=False)

top_component = allowed_components[0]
top_component_precision = average_precision_score(train_data[2].numpy(),latentTraining[:,top_component])
for new_component in allowed_components[1:]:
    new_component_precision = average_precision_score(train_data[2].numpy(),latentTraining[:,new_component])
    if new_component_precision > top_component_precision:
        top_component=new_component
        top_component_precision=new_component_precision

# compare the performance of the spectrum and outlier models
precision_spectrum_model=average_precision_score(test_data[2].numpy(),latentTest[:,top_component])
precision_outlier_model=average_precision_score(test_data[2].numpy(),perplexityTest)


# make the figures
f,axes = plt.subplots(1, 1,figsize=(8,16))
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

#
fig, ax = plt.subplots(figsize=(10, 8))

xmin=np.floor(inferredCrypticPhenotypes[:,allowed_components].min())
xmax=np.ceil(inferredCrypticPhenotypes[:,allowed_components].max())


g = ax.hexbin(inferredCrypticPhenotypes[:,allowed_components[0]],inferredCrypticPhenotypes[:,allowed_components[1]],cmap=cmap,marginals=False,mincnt=None,bins='log',gridsize=30,extent=[xmin,xmax,xmin,xmax])
ax.plot(inferredCrypticPhenotypes[simData['target_dis_dx']==1,allowed_components[0]],inferredCrypticPhenotypes[simData['target_dis_dx']==1,allowed_components[1]],lw=0.0,marker='^',ms=15,color=red_color,label='Dx with Rare Disease')
ax.set_xlim(xmin,xmax)
ax.set_ylim(xmin,xmax)

l=ax.legend(loc='best',fontsize=14,frameon=False)
for text in l.get_texts():
    text.set_color('w')

cb=fig.colorbar(g)
cb.set_label('Frequency\n'+r'($\log_{10}$-Scale)',fontsize=30,fontweight='bold')
ax.set_xlabel(r'Latent Component $\mathbf{Z}_{1}$',fontsize=40,fontweight='bold')
ax.set_ylabel(r'Latent Component $\mathbf{Z}_{2}$',fontsize=40,fontweight='bold')
ax.set_title('Inferred\nCryptic Phenotypes',fontsize=50,fontweight='bold')
plt.savefig('cryptic_phenotypes.svg')
plt.close()
#
#
fig, ax = plt.subplots(figsize=(10, 8))
ax.tick_params(axis='x',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False)
ax.tick_params(axis='y',which='both',left=False,right=False,bottom=False,top=False,labelleft=False)

im=ax.imshow(riskFunction,cmap=cmap)

plt.savefig('risk_func.svg')
plt.close()

fig, ax = plt.subplots(figsize=(10, 2))
fig.subplots_adjust(bottom=0.5)
norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='horizontal',ticks=[0.0,0.5,1.0])
cb1.set_label('Symptom Risk\nFunction '+r'[i.e. $f(\mathbf{Z};\hat{\mathbf{\theta}})$]',fontsize=30,fontweight='bold')
cb1.set_ticklabels(['Low','Intermediate','High'])
plt.savefig('risk_func_cbar.svg')
plt.close()


pr_spectrum=precision_recall_curve(test_data[2].numpy(),latentTest[:,top_component])
pr_outlier=precision_recall_curve(test_data[2].numpy(),perplexityTest)

f, axis = plt.subplots(1, 1,figsize=(10,8))
axis.spines['right'].set_visible(False)
axis.spines['top'].set_visible(False)

axis.step(pr_spectrum[1],pr_spectrum[0],lw=5.0,color=color_list[0],label='Spectrum Score (Top Performing Latent Component)')
axis.step(pr_outlier[1],pr_outlier[0],lw=5.0,color=color_list[3],label='Outlier Score (Reconstruction Error)')
axis.set_xlabel('Recall',fontsize=24)
axis.set_ylabel('Precision',fontsize=24)
axis.legend(loc='best',frameon=False,prop={'size': 18})
plt.savefig('pr_curve.svg')
plt.close()

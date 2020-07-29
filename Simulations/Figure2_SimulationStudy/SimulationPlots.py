import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import cm
import seaborn as sns
import pickle
import pandas as pd
from matplotlib import gridspec
from vlpi.data.ClinicalDataset import ClinicalDataset
import textwrap
from sklearn.metrics import roc_auc_score, roc_curve

sns.set(context='talk',color_codes=True,style='ticks',font='Arial',font_scale=2.5,rc={'axes.linewidth':5,"font.weight":"bold",'axes.labelweight':"bold",'xtick.major.width':4,'xtick.minor.width': 2})
cmap = cm.get_cmap('viridis', 12)
grey_color=(0.5490196078431373, 0.5490196078431373, 0.5490196078431373,1.0)
grey_color_dark=(0.2745098, 0.2745098, 0.2745098,1.0)
red_color = (0.70588235, 0.18431373, 0.18431373,1.0)
color_list=[cmap(x) for x in [0.0,0.1,0.25,0.5,0.75,0.9,1.0]]

sim_table=pd.read_pickle('SimulationStudyResults.pth')

f, axes = plt.subplots(1, 3,figsize=(30,8))
for axis in axes:
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

sim_table['Rare Disease Type']=sim_table['Is Outlier'].apply(lambda x: 'Outlier' if x is True else 'Spectrum')

axes[0].set_ylim(0,7)
axes[0].set_xlim(0,7)
axes[0].plot([0,7],[0,7],'--',lw=5.0,color=grey_color)

g=sns.pointplot(y='Inferred Rank',x='Simulated Rank',hue='Rare Disease Type',data=sim_table,ax=axes[0],palette=[color_list[0],color_list[3]],linestyles='',scale = 1.5,order=np.arange(7),errwidth=10.0,dodge=True)
plt.setp(g.collections, alpha=0.75) #for the markers
plt.setp(g.lines, alpha=0.75)

axes[0].set_xlabel('# of Simulated\nLatent Phenotypes',fontsize=48,fontweight='bold')
axes[0].set_ylabel('Inferred Effective Rank\n (Rank Required to Explain 99.999%\nof Symptom Variance on Log-Odds Scale)',fontsize=48,fontweight='bold')

axes[1].set_ylim(0,3.5)
axes[1].set_xlim(0,3.5)

symbol_list=['o','^','s','P']

legend_elements = [mpl.lines.Line2D([],[],marker='o',lw=0.0,color=color_list[0],ms=15,markeredgewidth=0.0,label='Spectrum'),
    mpl.lines.Line2D([],[],marker='o',lw=0.0,color=color_list[3],ms=15,markeredgewidth=0.0,label='Outlier')]
for i,numSimDim in enumerate([1,2,4,6]):
    subset_table=sim_table.loc[sim_table['Simulated Rank']==numSimDim]
    axes[1].plot(np.log10(subset_table.loc[subset_table['Is Outlier']==False]['Outlier Score']/subset_table.loc[subset_table['Is Outlier']==False]['Rare Disease Prev']),np.log10(subset_table.loc[subset_table['Is Outlier']==False]['Spectrum Score']/subset_table.loc[subset_table['Is Outlier']==False]['Rare Disease Prev']),marker=symbol_list[i],linestyle='',color=color_list[0],ms=15)
    axes[1].plot(np.log10(subset_table.loc[subset_table['Is Outlier']==True]['Outlier Score']/subset_table.loc[subset_table['Is Outlier']==False]['Rare Disease Prev']),np.log10(subset_table.loc[subset_table['Is Outlier']==True]['Spectrum Score']/subset_table.loc[subset_table['Is Outlier']==False]['Rare Disease Prev']),linestyle='',marker=symbol_list[i],color=color_list[3],ms=15)
    legend_elements+=[mpl.lines.Line2D([],[],marker=symbol_list[i],lw=0.0,color=grey_color,ms=15,markeredgewidth=0.0,label="{0:d} Latent Dim.".format(numSimDim))]

axes[1].plot([0,3.5],[0,3.5],'--',lw=5.0,color=grey_color)

axes[1].set_xlabel('Outlier Score\n'+r'($\log_{10}$-Scale, Prevalence Normalized)',fontsize=48,fontweight='bold')
axes[1].set_ylabel('Spectrum Score\n'+r'($\log_{10}$-Scale, Prevalence Normalized)',fontsize=48,fontweight='bold')
axes[1].legend(handles=legend_elements,loc='best',fontsize=32,frameon=False)

axes[2].set_ylim(-0.05,1.05)
axes[2].set_xlim(-0.05,1.05)

fpr,tpr,thresh=roc_curve(sim_table['Is Outlier'].values,np.log10(sim_table['Outlier Score']/sim_table['Spectrum Score']))
auc=roc_auc_score(sim_table['Is Outlier'].values,np.log10(sim_table['Outlier Score']/sim_table['Spectrum Score']))
axes[2].step(fpr,tpr,color=color_list[0],lw=10.0)
axes[2].set_xlabel('False Positive Rate',fontsize=64,fontweight='bold')
axes[2].set_ylabel('Sensitivity',fontsize=64,fontweight='bold')
axes[2].text(0.5,0.5,'AUC: {0:.2g}'.format(auc),fontsize=48,fontweight='bold')

plt.savefig('SimulationResults.svg')
plt.close()

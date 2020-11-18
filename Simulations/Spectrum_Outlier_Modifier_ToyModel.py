import numpy as np
from scipy.stats import norm
from scipy.stats.mstats import mquantiles
from QRankGWAS import QRank

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import cm
import seaborn as sns
from scipy.stats import linregress
import pandas as pd

"""

This script generates the toy, simulated datasets used in Figure 1.

"""


sns.set(context='talk',color_codes=True,style='ticks',font='Arial',font_scale=2,rc={'axes.linewidth':5,"font.weight":"bold",'axes.labelweight':"bold",'xtick.major.width':4,'xtick.minor.width': 2})
cmap = cm.get_cmap('viridis', 12)
color_list=[cmap(x) for x in [0.0,0.1,0.25,0.5,0.75,0.9,1.0]]
grey_color=(0.5490196078431373, 0.5490196078431373, 0.5490196078431373)
red_color = '#b42f2f'




N=500000
target_geno_freq=0.3
target_effect_size=1.05
mend_dis_freq=1e-4
np.random.seed(1023)
quantiles=np.array([0.5,0.9,0.999])

#Spectrum Model
genotype=np.random.choice(np.arange(3),p=[(1-target_geno_freq)*(1-target_geno_freq),2.0*target_geno_freq*(1-target_geno_freq),target_geno_freq*target_geno_freq],size=N)
genotype_strings=np.array(['A1/A1','A1/A2','A2/A2'])


additive_background=np.random.normal(0.0,1.0,N)
mendelian_genotypes=np.random.binomial(1,mend_dis_freq,N)
mendelian_disease_effect=norm(0.0,1.0).isf(mend_dis_freq)
general_modifier_effects=np.random.normal(0.0,1.0,N)



base_phenotype=additive_background+mendelian_genotypes*mendelian_disease_effect
obs_phenotype=base_phenotype+norm(mendelian_disease_effect,1.0).cdf(base_phenotype)*(general_modifier_effects+genotype*target_effect_size)

genotype_quantiles={}
genotype_means={}
for i,geno in enumerate(genotype_strings):
    genotype_quantiles[geno]=mquantiles(obs_phenotype[genotype==i],prob=quantiles)
    genotype_means[geno]=np.mean(obs_phenotype[genotype==i])



f, axis = plt.subplots(1, 1,figsize=(24,20))
gs = gridspec.GridSpec(3, 2,figure=f, width_ratios=[1,1], height_ratios=[8,4,8])
f.tight_layout(rect=[0.05,0.1,0.95,0.9])
axes=[plt.subplot(x) for x in gs]
plt.subplots_adjust( wspace=0.3, hspace=0.3)

axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)

axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)

axes[2].spines['left'].set_visible(False)
axes[2].spines['top'].set_visible(False)
axes[2].tick_params(axis='y', which='both', left=False,right=True, labelleft=False,labelright=True)

axes[3].spines['left'].set_visible(False)
axes[3].spines['top'].set_visible(False)
axes[3].tick_params(axis='y', which='both', left=False,right=True, labelleft=False,labelright=True)

axes[4].spines['right'].set_visible(False)
axes[4].spines['top'].set_visible(False)

axes[5].spines['right'].set_visible(False)
axes[5].spines['top'].set_visible(False)



#Histogram with Mendelian disease cases and non-cases
bins=np.linspace(np.floor(obs_phenotype.min()),np.ceil(obs_phenotype.max()),50)
est_quantile_values=mquantiles(obs_phenotype,prob=[0.5,0.9,0.999],alphap=0.0,betap=1.0)

output_wo_dis=sns.distplot(obs_phenotype[mendelian_genotypes==0],kde=False,hist_kws={'log':True,'alpha':1.0},color=color_list[0],label = 'No Mendelian Disease',bins=bins,ax=axes[0])
output_w_dis=sns.distplot(obs_phenotype[mendelian_genotypes==1],kde=False,hist_kws={'log':True,'alpha':1.0},color=color_list[-2],label = 'With Mendelian Disease',bins=bins,ax=axes[0])

for i,q in enumerate([0.5,0.9,0.999]):
    top_val=np.where(est_quantile_values[i]<=bins)[0][0]
    bot_val=np.where(est_quantile_values[i]>=bins)[0][-1]
    mid_bin_val=(bins[top_val]+bins[bot_val])/2.0
    try:
        y_val=max([h.get_height() for h in output_wo_dis.patches][bot_val],[h.get_height() for h in output_w_dis.patches][bot_val])
    except NameError:
        y_val=[h.get_height() for h in output_wo_dis.patches][bot_val]
    axes[0].plot(mid_bin_val,y_val,'o',ms=15,mew=0.0,color=red_color)
    axes[0].text(mid_bin_val,y_val,'{0:g}th Percentile'.format(100*q),verticalalignment='bottom', horizontalalignment='left',rotation=30,color=red_color,fontweight='bold',fontsize=18)


axes[0].set_xlabel('Cryptic Phenotype Severity',fontsize=24)
axes[0].set_ylabel('Number of Patients\n'+r'($\log_{10}$-Scale)',fontsize=24)
axes[0].legend(loc='best',frameon=False,prop={'size': 18})



#cumulative fraction with Mendelian Disease
#cumulative fraction of modifier variance
#cdf function for Mendelian Disease
sorted_phenotypes = np.argsort(obs_phenotype)
cumulative_cases = np.cumsum(mendelian_genotypes[sorted_phenotypes])/mendelian_genotypes.sum()

scaled_modifier_effects=norm(mendelian_disease_effect,1.0).cdf(base_phenotype)*(general_modifier_effects+genotype*target_effect_size)

var_vec = (scaled_modifier_effects-np.mean(scaled_modifier_effects))**2.0
cumulative_modifier_variance=np.cumsum(var_vec[sorted_phenotypes])/(np.var(scaled_modifier_effects)*obs_phenotype.shape[0])

axes[2].plot(obs_phenotype[sorted_phenotypes],cumulative_cases,'-',lw=6.0,color=color_list[2],label='Mendelian Disease Cases')
axes[2].plot(obs_phenotype[sorted_phenotypes],cumulative_modifier_variance,'-',lw=6.0,color=color_list[4],label='Scaled Modifier Effect Variance\n'+r'(i.e. Var[$\phi(\beta,\theta)\times\alpha$])')
axes[2].set_xlabel('Cryptic Phenotype Severity',fontsize=24)
axes[2].yaxis.set_label_position("right")
axes[2].set_ylabel('Cumulative\nFraction',fontsize=24,rotation=90)
axes[2].legend(loc='best',frameon=False,prop={'size': 18})




#boxen plot
fig_table = pd.DataFrame({'Cryptic Phenotype Severity':obs_phenotype,'Genotype':genotype_strings[genotype]})
sns.boxenplot(x='Genotype',y='Cryptic Phenotype Severity',data=fig_table,scale='linear',order=genotype_strings,color=grey_color,saturation=1.0,showfliers=False,ax=axes[4],k_depth='proportion',outlier_prop=0.001)
q_array=np.zeros((len(quantiles),3))


for i,geno in enumerate(genotype_quantiles.keys()):
    q_array[:,i]=genotype_quantiles[geno]


for i,q in enumerate(quantiles):
    axes[4].plot(range(len(genotype_strings)),q_array[i],'-o',ms=15,mew=0.0,label='{0:g}th Percentile'.format(q*100),color=color_list[i*2])
axes[4].plot(range(len(genotype_strings)),genotype_means.values(),'*',ms=25,mew=0.0,label='Mean',color=red_color)
axes[4].legend(loc='best',frameon=False,prop={'size': 18})

linmod_p=linregress(genotype,obs_phenotype).pvalue
qrank=QRank(pd.DataFrame({'P':obs_phenotype}),quantiles=[0.99,0.995,0.999])
qrank.FitNullModels()
qrank_p=qrank.ComputePValues(genotype)[1]
axes[4].text(axes[4].get_xlim()[0]+0.25*(axes[4].get_xlim()[1]-axes[4].get_xlim()[0]),axes[4].get_ylim()[0]+0.9*(axes[4].get_ylim()[1]-axes[4].get_ylim()[0]),r'Linear Model $P$-value: {0:.2g}'.format(linmod_p),fontsize=12)
axes[4].text(axes[4].get_xlim()[0]+0.25*(axes[4].get_xlim()[1]-axes[4].get_xlim()[0]),axes[4].get_ylim()[0]+0.95*(axes[4].get_ylim()[1]-axes[4].get_ylim()[0]),r'QRank$_{Ext}$ '+r'$P$-value: {0:.2g}'.format(qrank_p),fontsize=12)



#Outlier Model
genotype=np.random.choice(np.arange(3),p=[(1-target_geno_freq)*(1-target_geno_freq),2.0*target_geno_freq*(1-target_geno_freq),target_geno_freq*target_geno_freq],size=N)
genotype_strings=np.array(['A1/A1','A1/A2','A2/A2'])


additive_background=np.random.normal(0.0,0.1,N)
mendelian_genotypes=np.random.binomial(1,mend_dis_freq,N)
mendelian_disease_effect=norm(0.0,1.0).isf(mend_dis_freq)
general_modifier_effects=np.random.normal(0.0,1.0,N)



base_phenotype=additive_background+mendelian_genotypes*mendelian_disease_effect
obs_phenotype=base_phenotype+norm(mendelian_disease_effect,1.0).cdf(base_phenotype)*(general_modifier_effects+genotype*target_effect_size)

genotype_quantiles={}
genotype_means={}
for i,geno in enumerate(genotype_strings):
    genotype_quantiles[geno]=mquantiles(obs_phenotype[genotype==i],prob=quantiles)
    genotype_means[geno]=np.mean(obs_phenotype[genotype==i])

bins=np.linspace(np.floor(obs_phenotype.min()),np.ceil(obs_phenotype.max()),50)
est_quantile_values=mquantiles(obs_phenotype,prob=[0.5,0.9,0.999],alphap=0.0,betap=1.0)


output_wo_dis=sns.distplot(obs_phenotype[mendelian_genotypes==0],kde=False,hist_kws={'log':True,'alpha':1.0},color=color_list[0],label = 'No Mendelian Disease',bins=bins,ax=axes[1])
output_w_dis=sns.distplot(obs_phenotype[mendelian_genotypes==1],kde=False,hist_kws={'log':True,'alpha':1.0},color=color_list[-2],label = 'With Mendelian Disease',bins=bins,ax=axes[1])

for i,q in enumerate([0.5,0.9,0.999]):
    top_val=np.where(est_quantile_values[i]<=bins)[0][0]
    bot_val=np.where(est_quantile_values[i]>=bins)[0][-1]
    mid_bin_val=(bins[top_val]+bins[bot_val])/2.0
    try:
        y_val=max([h.get_height() for h in output_wo_dis.patches][bot_val],[h.get_height() for h in output_w_dis.patches][bot_val])
    except NameError:
        y_val=[h.get_height() for h in output_wo_dis.patches][bot_val]
    axes[1].plot(mid_bin_val,y_val,'o',ms=15,mew=0.0,color=red_color)
    axes[1].text(mid_bin_val,y_val,'{0:g}th Percentile'.format(100*q),verticalalignment='bottom', horizontalalignment='left',rotation=30,color=red_color,fontweight='bold',fontsize=18)


axes[1].set_xlabel('Cryptic Phenotype Severity',fontsize=24)
axes[1].set_ylabel('Number of Patients\n'+r'($\log_{10}$-Scale)',fontsize=24)
axes[1].legend(loc='best',frameon=False,prop={'size': 18})



#cumulative fraction with Mendelian Disease
#cumulative fraction of modifier variance
#cdf function for Mendelian Disease
sorted_phenotypes = np.argsort(obs_phenotype)
cumulative_cases = np.cumsum(mendelian_genotypes[sorted_phenotypes])/mendelian_genotypes.sum()

scaled_modifier_effects=norm(mendelian_disease_effect,1.0).cdf(base_phenotype)*(general_modifier_effects+genotype*target_effect_size)

var_vec = (scaled_modifier_effects-np.mean(scaled_modifier_effects))**2.0
cumulative_modifier_variance=np.cumsum(var_vec[sorted_phenotypes])/(np.var(scaled_modifier_effects)*obs_phenotype.shape[0])

axes[3].plot(obs_phenotype[sorted_phenotypes],cumulative_cases,'-',lw=6.0,color=color_list[2],label='Mendelian Disease Cases')
axes[3].plot(obs_phenotype[sorted_phenotypes],cumulative_modifier_variance,'-',lw=6.0,color=color_list[4],label='Scaled Modifier Effect Variance\n'+r'(i.e. Var[$\phi(\beta,\theta)\times\alpha$])')
axes[3].set_xlabel('Cryptic Phenotype Severity',fontsize=24)
axes[3].yaxis.set_label_position("right")
axes[3].set_ylabel('Cumulative\nFraction',fontsize=24,rotation=90)
axes[3].legend(loc='best',frameon=False,prop={'size': 18})




fig_table = pd.DataFrame({'Cryptic Phenotype Severity':obs_phenotype,'Genotype':genotype_strings[genotype]})
sns.boxenplot(x='Genotype',y='Cryptic Phenotype Severity',data=fig_table,scale='linear',order=genotype_strings,color=grey_color,saturation=1.0,showfliers=False,ax=axes[5],k_depth='proportion',outlier_prop=0.001)
q_array=np.zeros((len(quantiles),3))

for i,geno in enumerate(genotype_quantiles.keys()):
    q_array[:,i]=genotype_quantiles[geno]

for i,q in enumerate(quantiles):
    axes[5].plot(range(len(genotype_strings)),q_array[i],'-o',ms=15,mew=0.0,label='{0:g}th Percentile'.format(q*100),color=color_list[i*2])
axes[5].plot(range(len(genotype_strings)),genotype_means.values(),'*',ms=25,mew=0.0,label='Mean',color=red_color)
axes[5].legend(loc='best',frameon=False,prop={'size': 18})

linmod_p=linregress(genotype,obs_phenotype).pvalue
qrank=QRank(pd.DataFrame({'P':obs_phenotype}),quantiles=[0.99,0.995,0.999])
qrank.FitNullModels()
qrank_p=qrank.ComputePValues(genotype)[1]
axes[5].text(axes[5].get_xlim()[0]+0.25*(axes[5].get_xlim()[1]-axes[5].get_xlim()[0]),axes[5].get_ylim()[0]+0.9*(axes[5].get_ylim()[1]-axes[5].get_ylim()[0]),r'Linear Model $P$-value: {0:.2g}'.format(linmod_p),fontsize=12)
axes[5].text(axes[5].get_xlim()[0]+0.25*(axes[5].get_xlim()[1]-axes[5].get_xlim()[0]),axes[5].get_ylim()[0]+0.95*(axes[5].get_ylim()[1]-axes[5].get_ylim()[0]),r'QRank$_{Ext}$ '+r'$P$-value: {0:.2g}'.format(qrank_p),fontsize=12)


f.savefig("ToyModelBase.svg")
plt.close('all')

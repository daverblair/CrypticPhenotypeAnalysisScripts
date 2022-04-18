import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from scipy.stats import linregress,ttest_ind, ttest_rel, pearsonr


"""
This script reproduces the analysis in Figures 2d, 2e and 2f.

"""

sns.set(context='talk',color_codes=True,style='ticks',font='Arial',font_scale=2,rc={'axes.linewidth':5,"font.weight":"bold",'axes.labelweight':"bold",'xtick.major.width':4,'xtick.minor.width': 2})
cmap = cm.get_cmap('viridis', 12)
color_list=[cmap(x) for x in [0.0,0.1,0.25,0.5,0.75,0.9,1.0]]
grey_color=(0.75, 0.75, 0.75)
red_color = '#FF5D5D'
blue_color='#5DA4FF'


ucsfResults=pd.read_pickle('path/to/UCSF/Step6/FinalModels_UCSFPerformanceResults.pth')
ukbbResults=pd.read_pickle('path/to/UKBB/Step6/UKBB-EffectiveRankReplicates/FinalModels_UKBBPerformanceResults.pth')
combined_inference_results=pd.read_pickle('path/to/table/containing/modeling/results/summary/ModelInferenceCombinedResults.pth'')
namesInheritanceFreq = pd.read_csv('path/to/SupplementaryDataFile_1.txt',sep='\t',index_col='Disease ID')


diseasesOfInterest=ucsfResults.loc[(ukbbResults['Top Component-UKBB']==ukbbResults['Top Component-UCSF'])].index
non_nan=ukbbResults.loc[diseasesOfInterest].loc[pd.isna(ukbbResults['UKBB Case Severity Increase'])==False].index
ukbb_v_ucsf_ucsf=ttest_ind(ukbbResults.loc[non_nan]['UKBB Case Severity Increase'].values,ucsfResults.loc[non_nan]['Case Severity Increase'],equal_var=False)
ukbb_v_ucsf_ukbb=ttest_ind(ukbbResults.loc[non_nan]['UKBB Case Severity Increase'].values,ukbbResults.loc[non_nan]['UCSF Case Severity Increase'],equal_var=False)
ucsf_ucsf_v_ucsf_ukbb=ttest_rel(ucsfResults.loc[diseasesOfInterest]['Case Severity Increase'].values,ukbbResults.loc[diseasesOfInterest]['UCSF Case Severity Increase'])

ukbb_v_ucsf_ucsf_corr=pearsonr(ukbbResults.loc[non_nan]['UKBB Case Severity Increase'].values,ucsfResults.loc[non_nan]['Case Severity Increase'])
ukbb_v_ucsf_ukbb_corr=pearsonr(ukbbResults.loc[non_nan]['UKBB Case Severity Increase'].values,ukbbResults.loc[non_nan]['UCSF Case Severity Increase'])
ucsf_ucsf_v_ucsf_ukbb_corr=pearsonr(ucsfResults.loc[diseasesOfInterest]['Case Severity Increase'].values,ukbbResults.loc[diseasesOfInterest]['UCSF Case Severity Increase'])


with open('Figures/Model_Dataset_Stats.txt','w') as f:
    f.write('Comparison\tModel/Dataset-1 Mean\tModel/Dataset-2 Mean\tTest\tStatistic\tP-value\tPearson Corr.\tCorr. P-value\n')

    f.write('UKBB Data/UKBB Model vs UCSF Data/UCSF Model\t{0:.3f}\t{1:.4f}\tUn-paired (unequal var)\t{2:.3f}\t{3:.3f}\t{4:.2f}\t{5:.2e}\n'.format(ukbbResults.loc[non_nan]['UKBB Case Severity Increase'].mean(),ucsfResults.loc[non_nan]['Case Severity Increase'].mean(),ukbb_v_ucsf_ucsf[0],ukbb_v_ucsf_ucsf[1],ukbb_v_ucsf_ucsf_corr[0],ukbb_v_ucsf_ucsf_corr[1]))

    f.write('UKBB Data/UKBB Model vs UCSF Data/UKBB Model\t{0:.3f}\t{1:.3f}\tUn-paired (unequal var)\t{2:.3f}\t{3:.4f}\t{4:.2f}\t{5:.2e}\n'.format(ukbbResults.loc[diseasesOfInterest]['UKBB Case Severity Increase'].mean(),ukbbResults.loc[diseasesOfInterest]['UCSF Case Severity Increase'].mean(),ukbb_v_ucsf_ukbb[0],ukbb_v_ucsf_ukbb[1],ukbb_v_ucsf_ukbb_corr[0],ukbb_v_ucsf_ukbb_corr[1]))


    f.write('UCSF Data/UCSF Model vs UCSF Data/UKBB Model\t{0:.3f}\t{1:.3f}\tPaired\t{2:.3f}\t{3:.4f}\t{4:.2f}\t{5:.2e}\n'.format(ucsfResults.loc[diseasesOfInterest]['Case Severity Increase'].mean(),ukbbResults.loc[diseasesOfInterest]['UCSF Case Severity Increase'].mean(),ucsf_ucsf_v_ucsf_ukbb[0],ucsf_ucsf_v_ucsf_ukbb[1],ucsf_ucsf_v_ucsf_ukbb_corr[0],ucsf_ucsf_v_ucsf_ukbb_corr[1]))



f, axes = plt.subplots(1, 3,figsize=(30,8))
f.tight_layout(rect=[0.1,0.1,0.9,0.9])
for axis in axes:
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)


axes[0].set_xlim([-0.5,2.5])
axes[0].set_ylim([-0.5,2.5])
axes[0].hlines(0.0,xmin=-0.5,xmax=2.5,lw=5.0,ls='--',color=red_color)
axes[0].vlines(0.0,ymin=-0.5,ymax=2.5,lw=5.0,ls='--',color=red_color)
axes[0].scatter(ucsfResults.loc[diseasesOfInterest]['Case Severity Increase'],ukbbResults.loc[diseasesOfInterest]['UKBB Case Severity Increase'],color=color_list[0],marker='o',s=300.0)
axes[0].plot([-0.5,2.5],[-0.5,2.5],'--',lw=5.0,color=grey_color)


axes[0].vlines(ucsfResults.loc[diseasesOfInterest]['Case Severity Increase'],ymin=[x[0] for x in ukbbResults.loc[diseasesOfInterest]['UKBB Case Severity Increase (95% CI)']],ymax=[x[1] for x in ukbbResults.loc[diseasesOfInterest]['UKBB Case Severity Increase (95% CI)']],ls='-',color=color_list[0],alpha=0.5,lw=5.0)


axes[0].hlines(ukbbResults.loc[diseasesOfInterest]['UKBB Case Severity Increase'],xmin=[x[0] for x in ucsfResults.loc[diseasesOfInterest]['Case Severity Increase (95% CI)']],xmax=[x[1] for x in ucsfResults.loc[diseasesOfInterest]['Case Severity Increase (95% CI)']],ls='-',color=color_list[0],alpha=0.5,lw=5.0)
axes[0].set_xlabel('UCSF Dataset,\nUCSF Model')
axes[0].set_ylabel('UKBB Dataset,\nUKBB Model')
axes[0].set_title('Increase in\nCryptic Phenotype Severity\nAmong Diagnosed Cases',fontweight='bold',fontsize=24)


axes[1].set_xlim([-0.5,2.5])
axes[1].set_ylim([-0.5,2.5])
axes[1].hlines(0.0,xmin=-0.5,xmax=2.5,lw=5.0,ls='--',color=red_color)
axes[1].vlines(0.0,ymin=-0.5,ymax=2.5,lw=5.0,ls='--',color=red_color)
axes[1].scatter(ukbbResults.loc[diseasesOfInterest]['UCSF Case Severity Increase'],ukbbResults.loc[diseasesOfInterest]['UKBB Case Severity Increase'],color=color_list[2],marker='o',s=300.0)
axes[1].plot([-0.5,2.5],[-0.5,2.5],'--',lw=5.0,color=grey_color)


axes[1].hlines(ukbbResults.loc[diseasesOfInterest]['UKBB Case Severity Increase'],xmin=[x[0] for x in ukbbResults.loc[diseasesOfInterest]['UCSF Case Severity Increase (95% CI)']],xmax=[x[1] for x in ukbbResults.loc[diseasesOfInterest]['UCSF Case Severity Increase (95% CI)']],ls='-',color=color_list[2],alpha=0.5,lw=5.0)


axes[1].vlines(ukbbResults.loc[diseasesOfInterest]['UCSF Case Severity Increase'],ymin=[x[0] for x in ukbbResults.loc[diseasesOfInterest]['UKBB Case Severity Increase (95% CI)']],ymax=[x[1] for x in ukbbResults.loc[diseasesOfInterest]['UKBB Case Severity Increase (95% CI)']],ls='-',color=color_list[2],alpha=0.5,lw=5.0)
axes[1].set_xlabel('UCSF Dataset,\nUKBB Model')
axes[1].set_ylabel('UKBB Dataset,\nUKBB Model')
axes[1].set_title('Increase in\nCryptic Phenotype Severity\nAmong Diagnosed Cases',fontweight='bold',fontsize=24)


output_w_dis=sns.histplot(ukbbResults['UCSF-UKBB Model R^2'],kde=False,color=color_list[0],bins=np.linspace(0,1.0,6),ax=axes[2])
axes[2].vlines(0.2,ymin=0,ymax=axes[2].get_ylim()[1],color=red_color,lw=5.0,ls='--')
axes[2].set_xlabel(r'$r^{2}$ Between'+'\nUCSF and UKBB Models')
axes[2].set_ylabel('# of Diseases')
axes[2].text(0.2,axes[2].get_ylim()[1]+0.5,r'$r^{2}$=0.2',fontweight='bold',fontsize=20)
axes[2].set_title('Comparison of\nCryptic Phenotypes\nAcross Models',fontweight='bold',fontsize=24)

plt.savefig('Figures/GlobalModelComparison.svg')
plt.close()
## Compare Phenotype increase among cases in two datasets

import pandas as pd
import argparse
import numpy as np
from scipy.stats.mstats import mquantiles
from scipy.stats import spearmanr,chi2,beta
import sys
new_path='/Users/davidblair/Desktop/Research/MendelianDiseaseProject/Software/AuxillaryFunctions'
if new_path not in sys.path:
    sys.path.append(new_path)
from FirthRegression import FirthRegression
from GWASPlots import ManhattanPlot,QQPlot
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import os

sns.set(context='talk',color_codes=True,style='ticks',font='Arial',font_scale=2,rc={'axes.linewidth':5,"font.weight":"bold",'axes.labelweight':"bold",'xtick.major.width':4,'xtick.minor.width': 2})
cmap = cm.get_cmap('viridis', 12)
color_list=[cmap(x) for x in [0.0,0.1,0.25,0.5,0.75,0.9,1.0]]
grey_color=(0.25, 0.25, 0.25)
red_color = '#FF5D5D'
blue_color='#5DA4FF'


name_dict={'OMIM_ICD:86':'A1ATD','OMIM_ICD:108':'HHT','OMIM_ICD:120':'MFS','OMIM_ICD:121':'AS','OMIM_ICD:132':'AD-PCKD'}

for dis_ind in ['OMIM_ICD:108','OMIM_ICD:120']:
    results=pd.read_csv('../'+dis_ind.replace(':','_')+'/SummaryStats_Training.txt',sep='\t')
    results.set_index('Predictor',inplace=True)
    heritability = pd.read_csv('../'+dis_ind.replace(':','_')+'/LDAK_Heritability/ldak-thin-genotyped.hers',sep=' ')
    f,ax=QQPlot(results,error_type='theoretical',freq_bins=[0.01,0.05,0.5],lambda_gc_scale=10000)
    ax.set_title(name_dict[dis_ind],fontsize=40,fontweight='bold')
    ax.text(0.2,8.0,r'$h^{2}=$'+'{0:.3f} ({1:.4f} s.d.)'.format(heritability.iloc[1]['Heritability'], heritability.iloc[1]['Her_SD']),fontsize=20)
    plt.savefig('../'+dis_ind.replace(':','_')+'/QQPlot.svg')
    plt.close()

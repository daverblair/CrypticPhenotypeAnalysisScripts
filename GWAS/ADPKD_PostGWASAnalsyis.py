import pandas as pd
from statsmodels.api import OLS,Logit
import statsmodels as sm
from lifelines import CoxPHFitter,KaplanMeierFitter
import sys
import os
import copy
new_path='/Users/davidblair/Desktop/Research/MendelianDiseaseProject/Software/AuxillaryFunctions'
if new_path not in sys.path:
    sys.path.append(new_path)
from FirthRegression import FirthRegression
from GWASPlots import ManhattanPlot,QQPlot
import numpy as np
from vlpi.data.ClinicalDataset import ClinicalDataset
from AuxFuncs import assign_quantiles,_parseFloats,_dateParse,LRTest,_bpParse,_compute_eGFR

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import argparse
from scipy.stats import ttest_ind

parser = argparse.ArgumentParser(description='Script that performs post-hoc analysis for alport syndrome')
parser.add_argument("--includeFlagged", help="Indicates whether or not to include flagged variants in the analysis. Note, the analyses in Blair et al. 2022 excluded all flagged variants, as they did not have a detectable phenotypic effect in exome validation analyses. Therefore, they likely represent annotation noise. To replicate, set ---includeFlagged=False.")
args = parser.parse_args()

if args.includeFlagged:
    includeFlagged=True
else:
    includeFlagged=False

sns.set(context='talk',color_codes=True,style='ticks',font='Arial',font_scale=2,rc={'axes.linewidth':5,"font.weight":"bold",'axes.labelweight':"bold",'xtick.major.width':4,'xtick.minor.width': 2})
cmap = cm.get_cmap('viridis', 12)
color_list=[cmap(x) for x in [0.0,0.1,0.25,0.5,0.75,0.9,1.0]]
grey_color=(0.25, 0.25, 0.25)
red_color = '#d10e00'
blue_color='#5DA4FF'


fig_direc='/path/to/output/Figures'+includeFlagged*'_FlaggedIncluded'+'/'

table_direc='/path/to/output/Tables'+includeFlagged*'_FlaggedIncluded'+'/'
try:
    os.mkdir(fig_direc)
except FileExistsError:
    pass

try:
    os.mkdir(table_direc)
except FileExistsError:
    pass

#################### Section 1: Build Post-hoc Dataset ################


gwas_data=pd.read_pickle('/path/to/gwas/data/GWASCovariatesPhenotypes.pth')
validation_scores=pd.read_csv('path/to/validation/scores/bolt_model_validation.profile',sep='\t')
validation_scores.set_index('ID1',inplace=True)
target_scores=pd.read_csv('~path/to/target/scores/bolt_model_target.profile',sep='\t')
target_scores.set_index('ID1',inplace=True)
aux_data=pd.read_csv('/path/to/additional/pheno/data/ADPCKD_FollowUpData.txt',sep='\t')
aux_data.set_index('eid',inplace=True)


genotype_table=pd.read_pickle('path/to/ADPKD/rare/variant/data/GeneticCarriers.pth')
genotype_table=genotype_table[~genotype_table.index.duplicated(keep='first')]
genotype_table.set_index(np.array(genotype_table.index,dtype=np.int),inplace=True)
if includeFlagged==False:
    unflagged=genotype_table.loc[genotype_table.FLAGS=='NaN'].index
    flagged=genotype_table.loc[genotype_table.FLAGS!='NaN'].index
    genotype_table.drop(labels=flagged,inplace=True)
    gwas_data.drop(labels=flagged,inplace=True, errors='ignore')
    aux_data.drop(labels=flagged,inplace=True, errors='ignore')
    validation_scores.drop(labels=flagged,inplace=True, errors='ignore')
    target_scores.drop(labels=flagged,inplace=True, errors='ignore')


obs_years=pd.Series(np.zeros(aux_data.shape[0]),index=aux_data.index)
obs_years[pd.isnull(aux_data['Death Date'])]=2020-aux_data.loc[pd.isnull(aux_data['Death Date'])]['Birth Year']
obs_years[pd.isnull(aux_data['Death Date'])==False]=aux_data.loc[pd.isnull(aux_data['Death Date'])==False]['Death Date'].apply(lambda x:int(x.split('-')[0]))-aux_data.loc[pd.isnull(aux_data['Death Date'])==False]['Birth Year']
aux_data['Observation Window']=obs_years


# build the analysis table
analysis_table=gwas_data.loc[validation_scores.index.union(target_scores.index)][['sex','age_normalized','array']+['PC{0:d}'.format(x) for x in range(1,11)]+['CrypticPhenotype_OMIM_ICD:132']]


analysis_table.rename(columns={'CrypticPhenotype_OMIM_ICD:132':'Cryptic Phenotype'},inplace=True)

geno_vec=pd.Series(['Control']*analysis_table.shape[0],index=analysis_table.index)
geno_vec.loc[genotype_table.index.intersection(analysis_table.index)]='P/LP Carrier'
analysis_table['Genotype']=geno_vec
analysis_table=pd.concat([analysis_table,pd.get_dummies(analysis_table['Genotype'], drop_first=False)],axis=1)
gene_vec=pd.Series([np.nan]*analysis_table.shape[0],index=analysis_table.index)
gene_vec.loc[genotype_table.index.intersection(analysis_table.index)]=genotype_table.loc[genotype_table.index.intersection(analysis_table.index)]['GENE']
analysis_table['Affected Gene']=gene_vec

#smoking
smoker_key={'0':0,'1':1,'0,1':1,'1,0':1,np.nan:np.nan}
analysis_table['Smoking-Ever Smoked']=np.array([smoker_key[x] for x in aux_data.loc[analysis_table.index]['Smoking-Ever Smoked'].values])
pack_years=np.array([_parseFloats(x) for x  in aux_data.loc[analysis_table.index]['Smoking-Pack Years']])
analysis_table['Smoking-Pack Years']=pack_years
analysis_table.loc[analysis_table['Smoking-Ever Smoked']==0,'Smoking-Pack Years']=0.0


#blood pressure
sys_bp=aux_data.loc[analysis_table.index]['Systolic BP'].apply(_bpParse)
dia_bp=aux_data.loc[analysis_table.index]['Diastolic BP'].apply(_bpParse)
mean_art_pressure=(2/3)*dia_bp+(1/3)*sys_bp
analysis_table['Systolic BP']=sys_bp
analysis_table['Diastolic BP']=dia_bp
analysis_table['Mean Arterial Pressure']=mean_art_pressure

#lab values
#
#for microalbumin, use most recent measurement given known progression over disease with time. quantile normalized as distribution is very skewed
microalbumin=np.array([_parseFloats(x,useMax=False) for x  in aux_data.loc[analysis_table.index]['Urine Microalbumin']],dtype=np.float)
micro_age=np.array([_dateParse(x) for x  in aux_data.loc[analysis_table.index]['Urine Collection Date']])-aux_data.loc[analysis_table.index]['Birth Year'].values
analysis_table['Urine Microalbumin']=microalbumin
analysis_table['Urine Measurement Age']=micro_age


#for eGFR, again use most recent measurement given known progression over time.
creatinine=np.array([_parseFloats(x,useMax=False) for x  in aux_data.loc[analysis_table.index]['Serum Creatinine']])
creatinine_age=np.array([_dateParse(x) for x  in aux_data.loc[analysis_table.index]['Serum Collection Date']])-aux_data.loc[analysis_table.index]['Birth Year'].values
analysis_table['Serum Creatinine']=creatinine
analysis_table['Serum Measurement Age']=creatinine_age
egfr=np.array([_compute_eGFR(x[1][0],x[1][1],x[1][2]) if pd.isnull(x[1][2])==False else np.nan for x in analysis_table[['Serum Measurement Age','sex','Serum Creatinine']].iterrows()])
analysis_table['eGFR']=egfr

#algorithmically defined outcome ESRD
analysis_table['Obs Windows-ESRD (Algorithm)']=aux_data.loc[analysis_table.index]['Age ESRD (Algorithm)'].apply(_dateParse)-aux_data.loc[analysis_table.index]['Birth Year']
analysis_table.loc[analysis_table['Obs Windows-ESRD (Algorithm)']<0,'Obs Windows-ESRD (Algorithm)']=0


has_esrd=pd.Series(np.zeros(analysis_table.shape[0]),index=analysis_table.index)
has_esrd.loc[analysis_table.loc[pd.isnull(analysis_table['Obs Windows-ESRD (Algorithm)'])==False].index]=1
analysis_table['ESRD Dx (Algorithm)']=has_esrd
analysis_table.loc[pd.isnull(analysis_table['Obs Windows-ESRD (Algorithm)']),'Obs Windows-ESRD (Algorithm)']=aux_data.loc[analysis_table.loc[pd.isnull(analysis_table['Obs Windows-ESRD (Algorithm)'])].index]['Observation Window']

#first cystic kidney disease dx
analysis_table['Obs Windows-Cystic Kidney Disease (Q61)']=aux_data.loc[analysis_table.index]['Cystic Kidney Disease (Q61) Dx Data'].apply(_dateParse)-aux_data.loc[analysis_table.index]['Birth Year']
analysis_table.loc[analysis_table['Obs Windows-Cystic Kidney Disease (Q61)']<0,'Obs Windows-Cystic Kidney Disease (Q61)']=0

has_cystic_kidney=pd.Series(np.zeros(analysis_table.shape[0]),index=analysis_table.index)
has_cystic_kidney.loc[analysis_table.loc[pd.isnull(analysis_table['Obs Windows-Cystic Kidney Disease (Q61)'])==False].index]=1
analysis_table['Cystic Kidney Dx (Q61)']=has_cystic_kidney
analysis_table.loc[pd.isnull(analysis_table['Obs Windows-Cystic Kidney Disease (Q61)']),'Obs Windows-Cystic Kidney Disease (Q61)']=aux_data.loc[analysis_table.loc[pd.isnull(analysis_table['Obs Windows-Cystic Kidney Disease (Q61)'])].index]['Observation Window']

#specific PCKD diagnosis

analysis_table['PCKD Dx']=gwas_data.loc[analysis_table.index]['has_OMIM_ICD:132']


#add PGS scores
pgs_scores=pd.Series(np.zeros((analysis_table.shape[0])),index=analysis_table.index)
pgs_scores.loc[validation_scores.index]=validation_scores.Profile_1
pgs_scores.loc[target_scores.index]=target_scores.Profile_1
analysis_table['PGS']=(pgs_scores-pgs_scores.mean())/pgs_scores.std(ddof=0)

#add interaction terms
analysis_table['PGS x P/LP Carrier']=analysis_table['PGS']*analysis_table['P/LP Carrier']
analysis_table['Smoking x P/LP Carrier']=analysis_table['Smoking-Ever Smoked']*analysis_table['P/LP Carrier']
analysis_table['Pack-years x P/LP Carrier']=analysis_table['Smoking-Pack Years']*analysis_table['P/LP Carrier']

#### drop related samples
allowed_samples=pd.read_csv('filtered_subject_ids_target_related.txt',sep='\t')
analysis_table=analysis_table.loc[analysis_table.index.intersection(allowed_samples.IID)]


pack_year_table=analysis_table[pd.isnull(analysis_table['Smoking-Pack Years'])==False].copy()
ever_smoked_table=analysis_table[pd.isnull(analysis_table['Smoking-Ever Smoked'])==False].copy()
gfr_table=ever_smoked_table[pd.isnull(ever_smoked_table['eGFR'])==False].copy()
albumin_table=ever_smoked_table[pd.isnull(ever_smoked_table['Urine Microalbumin'])==False].copy()
bp_table=ever_smoked_table[pd.isnull(ever_smoked_table['Mean Arterial Pressure'])==False].copy()


#################### Section 1: GWAS Plots #############
#######################################################################

gwas_data=pd.read_csv('/path/to/summary/stats/SummaryStats_Training.txt',sep='\t')
gwas_data['CHROM']=gwas_data.Predictor.apply(lambda x:int(x.split(':')[0]))
gwas_data['POS']=gwas_data.Predictor.apply(lambda x:int(x.split(':')[1]))
gwas_data.set_index('Predictor',inplace=True)

ind_snp_file=pd.read_csv('/path/to/FUMA/results/FUMA/IndSigSNPs.txt',sep='\t')
ind_snp_file.set_index('rsID',inplace=True)
ind_snps=ind_snp_file['uniqID'].apply(lambda x:':'.join(x.split(':')[0:2]))


corr_snp_file=pd.read_csv('/path/to/FUMA/results/FUMA/snps.txt',sep='\t')
corr_snp_file.set_index('rsID',inplace=True)
corr_snps=corr_snp_file['uniqID'].apply(lambda x:':'.join(x.split(':')[0:2]))

is_sig=pd.Series(np.zeros(gwas_data.shape[0],dtype=np.int32),index=gwas_data.index)
is_sig.loc[is_sig.index.intersection(corr_snps)]=1
is_sig.loc[is_sig.index.intersection(ind_snps)]=1
is_sig.loc[gwas_data.P<=5e-8]=1 #FUMA excludes MHC by default, just adds it back
gwas_data['Sig_SNPs']=is_sig
gene_file=pd.read_csv('/path/to/FUMA/results/FUMA/genes.txt',sep='\t')


f,axis=ManhattanPlot(gwas_data,all_sig_thresh=[5e-8],marked_column='Sig_SNPs',hide_hla=False)
f.savefig(fig_direc+'ManhattanPlot_wGenes.svg')
plt.close()

heritability = pd.read_csv('/path/to/ADPKD/LDAK_Heritability/ldak-thin-genotyped.hers',sep=' ')
f,ax=QQPlot(gwas_data,error_type='theoretical',freq_bins=[0.01,0.05,0.5],lambda_gc_scale=10000)
ax.set_title('AD-PCKD Syndrome',fontsize=40,fontweight='bold')
ax.text(0.2,8.0,r'$h^{2}=$'+'{0:.3f} ({1:.4f} s.d.)'.format(heritability.iloc[1]['Heritability'], heritability.iloc[1]['Her_SD']),fontsize=20)
plt.savefig(fig_direc+'QQPlot.svg')
plt.close()

#######################################################################
#######################################################################


#######################################################################
#################### Section 2: PGS Replication and Validation ########
#######################################################################
with open(table_direc+'PGS_Validation_Results.txt','w') as ofile:
    pgs_lin_mod=OLS(analysis_table['PGS'],exog=sm.tools.add_constant(analysis_table[['P/LP Carrier']]),hasconst=True).fit()


    ofile.write('P/LP Carrier Effect on PGS (N={3:d}): {0:.3f} ({1:.3f}; {2:.2e})\n'.format(pgs_lin_mod.params['P/LP Carrier'],pgs_lin_mod.bse['P/LP Carrier'],pgs_lin_mod.pvalues['P/LP Carrier'],analysis_table.shape[0]))
    ofile.write('\n\n')



    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=2)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    v=sns.violinplot(x='Genotype',y='PGS',data=analysis_table, palette=[color_list[0],color_list[2]],ax=axis)
    for violin, alpha in zip(v.collections[::2], [0.4,0.6]):
        violin.set_alpha(alpha)

    axis.set_ylabel('Cryptic Phenotype\nPolygenic Score')
    axis.text(1,4.0,r'P-Value='+'{0:.2f}'.format(pgs_lin_mod.pvalues['P/LP Carrier']),fontsize=18)
    plt.savefig(fig_direc+'PGS_vs_AS_Geno.svg')
    plt.close()


    ######## Main Cryptic Phenotype Modeling ########
    #################################################

    cp_lin_mod_base=OLS(analysis_table['Cryptic Phenotype'],exog=sm.tools.add_constant(analysis_table[['sex','age_normalized','array']+['PC{0:d}'.format(i) for i in range(1,11)]]),hasconst=True).fit()

    cp_lin_mod_geno_marg=OLS(analysis_table['Cryptic Phenotype'],exog=sm.tools.add_constant(analysis_table[['sex','age_normalized','array']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier']]),hasconst=True).fit()


    ofile.write("#"*5+" Baseline Model (N={0:d}) ".format(analysis_table.shape[0])+"#"*5+'\n')
    ofile.write('Marginal P/LP Carrier CP Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(cp_lin_mod_geno_marg.params['P/LP Carrier'],cp_lin_mod_geno_marg.bse['P/LP Carrier'],cp_lin_mod_geno_marg.pvalues['P/LP Carrier']))
    ofile.write('\n'*3)

    cp_lin_mod_geno_pgs_int=OLS(analysis_table['Cryptic Phenotype'],exog=sm.tools.add_constant(analysis_table[['sex','age_normalized','array']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier','PGS','PGS x P/LP Carrier']]),hasconst=True).fit()
    ofile.write("#"*5+" Carrier Status plus PGS Model (N={0:d}) ".format(analysis_table.shape[0])+"#"*5+'\n')
    ofile.write('Marginal P/LP Carrier CP Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(cp_lin_mod_geno_pgs_int.params['P/LP Carrier'],cp_lin_mod_geno_pgs_int.bse['P/LP Carrier'],cp_lin_mod_geno_pgs_int.pvalues['P/LP Carrier']))
    ofile.write('Marginal PGS CP Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(cp_lin_mod_geno_pgs_int.params['PGS'],cp_lin_mod_geno_pgs_int.bse['PGS'],cp_lin_mod_geno_pgs_int.pvalues['PGS']))
    ofile.write('PGS x P/LP Carrier CP Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(cp_lin_mod_geno_pgs_int.params['PGS x P/LP Carrier'],cp_lin_mod_geno_pgs_int.bse['PGS x P/LP Carrier'],cp_lin_mod_geno_pgs_int.pvalues['PGS x P/LP Carrier']))
    ofile.write('\n'*3)

    firth = FirthRegression(ever_smoked_table,['sex','age_normalized','array']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier'],'Smoking-Ever Smoked',hasconst=False)
    plp_smoke_effect=firth.FirthInference('P/LP Carrier')


    pack_year_model=OLS(pack_year_table['Smoking-Pack Years'],exog=sm.tools.add_constant(pack_year_table[['sex','age_normalized','array']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier']]),hasconst=True).fit()


    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=4)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    p=sns.pointplot(y='Smoking-Ever Smoked',x='Genotype',data=ever_smoked_table,palette=[color_list[0],color_list[2]],ax=axis,errwidth=10.0,scale=2.0,join=False,markers=['o','^'])
    axis.text(0.5,axis.get_ylim()[0]+1.25*(axis.get_ylim()[1]-axis.get_ylim()[0]),'P/LP P-value={0:.2e}'.format(plp_smoke_effect['PVal']),fontsize=30,fontweight='bold')
    axis.set_ylabel('Proportion\nof Smokers')
    axis.set_xlabel('Genotype')
    f.savefig(fig_direc+'Genotype_v_EverSmoked.svg')
    plt.close()

    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=4)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    p=sns.pointplot(y='Smoking-Pack Years',x='Genotype',data=pack_year_table,palette=[color_list[0],color_list[2]],ax=axis,errwidth=10.0,scale=2.0,join=False,markers=['o','^'])
    axis.text(0.5,axis.get_ylim()[0]+1.05*(axis.get_ylim()[1]-axis.get_ylim()[0]),'P/LP P-value={0:.2e}'.format(pack_year_model.pvalues['P/LP Carrier']),fontsize=30,fontweight='bold')
    axis.set_ylabel('Smoking\nPack-Years')
    axis.set_xlabel('Genotype')
    f.savefig(fig_direc+'Genotype_v_PackYears.svg')
    plt.close()


    firth = FirthRegression(ever_smoked_table,['sex','age_normalized','array']+['PC{0:d}'.format(i) for i in range(1,11)]+['PGS'],'Smoking-Ever Smoked',hasconst=False)
    pgs_smoke_effect=firth.FirthInference('PGS')
    ever_smoked_table['PGS Quantiles']=assign_quantiles(ever_smoked_table,'PGS',5)

    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=2)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    sns.pointplot(x='PGS Quantiles',y='Smoking-Ever Smoked',data=ever_smoked_table,color=blue_color,ax=axis,scale=2.0,errwidth=10.0,join=False)
    axis.text(axis.get_xlim()[0]+0.25*(axis.get_xlim()[1]-axis.get_xlim()[0]),axis.get_ylim()[0]+0.75*(axis.get_ylim()[1]-axis.get_ylim()[0]),'P-value={0:.2e}'.format(pgs_smoke_effect['PVal']))
    axis.set_ylabel('Proportion\nEver-Smoked')
    axis.set_xlabel('Polygenic Score\nQuantiles')
    f.savefig(fig_direc+'SmokingProportion_v_PGS.svg')
    plt.close()

    ofile.write('PGS Ever-Smoked Effect (N={3:d}): {0:.3f} ({1:.3f}; {2:.2e})\n'.format(pgs_smoke_effect['ParamTable'].loc['PGS'].BETA,pgs_smoke_effect['ParamTable'].loc['PGS'].SE,pgs_smoke_effect['PVal'],ever_smoked_table.shape[0]))


    pack_years_mod=OLS(pack_year_table['Smoking-Pack Years'],exog=sm.tools.add_constant(pack_year_table[['sex','age_normalized','array']+['PC{0:d}'.format(i) for i in range(1,11)]+['PGS']]),hasconst=True).fit()
    pack_year_table['Pack-Years (Residuals)']=OLS(pack_year_table['Smoking-Pack Years'],exog=sm.tools.add_constant(pack_year_table[['sex','age_normalized','array']+['PC{0:d}'.format(i) for i in range(1,11)]]),hasconst=True).fit().resid
    pack_year_table['PGS Quantiles']=assign_quantiles(pack_year_table,'PGS',5)

    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=2)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    sns.pointplot(x='PGS Quantiles',y='Pack-Years (Residuals)',data=pack_year_table,color=red_color,ax=axis,scale=2.0,errwidth=10.0,join=False)
    axis.text(axis.get_xlim()[0]+0.25*(axis.get_xlim()[1]-axis.get_xlim()[0]),axis.get_ylim()[0]+0.75*(axis.get_ylim()[1]-axis.get_ylim()[0]),'P-value={0:.2e}'.format(pack_years_mod.pvalues.PGS))
    axis.set_ylabel('Residual Pack-years\n(Quantile-Normalized)')
    axis.set_xlabel('Polygenic Score\nQuantiles')
    f.savefig(fig_direc+'Packyears_v_PGS.svg')
    plt.close()


    ofile.write('PGS Pack-years Effect (N={3:d}): {0:.3f} ({1:.3f}; {2:.2e})\n'.format(pack_years_mod.params.PGS,pack_years_mod.bse.PGS,pack_years_mod.pvalues.PGS,pack_year_table.shape[0]))
    ofile.write('\n'*3)


    cp_lin_mod_w_smoking=OLS(ever_smoked_table['Cryptic Phenotype'],exog=sm.tools.add_constant(ever_smoked_table[['sex','age_normalized','array','Smoking-Ever Smoked']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier', 'Smoking x P/LP Carrier']]),hasconst=True).fit()

    ofile.write("#"*5+" Ever-Smoked Model (N={0:d}) ".format(ever_smoked_table.shape[0])+"#"*5+'\n')
    ofile.write('Marginal P/LP CP Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(cp_lin_mod_w_smoking.params['P/LP Carrier'],cp_lin_mod_w_smoking.bse['P/LP Carrier'],cp_lin_mod_w_smoking.pvalues['P/LP Carrier']))
    ofile.write('P/LP x Smoking CP Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(cp_lin_mod_w_smoking.params['Smoking x P/LP Carrier'],cp_lin_mod_w_smoking.bse['Smoking x P/LP Carrier'],cp_lin_mod_w_smoking.pvalues['Smoking x P/LP Carrier']))
    ofile.write('\n'*3)



    cp_lin_mod_w_pack_years=OLS(pack_year_table['Cryptic Phenotype'],exog=sm.tools.add_constant(pack_year_table[['sex','age_normalized','array']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier','Smoking-Pack Years','Pack-years x P/LP Carrier']]),hasconst=True).fit()

    ofile.write("#"*5+" Pack-Years Model (N={0:d}) ".format(pack_year_table.shape[0])+"#"*5+'\n')
    ofile.write('Marginal P/LP CP Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(cp_lin_mod_w_pack_years.params['P/LP Carrier'],cp_lin_mod_w_pack_years.bse['P/LP Carrier'],cp_lin_mod_w_pack_years.pvalues['P/LP Carrier']))
    ofile.write('P/LP x Pack-years CP Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(cp_lin_mod_w_pack_years.params['Pack-years x P/LP Carrier'],cp_lin_mod_w_pack_years.bse['Pack-years x P/LP Carrier'],cp_lin_mod_w_pack_years.pvalues['Pack-years x P/LP Carrier']))
    ofile.write('\n'*3)


    global_cp_model=OLS(ever_smoked_table['Cryptic Phenotype'],exog=sm.tools.add_constant(ever_smoked_table[['sex','age_normalized','array','Smoking-Ever Smoked']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier','PGS','PGS x P/LP Carrier','Smoking x P/LP Carrier']]),hasconst=True).fit()
    global_cp_model=OLS(pack_year_table['Cryptic Phenotype'],exog=sm.tools.add_constant(pack_year_table[['sex','age_normalized','array','Smoking-Pack Years']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier','PGS','PGS x P/LP Carrier','Pack-Years x P/LP Carrier']]),hasconst=True).fit()


    ofile.write("#"*5+" Final Model (N={0:d}) ".format(ever_smoked_table.shape[0])+"#"*5+'\n')
    for param in global_cp_model.params.keys():
        ofile.write('{3:s} Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(global_cp_model.params[param],global_cp_model.bse[param],global_cp_model.pvalues[param],param))
    ofile.write('\n')

    global_cp_model_minus_pgs=OLS(ever_smoked_table['Cryptic Phenotype'],exog=sm.tools.add_constant(ever_smoked_table[['sex','age_normalized','array','Smoking-Ever Smoked','Smoking x P/LP Carrier']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier']]),hasconst=True).fit()

    global_cp_model_minus_pgs_interaction=OLS(ever_smoked_table['Cryptic Phenotype'],exog=sm.tools.add_constant(ever_smoked_table[['sex','age_normalized','array','Smoking-Ever Smoked','Smoking x P/LP Carrier']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier','PGS']]),hasconst=True).fit()

    lr_test_all_pgs=global_cp_model.compare_lr_test(global_cp_model_minus_pgs)
    lr_test_pgs_interaction=global_cp_model.compare_lr_test(global_cp_model_minus_pgs_interaction)

    ofile.write('Aggregate PGS Effect P-Value (LR Test): {0:.2e}\n'.format(lr_test_all_pgs[1]))
    ofile.write('PGS Interaction Effects P-Value (LR Test): {0:.2e}\n'.format(lr_test_pgs_interaction[1]))
    ofile.write('\n'*3)

    ever_smoked_table['CP Residuals (Adj. for Smoking Status)']=global_cp_model_minus_pgs.resid
    ever_smoked_table['PGS Quantiles']=assign_quantiles(ever_smoked_table,'PGS',3)

    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=2)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    sns.pointplot(x='PGS Quantiles',y='CP Residuals (Adj. for Smoking Status)',data=ever_smoked_table,hue='Genotype',palette=[color_list[0],color_list[2]],ax=axis,scale=3.0,errwidth=15.0,join=False,markers=['o','^'],dodge=0.25)
    axis.set_ylabel('Cryptic Phenotype\nResiduals\n(Adj. for Smoking Status)')
    axis.set_xlabel('PGS Quantiles')
    f.savefig(fig_direc+'CP_v_PGS.svg')
    plt.close()


    ######## Ancillary Modeling ########
    ####################################

    ofile.write("#"*5+" Ancillary Modeling "+"#"*5+'\n\n\n')


    albumin_full_model=OLS(albumin_table['Urine Microalbumin'],exog=sm.tools.add_constant(albumin_table[['sex','Urine Measurement Age','array','Smoking-Ever Smoked','Smoking x P/LP Carrier']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier','PGS','PGS x P/LP Carrier']]),hasconst=True).fit()

    ofile.write("#"*5+" Urine Microalbumin Modeling (N={0:d}) ".format(albumin_table.shape[0])+"#"*5+'\n')
    for param in albumin_full_model.params.keys():
        ofile.write('{3:s} Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(albumin_full_model.params[param],albumin_full_model.bse[param],albumin_full_model.pvalues[param],param))
    ofile.write('\n')

    albumin_model_minus_pgs=OLS(albumin_table['Urine Microalbumin'],exog=sm.tools.add_constant(albumin_table[['sex','Urine Measurement Age','array','Smoking-Ever Smoked','Smoking x P/LP Carrier']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier']]),hasconst=True).fit()

    albumin_model_minus_pgs_interaction=OLS(albumin_table['Urine Microalbumin'],exog=sm.tools.add_constant(albumin_table[['sex','Urine Measurement Age','array','Smoking-Ever Smoked','Smoking x P/LP Carrier']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier','PGS']]),hasconst=True).fit()

    lr_test_all_pgs=albumin_full_model.compare_lr_test(albumin_model_minus_pgs)
    lr_test_pgs_interaction=albumin_full_model.compare_lr_test(albumin_model_minus_pgs_interaction)

    ofile.write('Aggregate PGS Effect P-Value (LR Test): {0:.2e}\n'.format(lr_test_all_pgs[1]))
    ofile.write('PGS Interaction Effects P-Value (LR Test): {0:.2e}\n'.format(lr_test_pgs_interaction[1]))
    ofile.write('\n'*3)

    albumin_table['Microalbumin Residuals']=OLS(albumin_table['Urine Microalbumin'],exog=sm.tools.add_constant(albumin_table[['sex','Urine Measurement Age','array','Smoking-Ever Smoked','Smoking x P/LP Carrier']+['PC{0:d}'.format(i) for i in range(1,11)]]),hasconst=True).fit().resid
    albumin_table['PGS Quantiles']=assign_quantiles(albumin_table,'PGS',3)

    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=2)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    sns.pointplot(x='PGS Quantiles',y='Microalbumin Residuals',data=albumin_table,hue='Genotype',palette=[color_list[0],color_list[2]],ax=axis,scale=2.0,errwidth=15.0,join=False,markers=['o','^'],dodge=0.25)
    axis.set_ylabel('Urine Microalbumin\nResiduals')
    axis.set_xlabel('PGS Quantiles')
    f.savefig(fig_direc+'Albumin_v_PGS.svg')
    plt.close()


    map_full_model=OLS(bp_table['Mean Arterial Pressure'],exog=sm.tools.add_constant(bp_table[['sex','age_normalized','array','Smoking-Ever Smoked','Smoking x P/LP Carrier']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier','PGS','PGS x P/LP Carrier']]),hasconst=True).fit()

    ofile.write("#"*5+" Mean Arterial Pressure Modeling (N={0:d}) ".format(bp_table.shape[0])+"#"*5+'\n')
    for param in map_full_model.params.keys():
        ofile.write('{3:s} Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(map_full_model.params[param],map_full_model.bse[param],map_full_model.pvalues[param],param))
    ofile.write('\n')

    map_model_minus_pgs=OLS(bp_table['Mean Arterial Pressure'],exog=sm.tools.add_constant(bp_table[['sex','age_normalized','array','Smoking-Ever Smoked','Smoking x P/LP Carrier']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier']]),hasconst=True).fit()

    map_model_minus_pgs_interaction=OLS(bp_table['Mean Arterial Pressure'],exog=sm.tools.add_constant(bp_table[['sex','age_normalized','array','Smoking-Ever Smoked','Smoking x P/LP Carrier']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier','PGS']]),hasconst=True).fit()

    lr_test_all_pgs=map_full_model.compare_lr_test(map_model_minus_pgs)
    lr_test_pgs_interaction=map_full_model.compare_lr_test(map_model_minus_pgs_interaction)

    ofile.write('Aggregate PGS Effect P-Value (LR Test): {0:.2e}\n'.format(lr_test_all_pgs[1]))
    ofile.write('PGS Interaction Effects P-Value (LR Test): {0:.2e}\n'.format(lr_test_pgs_interaction[1]))
    ofile.write('\n'*3)

    bp_table['MAP Residuals']=OLS(bp_table['Mean Arterial Pressure'],exog=sm.tools.add_constant(bp_table[['sex','age_normalized','array','Smoking-Ever Smoked','Smoking x P/LP Carrier']+['PC{0:d}'.format(i) for i in range(1,11)]]),hasconst=True).fit().resid
    bp_table['PGS Quantiles']=assign_quantiles(bp_table,'PGS',3)

    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=2)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    sns.pointplot(x='PGS Quantiles',y='MAP Residuals',data=bp_table,hue='Genotype',palette=[color_list[0],color_list[2]],ax=axis,scale=2.0,errwidth=15.0,join=False,markers=['o','^'],dodge=0.25)
    axis.set_ylabel('MAP Residuals')
    axis.set_xlabel('PGS Quantiles')
    f.savefig(fig_direc+'MAP_v_PGS.svg')
    plt.close()


    gfr_full_model=OLS(gfr_table['eGFR'],exog=sm.tools.add_constant(gfr_table[['sex','age_normalized','array','Smoking-Ever Smoked','Smoking x P/LP Carrier']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier','PGS','PGS x P/LP Carrier']]),hasconst=True).fit()

    ofile.write("#"*5+" eGFR Modeling (N={0:d}) ".format(gfr_table.shape[0])+"#"*5+'\n')
    for param in gfr_full_model.params.keys():
        ofile.write('{3:s} Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(gfr_full_model.params[param],gfr_full_model.bse[param],gfr_full_model.pvalues[param],param))
    ofile.write('\n')

    gfr_model_minus_pgs=OLS(gfr_table['eGFR'],exog=sm.tools.add_constant(gfr_table[['sex','age_normalized','array','Smoking-Ever Smoked','Smoking x P/LP Carrier']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier']]),hasconst=True).fit()

    gfr_model_minus_pgs_interaction=OLS(gfr_table['eGFR'],exog=sm.tools.add_constant(gfr_table[['sex','age_normalized','array','Smoking-Ever Smoked','Smoking x P/LP Carrier']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier','PGS']]),hasconst=True).fit()

    lr_test_all_pgs=gfr_full_model.compare_lr_test(gfr_model_minus_pgs)
    lr_test_pgs_interaction=gfr_full_model.compare_lr_test(gfr_model_minus_pgs_interaction)

    ofile.write('Aggregate PGS Effect P-Value (LR Test): {0:.2e}\n'.format(lr_test_all_pgs[1]))
    ofile.write('PGS Interaction Effects P-Value (LR Test): {0:.2e}\n'.format(lr_test_pgs_interaction[1]))
    ofile.write('\n'*3)

    gfr_table['eGFR Residuals']=OLS(gfr_table['eGFR'],exog=sm.tools.add_constant(gfr_table[['sex','age_normalized','array','Smoking-Ever Smoked']+['PC{0:d}'.format(i) for i in range(1,11)]]),hasconst=True).fit().resid
    gfr_table['PGS Quantiles']=assign_quantiles(gfr_table,'PGS',4)

    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=2)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    sns.pointplot(x='PGS Quantiles',y='eGFR Residuals',data=gfr_table,hue='Genotype',palette=[color_list[0],color_list[2]],ax=axis,scale=2.0,errwidth=15.0,join=False,markers=['o','^'],dodge=0.25)
    axis.set_ylabel('eGFR Residuals')
    axis.set_xlabel('PGS Quantiles')
    f.savefig(fig_direc+'eGFR_v_PGS.svg')
    plt.close()



################ Section 3: PCKD Analysis ########################
#######################################################################

with open(table_direc+'PCKD_Dx_Stats.txt','w') as ofile:
    num_total=analysis_table['PCKD Dx'].sum()
    num_carriers=((analysis_table['PCKD Dx']==1.0)&(analysis_table['P/LP Carrier']==1)).sum()
    num_control=int(num_total-num_carriers)

    ofile.write('Total Number PCKD Dx (N={1:d}): {0:d}\n'.format(int(num_total),analysis_table.shape[0]))
    ofile.write('PCKD Dx By Genotype:\n')
    ofile.write('\tControl: {0:d}\n'.format(num_control))
    ofile.write('\tP/LP Carrier: {0:d}\n'.format(num_carriers))
    ofile.write('\n\n')

    ofile.write('\n\n')
    #
    firth = FirthRegression(analysis_table,['sex','age_normalized','array']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier','PGS','PGS x P/LP Carrier'],'PCKD Dx',hasconst=False)
    plp_add_pckd_w_ints=firth.FirthInference(['P/LP Carrier'])
    pgs_add_pckd_w_ints=firth.FirthInference(['PGS'])
    pgs_int_plp_pckd=firth.FirthInference(['PGS x P/LP Carrier'])


    ofile.write('##### Logistic (Firth) Regression Effects--No Smoking (N={0:d}) #####\n'.format(analysis_table.shape[0]))
    ofile.write('Marginal P/LP Dx Effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(plp_add_pckd_w_ints['ParamTable'].loc['P/LP Carrier'].BETA,plp_add_pckd_w_ints['ParamTable'].loc['P/LP Carrier'].SE,plp_add_pckd_w_ints['PVal']))
    ofile.write('Additive PGS effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(pgs_add_pckd_w_ints['ParamTable'].loc['PGS'].BETA,pgs_add_pckd_w_ints['ParamTable'].loc['PGS'].SE,pgs_add_pckd_w_ints['PVal']))
    ofile.write('PGS x P/LP effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(pgs_int_plp_pckd['ParamTable'].loc['PGS x P/LP Carrier'].BETA,pgs_int_plp_pckd['ParamTable'].loc['PGS x P/LP Carrier'].SE,pgs_int_plp_pckd['PVal']))

    ofile.write('\n\n')




    firth = FirthRegression(ever_smoked_table,['sex','age_normalized','array','Smoking-Ever Smoked','Smoking x P/LP Carrier']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier','PGS','PGS x P/LP Carrier'],'PCKD Dx',hasconst=False)
    plp_add_pckd_w_ints=firth.FirthInference(['P/LP Carrier'])
    pgs_add_pckd_w_ints=firth.FirthInference(['PGS'])
    pgs_int_plp_pckd=firth.FirthInference(['PGS x P/LP Carrier'])
    smoking_plp_pckd=firth.FirthInference(['Smoking-Ever Smoked'])
    smoking_int_plp_pckd=firth.FirthInference(['Smoking x P/LP Carrier'])


    ofile.write('##### Logistic (Firth) Regression Effects--With Smoking (N={0:d}) #####\n'.format(ever_smoked_table.shape[0]))
    ofile.write('Marginal P/LP Dx Effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(plp_add_pckd_w_ints['ParamTable'].loc['P/LP Carrier'].BETA,plp_add_pckd_w_ints['ParamTable'].loc['P/LP Carrier'].SE,plp_add_pckd_w_ints['PVal']))
    ofile.write('Additive PGS effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(pgs_add_pckd_w_ints['ParamTable'].loc['PGS'].BETA,pgs_add_pckd_w_ints['ParamTable'].loc['PGS'].SE,pgs_add_pckd_w_ints['PVal']))
    ofile.write('PGS x P/LP effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(pgs_int_plp_pckd['ParamTable'].loc['PGS x P/LP Carrier'].BETA,pgs_int_plp_pckd['ParamTable'].loc['PGS x P/LP Carrier'].SE,pgs_int_plp_pckd['PVal']))
    ofile.write('Smoking-Ever Smoked effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(smoking_plp_pckd['ParamTable'].loc['Smoking-Ever Smoked'].BETA,smoking_plp_pckd['ParamTable'].loc['Smoking-Ever Smoked'].SE,smoking_plp_pckd['PVal']))
    ofile.write('Smoking-Ever Smoked x P/LP effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(smoking_int_plp_pckd['ParamTable'].loc['Smoking x P/LP Carrier'].BETA,smoking_int_plp_pckd['ParamTable'].loc['Smoking x P/LP Carrier'].SE,smoking_int_plp_pckd['PVal']))



################ Section 4: Q61 Analysis ########################
##################################################################
with open(table_direc+'GeneralCysticKidney_Dx_Stats.txt','w') as ofile:
    num_total=analysis_table['Cystic Kidney Dx (Q61)'].sum()
    num_carriers=((analysis_table['Cystic Kidney Dx (Q61)']==1.0)&(analysis_table['P/LP Carrier']==1)).sum()
    num_control=int(num_total-num_carriers)

    ofile.write('Total Number Cystic Kidney Disease (Q61) Dx (Algorithm; N={1:d}): {0:d}\n'.format(int(num_total),analysis_table.shape[0]))
    ofile.write('Cystic Kidney Dx (Q61) Dx By Genotype:\n')
    ofile.write('\tControl: {0:d}\n'.format(num_control))
    ofile.write('\tP/LP Carrier: {0:d}\n'.format(num_carriers))
    ofile.write('\n\n')
    #
    firth = FirthRegression(analysis_table,['sex','age_normalized','array']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier','PGS','PGS x P/LP Carrier'],'Cystic Kidney Dx (Q61)',hasconst=False)
    plp_add_cystkd_w_ints=firth.FirthInference(['P/LP Carrier'])
    pgs_add_cystkd_w_ints=firth.FirthInference(['PGS'])
    pgs_int_plp_cystkd=firth.FirthInference(['PGS x P/LP Carrier'])


    ofile.write('##### Logistic (Firth) Regression Effects--No Smoking (N={0:d}) #####\n'.format(analysis_table.shape[0]))
    ofile.write('Marginal P/LP Dx Effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(plp_add_cystkd_w_ints['ParamTable'].loc['P/LP Carrier'].BETA,plp_add_cystkd_w_ints['ParamTable'].loc['P/LP Carrier'].SE,plp_add_cystkd_w_ints['PVal']))
    ofile.write('Additive PGS effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(pgs_add_cystkd_w_ints['ParamTable'].loc['PGS'].BETA,pgs_add_cystkd_w_ints['ParamTable'].loc['PGS'].SE,pgs_add_cystkd_w_ints['PVal']))
    ofile.write('PGS x P/LP effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(pgs_int_plp_cystkd['ParamTable'].loc['PGS x P/LP Carrier'].BETA,pgs_int_plp_cystkd['ParamTable'].loc['PGS x P/LP Carrier'].SE,pgs_int_plp_cystkd['PVal']))
    ofile.write('\n\n')


    firth = FirthRegression(ever_smoked_table,['sex','age_normalized','array','Smoking-Ever Smoked','Smoking x P/LP Carrier']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier','PGS','PGS x P/LP Carrier'],'Cystic Kidney Dx (Q61)',hasconst=False)
    plp_add_cystkd_w_ints=firth.FirthInference(['P/LP Carrier'])
    pgs_add_cystkd_w_ints=firth.FirthInference(['PGS'])
    pgs_int_plp_cystkd=firth.FirthInference(['PGS x P/LP Carrier'])
    smoking_plp_cystkd=firth.FirthInference(['Smoking-Ever Smoked'])
    smoking_int_plp_cystkd=firth.FirthInference(['Smoking x P/LP Carrier'])

    ofile.write('##### Logistic (Firth) Regression Effects--With Smoking (N={0:d}) #####\n'.format(ever_smoked_table.shape[0]))
    ofile.write('Marginal P/LP Dx Effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(plp_add_cystkd_w_ints['ParamTable'].loc['P/LP Carrier'].BETA,plp_add_cystkd_w_ints['ParamTable'].loc['P/LP Carrier'].SE,plp_add_cystkd_w_ints['PVal']))
    ofile.write('Additive PGS effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(pgs_add_cystkd_w_ints['ParamTable'].loc['PGS'].BETA,pgs_add_cystkd_w_ints['ParamTable'].loc['PGS'].SE,pgs_add_cystkd_w_ints['PVal']))
    ofile.write('PGS x P/LP effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(pgs_int_plp_cystkd['ParamTable'].loc['PGS x P/LP Carrier'].BETA,pgs_int_plp_cystkd['ParamTable'].loc['PGS x P/LP Carrier'].SE,pgs_int_plp_cystkd['PVal']))
    ofile.write('Smoking-Ever Smoked effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(smoking_plp_cystkd['ParamTable'].loc['Smoking-Ever Smoked'].BETA,smoking_plp_cystkd['ParamTable'].loc['Smoking-Ever Smoked'].SE,smoking_plp_cystkd['PVal']))
    ofile.write('Smoking-Ever Smoked x P/LP effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(smoking_plp_cystkd['ParamTable'].loc['Smoking x P/LP Carrier'].BETA,smoking_plp_cystkd['ParamTable'].loc['Smoking x P/LP Carrier'].SE,smoking_plp_cystkd['PVal']))




################ Section 5: ESRD Analysis ########################
#######################################################################

with open(table_direc+'ESRD_Dx_Stats.txt','w') as ofile:
    num_total=analysis_table['ESRD Dx (Algorithm)'].sum()
    num_carriers=((analysis_table['ESRD Dx (Algorithm)']==1.0)&(analysis_table['P/LP Carrier']==1)).sum()
    num_control=int(num_total-num_carriers)

    ofile.write('ESRD Dx (Algorithm) By Genotype:\n')
    ofile.write('\tControl: {0:d}\n'.format(num_control))
    ofile.write('\tP/LP Carrier: {0:d}\n'.format(num_carriers))
    ofile.write('\n\n')

    firth = FirthRegression(analysis_table,['sex','age_normalized','array']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier','PGS','PGS x P/LP Carrier'],'ESRD Dx (Algorithm)',hasconst=False)


    plp_add_esrd_w_ints=firth.FirthInference(['P/LP Carrier'])
    pgs_add_esrd_w_ints=firth.FirthInference(['PGS'])
    pgs_int_plp_esrd=firth.FirthInference(['PGS x P/LP Carrier'])


    ofile.write('##### Logistic (Firth) Regression Effects--No Smoking (N={0:d}) #####\n'.format(analysis_table.shape[0]))
    ofile.write('Marginal P/LP Dx Effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(plp_add_esrd_w_ints['ParamTable'].loc['P/LP Carrier'].BETA,plp_add_esrd_w_ints['ParamTable'].loc['P/LP Carrier'].SE,plp_add_esrd_w_ints['PVal']))
    ofile.write('Additive PGS effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(pgs_add_esrd_w_ints['ParamTable'].loc['PGS'].BETA,pgs_add_esrd_w_ints['ParamTable'].loc['PGS'].SE,pgs_add_esrd_w_ints['PVal']))
    ofile.write('PGS x P/LP effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(pgs_int_plp_esrd['ParamTable'].loc['PGS x P/LP Carrier'].BETA,pgs_int_plp_esrd['ParamTable'].loc['PGS x P/LP Carrier'].SE,pgs_int_plp_esrd['PVal']))
    ofile.write('\n\n')


    firth = FirthRegression(ever_smoked_table,['sex','age_normalized','array','Smoking-Ever Smoked','Smoking x P/LP Carrier']+['PC{0:d}'.format(i) for i in range(1,11)]+['P/LP Carrier','PGS','PGS x P/LP Carrier'],'ESRD Dx (Algorithm)',hasconst=False)
    plp_add_esrd_w_ints=firth.FirthInference(['P/LP Carrier'])
    pgs_add_esrd_w_ints=firth.FirthInference(['PGS'])
    pgs_int_plp_esrd=firth.FirthInference(['PGS x P/LP Carrier'])
    smoking_plp_esrd=firth.FirthInference(['Smoking-Ever Smoked'])
    smoking_int_plp_esrd=firth.FirthInference(['Smoking x P/LP Carrier'])

    ofile.write('##### Logistic (Firth) Regression Effects--With Smoking (N={0:d}) #####\n'.format(ever_smoked_table.shape[0]))
    ofile.write('Marginal P/LP Dx Effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(plp_add_esrd_w_ints['ParamTable'].loc['P/LP Carrier'].BETA,plp_add_esrd_w_ints['ParamTable'].loc['P/LP Carrier'].SE,plp_add_esrd_w_ints['PVal']))
    ofile.write('Additive PGS effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(pgs_add_esrd_w_ints['ParamTable'].loc['PGS'].BETA,pgs_add_esrd_w_ints['ParamTable'].loc['PGS'].SE,pgs_add_esrd_w_ints['PVal']))
    ofile.write('PGS x P/LP effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(pgs_int_plp_esrd['ParamTable'].loc['PGS x P/LP Carrier'].BETA,pgs_int_plp_esrd['ParamTable'].loc['PGS x P/LP Carrier'].SE,pgs_int_plp_esrd['PVal']))
    ofile.write('Smoking-Ever Smoked effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(smoking_plp_esrd['ParamTable'].loc['Smoking-Ever Smoked'].BETA,smoking_plp_esrd['ParamTable'].loc['Smoking-Ever Smoked'].SE,smoking_plp_esrd['PVal']))
    ofile.write('Smoking-Ever Smoked x P/LP effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(smoking_int_plp_esrd['ParamTable'].loc['Smoking x P/LP Carrier'].BETA,smoking_int_plp_esrd['ParamTable'].loc['Smoking x P/LP Carrier'].SE,smoking_int_plp_esrd['PVal']))

#
#################### Section 6B: Q61 Survival Analysis (With Smoking) ###########
#######################################################################

cph_table=pd.DataFrame(index=ever_smoked_table.index)
cph_table['Dx']=ever_smoked_table['Cystic Kidney Dx (Q61)']
cph_table['ObsWindow']=ever_smoked_table['Obs Windows-Cystic Kidney Disease (Q61)']
cph_table['Sex']=ever_smoked_table['sex']
cph_table['Array']=ever_smoked_table['array']
cph_table['Smoke']=ever_smoked_table['Smoking-Ever Smoked']
for i in range(1,11):
    cph_table['PC{0:d}'.format(i)]=ever_smoked_table['PC{0:d}'.format(i)]
cph_table['Genotype']=ever_smoked_table['Genotype']
cph_table['P_LP']=ever_smoked_table['P/LP Carrier']
cph_table['Control']=ever_smoked_table['Control']
cph_table['PGS']=ever_smoked_table['PGS']

cph_table['PGSxP_LP']=ever_smoked_table['PGS']*ever_smoked_table['P/LP Carrier']
cph_table['SmokexP_LP']=ever_smoked_table['Smoking-Ever Smoked']*ever_smoked_table['P/LP Carrier']

cph = CoxPHFitter(penalizer=0.0)
kmf=KaplanMeierFitter(alpha=0.05)



with open(table_direc+'CysticKidney_WithSmoking_CoxPH_Analysis.txt','w') as ofile:
    full_model=copy.deepcopy(cph.fit(cph_table, duration_col='ObsWindow', event_col='Dx',formula="Sex+Array+"+'+'.join(['PC{0:d}'.format(i) for i in range(1,11)])+"+Smoke+SmokexP_LP+P_LP+PGS+PGSxP_LP",step_size=0.1))


    no_geno_model=copy.deepcopy(cph.fit(cph_table, duration_col='ObsWindow', event_col='Dx',formula="Sex+Array+"+'+'.join(['PC{0:d}'.format(i) for i in range(1,11)])+"+Smoke+PGS",step_size=0.1))
    test_stats_geno=LRTest(full_model.log_likelihood_,no_geno_model.log_likelihood_,3)

    no_pgs_model=copy.deepcopy(cph.fit(cph_table, duration_col='ObsWindow', event_col='Dx',formula="Sex+Array+"+'+'.join(['PC{0:d}'.format(i) for i in range(1,11)])+"+Smoke+SmokexP_LP+P_LP",step_size=0.05))
    test_stats_PGS=LRTest(full_model.log_likelihood_,no_pgs_model.log_likelihood_,2)

    no_smoke_model=copy.deepcopy(cph.fit(cph_table, duration_col='ObsWindow', event_col='Dx',formula="Sex+Array+"+'+'.join(['PC{0:d}'.format(i) for i in range(1,11)])+"+P_LP+PGS+PGSxP_LP",step_size=0.05))
    test_stats_smoke=LRTest(full_model.log_likelihood_,no_smoke_model.log_likelihood_,2)

    no_pgs_int_model=copy.deepcopy(cph.fit(cph_table, duration_col='ObsWindow', event_col='Dx',formula="Sex+Array+"+'+'.join(['PC{0:d}'.format(i) for i in range(1,11)])+"+Smoke+SmokexP_LP+P_LP+PGS",step_size=0.1))
    test_stats_PGS_int=LRTest(full_model.log_likelihood_,no_pgs_int_model.log_likelihood_,1)




    ofile.write("Smoking Effect on Cystic Kidney Disease Risk P-value (LR Test): {0:.2e}\n".format(test_stats_smoke[1]))
    ofile.write("Genotypic Effect on Cystic Kidney Disease Risk P-value (LR Test): {0:.2e}\n".format(test_stats_geno[1]))
    ofile.write("Marginal PGS Effect on Cystic Kidney Disease Risk P-value (LR Test): {0:.2e}\n".format(test_stats_PGS[1]))
    ofile.write("PGSxP/LP Effect on Cystic Kidney Disease Risk P-value (LR Test): {0:.2e}\n".format(test_stats_PGS_int[1]))
    ofile.write('\n\n')

    ofile.write("#"*5+" Final Model "+'#'*5+'\n')
    ofile.write(full_model.summary.to_string())

    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=2)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    kmf=KaplanMeierFitter()
    kmf.fit(cph_table.loc[cph_table.Genotype=='Control']['ObsWindow'],event_observed=cph_table.loc[cph_table.Genotype=='Control']['Dx'], label='Control')
    kmf.plot_survival_function(ax=axis,color=color_list[0],lw=2.0)


    kmf.fit(cph_table.loc[cph_table.Genotype=='P/LP Carrier']['ObsWindow'],event_observed=cph_table.loc[cph_table.Genotype=='P/LP Carrier']['Dx'], label='P/LP Carrier')
    kmf.plot_survival_function(ax=axis,color=color_list[4],lw=2.0)


    axis.text(axis.get_xlim()[0]+0.25*(axis.get_xlim()[1]-axis.get_xlim()[0]),axis.get_ylim()[0]+0.35*(axis.get_ylim()[1]-axis.get_ylim()[0]),'P/LP P-value={0:.2e}'.format(test_stats_geno[1]),fontsize=20,fontweight='bold')
    axis.set_title('Smoking Test Dataset (N={0:d})'.format(cph_table.shape[0]),fontsize=20,fontweight='bold')
    axis.set_ylabel('Fraction of Patients\nUnaffected by\nCystic Kidney Disease')
    axis.set_xlabel('Patient Age')
    plt.savefig(fig_direc+'Genotype_CysticKidney_WithSmoking_KaplanMeier.svg')
    plt.close()




    plp_only=cph_table.loc[cph_table.P_LP==1].copy()
    plp_only['PGS_Quantiles']=assign_quantiles(plp_only,'PGS',5)

    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=2)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    kmf=KaplanMeierFitter()

    kmf.fit(plp_only.loc[plp_only.PGS_Quantiles=='1st']['ObsWindow'],event_observed=plp_only.loc[plp_only.PGS_Quantiles=='1st']['Dx'], label='1st PGS Quintile')
    kmf.plot_survival_function(ax=axis,color=blue_color,lw=2.0)

    kmf.fit(plp_only.loc[plp_only.PGS_Quantiles=='5th']['ObsWindow'],event_observed=plp_only.loc[plp_only.PGS_Quantiles=='5th']['Dx'], label='5th PGS Quintile')
    kmf.plot_survival_function(ax=axis,color=red_color,lw=2.0)

    axis.text(axis.get_xlim()[0]+0.25*(axis.get_xlim()[1]-axis.get_xlim()[0]),axis.get_ylim()[0]+0.25*(axis.get_ylim()[1]-axis.get_ylim()[0]),'PGS P-value={0:.2e}'.format(test_stats_PGS_int[1]),fontsize=20,fontweight='bold')

    axis.set_title('Smoking Test Dataset,\nP/LP Carriers Only (N={0:d})'.format(plp_only.shape[0]),fontsize=20,fontweight='bold')
    axis.set_ylabel('Fraction of Patients\nUnaffected by\nCystic Kidney Disease')
    axis.set_xlabel('Patient Age')
    plt.savefig(fig_direc+'PGS_CysticKidney_WithSmoking_PLPOnly_KaplanMeier.svg')
    plt.close()



#################### Section 7B: ESRD Survival Analysis (With Smoking) ###########
#######################################################################

cph_table=pd.DataFrame(index=ever_smoked_table.index)
cph_table['Dx']=ever_smoked_table['ESRD Dx (Algorithm)']
cph_table['ObsWindow']=ever_smoked_table['Obs Windows-ESRD (Algorithm)']
cph_table['Sex']=ever_smoked_table['sex']
cph_table['Array']=ever_smoked_table['array']
cph_table['Smoke']=ever_smoked_table['Smoking-Ever Smoked']
for i in range(1,11):
    cph_table['PC{0:d}'.format(i)]=ever_smoked_table['PC{0:d}'.format(i)]
cph_table['Genotype']=ever_smoked_table['Genotype']
cph_table['P_LP']=ever_smoked_table['P/LP Carrier']
cph_table['Control']=ever_smoked_table['Control']
cph_table['PGS']=ever_smoked_table['PGS']

cph_table['PGSxP_LP']=ever_smoked_table['PGS']*ever_smoked_table['P/LP Carrier']
cph_table['SmokexP_LP']=ever_smoked_table['Smoking-Ever Smoked']*ever_smoked_table['P/LP Carrier']

cph = CoxPHFitter(penalizer=0.0)
kmf=KaplanMeierFitter(alpha=0.05)



with open(table_direc+'ESRD_WithSmoking_CoxPH_Analysis.txt','w') as ofile:

    full_model=copy.deepcopy(cph.fit(cph_table, duration_col='ObsWindow', event_col='Dx',formula="Sex+Array+"+'+'.join(['PC{0:d}'.format(i) for i in range(1,11)])+"+Smoke+SmokexP_LP+P_LP+PGS+PGSxP_LP",step_size=0.1))


    no_geno_model=copy.deepcopy(cph.fit(cph_table, duration_col='ObsWindow', event_col='Dx',formula="Sex+Array+"+'+'.join(['PC{0:d}'.format(i) for i in range(1,11)])+"+Smoke+PGS",step_size=0.1))
    test_stats_geno=LRTest(full_model.log_likelihood_,no_geno_model.log_likelihood_,3)

    no_pgs_model=copy.deepcopy(cph.fit(cph_table, duration_col='ObsWindow', event_col='Dx',formula="Sex+Array+"+'+'.join(['PC{0:d}'.format(i) for i in range(1,11)])+"+Smoke+SmokexP_LP+P_LP",step_size=0.05))
    test_stats_PGS=LRTest(full_model.log_likelihood_,no_pgs_model.log_likelihood_,2)

    no_smoke_model=copy.deepcopy(cph.fit(cph_table, duration_col='ObsWindow', event_col='Dx',formula="Sex+Array+"+'+'.join(['PC{0:d}'.format(i) for i in range(1,11)])+"+P_LP+PGS+PGSxP_LP",step_size=0.05))
    test_stats_smoke=LRTest(full_model.log_likelihood_,no_smoke_model.log_likelihood_,2)

    no_pgs_int_model=copy.deepcopy(cph.fit(cph_table, duration_col='ObsWindow', event_col='Dx',formula="Sex+Array+"+'+'.join(['PC{0:d}'.format(i) for i in range(1,11)])+"+Smoke+SmokexP_LP+P_LP+PGS",step_size=0.1))
    test_stats_PGS_int=LRTest(full_model.log_likelihood_,no_pgs_int_model.log_likelihood_,1)




    ofile.write("Smoking Effect on ESRD Risk P-value (LR Test): {0:.2e}\n".format(test_stats_smoke[1]))
    ofile.write("Genotypic Effect on ESRD Risk P-value (LR Test): {0:.2e}\n".format(test_stats_geno[1]))
    ofile.write("Marginal PGS Effect on ESRD Risk P-value (LR Test): {0:.2e}\n".format(test_stats_PGS[1]))
    ofile.write("PGSxP/LP Effect on ESRD Risk P-value (LR Test): {0:.2e}\n".format(test_stats_PGS_int[1]))
    ofile.write('\n\n')

    ofile.write("#"*5+" Final Model "+'#'*5+'\n')
    ofile.write(full_model.summary.to_string())



    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=2)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    kmf=KaplanMeierFitter()
    kmf.fit(cph_table.loc[cph_table.Genotype=='Control']['ObsWindow'],event_observed=cph_table.loc[cph_table.Genotype=='Control']['Dx'], label='Control')
    kmf.plot_survival_function(ax=axis,color=color_list[0],lw=2.0)


    kmf.fit(cph_table.loc[cph_table.Genotype=='P/LP Carrier']['ObsWindow'],event_observed=cph_table.loc[cph_table.Genotype=='P/LP Carrier']['Dx'], label='P/LP Carrier')
    kmf.plot_survival_function(ax=axis,color=color_list[4],lw=2.0)


    axis.text(axis.get_xlim()[0]+0.25*(axis.get_xlim()[1]-axis.get_xlim()[0]),axis.get_ylim()[0]+0.35*(axis.get_ylim()[1]-axis.get_ylim()[0]),'P/LP P-value={0:.2e}'.format(test_stats_geno[1]),fontsize=20,fontweight='bold')

    axis.set_title('Smoking Test Dataset (N={0:d})'.format(cph_table.shape[0]),fontsize=20,fontweight='bold')
    axis.set_ylabel('Fraction of Patients\nUnaffected by\nESRD')
    axis.set_xlabel('Patient Age')
    plt.savefig(fig_direc+'Genotype_ESRD_WithSmoking_KaplanMeier.svg')
    plt.close()




    plp_only=cph_table.loc[cph_table.P_LP==1].copy()
    plp_only['PGS_Quantiles']=assign_quantiles(plp_only,'PGS',5)


    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=2)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    kmf=KaplanMeierFitter()

    kmf.fit(plp_only.loc[plp_only.PGS_Quantiles=='1st']['ObsWindow'],event_observed=plp_only.loc[plp_only.PGS_Quantiles=='1st']['Dx'], label='1st PGS Quintile')
    kmf.plot_survival_function(ax=axis,color=blue_color,lw=2.0)

    kmf.fit(plp_only.loc[plp_only.PGS_Quantiles=='5th']['ObsWindow'],event_observed=plp_only.loc[plp_only.PGS_Quantiles=='5th']['Dx'], label='5th PGS Quintile')
    kmf.plot_survival_function(ax=axis,color=red_color,lw=2.0)

    axis.text(axis.get_xlim()[0]+0.25*(axis.get_xlim()[1]-axis.get_xlim()[0]),axis.get_ylim()[0]+0.25*(axis.get_ylim()[1]-axis.get_ylim()[0]),'PGS P-value={0:.2e}'.format(test_stats_PGS_int[1]),fontsize=20,fontweight='bold')

    axis.set_title('Smoking Test Dataset,\nP/LP Carriers Only (N={0:d})'.format(plp_only.shape[0]),fontsize=20,fontweight='bold')
    axis.set_ylabel('Fraction of Patients\nUnaffected by\nESRD')
    axis.set_xlabel('Patient Age')
    plt.savefig(fig_direc+'PGS_ESRD_WithSmoking_PLPOnly_KaplanMeier.svg')
    plt.close()

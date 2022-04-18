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
from AuxFuncs import assign_quantiles,_parseFloats,_dateParse,InverseNormalTransform,LRTest


import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns



sns.set(context='talk',color_codes=True,style='ticks',font='Arial',font_scale=2,rc={'axes.linewidth':5,"font.weight":"bold",'axes.labelweight':"bold",'xtick.major.width':4,'xtick.minor.width': 2})
cmap = cm.get_cmap('viridis', 12)
color_list=[cmap(x) for x in [0.0,0.1,0.25,0.5,0.75,0.9,1.0]]
grey_color=(0.25, 0.25, 0.25)
red_color = '#d10e00'
blue_color='#5DA4FF'



try:
    os.mkdir('path/to/output/Figures')
except FileExistsError:
    pass

try:
    os.mkdir('path/to/output/Tables')
except FileExistsError:
    pass

fig_direc='path/to/output/Figures'

table_direc='path/to/output/Tables'

#################### Section 1: Build Post-hoc Dataset ################
#######################################################################
# first load covariate data, genotype data, polygenic score data, and auxillary data

gwas_data=pd.read_pickle('path/to/covariates/phenos/GWASCovariatesPhenotypes.pth')
genotype_table=pd.read_pickle('path/to/rare/variant/data/OMIM_ICD_86/GeneticCarriers.pth')
genotype_table=genotype_table[~genotype_table.index.duplicated(keep='first')]
genotype_table.set_index(np.array(genotype_table.index,dtype=np.int),inplace=True)
validation_scores=pd.read_csv('path/to/validation/PGS/bolt_model_validation.profile',sep='\t')
validation_scores.set_index('ID1',inplace=True)
target_scores=pd.read_csv('/path/to/target/PGS/bolt_model_target.profile',sep='\t')
target_scores.set_index('ID1',inplace=True)

#Auxillary phenotypic information related to smoking and symptom onset
aux_data=pd.read_csv('/path/to/additional/pheno/info/A1ATD_FollowUpData.txt',sep='\t')
aux_data.set_index('eid',inplace=True)

obs_years=pd.Series(np.zeros(aux_data.shape[0]),index=aux_data.index)
obs_years[pd.isnull(aux_data['Death Date'])]=2020-aux_data.loc[pd.isnull(aux_data['Death Date'])]['Birth Year']
obs_years[pd.isnull(aux_data['Death Date'])==False]=aux_data.loc[pd.isnull(aux_data['Death Date'])==False]['Death Date'].apply(lambda x:int(x.split('-')[0]))-aux_data.loc[pd.isnull(aux_data['Death Date'])==False]['Birth Year']
aux_data['Observation Window']=obs_years

# build the analysis table
analysis_table=gwas_data.loc[validation_scores.index.union(target_scores.index)][['sex','age_normalized','array']+['PC{0:d}'.format(x) for x in range(1,11)]+['CrypticPhenotype_OMIM_ICD:86']]
analysis_table.rename(columns={'CrypticPhenotype_OMIM_ICD:86':'Cryptic Phenotype'},inplace=True)

#add PiZ genotype information, stored as string and as binary array
geno_vec=pd.Series(['PiMM']*analysis_table.shape[0],index=analysis_table.index)
geno_vec.loc[genotype_table.loc[genotype_table.GENOTYPE==1].index.intersection(analysis_table.index)]='PiMZ'
geno_vec.loc[genotype_table.loc[genotype_table.GENOTYPE==2].index.intersection(analysis_table.index)]='PiZZ'
analysis_table['Genotype']=geno_vec
analysis_table=pd.concat([analysis_table,pd.get_dummies(analysis_table['Genotype'], drop_first=False)],axis=1)


# add smoking information, obtained from auxillary data
smoker_key={'0':0,'1':1,'0,1':1,'1,0':1,np.nan:np.nan}
analysis_table['Smoking-Ever Smoked']=np.array([smoker_key[x] for x in aux_data.loc[analysis_table.index]['Smoking-Ever Smoked'].values])
pack_years=np.array([_parseFloats(x) for x  in aux_data.loc[analysis_table.index]['Smoking-Pack Years']])
analysis_table['Smoking-Pack Years']=pack_years
analysis_table.loc[analysis_table['Smoking-Ever Smoked']==0,'Smoking-Pack Years']=0.0
analysis_table['Smoking-Pack Years']=(analysis_table['Smoking-Pack Years']-analysis_table.loc[pd.isna(analysis_table['Smoking-Pack Years'])==False]['Smoking-Pack Years'].mean())/analysis_table.loc[pd.isna(analysis_table['Smoking-Pack Years'])==False]['Smoking-Pack Years'].std()



#FEV/FVC

analysis_table['FEV1/FVC Ratio']=aux_data.loc[analysis_table.index]['FEV1/FVC (Z-score)']
analysis_table['FEV1']=aux_data.loc[analysis_table.index]['FEV1 (Z-score)']
analysis_table['FVC']=aux_data.loc[analysis_table.index]['FVC (Z-score)']

#diagnostic status age for A1ATD and associated lung diseases. Note, these are self reported in survey data
analysis_table['Obs Windows-A1ATD Survey']=aux_data.loc[analysis_table.index]['Age-A1ATD Dx (Survey)'].copy()
has_a1atd=pd.Series(np.zeros(analysis_table.shape[0]),index=analysis_table.index)
has_a1atd.loc[analysis_table.loc[pd.isnull(analysis_table['Obs Windows-A1ATD Survey'])==False].index]=1
analysis_table['A1ATD Dx (Survey)']=has_a1atd
analysis_table.loc[pd.isnull(analysis_table['Obs Windows-A1ATD Survey']),'Obs Windows-A1ATD Survey']=aux_data.loc[analysis_table.loc[pd.isnull(analysis_table['Obs Windows-A1ATD Survey'])].index]['Observation Window']


#algorithmically defined outcomes, i.e. COPD
analysis_table['Obs Windows-COPD (Algorithm)']=aux_data.loc[analysis_table.index]['Age-COPD (Algorithm)'].apply(_dateParse)-aux_data.loc[analysis_table.index]['Birth Year']
analysis_table.loc[analysis_table['Obs Windows-COPD (Algorithm)']<0,'Obs Windows-COPD (Algorithm)']=0

has_copd=pd.Series(np.zeros(analysis_table.shape[0]),index=analysis_table.index)
has_copd.loc[analysis_table.loc[pd.isnull(analysis_table['Obs Windows-COPD (Algorithm)'])==False].index]=1
analysis_table['COPD Dx (Algorithm)']=has_copd
analysis_table.loc[pd.isnull(analysis_table['Obs Windows-COPD (Algorithm)']),'Obs Windows-COPD (Algorithm)']=aux_data.loc[analysis_table.loc[pd.isnull(analysis_table['Obs Windows-COPD (Algorithm)'])].index]['Observation Window']



#add PGS scores (std normalized)
pgs_scores=pd.Series(np.zeros((analysis_table.shape[0])),index=analysis_table.index)
pgs_scores.loc[validation_scores.index]=validation_scores.Profile_1
pgs_scores.loc[target_scores.index]=target_scores.Profile_1
analysis_table['PGS']=(pgs_scores-pgs_scores.mean())/pgs_scores.std(ddof=0)

#add interaction terms
analysis_table['PGS x PiMZ']=analysis_table['PGS']*analysis_table['PiMZ']
analysis_table['PGS x PiZZ']=analysis_table['PGS']*analysis_table['PiZZ']
analysis_table['Smoking x PiZZ']= analysis_table['PiZZ']*analysis_table['Smoking-Ever Smoked']
analysis_table['Smoking x PiMZ']= analysis_table['PiMZ']*analysis_table['Smoking-Ever Smoked']

analysis_table['Pack-years x PiZZ']= analysis_table['PiZZ']*analysis_table['Smoking-Pack Years']
analysis_table['Pack-years x PiMZ']= analysis_table['PiMZ']*analysis_table['Smoking-Pack Years']


#filter out related target_sample_indices
allowed_samples=pd.read_csv('filtered_subject_ids_target_related.txt',sep='\t')
analysis_table=analysis_table.loc[allowed_samples.IID]


### smoking analysis, first eliminate missing data
pack_year_table=analysis_table[pd.isnull(analysis_table['Smoking-Pack Years'])==False].copy()
ever_smoked_table=analysis_table[pd.isnull(analysis_table['Smoking-Ever Smoked'])==False].copy()
fvc_table=pack_year_table[pd.isnull(pack_year_table['FEV1/FVC Ratio'])==False].copy()

smokers=pack_year_table.loc[pack_year_table['Smoking-Ever Smoked']==1.0]
non_smokers=pack_year_table.loc[pack_year_table['Smoking-Ever Smoked']==0.0]


#######################################################################
#######################################################################

#################### Section 1: GWAS Plots #############
#######################################################################

gwas_data=pd.read_csv('/path/to/SummaryStats_Training.txt',sep='\t')
gwas_data['CHROM']=gwas_data.Predictor.apply(lambda x:int(x.split(':')[0]))
gwas_data['POS']=gwas_data.Predictor.apply(lambda x:int(x.split(':')[1]))
gwas_data.set_index('Predictor',inplace=True)

ind_snp_file=pd.read_csv('/path/to/FUMA/Results/FUMA/IndSigSNPs.txt',sep='\t')
ind_snp_file.set_index('rsID',inplace=True)
ind_snps=ind_snp_file['uniqID'].apply(lambda x:':'.join(x.split(':')[0:2]))


corr_snp_file=pd.read_csv('/path/to/FUMA/Results/FUMA/snps.txt',sep='\t')
corr_snp_file.set_index('rsID',inplace=True)
corr_snps=corr_snp_file['uniqID'].apply(lambda x:':'.join(x.split(':')[0:2]))

is_sig=pd.Series(np.zeros(gwas_data.shape[0],dtype=np.int32),index=gwas_data.index)
is_sig.loc[is_sig.index.intersection(corr_snps)]=1
is_sig.loc[is_sig.index.intersection(ind_snps)]=1
is_sig.loc[gwas_data.P<=5e-8]=1 #FUMA excludes MHC by default, just adds it back
gwas_data['Sig_SNPs']=is_sig
gene_file=pd.read_csv('/path/to/FUMA/Results/FUMA/genes.txt',sep='\t')


snps=gene_file['IndSigSNPs'].apply(lambda x:x.split(';')[0])
genes=gene_file['symbol']

snp_to_gene={}
for i,snp in enumerate(snps):
    try:
        snp_to_gene[':'.join(ind_snp_file.loc[snp]['uniqID'].split(':')[0:2])]+=[genes[i]]
    except KeyError:
        snp_to_gene[':'.join(ind_snp_file.loc[snp]['uniqID'].split(':')[0:2])]=[genes[i]]


f,axis=ManhattanPlot(gwas_data,all_sig_thresh=[5e-8],marked_column='Sig_SNPs',hide_hla=False,snp_to_gene=snp_to_gene)
f.savefig(fig_direc+'ManhattanPlot_wGenes.svg')
plt.close()

heritability = pd.read_csv('/path/to/LDAK/Results/LDAK_Heritability/ldak-thin-genotyped.hers',sep=' ')
f,ax=QQPlot(gwas_data,error_type='theoretical',freq_bins=[0.01,0.05,0.5],lambda_gc_scale=10000)
ax.set_title('A1ATD',fontsize=40,fontweight='bold')
ax.text(0.2,8.0,r'$h^{2}=$'+'{0:.3f} ({1:.4f} s.d.)'.format(heritability.iloc[1]['Heritability'], heritability.iloc[1]['Her_SD']),fontsize=20)
plt.savefig(fig_direc+'QQPlot.svg')
plt.close()

#######################################################################
#######################################################################



#######################################################################
#################### Section 2: PGS Replication and Validation ########
#######################################################################

with open(table_direc+'PGS_Validation_Results.txt','w') as ofile:

    # first check to make sure that PGS is not correlated with SERPINA1 genotype
    pgs_lin_mod=OLS(analysis_table['PGS'],exog=sm.tools.add_constant(analysis_table[['PiMZ','PiZZ']]),hasconst=True)
    pgs_relationships=pgs_lin_mod.fit()
    p_value=pgs_relationships.compare_lr_test(OLS(analysis_table['PGS'],exog=np.ones(analysis_table.shape[0]),hasconst=True).fit())[1]

    ofile.write('PiZZ Effect on PGS: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(pgs_relationships.params.PiZZ,pgs_relationships.bse.PiZZ,pgs_relationships.pvalues.PiZZ))
    ofile.write('PiMZ Effect on PGS: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(pgs_relationships.params.PiMZ,pgs_relationships.bse.PiMZ,pgs_relationships.pvalues.PiMZ))
    ofile.write('SERPINA1 Global Genotypic Effect on PGS (P-value; N={1:d}): {0:.2e}\n'.format(p_value,analysis_table.shape[0]))
    ofile.write('\n\n')

    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=2)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    v=sns.violinplot(x='Genotype',y='PGS',data=analysis_table, palette=[color_list[0],color_list[2],color_list[4]],ax=axis)
    for violin, alpha in zip(v.collections[::2], [0.4,0.6,0.8]):
        violin.set_alpha(alpha)

    axis.set_ylabel('Cryptic Phenotype\nPolygenic Score')
    axis.text(1,4.0,r'P-Value='+'{0:.2f}'.format(p_value),fontsize=18)
    plt.savefig(fig_direc+'PGS_vs_A1ATD_Geno.svg')
    plt.close()



    ######## Main Cryptic Phenotype Modeling ########
    #################################################
    # baseline genotype modeling
    cp_lin_mod_base=OLS(analysis_table['Cryptic Phenotype'],exog=sm.tools.add_constant(analysis_table[['sex','age_normalized','array']+['PC{0:d}'.format(i) for i in range(1,11)]]),hasconst=True).fit()
    cp_lin_mod_geno_marg=OLS(analysis_table['Cryptic Phenotype'],exog=sm.tools.add_constant(analysis_table[['sex','age_normalized','array']+['PC{0:d}'.format(i) for i in range(1,11)]+['PiMZ','PiZZ']]),hasconst=True).fit()
    cp_lin_mod_basic_pgs=OLS(analysis_table['Cryptic Phenotype'],exog=sm.tools.add_constant(analysis_table[['sex','age_normalized','array']+['PC{0:d}'.format(i) for i in range(1,11)]+['PiMZ','PiZZ','PGS']]),hasconst=True).fit()

    ofile.write("#"*5+" Baseline Model (N={0:d}) ".format(analysis_table.shape[0])+"#"*5+'\n')
    ofile.write('\tMarginal PiZZ CP Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(cp_lin_mod_geno_marg.params.PiZZ,cp_lin_mod_geno_marg.bse.PiZZ,cp_lin_mod_geno_marg.pvalues.PiZZ))
    ofile.write('\tMarginal PiMZ CP Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(cp_lin_mod_geno_marg.params.PiMZ,cp_lin_mod_geno_marg.bse.PiMZ,cp_lin_mod_geno_marg.pvalues.PiMZ))
    ofile.write('\n'*3)


    ofile.write("#"*5+" Simple PGS Model (N={0:d}) ".format(analysis_table.shape[0])+"#"*5+'\n')
    ofile.write('\tMarginal PiZZ CP Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(cp_lin_mod_basic_pgs.params.PiZZ,cp_lin_mod_basic_pgs.bse.PiZZ,cp_lin_mod_basic_pgs.pvalues.PiZZ))
    ofile.write('\tMarginal PiMZ CP Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(cp_lin_mod_basic_pgs.params.PiMZ,cp_lin_mod_basic_pgs.bse.PiMZ,cp_lin_mod_basic_pgs.pvalues.PiMZ))
    ofile.write('\tMarginal PGS CP Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(cp_lin_mod_basic_pgs.params.PGS,cp_lin_mod_basic_pgs.bse.PGS,cp_lin_mod_basic_pgs.pvalues.PGS))
    ofile.write('\n'*3)



    #smoke proportion
    firth = FirthRegression(ever_smoked_table,['sex','age_normalized','array']+['PC{0:d}'.format(i) for i in range(1,11)]+['PiMZ','PiZZ'],'Smoking-Ever Smoked',hasconst=False)
    pimz_smoke_effect=firth.FirthInference('PiMZ')
    pizz_smoke_effect=firth.FirthInference('PiZZ')

    #smoke pack years
    pack_year_model=OLS(pack_year_table['Smoking-Pack Years'],exog=sm.tools.add_constant(pack_year_table[['sex','age_normalized','array']+['PC{0:d}'.format(i) for i in range(1,11)]+['PiMZ','PiZZ']]),hasconst=True).fit()

    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=4)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    p=sns.pointplot(y='Smoking-Ever Smoked',x='Genotype',data=ever_smoked_table,palette=[color_list[0],color_list[2],color_list[4]],ax=axis,errwidth=10.0,scale=2.0,join=False,markers=['o','^','s'])
    axis.text(0.5,axis.get_ylim()[0]+1.25*(axis.get_ylim()[1]-axis.get_ylim()[0]),'PiZZ P-value={0:.2e}'.format(pizz_smoke_effect['PVal']),fontsize=30,fontweight='bold')
    axis.text(0.5,axis.get_ylim()[0]+1.05*(axis.get_ylim()[1]-axis.get_ylim()[0]),'PiMZ P-value={0:.2e}'.format(pimz_smoke_effect['PVal']),fontsize=30,fontweight='bold')
    axis.set_ylabel('Proportion\nof Smokers')
    axis.set_xlabel('Genotype')
    f.savefig(fig_direc+'Genotype_v_EverSmoked.svg')
    plt.close()


    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=4)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    p=sns.pointplot(y='Smoking-Pack Years',x='Genotype',data=pack_year_table,palette=[color_list[0],color_list[2],color_list[4]],ax=axis,errwidth=10.0,scale=2.0,join=False,markers=['o','^','s'])
    axis.text(0.5,axis.get_ylim()[0]+1.25*(axis.get_ylim()[1]-axis.get_ylim()[0]),'PiZZ P-value={0:.2e}'.format(pack_year_model.pvalues.PiZZ),fontsize=30,fontweight='bold')
    axis.text(0.5,axis.get_ylim()[0]+1.05*(axis.get_ylim()[1]-axis.get_ylim()[0]),'PiMZ P-value={0:.2e}'.format(pack_year_model.pvalues.PiMZ),fontsize=30,fontweight='bold')
    axis.set_ylabel('Smoking\nPack-Years')
    axis.set_xlabel('Genotype')
    f.savefig(fig_direc+'Genotype_v_PackYears.svg')
    plt.close()



    cp_lin_mod_w_pack_years=OLS(pack_year_table['Cryptic Phenotype'],exog=sm.tools.add_constant(pack_year_table[['sex','age_normalized','array']+['PC{0:d}'.format(i) for i in range(1,11)]+['PiMZ','PiZZ','Smoking-Pack Years','Pack-years x PiMZ','Pack-years x PiZZ']]),hasconst=True).fit()

    ofile.write("#"*5+" Pack-Years Model (N={0:d}) ".format(pack_year_table.shape[0])+"#"*5+'\n')
    ofile.write('\tMarginal PiZZ CP Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(cp_lin_mod_w_pack_years.params.PiZZ,cp_lin_mod_w_pack_years.bse.PiZZ,cp_lin_mod_w_pack_years.pvalues.PiZZ))
    ofile.write('\tMarginal PiMZ CP Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(cp_lin_mod_w_pack_years.params.PiMZ,cp_lin_mod_w_pack_years.bse.PiMZ,cp_lin_mod_w_pack_years.pvalues.PiMZ))
    ofile.write('\tPiZZ x Pack-years CP Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(cp_lin_mod_w_pack_years.params['Pack-years x PiZZ'],cp_lin_mod_w_pack_years.bse['Pack-years x PiZZ'],cp_lin_mod_w_pack_years.pvalues['Pack-years x PiZZ']))
    ofile.write('\tPiMZ x Pack-years CP Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(cp_lin_mod_w_pack_years.params['Pack-years x PiMZ'],cp_lin_mod_w_pack_years.bse['Pack-years x PiMZ'],cp_lin_mod_w_pack_years.pvalues['Pack-years x PiMZ']))
    ofile.write('\n'*3)

    pack_year_table['CP Residuals']=cp_lin_mod_base.resid
    no_pizz_geno_table=pack_year_table.loc[pack_year_table.Genotype!='PiZZ'].copy()
    no_pizz_geno_table['Pack-Years Quantiles']=assign_quantiles(no_pizz_geno_table.loc[no_pizz_geno_table['Smoking-Ever Smoked']>0.0],'Smoking-Pack Years',4)
    no_pizz_geno_table['Pack-Years Quantiles'].cat.add_categories('None',inplace=True)
    no_pizz_geno_table.loc[no_pizz_geno_table['Smoking-Ever Smoked']==0.0,'Pack-Years Quantiles']='None'
    no_pizz_geno_table['Pack-Years Quantiles'].cat.reorder_categories(list(no_pizz_geno_table['Pack-Years Quantiles'].cat.categories[-1:])+list(no_pizz_geno_table['Pack-Years Quantiles'].cat.categories[0:-1]),inplace=True)


    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=2)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    sns.pointplot(x='Pack-Years Quantiles',y='CP Residuals',data=no_pizz_geno_table,hue='Genotype',palette=[color_list[0],color_list[2]],ax=axis,scale=3.0,errwidth=15.0,join=False,markers=['o','^'],dodge=0.25)
    axis.set_ylabel('Cryptic Phenotype\nResiduals')
    axis.set_xlabel('Pack-Year Quantiles')
    f.savefig(fig_direc+'CP_PiMZ_v_Packyears.svg')
    plt.close()

    no_pimz_geno_table=pack_year_table.loc[pack_year_table.Genotype!='PiMZ'].copy()
    no_pimz_geno_table['Pack-Years Quantiles']=assign_quantiles(no_pimz_geno_table.loc[no_pimz_geno_table['Smoking-Ever Smoked']>0.0],'Smoking-Pack Years',3)
    no_pimz_geno_table['Pack-Years Quantiles'].cat.add_categories('None',inplace=True)
    no_pimz_geno_table.loc[no_pimz_geno_table['Smoking-Ever Smoked']==0.0,'Pack-Years Quantiles']='None'
    no_pimz_geno_table['Pack-Years Quantiles'].cat.reorder_categories(list(no_pimz_geno_table['Pack-Years Quantiles'].cat.categories[-1:])+list(no_pimz_geno_table['Pack-Years Quantiles'].cat.categories[0:-1]),inplace=True)


    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=2)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    sns.pointplot(x='Pack-Years Quantiles',y='CP Residuals',data=no_pimz_geno_table,hue='Genotype',palette=[color_list[0],color_list[4]],ax=axis,scale=3.0,errwidth=15.0,join=False,markers=['o','s'],dodge=0.25)
    axis.set_ylabel('Cryptic Phenotype\nResiduals')
    axis.set_xlabel('Pack-Year Quantiles')
    f.savefig(fig_direc+'CP_PiZZ_v_Packyears.svg')
    plt.close()


    global_cp_model=OLS(pack_year_table['Cryptic Phenotype'],exog=sm.tools.add_constant(pack_year_table[['sex','age_normalized','array','Smoking-Pack Years']+['PC{0:d}'.format(i) for i in range(1,11)]+['PiMZ','PiZZ','Pack-years x PiMZ','Pack-years x PiZZ','PGS','PGS x PiMZ','PGS x PiZZ']]),hasconst=True).fit()


    ofile.write("#"*5+" Final Model (N={0:d}) ".format(pack_year_table.shape[0])+"#"*5+'\n')
    for param in global_cp_model.params.keys():
        ofile.write('{3:s} Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(global_cp_model.params[param],global_cp_model.bse[param],global_cp_model.pvalues[param],param))
    ofile.write('\n')


    global_cp_model_minus_pgs=OLS(pack_year_table['Cryptic Phenotype'],exog=sm.tools.add_constant(pack_year_table[['sex','age_normalized','array','Smoking-Pack Years']+['PC{0:d}'.format(i) for i in range(1,11)]+['PiMZ','PiZZ','Pack-years x PiMZ','Pack-years x PiZZ']]),hasconst=True).fit()

    global_cp_model_minus_pgs_interaction=OLS(pack_year_table['Cryptic Phenotype'],exog=sm.tools.add_constant(pack_year_table[['sex','age_normalized','array','Smoking-Pack Years']+['PC{0:d}'.format(i) for i in range(1,11)]+['PiMZ','PiZZ','Pack-years x PiMZ','Pack-years x PiZZ','PGS']]),hasconst=True).fit()

    lr_test_all_pgs=global_cp_model.compare_lr_test(global_cp_model_minus_pgs)
    lr_test_pgs_interaction=global_cp_model.compare_lr_test(global_cp_model_minus_pgs_interaction)

    ofile.write('Aggregate PGS Effect P-Value (LR Test): {0:.2e}\n'.format(lr_test_all_pgs[1]))
    ofile.write('PGS Interaction Effects P-Value (LR Test): {0:.2e}\n'.format(lr_test_pgs_interaction[1]))
    ofile.write('\n'*3)


    pack_year_table['CP Residuals (Pack-Year Adj.)']=global_cp_model_minus_pgs.resid

    no_pizz_geno_table=pack_year_table.loc[pack_year_table.Genotype!='PiZZ'].copy()
    no_pizz_geno_table['PGS Quantiles']=assign_quantiles(no_pizz_geno_table,'PGS',5)

    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=2)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    sns.pointplot(x='PGS Quantiles',y='CP Residuals (Pack-Year Adj.)',data=no_pizz_geno_table,hue='Genotype',palette=[color_list[0],color_list[2]],ax=axis,scale=3.0,errwidth=15.0,join=False,markers=['o','^'],dodge=0.25)
    axis.set_ylabel('Cryptic Phenotype\nResiduals (Pack-years Adj.)')
    axis.set_xlabel('PGS Quantiles')
    f.savefig(fig_direc+'CP_wPiMZ_v_PGS.svg')
    plt.close()

    no_pimz_geno_table=pack_year_table.loc[pack_year_table.Genotype!='PiMZ'].copy()
    no_pimz_geno_table['PGS Quantiles']=assign_quantiles(no_pimz_geno_table,'PGS',3)

    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=2)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    sns.pointplot(x='PGS Quantiles',y='CP Residuals (Pack-Year Adj.)',data=no_pimz_geno_table,hue='Genotype',palette=[color_list[0],color_list[4]],ax=axis,scale=3.0,errwidth=15.0,join=False,markers=['o','s'],dodge=0.25)
    axis.set_ylabel('Cryptic Phenotype\nResiduals (Pack-years Adj.)')
    axis.set_xlabel('PGS Quantiles')
    f.savefig(fig_direc+'CP_wPiZZ_v_PGS.svg')
    plt.close()

    ######## Pathogenic Modeling ########
    ####################################
    pathogenic_only_table=pack_year_table.loc[pack_year_table.Genotype!='PiMM'].copy()
    pathogenic_only_table['Smoke x PGS']=pathogenic_only_table['PGS']*pathogenic_only_table['Smoking-Ever Smoked']
    pathogenic_only_table['PGS x PiZZ']=pathogenic_only_table['PGS']*pathogenic_only_table['PiZZ']
    pathogenic_only_table['Smoke x PGS x PiZZ']=pathogenic_only_table['PGS']*pathogenic_only_table['Smoking-Ever Smoked']*pathogenic_only_table['PiZZ']

    pathogenic_only_cp_model=OLS(pathogenic_only_table['Cryptic Phenotype'],exog=sm.tools.add_constant(pathogenic_only_table[['sex','age_normalized','array','Smoking-Pack Years']+['PC{0:d}'.format(i) for i in range(1,11)]+['PiZZ','Pack-years x PiZZ','PGS','Smoke x PGS','PGS x PiZZ','Smoke x PGS x PiZZ']]),hasconst=True).fit()

    pathogenic_only_no_pgs_smoke_int=OLS(pathogenic_only_table['Cryptic Phenotype'],exog=sm.tools.add_constant(pathogenic_only_table[['sex','age_normalized','array','Smoking-Pack Years']+['PC{0:d}'.format(i) for i in range(1,11)]+['PiZZ','Pack-years x PiZZ','PGS','PGS x PiZZ']]),hasconst=True).fit()

    smoke_x_pgs_effect=pathogenic_only_cp_model.compare_lr_test(pathogenic_only_no_pgs_smoke_int)

    ofile.write("#"*5+" Smoke x PGS Pathogenic Model (N={0:d}) ".format(pathogenic_only_table.shape[0])+"#"*5+'\n')
    for param in pathogenic_only_cp_model.params.keys():
        ofile.write('{3:s} Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(pathogenic_only_cp_model.params[param],pathogenic_only_cp_model.bse[param],pathogenic_only_cp_model.pvalues[param],param))
    ofile.write('\n')

    ofile.write('Aggregate PGS x Smoking Effect P-Value (LR Test): {0:.2e}\n'.format(smoke_x_pgs_effect[1]))
    ofile.write('\n'*3)



    pathogenic_only_table['CP Residuals (Pack-Year Adj.)']=OLS(pathogenic_only_table['Cryptic Phenotype'],exog=sm.tools.add_constant(pathogenic_only_table[['sex','age_normalized','array','Smoking-Pack Years']+['PC{0:d}'.format(i) for i in range(1,11)]+['PiZZ','Pack-years x PiZZ']]),hasconst=True).fit().resid


    pathogenic_only_table['PGS Quantiles']=assign_quantiles(pathogenic_only_table,'PGS',2)
    new_label=pd.Series(['PiMZ-Smoke Neg.']*pathogenic_only_table.shape[0],index=pathogenic_only_table.index)
    new_label.loc[(pathogenic_only_table.PiZZ==1)&(pathogenic_only_table['Smoking-Ever Smoked']==0)]='PiZZ-Smoke Neg.'
    new_label.loc[(pathogenic_only_table.PiMZ==1)&(pathogenic_only_table['Smoking-Ever Smoked']==1)]='PiMZ-Smoke Pos.'
    new_label.loc[(pathogenic_only_table.PiZZ==1)&(pathogenic_only_table['Smoking-Ever Smoked']==1)]='PiZZ-Smoke Pos.'
    pathogenic_only_table['Geno. x Smoking']=new_label



    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=2)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    sns.pointplot(x='PGS Quantiles',y='CP Residuals (Pack-Year Adj.)',data=pathogenic_only_table,hue='Geno. x Smoking',palette=[color_list[0],color_list[1],color_list[2],color_list[4]],ax=axis,scale=3.0,errwidth=15.0,join=False,markers=['o','^','s','d'],dodge=0.5)
    axis.set_ylabel('Cryptic Phenotype\nResiduals (Pack-years Adj.)')
    axis.set_xlabel('PGS Quantiles')
    f.savefig(fig_direc+'CP_v_PGS_gene_smoke_int.svg')
    plt.close()

    ######## Ancillary Modeling ########
    ####################################

    # first check PGS effect on Ever-smoking

    ofile.write("#"*5+" Ancillary Modeling "+"#"*5+'\n')

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
    axis.set_ylabel('Residual Pack-years')
    axis.set_xlabel('Polygenic Score\nQuantiles')
    f.savefig(fig_direc+'Packyears_v_PGS.svg')
    plt.close()

    ofile.write('PGS Pack-years Effect (N={3:d}): {0:.3f} ({1:.3f}; {2:.2e})\n'.format(pack_years_mod.params.PGS,pack_years_mod.bse.PGS,pack_years_mod.pvalues.PGS,pack_year_table.shape[0]))
    ofile.write('\n'*3)


    fvc_table_patho_only=fvc_table.loc[(fvc_table.Genotype!='PiMM')].copy()

    fvc_ratio_full_model=OLS(fvc_table_patho_only['FEV1/FVC Ratio'],exog=sm.tools.add_constant(fvc_table_patho_only[['sex','age_normalized','array','Smoking-Pack Years']+['PC{0:d}'.format(i) for i in range(1,11)]+['PiZZ','Pack-years x PiZZ','PGS','PGS x PiZZ']]),hasconst=True).fit()



    ofile.write("#"*5+" FVC1/FEV Ratio Modeling (Pathogenic Only) (N={0:d}) ".format(fvc_table_patho_only.shape[0])+"#"*5+'\n')
    for param in fvc_ratio_full_model.params.keys():
        ofile.write('{3:s} Effect: {0:.3f} ({1:.3f}; {2:.2e})\n'.format(fvc_ratio_full_model.params[param],fvc_ratio_full_model.bse[param],fvc_ratio_full_model.pvalues[param],param))
    ofile.write('\n')


    fvc_ratio_model_minus_pgs=OLS(fvc_table_patho_only['FEV1/FVC Ratio'],exog=sm.tools.add_constant(fvc_table_patho_only[['sex','age_normalized','array','Smoking-Pack Years']+['PC{0:d}'.format(i) for i in range(1,11)]+['PiZZ','Pack-years x PiZZ']]),hasconst=True).fit()

    fvc_ratio_model_minus_pgs_interaction=OLS(fvc_table_patho_only['FEV1/FVC Ratio'],exog=sm.tools.add_constant(fvc_table_patho_only[['sex','age_normalized','array','Smoking-Pack Years']+['PC{0:d}'.format(i) for i in range(1,11)]+['PiZZ','Pack-years x PiZZ','PGS']]),hasconst=True).fit()

    lr_test_all_pgs=fvc_ratio_full_model.compare_lr_test(fvc_ratio_model_minus_pgs)
    lr_test_pgs_interaction=fvc_ratio_full_model.compare_lr_test(fvc_ratio_model_minus_pgs_interaction)

    ofile.write('Aggregate PGS Effect P-Value (LR Test): {0:.2e}\n'.format(lr_test_all_pgs[1]))
    ofile.write('PGS Interaction Effects P-Value (LR Test): {0:.2e}\n'.format(lr_test_pgs_interaction[1]))
    ofile.write('\n'*3)


    fvc_table_patho_only['FEV1/FVC Residuals']=fvc_ratio_model_minus_pgs.resid
    fvc_table_patho_only['PGS Quantiles']=assign_quantiles(fvc_table_patho_only,'PGS',3)


    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=2)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    sns.pointplot(x='PGS Quantiles',y='FEV1/FVC Residuals',data=fvc_table_patho_only,palette=[color_list[2]],ax=axis,scale=2.0,errwidth=15.0,join=False,markers=['o','^'],dodge=0.25)
    axis.set_ylabel('FEV1/FVC Ratio\n(Pack-years Adj.)')
    axis.set_xlabel('PGS Quantiles')
    f.savefig(fig_direc+'FEV_v_PGS_PathoOnly.svg')
    plt.close()



#################### Section 3: A1ATD Analysis ########################
#######################################################################

with open(table_direc+'A1ATD_Dx_Stats.txt','w') as ofile:
    num_total=analysis_table['A1ATD Dx (Survey)'].sum()
    num_PiZZ=((analysis_table['A1ATD Dx (Survey)']==1.0)&(analysis_table['PiZZ']==1)).sum()
    num_PiMZ=((analysis_table['A1ATD Dx (Survey)']==1.0)&(analysis_table['PiMZ']==1)).sum()
    num_PiMM=int(num_total-num_PiZZ-num_PiMZ)

    ofile.write('Total Number A1ATD Dx (self-reported; N={1:d}): {0:d}\n'.format(int(num_total),analysis_table.shape[0]))
    ofile.write('A1ATD Dx By Genotype:\n')
    ofile.write('\tPiMM: {0:d}\n'.format(num_PiMM))
    ofile.write('\tPiMZ: {0:d}\n'.format(num_PiMZ))
    ofile.write('\tPiZZ: {0:d}\n'.format(num_PiZZ))
    ofile.write('\n\n')
    #
    firth = FirthRegression(pack_year_table,['sex','age_normalized','array','Smoking-Pack Years']+['PC{0:d}'.format(i) for i in range(1,11)]+['PiMZ','PiZZ','Pack-years x PiMZ','Pack-years x PiZZ','PGS'],'A1ATD Dx (Survey)',hasconst=False)
    pizz_a1atd=firth.FirthInference(['PiZZ'])
    pimz_a1atd=firth.FirthInference(['PiMZ'])
    marg_packyears_a1atd=firth.FirthInference(['Smoking-Pack Years'])
    packyears_int_a1atd=firth.FirthInference(['Pack-years x PiMZ','Pack-years x PiZZ'])
    pgs_add_a1atd=firth.FirthInference(['PGS'])



    ofile.write('##### Logistic (Firth) Regression Effects (N={0:d}) #####\n'.format(pack_year_table.shape[0]))
    ofile.write('Marginal PiZZ Dx Effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(pizz_a1atd['ParamTable'].loc['PiZZ'].BETA,pizz_a1atd['ParamTable'].loc['PiZZ'].SE,pizz_a1atd['PVal']))
    ofile.write('Marginal PiMZ Dx Effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(pimz_a1atd['ParamTable'].loc['PiMZ'].BETA,pimz_a1atd['ParamTable'].loc['PiMZ'].SE,pimz_a1atd['PVal']))
    ofile.write('Pack-years effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(marg_packyears_a1atd['ParamTable'].loc['Smoking-Pack Years'].BETA,marg_packyears_a1atd['ParamTable'].loc['Smoking-Pack Years'].SE,marg_packyears_a1atd['PVal']))
    ofile.write('Additive PGS effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(pgs_add_a1atd['ParamTable'].loc['PGS'].BETA,pgs_add_a1atd['ParamTable'].loc['PGS'].SE,pgs_add_a1atd['PVal']))
    ofile.write('\n\n')

    firth = FirthRegression(pack_year_table,['sex','age_normalized','array','Smoking-Pack Years']+['PC{0:d}'.format(i) for i in range(1,11)]+['PiMZ','PiZZ','Pack-years x PiMZ','Pack-years x PiZZ','PGS','PGS x PiMZ','PGS x PiZZ'],'A1ATD Dx (Survey)',hasconst=False)
    pizz_a1atd=firth.FirthInference(['PiZZ','PGS x PiZZ'],convergence_limit=1e-6)
    pimz_a1atd=firth.FirthInference(['PiMZ','PGS x PiMZ'],convergence_limit=1e-6)
    pgs_add_a1atd_w_ints=firth.FirthInference(['PGS','PGS x PiMZ','PGS x PiZZ'],convergence_limit=1e-6)
    pgs_int_pimz_a1atd=firth.FirthInference(['PGS x PiMZ'],convergence_limit=1e-6)
    pgs_int_pizz_a1atd=firth.FirthInference(['PGS x PiZZ'],convergence_limit=1e-6)
    pgs_int_both_a1atd=firth.FirthInference(['PGS x PiMZ','PGS x PiZZ'],convergence_limit=1e-6)


    ofile.write('##### Logistic (Firth) Regression Effects--PGS Interactions (N={0:d}) #####\n'.format(pack_year_table.shape[0]))
    ofile.write('Marginal PiZZ Dx Effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(pizz_a1atd['ParamTable'].loc['PiZZ'].BETA,pizz_a1atd['ParamTable'].loc['PiZZ'].SE,pizz_a1atd['PVal']))
    ofile.write('Marginal PiMZ Dx Effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(pimz_a1atd['ParamTable'].loc['PiMZ'].BETA,pimz_a1atd['ParamTable'].loc['PiMZ'].SE,pimz_a1atd['PVal']))
    ofile.write('Additive PGS effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(pgs_add_a1atd_w_ints['ParamTable'].loc['PGS'].BETA,pgs_add_a1atd_w_ints['ParamTable'].loc['PGS'].SE,pgs_add_a1atd_w_ints['PVal']))
    ofile.write('PGS x PiMZ effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(pgs_int_pimz_a1atd['ParamTable'].loc['PGS x PiMZ'].BETA,pgs_int_pimz_a1atd['ParamTable'].loc['PGS x PiMZ'].SE,pgs_int_pimz_a1atd['PVal']))
    ofile.write('PGS x PiZZ effect (LR Test): {0:.2f} ({1:.2f}; {2:.2e})\n'.format(pgs_int_pizz_a1atd['ParamTable'].loc['PGS x PiZZ'].BETA,pgs_int_pizz_a1atd['ParamTable'].loc['PGS x PiZZ'].SE,pgs_int_pizz_a1atd['PVal']))
    ofile.write("Global PGS Interaction Effects (LR Test): {0:.2e}\n".format(pgs_int_both_a1atd['PVal']))



# # #######################################################################
# #######################################################################
#
# #################### Section 4: COPD Survival Analysis ################
# #######################################################################
#
#
cph_table=pd.DataFrame(index=pack_year_table.index)
cph_table['Dx']=pack_year_table['COPD Dx (Algorithm)']
cph_table['ObsWindow']=pack_year_table['Obs Windows-COPD (Algorithm)']
cph_table['Sex']=pack_year_table['sex']
cph_table['Array']=pack_year_table['array']
for i in range(1,11):
    cph_table['PC{0:d}'.format(i)]=pack_year_table['PC{0:d}'.format(i)]
cph_table['Genotype']=pack_year_table['Genotype']
cph_table['PiZZ']=pack_year_table['PiZZ']
cph_table['PiMZ']=pack_year_table['PiMZ']
cph_table['Smoking']=pack_year_table['Smoking-Pack Years']
cph_table['PGS']=pack_year_table['PGS']

cph_table['SmokingxPiZZ']=pack_year_table['Smoking-Pack Years']*pack_year_table['PiZZ']
cph_table['SmokingxPiMZ']=pack_year_table['Smoking-Pack Years']*pack_year_table['PiMZ']
cph_table['PGSxPiZZ']=pack_year_table['PGS']*pack_year_table['PiZZ']
cph_table['PGSxPiMZ']=pack_year_table['PGS']*pack_year_table['PiMZ']

# #
cph = CoxPHFitter(penalizer=0.0)
kmf=KaplanMeierFitter(alpha=0.05)

with open(table_direc+'COPD_CoxPH_Analysis.txt','w') as ofile:

    full_model=copy.deepcopy(cph.fit(cph_table, duration_col='ObsWindow', event_col='Dx',formula="Sex+Array+"+'+'.join(['PC{0:d}'.format(i) for i in range(1,11)])+"+Smoking+PiZZ+PiMZ+SmokingxPiZZ+SmokingxPiMZ+PGS",step_size=0.25))

    no_smoking_model=copy.deepcopy(cph.fit(cph_table, duration_col='ObsWindow', event_col='Dx',formula="Sex+Array+"+'+'.join(['PC{0:d}'.format(i) for i in range(1,11)])+"+PiZZ+PiMZ+PGS",step_size=0.25))
    test_stats_smoking_only=LRTest(full_model.log_likelihood_,no_smoking_model.log_likelihood_,3)


    no_pizz_model=copy.deepcopy(cph.fit(cph_table, duration_col='ObsWindow', event_col='Dx',formula="Sex+Array+"+'+'.join(['PC{0:d}'.format(i) for i in range(1,11)])+"+Smoking+PiMZ+SmokingxPiMZ+PGS",step_size=0.25))
    test_stats_pizz=LRTest(full_model.log_likelihood_,no_pizz_model.log_likelihood_,2)

    no_pimz_model=copy.deepcopy(cph.fit(cph_table, duration_col='ObsWindow', event_col='Dx',formula="Sex+Array+"+'+'.join(['PC{0:d}'.format(i) for i in range(1,11)])+"+Smoking+PiZZ+SmokingxPiZZ+PGS",step_size=0.25))
    test_stats_pimz=LRTest(full_model.log_likelihood_,no_pimz_model.log_likelihood_,2)

    no_pgs_model=copy.deepcopy(cph.fit(cph_table, duration_col='ObsWindow', event_col='Dx',formula="Sex+Array+"+'+'.join(['PC{0:d}'.format(i) for i in range(1,11)])+"+Smoking+PiZZ+PiMZ+SmokingxPiZZ+SmokingxPiMZ",step_size=0.25))
    test_stats_pgs=LRTest(full_model.log_likelihood_,no_pgs_model.log_likelihood_,1)

    no_smoke_int=copy.deepcopy(cph.fit(cph_table, duration_col='ObsWindow', event_col='Dx',formula="Sex+Array+"+'+'.join(['PC{0:d}'.format(i) for i in range(1,11)])+"+Smoking+PiZZ+PiMZ+PGS",step_size=0.25))
    smoke_int_stats=LRTest(full_model.log_likelihood_,no_smoke_int.log_likelihood_,2)


    ofile.write("Smoking Effect on COPD Risk P-value (LR Test): {0:.2e}\n".format(test_stats_smoking_only[1]))
    ofile.write("PiZZ Effect on COPD Risk P-value (LR Test): {0:.2e}\n".format(test_stats_pizz[1]))
    ofile.write("PiMZ Effect on COPD Risk P-value (LR Test): {0:.2e}\n".format(test_stats_pimz[1]))
    ofile.write("PGS Effect on COPD Risk P-value (LR Test): {0:.2e}\n".format(test_stats_pgs[1]))
    ofile.write("SmokexGenotype Effect on COPD Risk P-value (LR Test): {0:.2e}\n".format(smoke_int_stats[1]))


    ofile.write('\n\n')
    ofile.write("#"*5+" Final Model "+'#'*5+'\n')
    ofile.write(full_model.summary.to_string())

    cph_table['PGS_Quantiles']=assign_quantiles(cph_table,'PGS',5)
    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=2)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    kmf=KaplanMeierFitter()
    kmf.fit(cph_table.loc[cph_table.PGS_Quantiles=='1st']['ObsWindow'],event_observed=cph_table.loc[cph_table.PGS_Quantiles=='1st']['Dx'], label='1st PGS Quintile')
    kmf.plot_survival_function(ax=axis,color=blue_color,lw=2.0)


    kmf.fit(cph_table.loc[cph_table.PGS_Quantiles=='3rd']['ObsWindow'],event_observed=cph_table.loc[cph_table.PGS_Quantiles=='3rd']['Dx'], label='3rd PGS Quintile')
    kmf.plot_survival_function(ax=axis,color=grey_color,lw=2.0)

    kmf.fit(cph_table.loc[cph_table.PGS_Quantiles=='5th']['ObsWindow'],event_observed=cph_table.loc[cph_table.PGS_Quantiles=='5th']['Dx'], label='5th PGS Quintile')
    kmf.plot_survival_function(ax=axis,color=red_color,lw=2.0)

    axis.text(axis.get_xlim()[0]+0.25*(axis.get_xlim()[1]-axis.get_xlim()[0]),axis.get_ylim()[0]+0.25*(axis.get_ylim()[1]-axis.get_ylim()[0]),'PGS P-value={0:.2e}'.format(test_stats_pgs[1]),fontsize=20,fontweight='bold')


    axis.set_ylabel('Fraction of Patients\nUnaffected by COPD')
    axis.set_xlabel('Patient Age')
    axis.set_title('Full Test Dataset',fontweight='bold',fontsize=40)
    plt.savefig(fig_direc+'PGS_COPDKaplanMeier.svg')
    plt.close()


    #geno specific figures
    #
    pimm_only=cph_table.loc[(cph_table.PiMZ==0)&(cph_table.PiZZ==0)].copy()
    pathogenic_only=cph_table.loc[(cph_table.PiMZ==1)|(cph_table.PiZZ==1)].copy()
    pizz_only=cph_table.loc[cph_table.PiZZ==1].copy()

    pathogenic_only['PGS_Quantiles']=assign_quantiles(cph_table,'PGS',5)

    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=2)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    kmf=KaplanMeierFitter()
    kmf.fit(pathogenic_only.loc[pathogenic_only.PGS_Quantiles=='1st']['ObsWindow'],event_observed=pathogenic_only.loc[pathogenic_only.PGS_Quantiles=='1st']['Dx'], label='1st PGS Quintile')
    kmf.plot_survival_function(ax=axis,color=blue_color,lw=2.0)

    kmf.fit(pathogenic_only.loc[pathogenic_only.PGS_Quantiles=='5th']['ObsWindow'],event_observed=pathogenic_only.loc[pathogenic_only.PGS_Quantiles=='5th']['Dx'], label='5th PGS Quintile')
    kmf.plot_survival_function(ax=axis,color=red_color,lw=2.0)

    kmf.fit(pimm_only['ObsWindow'],event_observed=pimm_only['Dx'], label='PiMM Controls')
    kmf.plot_survival_function(ax=axis,color=grey_color,lw=2.0)

    pathogenic_only_model=copy.deepcopy(cph.fit(pathogenic_only, duration_col='ObsWindow', event_col='Dx',formula="Sex+Array+"+'+'.join(['PC{0:d}'.format(i) for i in range(1,11)])+"+Smoking+PiZZ+SmokingxPiZZ+PGS",step_size=0.25))
    pathogenic_only_no_pgs=copy.deepcopy(cph.fit(pathogenic_only, duration_col='ObsWindow', event_col='Dx',formula="Sex+Array+"+'+'.join(['PC{0:d}'.format(i) for i in range(1,11)])+"+Smoking+PiZZ+SmokingxPiZZ",step_size=0.25))
    p_val=LRTest(pathogenic_only_model.log_likelihood_,pathogenic_only_no_pgs.log_likelihood_,1)[1]
    axis.text(axis.get_xlim()[0]+0.25*(axis.get_xlim()[1]-axis.get_xlim()[0]),axis.get_ylim()[0]+0.25*(axis.get_ylim()[1]-axis.get_ylim()[0]),'Marginal PGS Effect: {0:.2f} ({1:.2f}; {2:.1e})'.format(pathogenic_only_model.params_['PGS'],pathogenic_only_model.standard_errors_['PGS'],p_val),fontsize=20,fontweight='bold')


    axis.set_ylabel('Fraction of Patients\nUnaffected by COPD')
    axis.set_xlabel('Patient Age')
    axis.set_title('PiZZ/PiMZ Genotypes (N:{0:d})'.format(pathogenic_only.shape[0]),fontweight='bold',fontsize=40)
    plt.savefig(fig_direc+'PGS_Pathogenic_COPDKaplanMeier.svg')
    plt.close()


    pizz_only['PGS_Quantiles']=assign_quantiles(pizz_only,'PGS',3)
    f, axis = plt.subplots(1, 1,figsize=(10,8))
    f.tight_layout(pad=2)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    kmf=KaplanMeierFitter()
    kmf.fit(pizz_only.loc[pizz_only.PGS_Quantiles=='1st']['ObsWindow'],event_observed=pizz_only.loc[pizz_only.PGS_Quantiles=='1st']['Dx'], label='1st PGS Tertile')
    kmf.plot_survival_function(ax=axis,color=blue_color,lw=2.0)

    kmf.fit(pizz_only.loc[pizz_only.PGS_Quantiles=='3rd']['ObsWindow'],event_observed=pizz_only.loc[pizz_only.PGS_Quantiles=='3rd']['Dx'], label='3rd PGS Tertile')
    kmf.plot_survival_function(ax=axis,color=red_color,lw=2.0)

    kmf.fit(pimm_only['ObsWindow'],event_observed=pimm_only['Dx'], label='PiMM Controls')
    kmf.plot_survival_function(ax=axis,color=grey_color,lw=2.0)

    pizz_only_model=copy.deepcopy(cph.fit(pizz_only, duration_col='ObsWindow', event_col='Dx',formula="Sex+Array+"+'+'.join(['PC{0:d}'.format(i) for i in range(1,11)])+"+Smoking+PGS",step_size=0.25))
    pizz_no_pgs=copy.deepcopy(cph.fit(pizz_only, duration_col='ObsWindow', event_col='Dx',formula="Sex+Array+"+'+'.join(['PC{0:d}'.format(i) for i in range(1,11)])+"+Smoking",step_size=0.25))
    p_val=LRTest(pizz_only_model.log_likelihood_,pizz_no_pgs.log_likelihood_,1)[1]
    axis.text(axis.get_xlim()[0]+0.25*(axis.get_xlim()[1]-axis.get_xlim()[0]),axis.get_ylim()[0]+0.25*(axis.get_ylim()[1]-axis.get_ylim()[0]),'Marginal PGS Effect: {0:.2f} ({1:.2f}; {2:.1e})'.format(pizz_only_model.params_['PGS'],pizz_only_model.standard_errors_['PGS'],p_val),fontsize=20,fontweight='bold')


    axis.set_ylabel('Fraction of Patients\nUnaffected by COPD')
    axis.set_xlabel('Patient Age')
    axis.set_title('PiZZ Genotypes (N:{0:d})'.format(pizz_only.shape[0]),fontweight='bold',fontsize=40)
    plt.savefig(fig_direc+'PGS_PiZZ_COPDKaplanMeier.svg')
    plt.close()

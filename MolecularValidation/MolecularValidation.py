import pandas as pd
import statsmodels.api as sm
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn import preprocessing
from scipy import stats
import argparse
import sys
import os
new_path='/Users/davidblair/Desktop/Research/MendelianDiseaseProject/Software/AuxillaryFunctions'
if new_path not in sys.path:
    sys.path.append(new_path)
from FirthRegression import FirthRegression

def convertResultsToString(resultsDict):
    return '{0:.2f} ({1:.2f}; {2:.2e})'.format(resultsDict['BETA'],resultsDict['SE'],resultsDict['PVAL'])


sns.set(context='talk',color_codes=True,style='ticks',font='Arial',font_scale=2,rc={'axes.linewidth':5,"font.weight":"bold",'axes.labelweight':"bold",'xtick.major.width':4,'xtick.minor.width': 2})
cmap = cm.get_cmap('viridis', 12)
color_list=[cmap(x) for x in [0.0,0.1,0.25,0.5,0.75,0.9,1.0]]
grey_color=(0.75, 0.75, 0.75)
red_color = '#FF5D5D'
blue_color='#5DA4FF'


table_path='/output/path/for/Tables/'
figure_path='/output/path/for/Figures/'

try:
    os.mkdir(table_path)
except FileExistsError:
    pass


try:
    os.mkdir(figure_path)
except FileExistsError:
    pass

follow_up_diseases = ['OMIM_ICD:86','OMIM_ICD:108','OMIM_ICD:120','OMIM_ICD:121','OMIM_ICD:132']
dis_abbrev = dict(zip(follow_up_diseases,['A1ATD','HHT','MFS','AS','AD-PCKD']))
masterPhenoTable=pd.read_pickle('path/to/crypticPheno/data/CrypticPhenotypeTable.pth')

#load subset of UKBB ids that represent unrelated caucasian subjects with exome data. This group of patients was selected to maintain unrelatedness AND to maximize the availability of unrelated exome data.
exome_unrelated_ids=pd.read_csv('filtered_subject_ids_exome_related.txt',sep='\t',index_col='IID')
exome_unrelated=pd.Series(np.zeros(masterPhenoTable.shape[0],dtype=np.bool),index=masterPhenoTable.index)
exome_unrelated.loc[exome_unrelated_ids.index]=True
masterPhenoTable['Exome_Unrelated']=exome_unrelated

#indvidual disease analysis
#all analysis is performed for three subsets of the exome data: all subjects with exome data, caucasian only, and caucasian unrelated. Only the caucasian unrelated analyses were reported reported in the text, as they are the most concservative (due to smallest N).
for sample_subset in ['All','CaucasianOnly','CaucasianUnrelated']:

    basic_stats_table={'Disease':[],'# of Subjects':[],'Num Dx':[],'Dx Prev':[],'# of P/LP Carriers':[],'# Unflagged Carriers':[],'P/LP AF':[]}
    cp_regression_stats={'Disease':[],'Marginal P/LP CP Effect':[],'Sex-Specific P/LP CP Effect':[],'Baseline Variant Effect':[],'Unflagged Variant Effect':[],'Within Carrier Dx Effect':[]}
    dx_regression_stats={'Disease':[],'P/LP Dx Effect':[],'CP Dx Effect':[],'CP x LP/P Dx Effect':[],'P/LP Dx Effect (Flagged Dropped)':[],'CP Dx Effect (Flagged Dropped)':[],'CP x LP/P Dx Effect (Flagged Dropped)':[]}


    for disease in follow_up_diseases:

        try:
            os.mkdir(figure_path+disease.replace(':','_'))
        except FileExistsError:
            pass

        #start the tables
        disease_name = dis_abbrev[disease]
        basic_stats_table['Disease']+=[disease_name]
        cp_regression_stats['Disease']+=[disease_name]
        dx_regression_stats['Disease']+=[disease_name]

        #load rare variant carrier data
        geneticData = pd.read_pickle(disease.replace(':','_')+'/GeneticCarriers.pth')

        #remove duplicate entries (arise from duplicate ClinVar variant entries) and set index to integer
        geneticData=geneticData[~geneticData.index.duplicated(keep='first')]
        geneticData=geneticData.set_index(np.array(geneticData.index,dtype=np.int))

        #A1ATD must be handled differently, as it's phased autosomal recessive from SNP data
        if disease!='OMIM_ICD:86':
            if sample_subset=='All':
                geneticAnalysisTable=copy.deepcopy(masterPhenoTable.loc[(masterPhenoTable.Has_Exome==True)&(masterPhenoTable.Pass_Basic_QC==True)])
            elif sample_subset=='CaucasianOnly':
                geneticAnalysisTable=copy.deepcopy(masterPhenoTable.loc[(masterPhenoTable.Has_Exome==True)&(masterPhenoTable.Caucasian_QC==True)])
            else:
                geneticAnalysisTable=copy.deepcopy(masterPhenoTable.loc[(masterPhenoTable.Has_Exome==True)&(masterPhenoTable.Exome_Unrelated==True)&(masterPhenoTable.Caucasian_QC==True)])

            genotype_vec=pd.Series(np.zeros((geneticAnalysisTable.shape[0])),index=geneticAnalysisTable.index)
            unflagged_vec=pd.Series(np.zeros((geneticAnalysisTable.shape[0])),index=geneticAnalysisTable.index)
            genotype_vec.loc[geneticData.loc[(geneticData['GENOTYPE']==1)].index.intersection(genotype_vec.index)]=1
            unflagged_vec.loc[geneticData.loc[(geneticData['FLAGS']=='NaN')&(geneticData['GENOTYPE']==1)].index.intersection(genotype_vec.index)]=1
            #add genotypes
            geneticAnalysisTable.insert(geneticAnalysisTable.shape[1], "genotype", genotype_vec, True)
            geneticAnalysisTable.insert(geneticAnalysisTable.shape[1], "unflagged", unflagged_vec, True)

        else:
            # in the case of A1ATD, we can keep all samples that pass QC on arrays
            if sample_subset=='All':
                geneticAnalysisTable=copy.deepcopy(masterPhenoTable.loc[(masterPhenoTable.Pass_Basic_QC==True)])
            elif sample_subset=='CaucasianOnly':
                geneticAnalysisTable=copy.deepcopy(masterPhenoTable.loc[(masterPhenoTable.Caucasian_QC==True)])
            else:
                geneticAnalysisTable=copy.deepcopy(masterPhenoTable.loc[(masterPhenoTable.Unrelated_QC==True)])

            genotype_vec=pd.Series(np.zeros((geneticAnalysisTable.shape[0])),index=geneticAnalysisTable.index)
            genotype_vec.loc[geneticData.loc[geneticData['GENOTYPE']==2].index.intersection(genotype_vec.index)]=1
            geneticAnalysisTable.insert(geneticAnalysisTable.shape[1], "genotype", genotype_vec, True)
            aux_geno_vec=pd.Series(np.zeros((geneticAnalysisTable.shape[0])),index=geneticAnalysisTable.index)
            aux_geno_vec.loc[geneticData.loc[geneticData['GENOTYPE']==1].index.intersection(aux_geno_vec.index)]=1
            aux_geno_vec.loc[geneticData.loc[geneticData['GENOTYPE']==2].index.intersection(aux_geno_vec.index)]=2
            geneticAnalysisTable.insert(geneticAnalysisTable.shape[1], "aux_genotype", aux_geno_vec, True)




        basic_stats_table['# of Subjects']+=[int(geneticAnalysisTable.shape[0])]
        basic_stats_table['# of P/LP Carriers']+=[int(geneticAnalysisTable['genotype'].sum())]
        if disease!='OMIM_ICD:86':
            basic_stats_table['# Unflagged Carriers']+=[int(geneticAnalysisTable['unflagged'].sum())]
        else:
            basic_stats_table['# Unflagged Carriers']+=[0]

        basic_stats_table['P/LP AF']+=[geneticAnalysisTable['genotype'].sum()/geneticAnalysisTable.shape[0]]
        basic_stats_table['Dx Prev']+=[geneticAnalysisTable['has_'+disease].sum()/geneticAnalysisTable.shape[0]]
        basic_stats_table['Num Dx']+=[geneticAnalysisTable['has_'+disease].sum()]

        ############# CP Regression Analysis ##############
        ###################################################
        Y_CP=np.array(geneticAnalysisTable[['CrypticPhenotype_'+disease]].values,dtype=np.float)

        #build covariate matrix
        cp_regression_matrix=geneticAnalysisTable[['age_normalized','sex','array']+['PC{0:d}'.format(i) for i in range(1,11)]+['genotype']]
        cp_regression_matrix.insert(0,'intercept',np.ones((cp_regression_matrix.shape[0])),True)
        base_lin_mod=sm.OLS(Y_CP,exog=cp_regression_matrix,hasconst=True).fit()



        #AS has an X-linked architecture (COL4A5), although female sex also develop symptoms. Include an interaction term.

        if disease=='OMIM_ICD:121':
            #find all male carriers of COL4A5 variants, include this as an extra covariate in the model
            x_carriers = geneticData.loc[geneticData['GENE']=='COL4A5'].index
            interaction_vec=pd.Series(np.zeros((geneticAnalysisTable.shape[0])),index=geneticAnalysisTable.index)
            interaction_vec.loc[interaction_vec.index.intersection(x_carriers)]=1
            interaction_vec=interaction_vec*geneticAnalysisTable['sex']
            cp_regression_matrix.insert(cp_regression_matrix.shape[1],'genotype x sex',interaction_vec,True)

        #Marginal genotype effect
        base_lin_mod=sm.OLS(Y_CP,exog=cp_regression_matrix,hasconst=True).fit()
        cp_regression_stats['Marginal P/LP CP Effect']+=[{'BETA':base_lin_mod.params['genotype'],'SE':base_lin_mod.bse['genotype'],'PVAL':base_lin_mod.pvalues['genotype']}]
        if disease=='OMIM_ICD:121':
            cp_regression_stats['Sex-Specific P/LP CP Effect']+=[{'BETA':base_lin_mod.params['genotype x sex'],'SE':base_lin_mod.bse['genotype x sex'],'PVAL':base_lin_mod.pvalues['genotype x sex']}]
        else:
            cp_regression_stats['Sex-Specific P/LP CP Effect']+=[{'BETA':np.nan,'SE':np.nan,'PVAL':np.nan}]

        #genotype effect stratifed by flagged vs unflagged
        if disease!='OMIM_ICD:86':
            cp_regression_matrix.insert(cp_regression_matrix.shape[1],'genotype x unflagged',geneticAnalysisTable['unflagged']*geneticAnalysisTable['genotype'],True)
            cp_lin_mod_flag_strat=sm.OLS(Y_CP,exog=cp_regression_matrix,hasconst=True).fit()
            cp_regression_stats['Baseline Variant Effect']+=[{'BETA':cp_lin_mod_flag_strat.params['genotype'],'SE':cp_lin_mod_flag_strat.bse['genotype'],'PVAL':cp_lin_mod_flag_strat.pvalues['genotype']}]
            cp_regression_stats['Unflagged Variant Effect']+=[{'BETA':cp_lin_mod_flag_strat.params['genotype x unflagged'],'SE':cp_lin_mod_flag_strat.bse['genotype x unflagged'],'PVAL':cp_lin_mod_flag_strat.pvalues['genotype x unflagged']}]
        else:
            cp_regression_stats['Baseline Variant Effect']+=[{'BETA':np.nan,'SE':np.nan,'PVAL':np.nan}]
            cp_regression_stats['Unflagged Variant Effect']+=[{'BETA':np.nan,'SE':np.nan,'PVAL':np.nan}]

        #dx effect wtihin carriers
        if geneticAnalysisTable['has_'+disease].sum()>0:
            cp_regression_matrix.insert(cp_regression_matrix.shape[1],'has_dx',geneticAnalysisTable['has_'+disease]*cp_regression_matrix['intercept'],True)
            carriers_only=copy.deepcopy(cp_regression_matrix.loc[cp_regression_matrix.genotype==1])
            Y_CP_carriers=np.array(geneticAnalysisTable.loc[cp_regression_matrix.genotype==1][['CrypticPhenotype_'+disease]].values,dtype=np.float)
            carriers_only.drop(columns=['genotype'],inplace=True)
            cp_lin_mod_dx=sm.OLS(Y_CP_carriers,exog=carriers_only,hasconst=True).fit()
            cp_regression_stats['Within Carrier Dx Effect']+=[{'BETA':cp_lin_mod_dx.params['has_dx'],'SE':cp_lin_mod_dx.bse['has_dx'],'PVAL':cp_lin_mod_dx.pvalues['has_dx']}]
        else:
            cp_regression_stats['Within Carrier Dx Effect']+=[{'BETA':np.nan,'SE':np.nan,'PVAL':np.nan}]

        ############# Figures ##########################
        ################################################
        if disease!='OMIM_ICD:86':



            fig_table=pd.DataFrame(index=geneticAnalysisTable.index)
            fig_table['Cryptic Phenotype Severity']=geneticAnalysisTable['CrypticPhenotype_'+disease]
            fig_table['Genotype']=pd.Series(['NaN']*geneticAnalysisTable.shape[0],geneticAnalysisTable.index)
            fig_table.loc[geneticAnalysisTable.genotype==0,'Genotype']='Reference'
            fig_table.loc[(geneticAnalysisTable.genotype==1)&(geneticAnalysisTable.unflagged==0),'Genotype']='Flagged P/LP Carrier'
            fig_table.loc[(geneticAnalysisTable.genotype==1)&(geneticAnalysisTable.unflagged==1),'Genotype']='P/LP Carrier'

            fig_table['Genotype']=fig_table['Genotype'].astype('category')
            fig_table['Genotype'].cat.reorder_categories(['P/LP Carrier','Flagged P/LP Carrier','Reference'], inplace=True)

            fig_table['Clinical Diagnosis']=pd.Series(['NaN']*geneticAnalysisTable.shape[0],geneticAnalysisTable.index)



            f, axes = plt.subplots(1, 2,figsize=(20,8))
            f.tight_layout(rect=[0.1,0.1,0.9,0.9])
            for axis in axes:
                axis.spines['right'].set_visible(False)
                axis.spines['top'].set_visible(False)

            if geneticAnalysisTable['has_'+disease].sum()>0:
                fig_table.loc[geneticAnalysisTable['has_'+disease]==True,'Clinical Diagnosis']='Pres.'
                fig_table.loc[geneticAnalysisTable['has_'+disease]==False,'Clinical Diagnosis']='Abs.'
                sns.stripplot(data=fig_table[fig_table.Genotype!='Reference'],y='Cryptic Phenotype Severity',x='Clinical Diagnosis',order=['Abs.','Pres.'],size=25,palette=[color_list[0],color_list[2]],ax=axes[1])

                dx_only=fig_table[fig_table.Genotype!='Reference']

                dx_mean=dx_only.loc[dx_only['Clinical Diagnosis']=='Pres.']['Cryptic Phenotype Severity'].mean()
                udx_mean=dx_only.loc[dx_only['Clinical Diagnosis']=='Abs.']['Cryptic Phenotype Severity'].mean()
                l1=axes[1].hlines(udx_mean,xmin=-0.25,xmax=0.25, lw=12.0,color='0.5')
                l2=axes[1].hlines(dx_mean,xmin=0.75,xmax=1.25,lw=12.0,color='0.5')
                l1.set_capstyle('round')
                l2.set_capstyle('round')

                lt=axes[1].get_ylim()[1]*1.05
                ht=axes[1].get_ylim()[1]*1.1
                axes[1].plot([0,0,1,1],[lt,ht,ht,lt],color='k',lw=5.0,solid_joinstyle='round')
                axes[1].text(0.5,ht*1.025,r'$P$={0:.2e}'.format(cp_lin_mod_dx.pvalues['has_dx']),horizontalalignment='center')
                axes[1].set_ylabel('Cryptic Phenotype\nSeverity')

            bins=np.linspace(np.floor(fig_table['Cryptic Phenotype Severity'].min()*2)/2,np.ceil(fig_table['Cryptic Phenotype Severity'].max()*2)/2,5).ravel()
            axes[0].hist([fig_table.loc[fig_table['Genotype']=='P/LP Carrier','Cryptic Phenotype Severity'],fig_table.loc[fig_table['Genotype']=='Flagged P/LP Carrier','Cryptic Phenotype Severity'],fig_table.loc[fig_table['Genotype']=='Reference','Cryptic Phenotype Severity']],log=True,histtype='barstacked',density=False,color=[color_list[0],color_list[4],grey_color],label=['P/LP Carrier','Flagged P/LP Carrier','Reference'],bins=bins)
            axes[0].set_xlabel('Cryptic Phenotype\nSeverity')
            axes[0].set_ylabel('# of Patients\n'+r'($\log_{10}$-Scale)')


            axes[0].text(axes[0].get_xlim()[0]-0.25*(axes[0].get_xlim()[1]-axes[0].get_xlim()[0]),axes[0].get_ylim()[0]+(axes[0].get_ylim()[1]-axes[0].get_ylim()[0])**1.25,r'Marginal Variant Effect: {0:.2f} ($P$={1:.2e})'.format(base_lin_mod.params['genotype'],base_lin_mod.pvalues['genotype']),fontsize=30,fontweight='bold')
            axes[0].text(axes[0].get_xlim()[0]-0.1*(axes[0].get_xlim()[1]-axes[0].get_xlim()[0]),axes[0].get_ylim()[0]+(axes[0].get_ylim()[1]-axes[0].get_ylim()[0])**1.15,r'Baseline Variant Effect: {0:.2f} ($P$={1:.2e})'.format(cp_lin_mod_flag_strat.params['genotype'],cp_lin_mod_flag_strat.pvalues['genotype']),fontsize=20,fontweight='bold')
            axes[0].text(axes[0].get_xlim()[0]-0.1*(axes[0].get_xlim()[1]-axes[0].get_xlim()[0]),axes[0].get_ylim()[0]+(axes[0].get_ylim()[1]-axes[0].get_ylim()[0])**1.075,r'Unflagged Effect: {0:.2f} ($P$={1:.2e})'.format(cp_lin_mod_flag_strat.params['genotype x unflagged'],cp_lin_mod_flag_strat.pvalues['genotype x unflagged']),fontsize=20,fontweight='bold')
            axes[0].legend(loc='best',frameon=False)



            plt.savefig(figure_path+disease.replace(':','_')+'/'+sample_subset+'_SummaryFigures.svg')

        else:


            #build covariate matrix
            new_regression_matrix=geneticAnalysisTable[['age_normalized','sex','array']+['PC{0:d}'.format(i) for i in range(1,11)]]

            pimz=pd.Series(np.zeros(new_regression_matrix.shape[0]),index=new_regression_matrix.index)
            pimz[geneticAnalysisTable.aux_genotype==1]=1
            pizz=pd.Series(np.zeros(new_regression_matrix.shape[0]),index=new_regression_matrix.index)
            pizz[geneticAnalysisTable.aux_genotype==2]=1

            new_regression_matrix.insert(new_regression_matrix.shape[1],'PiMZ',pimz,True)
            new_regression_matrix.insert(new_regression_matrix.shape[1],'PiZZ',pizz,True)

            new_regression_matrix.insert(0,'intercept',np.ones((new_regression_matrix.shape[0])),True)
            new_lin_mod=sm.OLS(Y_CP,exog=new_regression_matrix,hasconst=True).fit()


            fig_table=pd.DataFrame(index=geneticAnalysisTable.index)
            fig_table['Cryptic Phenotype Severity']=geneticAnalysisTable['CrypticPhenotype_'+disease]
            fig_table['Genotype']=pd.Series(['NaN']*geneticAnalysisTable.shape[0],geneticAnalysisTable.index)
            fig_table.loc[geneticAnalysisTable.aux_genotype==0,'Genotype']='PiMM'
            fig_table.loc[(geneticAnalysisTable.aux_genotype==1),'Genotype']='PiMZ'
            fig_table.loc[(geneticAnalysisTable.aux_genotype==2),'Genotype']='PiZZ'

            fig_table['Genotype']=fig_table['Genotype'].astype('category')
            fig_table['Genotype'].cat.reorder_categories(['PiMM','PiMZ','PiZZ'], inplace=True)




            f, axis = plt.subplots(1, 1,figsize=(10,8))
            f.tight_layout(rect=[0.1,0.1,0.9,0.9])
            axis.spines['right'].set_visible(False)
            axis.spines['top'].set_visible(False)


            bins=np.linspace(np.floor(fig_table['Cryptic Phenotype Severity'].min()*2)/2,np.ceil(fig_table['Cryptic Phenotype Severity'].max()*2)/2,5).ravel()

            axis.hist([fig_table.loc[fig_table['Genotype']=='PiZZ','Cryptic Phenotype Severity'],fig_table.loc[fig_table['Genotype']=='PiMZ','Cryptic Phenotype Severity'],fig_table.loc[fig_table['Genotype']=='PiMM','Cryptic Phenotype Severity']],log=True,histtype='barstacked',density=False,color=[color_list[0],color_list[4],grey_color],label=['PiZZ','PiMZ','PiMM'],bins=bins)
            axis.set_xlabel('Cryptic Phenotype\nSeverity')
            axis.set_ylabel('# of Patients\n'+r'($\log_{10}$-Scale)')


            axis.text(axis.get_xlim()[0]-0.1*(axis.get_xlim()[1]-axis.get_xlim()[0]),axis.get_ylim()[0]+(axis.get_ylim()[1]-axis.get_ylim()[0])**1.15,r'PiZZ Effect: {0:.2f} ({2:.2f}; $P$={1:.2e})'.format(new_lin_mod.params['PiZZ'],new_lin_mod.pvalues['PiZZ'],new_lin_mod.bse['PiZZ']),fontsize=20,fontweight='bold')
            axis.text(axis.get_xlim()[0]-0.1*(axis.get_xlim()[1]-axis.get_xlim()[0]),axis.get_ylim()[0]+(axis.get_ylim()[1]-axis.get_ylim()[0])**1.075,r'PiMZ Effect: {0:.2f} ({2:.2f}; $P$={1:.2e})'.format(new_lin_mod.params['PiMZ'],new_lin_mod.pvalues['PiMZ'],new_lin_mod.bse['PiMZ']),fontsize=20,fontweight='bold')
            axis.legend(loc='best',frameon=False)


            plt.savefig(figure_path+disease.replace(':','_')+'/'+sample_subset+'_SummaryFigures.svg')







        ############# Dx Regression Analysis ##############
        ###################################################

        if geneticAnalysisTable['has_'+disease].sum()>0:

            #now logistic regrssion directly on the diagnostic codes, first with all variants

            dx_regression_matrix=geneticAnalysisTable[['age_normalized','sex','array']+['PC{0:d}'.format(i) for i in range(1,11)]+['genotype']]
            dx_regression_matrix.insert(0,'intercept',np.ones((dx_regression_matrix.shape[0])),True)
            dx_regression_matrix.insert(dx_regression_matrix.shape[1],'CP',Y_CP,True)
            dx_regression_matrix.insert(dx_regression_matrix.shape[1],'genotype x CP',Y_CP.ravel()*dx_regression_matrix['genotype'],True)
            dx_regression_matrix.insert(dx_regression_matrix.shape[1],'has_'+disease,np.array(geneticAnalysisTable['has_'+disease].values,dtype=np.int),True)

            firth = FirthRegression(dx_regression_matrix,['intercept','age_normalized','sex','array']+['PC{0:d}'.format(i) for i in range(1,11)]+['genotype','CP','genotype x CP'],'has_'+disease,hasconst=True)
            log_reg_variant_effect=firth.FirthInference('genotype')
            log_reg_cp_effect=firth.FirthInference('CP')
            log_reg_interaction_effect=firth.FirthInference('genotype x CP')



            dx_regression_stats['P/LP Dx Effect']+=[{'BETA':log_reg_variant_effect['ParamTable'].loc['genotype'].BETA,'SE':log_reg_variant_effect['ParamTable'].loc['genotype'].SE,'PVAL':log_reg_variant_effect['PVal']}]

            dx_regression_stats['CP Dx Effect']+=[{'BETA':log_reg_cp_effect['ParamTable'].loc['CP'].BETA,'SE':log_reg_cp_effect['ParamTable'].loc['CP'].SE,'PVAL':log_reg_cp_effect['PVal']}]
            dx_regression_stats['CP x LP/P Dx Effect']+=[{'BETA':log_reg_interaction_effect['ParamTable'].loc['genotype x CP'].BETA
            ,'SE':log_reg_interaction_effect['ParamTable'].loc['genotype x CP'].SE,'PVAL':log_reg_interaction_effect['PVal']}]

            #next with flagged dropped
            dx_regression_matrix_dropped=dx_regression_matrix.drop(index=geneticAnalysisTable.loc[(geneticAnalysisTable.genotype==1)&(geneticAnalysisTable.unflagged==0)].index)
            firth = FirthRegression(dx_regression_matrix_dropped,['intercept','age_normalized','sex','array']+['PC{0:d}'.format(i) for i in range(1,11)]+['genotype','CP','genotype x CP'],'has_'+disease,hasconst=True)
            log_reg_variant_effect=firth.FirthInference('genotype')
            log_reg_cp_effect=firth.FirthInference('CP')
            log_reg_interaction_effect=firth.FirthInference('genotype x CP')

            dx_regression_stats['P/LP Dx Effect (Flagged Dropped)']+=[{'BETA':log_reg_variant_effect['ParamTable'].loc['genotype'].BETA,'SE':log_reg_variant_effect['ParamTable'].loc['genotype'].SE,'PVAL':log_reg_variant_effect['PVal']}]

            dx_regression_stats['CP Dx Effect (Flagged Dropped)']+=[{'BETA':log_reg_cp_effect['ParamTable'].loc['CP'].BETA,'SE':log_reg_cp_effect['ParamTable'].loc['CP'].SE,'PVAL':log_reg_cp_effect['PVal']}]

            dx_regression_stats['CP x LP/P Dx Effect (Flagged Dropped)']+=[{'BETA':log_reg_interaction_effect['ParamTable'].loc['genotype x CP'].BETA
            ,'SE':log_reg_interaction_effect['ParamTable'].loc['genotype x CP'].SE,'PVAL':log_reg_interaction_effect['PVal']}]


        else:
            dx_regression_stats['P/LP Dx Effect']+=[{'BETA':np.nan,'SE':np.nan,'PVAL':np.nan}]
            dx_regression_stats['CP Dx Effect']+=[{'BETA':np.nan,'SE':np.nan,'PVAL':np.nan}]
            dx_regression_stats['CP x LP/P Dx Effect']+=[{'BETA':np.nan,'SE':np.nan,'PVAL':np.nan}]
            dx_regression_stats['P/LP Dx Effect (Flagged Dropped)']+=[{'BETA':np.nan,'SE':np.nan,'PVAL':np.nan}]
            dx_regression_stats['CP Dx Effect (Flagged Dropped)']+=[{'BETA':np.nan,'SE':np.nan,'PVAL':np.nan}]
            dx_regression_stats['CP x LP/P Dx Effect (Flagged Dropped)']+=[{'BETA':np.nan,'SE':np.nan,'PVAL':np.nan}]




    basic_stats_table=pd.DataFrame(basic_stats_table)
    basic_stats_table.set_index('Disease',inplace=True)
    basic_stats_table.to_pickle(table_path+sample_subset+'_BasicStats.pth')

    cp_regression_stats=pd.DataFrame(cp_regression_stats)
    cp_regression_stats.set_index('Disease',inplace=True)
    cp_regression_stats.to_pickle(table_path+sample_subset+'_CPRegressionStats.pth')

    dx_regression_stats=pd.DataFrame(dx_regression_stats)
    dx_regression_stats.set_index('Disease',inplace=True)
    dx_regression_stats.to_pickle(table_path+sample_subset+'_DxRegressionStats.pth')



    basic_stats_tableStrings=copy.deepcopy(basic_stats_table)
    basic_stats_tableStrings['P/LP AF']=basic_stats_tableStrings['P/LP AF'].apply(lambda x: '{0:.2e}'.format(x))
    basic_stats_tableStrings['Dx Prev']=basic_stats_tableStrings['Dx Prev'].apply(lambda x: '{0:.2e}'.format(x))
    basic_stats_tableStrings.to_csv(table_path+sample_subset+'_BasicStats.txt',sep='\t')

    cp_regression_statsStrings=copy.deepcopy(cp_regression_stats)
    cp_regression_statsStrings['Marginal P/LP CP Effect']=cp_regression_statsStrings['Marginal P/LP CP Effect'].apply(lambda x: convertResultsToString(x))
    cp_regression_statsStrings['Sex-Specific P/LP CP Effect']=cp_regression_statsStrings['Sex-Specific P/LP CP Effect'].apply(lambda x: convertResultsToString(x))
    cp_regression_statsStrings['Baseline Variant Effect']=cp_regression_statsStrings['Baseline Variant Effect'].apply(lambda x: convertResultsToString(x))
    cp_regression_statsStrings['Unflagged Variant Effect']=cp_regression_statsStrings['Unflagged Variant Effect'].apply(lambda x: convertResultsToString(x))
    cp_regression_statsStrings['Within Carrier Dx Effect']=cp_regression_statsStrings['Within Carrier Dx Effect'].apply(lambda x: convertResultsToString(x))
    cp_regression_statsStrings.to_csv(table_path+sample_subset+'_CPRegressionStats.txt',sep='\t')




    dx_regression_statsStrings=copy.deepcopy(dx_regression_stats)
    dx_regression_statsStrings['P/LP Dx Effect']=dx_regression_statsStrings['P/LP Dx Effect'].apply(lambda x: convertResultsToString(x))
    dx_regression_statsStrings['CP Dx Effect']=dx_regression_statsStrings['CP Dx Effect'].apply(lambda x: convertResultsToString(x))
    dx_regression_statsStrings['CP x LP/P Dx Effect']=dx_regression_statsStrings['CP x LP/P Dx Effect'].apply(lambda x: convertResultsToString(x))
    dx_regression_statsStrings['P/LP Dx Effect (Flagged Dropped)']=dx_regression_statsStrings['P/LP Dx Effect (Flagged Dropped)'].apply(lambda x: convertResultsToString(x))
    dx_regression_statsStrings['CP Dx Effect (Flagged Dropped)']=dx_regression_statsStrings['CP Dx Effect (Flagged Dropped)'].apply(lambda x: convertResultsToString(x))
    dx_regression_statsStrings['CP x LP/P Dx Effect (Flagged Dropped)']=dx_regression_statsStrings['CP x LP/P Dx Effect (Flagged Dropped)'].apply(lambda x: convertResultsToString(x))
    dx_regression_statsStrings.to_csv(table_path+sample_subset+'_DxRegressionStats.txt',sep='\t')

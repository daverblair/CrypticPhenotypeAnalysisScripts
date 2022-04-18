import pandas as pd
import numpy as np
from scipy.stats.mstats import mquantiles
from scipy.stats import spearmanr,chi2,beta
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

sns.set(context='talk',color_codes=True,style='ticks',font='Arial',font_scale=2,rc={'axes.linewidth':5,"font.weight":"bold",'axes.labelweight':"bold",'xtick.major.width':4,'xtick.minor.width': 2})
cmap = cm.get_cmap('viridis', 12)
color_list=[cmap(x) for x in [0.0,0.1,0.25,0.5,0.75,0.9,1.0]]
grey_color=(0.25, 0.25, 0.25)
red_color = '#d10e00'
blue_color='#5DA4FF'


def LambdaGC(p_val_vec,quantile_list=[0.5],scaling_factor=None):
    quantile_list=np.array(quantile_list)
    obs_pval_quantiles=mquantiles(p_val_vec,prob=quantile_list,alphap=1.0,betap=1.0)
    if scaling_factor is not None:
        lambda_unscaled = chi2.ppf(1.0-obs_pval_quantiles, 1) / chi2.ppf(1.0-quantile_list,1)
        return 1+(lambda_unscaled-1.0)*(scaling_factor/p_val_vec.shape[0])
    else:
        return chi2.ppf(1.0-obs_pval_quantiles, 1) / chi2.ppf(1.0-quantile_list,1)


def QQPlot(data_table,p_value_column=None,maf_column=None,freq_bins=None,n_quantiles=1000,error_ci=0.95,min_p=1e-30,hide_hla=False,error_type='experimental',lambda_gc_scale=None):
    f, axis = plt.subplots(1, 1,figsize=(8,8))
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    if p_value_column==None:
        p_value_column='P'
    if maf_column==None:
        if 'MAF' in data_table.columns:
            maf_column='MAF'
        else:
            data_table['MAF']=np.zeros(len(data_table))*np.nan

    if hide_hla:
        chr6 = data_table.loc[(data_table.CHROM==6)]
        excluded=chr6.index[np.logical_and(chr6.POS>=28477797,chr6.POS<=33448354)]
        p_maf_table=data_table.drop(excluded)[[maf_column,p_value_column]]
    elif maf_column is not None:
        p_maf_table=data_table[[maf_column,p_value_column]]
    else:
        p_maf_table=data_table[[p_value_column]]

    assert error_type in ['experimental','theoretical'],"Error type must be in ['experimental','theoretical']"



    min_vals_obs=[]
    min_vals_exp=[]
    if freq_bins is None:
        p_input= p_maf_table[p_value_column].values
        p_input[p_input<min_p]=min_p
        quantile_thresholds = np.concatenate([np.arange(1,np.floor(0.5*n_quantiles))/p_input.shape[0], np.logspace(np.log10(np.floor(0.5*n_quantiles)/p_input.shape[0]), 0, int(np.ceil(0.5*n_quantiles))+1)[:-1]])
        obs_quantiles = mquantiles(p_input, prob=quantile_thresholds, alphap=0.0, betap=1.0, limit=(0.0, 1.0))
        axis.plot(-np.log10(quantile_thresholds),-np.log10(obs_quantiles),'.',color=color_list[0],ms=15)
        if lambda_gc_scale is not None:
            axis.text(1,5,r'$\lambda_{IF}$'+'={0:1.2f}'.format(LambdaGC(p_input)[0])+' ('+r'$\lambda^{'+'{0:d}'.format(lambda_gc_scale)+'}_{IF}$'+'={0:1.3f}'.format(LambdaGC(p_input,scaling_factor=lambda_gc_scale)[0])+')',fontsize=24,fontweight='bold',color=color_list[0])
        else:
            axis.text(1,5,r'$\lambda_{IF}$'+'={0:1.2f}'.format(LambdaGC(p_input)[0]),fontsize=24,fontweight='bold',color=color_list[0])

        min_vals_obs+=[obs_quantiles.min()]
        min_vals_exp+=[quantile_thresholds.min()]
        if error_type=='experimental':
            ci_vecs = beta.interval(error_ci, len(p_maf_table)*quantile_thresholds, len(p_maf_table) - len(p_maf_table)*quantile_thresholds)


            axis.fill_between( -np.log10(quantile_thresholds), -np.log10(obs_quantiles/quantile_thresholds*ci_vecs[0]), -np.log10(obs_quantiles/quantile_thresholds*ci_vecs[1]), color=color_list[0], alpha=0.25, label='{0:2d}% CI'.format(int(100*error_ci)))


    else:
        for i in range(len(freq_bins)-2):
            p_input= p_maf_table[np.logical_and(p_maf_table[maf_column]>=freq_bins[i],p_maf_table[maf_column]<freq_bins[i+1])][p_value_column].values
            p_input[p_input<min_p]=min_p
            quantile_thresholds = np.concatenate([np.arange(1,np.floor(0.5*n_quantiles))/p_input.shape[0], np.logspace(np.log10(np.floor(0.5*n_quantiles)/p_input.shape[0]), 0, int(np.ceil(0.5*n_quantiles))+1)[:-1]])
            obs_quantiles = mquantiles(p_input, prob=quantile_thresholds, alphap=0.0, betap=1.0, limit=(0.0, 1.0))
            axis.plot(-np.log10(quantile_thresholds),-np.log10(obs_quantiles),'.',ms=15,color=color_list[(i*2)%len(color_list)],label=r'{0:.1e}$\leq$ MAF$<${1:.1e}'.format(freq_bins[i],freq_bins[i+1]))
            if error_type=='experimental':
                ci_vecs = beta.interval(error_ci, len(p_maf_table)*quantile_thresholds, len(p_maf_table) - len(p_maf_table)*quantile_thresholds)
                axis.fill_between( -np.log10(quantile_thresholds), -np.log10(obs_quantiles/quantile_thresholds*ci_vecs[0]), -np.log10(obs_quantiles/quantile_thresholds*ci_vecs[1]), color=color_list[(i*2)%len(color_list)], alpha=0.25, label='{0:2d}% CI'.format(int(100*error_ci)))

            if lambda_gc_scale is not None:
                axis.text(1,5-i,r'$\lambda_{IF}$'+'={0:1.2f}'.format(LambdaGC(p_input)[0])+' ('+r'$\lambda^{'+'{0:d}'.format(lambda_gc_scale)+'}_{IF}$'+'={0:1.3f}'.format(LambdaGC(p_input,scaling_factor=lambda_gc_scale)[0])+')',fontsize=24,fontweight='bold',color=color_list[i*2])
            else:
                axis.text(1,5-i,r'$\lambda_{IF}$'+'={0:1.2f}'.format(LambdaGC(p_input)[0]),fontsize=24,fontweight='bold',color=color_list[i*2])

            min_vals_obs+=[obs_quantiles.min()]
            min_vals_exp+=[quantile_thresholds.min()]

        i+=1
        p_input= p_maf_table[np.logical_and(p_maf_table[maf_column]>=freq_bins[i],p_maf_table[maf_column]<=freq_bins[i+1])][p_value_column].values
        p_input[p_input<min_p]=min_p
        quantile_thresholds = np.concatenate([np.arange(1,np.floor(0.5*n_quantiles))/p_input.shape[0], np.logspace(np.log10(np.floor(0.5*n_quantiles)/p_input.shape[0]), 0, int(np.ceil(0.5*n_quantiles))+1)[:-1]])
        obs_quantiles = mquantiles(p_input, prob=quantile_thresholds, alphap=0.0, betap=1.0, limit=(0.0, 1.0))
        axis.plot(-np.log10(quantile_thresholds),-np.log10(obs_quantiles),'o',color=color_list[(i*2)%len(color_list)],mew=0.0,label=r'{0:.1e}$\leq$ MAF$\leq${1:.1e}'.format(freq_bins[i],0.5))
        if error_type=='experimental':
            ci_vecs = beta.interval(error_ci, len(p_maf_table)*quantile_thresholds, len(p_maf_table) - len(p_maf_table)*quantile_thresholds)

            axis.fill_between( -np.log10(quantile_thresholds), -np.log10(obs_quantiles/quantile_thresholds*ci_vecs[0]), -np.log10(obs_quantiles/quantile_thresholds*ci_vecs[1]), color=color_list[(i*2)%len(color_list)], alpha=0.25, label='{0:2d}% CI'.format(int(100*error_ci)))

        if lambda_gc_scale is not None:
            axis.text(1,5-i,r'$\lambda_{IF}$'+'={0:1.2f}'.format(LambdaGC(p_input)[0])+' ('+r'$\lambda^{'+'{0:d}'.format(lambda_gc_scale)+'}_{IF}$'+'={0:1.3f}'.format(LambdaGC(p_input,scaling_factor=lambda_gc_scale)[0])+')',fontsize=24,fontweight='bold',color=color_list[i*2])
        else:
            axis.text(1,5-i,r'$\lambda_{IF}$'+'={0:1.2f}'.format(LambdaGC(p_input)[0]),fontsize=24,fontweight='bold',color=color_list[i*2])
        min_vals_obs+=[obs_quantiles.min()]
        min_vals_exp+=[quantile_thresholds.min()]



    axis.set_xlim(0.0,np.ceil(-np.log10(min(min_vals_exp))))



    exp_p_vals = np.linspace(0,axis.get_xlim()[1],100)
    if error_type=='theoretical':
        ci_vecs = beta.interval(error_ci, len(p_maf_table)*(10**(-1.0*exp_p_vals)), len(p_maf_table) - len(p_maf_table)*(10**(-1.0*exp_p_vals)))
        axis.fill_between(exp_p_vals, -np.log10(ci_vecs[0]), -np.log10(ci_vecs[1]), color=grey_color, alpha=0.25, label='{0:2d}% CI'.format(int(100*error_ci)))

    axis.plot(exp_p_vals,exp_p_vals,'--',color=red_color,lw=3.0)

    axis.set_ylim(0.0,np.ceil(-np.log10(min(min(min_vals_obs),ci_vecs[0].min(),min(min_vals_obs))))+1)

    axis.legend(loc='upper left',frameon=False,fontsize=14)
    axis.set_xlabel(r'$\log_{10}$(P-Value)'+'\nExpected',fontsize=24)
    axis.set_ylabel(r'$\log_{10}$(P-Value)'+'\nObserved',fontsize=24)
    return f,axis

def ManhattanPlot(data_table,p_value_column='P',chrom_column='CHROM',pos_column='POS',allele_freq_window=None,maf_column=None,marked_column=None,all_sig_thresh=[5e-8],chrom_colors = [color_list[0],color_list[3]],alpha_min=1.0,min_p=1e-30,hide_hla=False,thin_data=True,thin_data_thresh=1e-4):
    f, axis = plt.subplots(1, 1,figsize=(24,8))
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['bottom'].set_visible(False)

    included_columns=[chrom_column,pos_column,p_value_column]
    if marked_column is not None:
        included_columns+=[marked_column]

    if allele_freq_window is not None:
        if maf_column is None:
            fig_table=data_table[np.logical_and(data_table.MAF>=allele_freq_window[0],data_table.MAF<allele_freq_window[1])][included_columns]
        else:
            fig_table=data_table[np.logical_and(data_table[maf_column]>=allele_freq_window[0],data_table[maf_column]<allele_freq_window[1])][included_columns]

    else:
        fig_table=data_table[included_columns]

    if hide_hla:
        chr6 = fig_table.loc[(fig_table[chrom_column]==6)]
        excluded=chr6.index[np.logical_and(chr6[pos_column]>=28477797,chr6[pos_column]<=33448354)]
        fig_table=fig_table.drop(excluded)[included_columns]


    axis.set_xlim(0.0,1.05)
    axis.set_ylim(0.0,np.ceil(min(-np.log10(fig_table[p_value_column].min()),-np.log10(min_p))))


    all_chrom = np.unique(fig_table[chrom_column])
    all_chrom.sort()

    offsets = np.zeros(all_chrom.shape[0])
    total_bps=0.0
    for i,c in enumerate(all_chrom):
        offsets[i]=total_bps
        total_bps+=np.max(fig_table.loc[fig_table[chrom_column]==c][pos_column].values)
    offsets/=total_bps
    offsets=np.append(offsets,1.0)


    for i,c in enumerate(all_chrom):

        current_chrom = fig_table.loc[fig_table[chrom_column]==c]

        plot_table=pd.DataFrame(index=current_chrom.index)
        plot_table['logP']=-np.log10(current_chrom[p_value_column])
        plot_table.loc[plot_table.logP>(-np.log10(min_p))]=-np.log10(min_p)
        plot_table['pos']=(current_chrom[pos_column]/total_bps)+offsets[i]
        if marked_column is not None:
            plot_table['mark']=current_chrom[marked_column]

        if thin_data:
            #thins p-values less than 1e-5 by rounding and taking only unique values
            plot_table['logP_R']=plot_table['logP'].values
            plot_table.loc[plot_table.logP<(-np.log10(thin_data_thresh)),'logP_R']=np.round(plot_table.loc[plot_table.logP<(-np.log10(thin_data_thresh))]['logP_R'].values,1)
            plot_table['pos_R']=np.round(plot_table['pos']*5,2)
            plot_table=plot_table.drop_duplicates(['logP_R','pos_R'])


        rgba_colors = np.zeros((plot_table.shape[0],4))
        rgba_colors[:,0:4] = np.array(chrom_colors[i%2])
        alpha_levels = (1.0-alpha_min)*(plot_table.logP)/(-np.log10(min_p))+alpha_min
        alpha_levels[alpha_levels>1.0]=1.0
        rgba_colors[:,3]=alpha_levels
        axis.scatter(plot_table.pos,plot_table.logP,s=75.0,marker='o',color=rgba_colors,lw=0.0)
        if marked_column is not None:
            axis.scatter(plot_table.loc[plot_table['mark']==True].pos,plot_table.loc[plot_table['mark']==True].logP,s=75.0,marker='*',color=np.array(red_color),lw=0.0)

    for sig_thresh in all_sig_thresh:

        axis.hlines(-np.log10(sig_thresh),0.0,1.0,linestyle='--',color=red_color,alpha=0.75,lw=2.0)
        axis.text(0.1,-np.log10(sig_thresh)+0.05*axis.get_ylim()[1],r'Sig. Level {0:.1e}'.format(sig_thresh),fontsize=12)


    axis.set_ylabel(r"$P$-Value"+'\n'+r'($-\log_{10}$-Scale)',fontsize=24)
    axis.set_xlabel('Chromsome',fontsize=24)
    chrom_locators = offsets[:-1]+(offsets[1:]-offsets[:-1])/2.0
    axis.xaxis.set_major_locator(plt.FixedLocator(chrom_locators[0::2]))
    axis.xaxis.set_major_formatter(plt.FixedFormatter(np.array(all_chrom,dtype=np.str)[0::2]))
    axis.xaxis.set_minor_locator(plt.FixedLocator(chrom_locators[1::2]))

    return f,axis

def ManhattanPlot(data_table,p_value_column='P',chrom_column='CHROM',pos_column='POS',allele_freq_window=None,maf_column=None,marked_column=None,snp_to_gene=None,all_sig_thresh=[5e-8],chrom_colors = [color_list[0],color_list[3]],alpha_min=1.0,min_p=1e-30,hide_hla=False,thin_data=True,thin_data_thresh=1e-6):
    f, axis = plt.subplots(1, 1,figsize=(24,8))
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['bottom'].set_visible(False)

    included_columns=[chrom_column,pos_column,p_value_column]
    if marked_column is not None:
        included_columns+=[marked_column]

    if allele_freq_window is not None:
        if maf_column is None:
            fig_table=data_table[np.logical_and(data_table.MAF>=allele_freq_window[0],data_table.MAF<allele_freq_window[1])][included_columns]
        else:
            fig_table=data_table[np.logical_and(data_table[maf_column]>=allele_freq_window[0],data_table[maf_column]<allele_freq_window[1])][included_columns]

    else:
        fig_table=data_table[included_columns]

    if hide_hla:
        chr6 = fig_table.loc[(fig_table[chrom_column]==6)]
        excluded=chr6.index[np.logical_and(chr6[pos_column]>=28477797,chr6[pos_column]<=33448354)]
        fig_table=fig_table.drop(excluded)[included_columns]


    axis.set_xlim(0.0,1.05)
    axis.set_ylim(0.0,np.ceil(min(-np.log10(fig_table[p_value_column].min()),-np.log10(min_p))))


    all_chrom = np.unique(fig_table[chrom_column])
    all_chrom.sort()

    offsets = np.zeros(all_chrom.shape[0])
    total_bps=0.0
    for i,c in enumerate(all_chrom):
        offsets[i]=total_bps
        total_bps+=np.max(fig_table.loc[fig_table[chrom_column]==c][pos_column].values)
    offsets/=total_bps
    offsets=np.append(offsets,1.0)


    for i,c in enumerate(all_chrom):

        current_chrom = fig_table.loc[fig_table[chrom_column]==c]

        plot_table=pd.DataFrame(index=current_chrom.index)
        plot_table['logP']=-np.log10(current_chrom[p_value_column])
        plot_table.loc[plot_table.logP>(-np.log10(min_p))]=-np.log10(min_p)
        plot_table['pos']=(current_chrom[pos_column]/total_bps)+offsets[i]
        if marked_column is not None:
            plot_table['mark']=current_chrom[marked_column]

        if thin_data:
            #thins p-values less than thin_data_thresh by rounding and taking only unique values
            plot_table['logP_R']=plot_table['logP'].values
            plot_table.loc[plot_table.logP<(-np.log10(thin_data_thresh)),'logP_R']=np.round(plot_table.loc[plot_table.logP<(-np.log10(thin_data_thresh))]['logP_R'].values,1)
            plot_table['pos_R']=np.round(plot_table['pos']*5,2)
            plot_table=plot_table.drop_duplicates(['logP_R','pos_R'])

        rgba_colors = np.zeros((plot_table.shape[0],4))
        rgba_colors[:,0:4] = np.array(chrom_colors[i%2])
        alpha_levels = (1.0-alpha_min)*(plot_table.logP)/(-np.log10(min_p))+alpha_min
        alpha_levels[alpha_levels>1.0]=1.0
        rgba_colors[:,3]=alpha_levels
        axis.scatter(plot_table.pos,plot_table.logP,s=75.0,marker='o',color=rgba_colors,lw=0.0)
        if marked_column is not None:
            axis.scatter(plot_table.loc[plot_table['mark']==True].pos,plot_table.loc[plot_table['mark']==True].logP,s=75.0,marker='*',color=red_color,lw=0.0)

    for sig_thresh in all_sig_thresh:

        axis.hlines(-np.log10(sig_thresh),0.0,1.0,linestyle='--',color=red_color,alpha=0.75,lw=2.0)
        axis.text(0.1,-np.log10(sig_thresh)+0.05*axis.get_ylim()[1],r'Sig. Level {0:.1e}'.format(sig_thresh),fontsize=18)

    if snp_to_gene is not None:
        for snp,gene_list in snp_to_gene.items():
            x_loc=((fig_table.loc[snp][pos_column]/total_bps)+offsets[int(fig_table.loc[snp][chrom_column])-1])+0.001*axis.get_ylim()[1]
            y_loc=-np.log10(fig_table.loc[snp][p_value_column])
            axis.text(x_loc,y_loc,'\n'.join(gene_list),fontsize=18,fontstyle='italic')


    axis.set_ylabel(r"$P$-Value"+'\n'+r'($-\log_{10}$-Scale)',fontsize=24)
    axis.set_xlabel('Chromsome',fontsize=24)
    chrom_locators = offsets[:-1]+(offsets[1:]-offsets[:-1])/2.0
    axis.xaxis.set_major_locator(plt.FixedLocator(chrom_locators[0::2]))
    axis.xaxis.set_major_formatter(plt.FixedFormatter(np.array(all_chrom,dtype=np.str)[0::2]))
    axis.xaxis.set_minor_locator(plt.FixedLocator(chrom_locators[1::2]))

    return f,axis

import pandas as pd
import pickle
import os
import numpy as np
import argparse

"""
This script performs the filtering based on CPA performance that was discussed in the main text and methods of Blair et al. 2022.
"""

##### Selection Criteria #######
# 1) Must converge in both datasets (selection already made)
# 2) Dx rare disease cases must have significantly higher severities in held out test data in UCSF data
# 3) Top performing component in UKBB must be top performing component in UCSF
# 4) Dx rare disease cases must have significantly higher severities in held out test data in UCSF AND UKBB data
# 5) Ensure UKBB model produces correlated severity predictions above some threshold--passed in as R_2_thresh

################################

parser = argparse.ArgumentParser(description='Filters diseases')
parser.add_argument("R_2_thresh",help="R^2 threshold for downstream analyses",type=float)
args = parser.parse_args()


ucsfResults=pd.read_pickle('path/to/collected/results/for/Step6/FinalModels_UCSFPerformanceResults.pth')
ukbbResults=pd.read_pickle('path/to/collected/results/for/Step6/FinalModels_UKBBPerformanceResults.pth')
combined_inference_results=pd.read_pickle('path/to/table/containing/modeling/results/summary/ModelInferenceCombinedResults.pth')
namesInheritanceFreq = pd.read_csv('path/to/SupplementaryDataFile_1.txt',sep='\t',index_col='Disease ID')



follow_up_diseases = ucsfResults.index
oFile=open('FilteringStats.txt','w')
#Selection Step 1
pass_filter_1=ucsfResults['Case Severity P-valiue'].apply(lambda x:x<0.05/ucsfResults.shape[0])
oFile.write("Number of Rare Diseases with Significantly Higher Severity: {0:d}/{1:d}\n".format(pass_filter_1.sum(),len(pass_filter_1)))
follow_up_diseases=follow_up_diseases[pass_filter_1]

#Selection Step 2
pass_filter_2=(ukbbResults.loc[follow_up_diseases]['Top Component-UKBB']==ukbbResults.loc[follow_up_diseases]['Top Component-UCSF'])
oFile.write("Number of Diseases with Identical Top Components in UCSF/UKBB (UKBB Model): {0:d}/{1:d}\n".format(pass_filter_2.sum(),len(pass_filter_2)))
follow_up_diseases=follow_up_diseases[pass_filter_2]

#Selection Step 3
pass_filter_3=ukbbResults.loc[follow_up_diseases]['UKBB Case Severity P-valiue'].apply(lambda x: True if np.isnan(x) else (x<0.05/follow_up_diseases.shape[0]))
oFile.write("Number of Rare Diseases with Significantly Higher Severity in UKBB: {0:d}/{1:d}\n".format(pass_filter_3.sum(),len(pass_filter_3)))
follow_up_diseases=follow_up_diseases[pass_filter_3]

#Selection Step 4
pass_filter_4=ukbbResults.loc[follow_up_diseases]['UCSF Case Severity P-valiue'].apply(lambda x:x<(0.05/follow_up_diseases.shape[0]))
oFile.write("Number of Rare Diseases with Significantly Higher Severity in UCSF (UKBB Model): {0:d}/{1:d}\n".format(pass_filter_4.sum(),len(pass_filter_4)))
follow_up_diseases=follow_up_diseases[pass_filter_4]

#Selection Step 5
pass_filter_5=ukbbResults.loc[follow_up_diseases]['UCSF-UKBB Model R^2'].apply(lambda x:x>args.R_2_thresh)
oFile.write("Number of Rare Diseases with UCSF-UKBB Model R^2 > {2:f}: {0:d}/{1:d}\n".format(pass_filter_5.sum(),len(pass_filter_5),args.R_2_thresh))
follow_up_diseases=follow_up_diseases[pass_filter_5]
oFile.close()

def _collapseSeverity(dataRows,index):
    o=[]
    for d in dataRows.iterrows():
        d=d[1]
        o+=['{0:f} ({1:f}, [{2:f}, {3:f}])'.format(d[0],d[1],d[2][0],d[2][1])]
    return pd.Series(o,index=index)

final_table=namesInheritanceFreq.loc[follow_up_diseases]
final_table['UCSF Model-Top Component']=ucsfResults.loc[follow_up_diseases]['Top Component']
final_table['UCSF Model-Avg Precision (Fold-increase; Training Data)']=ucsfResults.loc[follow_up_diseases]['Top Component Avg. Precision']/namesInheritanceFreq.loc[follow_up_diseases]['UCSF Prevalence']
final_table['UCSF Model-Case Severity Increase (Testing Data)']=_collapseSeverity(ucsfResults.loc[follow_up_diseases][['Case Severity Increase','Case Severity P-valiue','Case Severity Increase (95% CI)']],follow_up_diseases)
final_table['UKBB Model-Top Component']=ukbbResults.loc[follow_up_diseases]['Top Component-UKBB']
final_table['UKBB Model-Avg Precision (Fold-increase; Training Data-UCSF)']=ukbbResults.loc[follow_up_diseases]['UCSF Avg. Precision']/namesInheritanceFreq.loc[follow_up_diseases]['UCSF Prevalence']
final_table['UKBB Model-Avg Precision (Fold-increase; Training Data-UKBB)']=ukbbResults.loc[follow_up_diseases]['UKBB Avg. Precision']/namesInheritanceFreq.loc[follow_up_diseases]['UKBB Prevalence']
final_table['UKBB Model-Case Severity Increase (Testing Data-UCSF)']=_collapseSeverity(ukbbResults.loc[follow_up_diseases][['UCSF Case Severity Increase','UCSF Case Severity P-valiue','UCSF Case Severity Increase (95% CI)']],follow_up_diseases)
final_table['UKBB Model-Case Severity Increase (Testing Data-UKBB)']=_collapseSeverity(ukbbResults.loc[follow_up_diseases][['UKBB Case Severity Increase','UKBB Case Severity P-valiue','UKBB Case Severity Increase (95% CI)']],follow_up_diseases)
final_table['UKBB Model-UCSF Model R^2 (UCSF Dataset)']=ukbbResults.loc[follow_up_diseases]['UCSF-UKBB Model R^2']
final_table['UCSF Max. Model Rank']=combined_inference_results['UCSF Max. Model Rank']
final_table['UKBB Max. Model Rank']=combined_inference_results['UKBB Max. Model Rank']
final_table['Annotated HPO Terms UKBB']=combined_inference_results['Annotated HPO Terms UKBB']
final_table['Annotated HPO Terms UCSF']=combined_inference_results['Annotated HPO Terms']
final_table['Covariate Set']=combined_inference_results['Covariate Set']

final_table.to_pickle('FilteredModelingResults.pth')

final_table['Annotated HPO Terms UKBB']=combined_inference_results['Annotated HPO Terms UKBB'].apply(lambda x:';'.join(x))
final_table['Annotated HPO Terms UCSF']=combined_inference_results['Annotated HPO Terms'].apply(lambda x:';'.join(x))
final_table.to_csv('FilteredModelingResults.txt',sep='\t')

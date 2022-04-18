import pandas as pd

"""
This script concatenates a summary of the parameters/results for the cryptic phenotype models fit within each dataset. The table is used downstream by CollectResults_FilterFinalDiseases.py
"""

UCSFTable=pd.read_pickle('../UCSF/SummaryTable-7/UCSFModelingResults.pth')
UKBBTable=pd.read_pickle('../UKBB/FinalModels-4/ConvergenceResultsTable.pth')



converged_in_both=UKBBTable[UKBBTable['Converged']==True].index.intersection(UCSFTable.index[UCSFTable['Inference Converged']==True])

combined_table=UCSFTable.loc[converged_in_both][['Annotated HPO Terms', 'Annotated HPO Terms UKBB','Covariate Set']]
combined_table['UCSF Max. Model Rank']=UCSFTable.loc[converged_in_both]['Max. Model Rank']
combined_table['UKBB Max. Model Rank']=UKBBTable.loc[converged_in_both]['Rank']
combined_table['UCSF Inference Parameters']=UCSFTable.loc[converged_in_both]['Inference Parameters']
combined_table['UKBB Inference Parameters']=UKBBTable.loc[converged_in_both]['Inference Parameters']
combined_table.to_pickle('ModelInferenceCombinedResults.pth')

with open('ConvergedInBoth.txt','w') as f:
    f.write('\n'.join(list(combined_table.index))+'\n')

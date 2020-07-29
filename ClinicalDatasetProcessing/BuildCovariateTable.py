import pandas as pd
import pickle
import numpy as np
from scipy.stats import rankdata,norm

def InverseNormalTransform(data_array,k=0.5):
    num_samps = data_array.shape[0]
    ranked_data=rankdata(data_array)
    return norm(0.0,1.0).ppf((ranked_data-k)/(num_samps-2.0*k+1.0))

num_pcs = 40

with open('UKBB_GeneticQC_Cov_Params.pth','rb') as f:
    cov_table=pickle.load(f)

with open('ukbb_encodings/coding22000.tsv','r') as f:
    batch_encoding = pd.read_csv(f,sep='\t')



batch_encoding['coding']=batch_encoding['coding'].apply(lambda x: str(x))
new_row = pd.DataFrame(dict(zip(list(batch_encoding.columns),[[x] for x in ['-12','NA','NA','NA','NA']])))
batch_encoding=pd.concat([batch_encoding,new_row],ignore_index=True)
batch_encoding.set_index('coding',drop=True,inplace=True)

with open('ukbb_encodings/coding10.tsv','r') as f:
    centre_encoding = pd.read_csv(f,sep='\t')
centre_encoding['coding']=centre_encoding['coding'].apply(lambda x: str(x))
centre_encoding.set_index('coding',drop=True,inplace=True)
centre_encoding['meaning']=centre_encoding['meaning'].apply(lambda x:x.replace(' ','_'))

final_cov_table=pd.DataFrame(cov_table[['eid']])
final_cov_table.rename(columns={'eid':'#FID'},inplace=True)
final_cov_table.set_index('#FID', inplace=True, drop=True)
final_cov_table['IID']=final_cov_table.index

final_cov_table['sex']=cov_table.loc[final_cov_table.index]['Sex (self-reported)']

age = 2020-cov_table['Birth Year'].apply(lambda x:float(x))
final_cov_table['age']= age
final_cov_table['age_normalized']=InverseNormalTransform(age)

cov_table['Genotype Array Batch']=cov_table['Genotype Array Batch'].apply(lambda x:x if x!='' else '-12')
final_cov_table['batch']=cov_table['Genotype Array Batch'].apply(lambda x: batch_encoding.loc[x]['meaning'])

final_cov_table['centre']=cov_table['Initial Assessment Center'].apply(lambda x: centre_encoding.loc[x]['meaning'])

for i in range(num_pcs):
    final_cov_table['PC{0}'.format(i+1)]=cov_table['Genetic PCs'].apply(lambda x:x[i])

with open('UKBBCovArray.txt','w') as f:
    final_cov_table.to_csv(f,sep='\t')

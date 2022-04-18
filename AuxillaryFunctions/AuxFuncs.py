import numpy as np
from scipy.stats.mstats import mquantiles
from scipy.stats import rankdata,norm,chi2
import pandas as pd


def assign_quantiles(data_frame,property,num_quantiles):
    quantile_thresh=mquantiles(data_frame[property],prob=np.linspace(0.0,1.0,num_quantiles+1))
    atypical_labels=['1st','2nd','3rd']
    labels=atypical_labels[0:min(num_quantiles,3)]+['{0:d}th'.format(i) for i in range(4,num_quantiles+1)]

    return pd.cut(data_frame[property],bins=quantile_thresh,include_lowest=True,labels=labels)

def _parseFloats(x,useMax=True):
    """
    Parses float strings. By default, if multiple floats, returns maximum. Can instead return the first.
    """
    try:
        return float(x)
    except ValueError:
        if useMax:
            return max(map(float,x.split(',')))
        else:
            list(map(float,x.split(',')))[0]

def _dateParse(x):
    if isinstance(x,str):
        if x.strip()=='1900-01-01' or x.strip()=='2037-07-07':
            return np.nan
        else:
            return float(x.split('-')[0])
    else:
        return np.nan

def InverseNormalTransform(data_array,k=0.5):
    num_samps = data_array.shape[0]
    ranked_data=rankdata(data_array)
    return norm(0.0,1.0).ppf((ranked_data-k)/(num_samps-2.0*k+1.0))

def LRTest(full_mod_ll,nested_mod_ll,param_diff):
    lrt=2.0*(full_mod_ll-nested_mod_ll)
    return lrt,chi2.sf(lrt, param_diff)

def _compute_eGFR(age,sex,SCr,black_race=False,in_micromol=True):
    if in_micromol:
        SCr*=0.0113
    if sex==1:
        k=0.9
        a=-0.411
    else:
        k=0.7
        a=-0.329
    eGFR=141.0*(min(SCr/k,1.0)**a)*(max(SCr/k, 1.0)**-1.209)*(0.993**age)*(1.018**(1.0-sex))
    if black_race:
        eGFR*=1.159
    return eGFR

def _bpParse(bp_string):
    try:
        bp_measurements=np.array(bp_string.split(','),dtype=np.float)
        return np.mean(bp_measurements)
    except AttributeError:
        return np.nan

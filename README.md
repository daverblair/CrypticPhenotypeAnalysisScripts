# CrypticPhenotypeAnalysisScripts

This directory contains the scripts that were used to perform the analyses described in Blair et al 2020 (Common Genetic Variants Associated with Mendelian Disease Morbidity Revealed Through Cryptic Phenotype Analysis). Note, these scripts are provided as is and have not been tested across multiple architectures. If software/python module installation is required, it can be done using pip/conda. Many of these scripts rely on multiple data files that cannot be directly shared due to data use agreements, so they are provided for illustrative purposes only. However, the scripts could potentially be adapted for use on customized datasets. Below, the function of each script is briefly described, separated by sub-directory.

# Sub-Directories

## Simulations

This directory contains scripts used to the simulate data that used to construct the illustrative Figure 1 and illustrative Figure 2A. It also contains the script used to conduct the simulations depicted in Figures 2B and C.

Spectrum_Outlier_Modifier_ToyModel.py: The toy model used to describe the morbidity-dependent genetic effects.

vLPI_Illustration.py: The simulation used to construct Figure 2A.

vLPI_Sim.py: Script used to generate arbitrary simulated, disease symptom datasets. Type 'python vLPI_Sim.py -h' for command line arguments.

## CrypticPhenotypeAnalysis

This directory contains the scripts used for Cryptic Phenotype Analysis (CPA). The details of this analysis are described in the Supplementary Methods. Supplementary Figure 5 depicts the overall CPA pipeline, and the steps performed by each script are provided in the file name. The final output of this analysis is depicted in Figures 2D and 2E.

CPA_vLPI_Fit_Step1.py: Fits the latent phenotype model to some observed symptom dataset. The model is saved to disk. Type 'python CPA_vLPI_Fit_Step1.py -h' for command line arguments.  

CPA_Model_Selection_Step1-2.py: Compares all the different models fit to a single disease and identifies the one with the lowest perplexity. Type 'python CPA_Model_Selection_Step1-2.py -h' for command line arguments.  

CPA_ConsistencyAnalysis_Step2.py: Performs the consistency checks for the different model inference trials. Corresponds to Step 2 of Supplementary Figure 5. Type 'python CPA_ConsistencyAnalysis_Step2.py -h' for command line arguments.

Note, there is no Step 3 script, as it corresponds to repeating Step 1 but with a different set hyper-parameters/set of manually curated symptoms.

CPA_Effective_Rank_Step4.py: Computes fraction of variance explained by each component of the top performing latent phenotype model. This information is used to estimate effective rank. Type 'python CPA_Effective_Rank_Step4.py -h' for command line arguments.

CPA_IdentifyCrypticPheno_AssessOutlier_Steps5-6.py: This script uses the output of the previous scripts to compare the outlier and spectrum models for the disease of interest. To do so, it first determines the effective rank of the top performing model using the output from the previous script. It then identifies the top performing (cryptic) phenotype for the rare disease of interest in the training dataset. Finally, it uses the testing dataset to compare the outlier and spectrum models. Type 'python CPA_IdentifyCrypticPheno_AssessOutlier_Steps5-6.py -h' for command line arguments.


## CrypticPhenotypeImputation

FitUKBBImputationModels_GBR.py: Python script used for generating the Gradient Boosted Regression models for phenotype imputation. Type 'python FitUKBBImputationModels_GBR.py -h' for command line arguments. Note, these models are included in the 'CrypticPhenoImpute' python package (see https://github.com/daverblair/CrypticPhenoImpute).

## GWAS

compute_qrank_gwas.sh: Shell script that uses the QRankGWAS python module (see https://github.com/daverblair/QRankGWAS) to perform quantile regression genome-wide association studies. Follow up analyses were conducted using FUMA (https://fuma.ctglab.nl/).

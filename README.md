# CrypticPhenotypeAnalysisScripts

This directory contains the scripts that were used to perform the analyses described in Blair et al 2022. Note, these scripts are provided as is and have not been tested across multiple architectures. If software/python module installation is required, it can be done using pip/conda. Many of these scripts rely on multiple data files that cannot be directly shared due to data use agreements, so they are provided with dummy paths in the place of such data. However, the scripts could be adapted for use on customized datasets. Below, the purpose of each script is briefly described, separated by sub-directory.

# Sub-Directories

## AuxillaryFunctions

Functions called by other scripts that are not part of a pip-installable module.

FirthRegression.py: Class that performs firth-corrected logistic regression based on the approach described in PMID: 12758140.

GWASPlots.py: Python functions for making the quantile-quantile and QQ plots used in this study (see Figures 4, 5, and 6).

AuxFuncs.py: A handful of functions used repeatedly by other scripts.

## Illustration

This directory contains script used to generate the simulated data and figure panels from Figure 1.

vLPI_Illustration.py: This script should reproduce the plots in Figures 1a and 1b.

IllustrativeExample.pth: Pickled simulated dataset for Figure 1.

## CPA

This directory contains the scripts used for Cryptic Phenotype Analysis (CPA). The details of this analysis are described in the Supplementary Methods. Supplementary Figure 5 depicts the overall CPA pipeline, and the steps performed by each script are provided in the file name. The final output of this analysis is depicted in Figures 2D and 2E.

CPA_vLPI_Fit_Step1.py: Fits the latent phenotype model to some observed symptom dataset. The model is saved to disk. Type 'python CPA_vLPI_Fit_Step1.py -h' for command line arguments.  


CPA_Model_Selection_Step2_1.py: Compares all the different models fit to a single disease and identifies the one with the lowest perplexity. This is done in the training dataset. Type 'python CPA_Model_Selection_Step1-2.py -h' for command line arguments.  

CPA_ConsistencyAnalysis_Step2_2.py: Performs the consistency checks for the different model inference trials. Corresponds to Step 2 of Supplementary Figure 5. Results of this analysis are diplayed in Supplementary Figure 3. Type 'python CPA_ConsistencyAnalysis_Step2.py -h' for command line arguments.

Note, there is no Step 3 script, as it corresponds to repeating Step 1 but with a different set of hyper-parameters/set of manually curated symptoms.

CPA_Effective_Rank_Step4.py: Computes fraction of variance explained by each component of the top performing latent phenotype model. This information is used to estimate effective rank. Type 'python CPA_Effective_Rank_Step4.py -h' for command line arguments.

CPA_IdentifyCrypticPheno_Step5.py: This script simply identifies the top performing cryptic phenotype for a given model.

CPA_Validation_Step6_UCSFModel.py:  This script computes the increase in case severity among withheld disease cases in the UCSF dataset for the UCSF model.

CPA_Validation_Step6_UKBBModel.py: This script computes the increase in case severity among withheld disease cases in the UCSF dataset for the UKBB model. When possible, it also computes this information for the UKBB model and dataset (diagnoses are not available for all diseases in the UKBB dataset).

BuildCombinedUCSF_UKBBTable.py: Simple script that concatenates summary tables for model inference into a single table. It is included for reference so that 'ModelInferenceCombinedResults.pth' can be reconstructed.

CollectResults_FilterFinalDiseases.py: This script performs the filtering describe in Blair et al. 2022, which results in the final 10 diseases that replicated in both datasets.

DatasetModelCompare.py: This script produced the analysis displayed in Figures 2d, 2e, and 2f.

## MolecularValidation

MolecularValidation.py: Script that performs the validation analysis for the cryptic phenotypes using exome sequencing data. These results are displayed in Figure 3 and Supplementary Figure 6.

## GWAS

plink_gwas.sh: shell script for per performing GWAS plink2.

calc_ldak_weightings_taggings.sh: Script used to calculate LDAK weights/taggings, which are needed for downstream analyses. See https://dougspeed.com/calculate-taggings/ for details.

ldak_sumher.sh: Script used to compute heritability estimates from GWAS summary statistics

ldak_bolt.sh: Script used to estimate the LDAK-BOLT model used for genomic predictions/PGS construction

ldak_scores.sh: Script used to impute polygenic scores into the training, validation, and target datasets. Note, only the validation/target datasets (not used for BOLT model inference) are analyzed in Figures 5/6.

GenerateQQPlots.py: Generates the 5 QQ plots shown in Figure 4

A1ATD_PostGWASAnalysis.py: Performs all PGS and P/LP-related analyses in the target/validation cohorts for A1ATD. Results are displayed in Figures 5 and Supplementary Figure 7.

AS_PostGWASAnalysis.py: Performs all PGS and P/LP-related analyses in the target/validation cohorts for AS. Results are displayed in Figures 6 and Supplementary Figure 8.


ADPKD_PostGWASAnalysis.py: Performs all PGS and P/LP-related analyses in the target/validation cohorts for ADPKD. Results are displayed in Figures 6 and Supplementary Figure 9.

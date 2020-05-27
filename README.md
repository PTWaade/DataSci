# DataSci
Repository for Data Science exam at Aarhus University, CogSci, 2020.

This README outlines the role of the different scripts of the analysis.

#### Analysis overview ####
We first ran a simulation using the MABE framework. We then calculated IIT and surprisal measures. We then made the generation scale analysis, and lastle the trial scale time series analysis. Note that running the simulation and the phi calculations takes days on normal laptops and output large amounts of data. One can therefore downloade the processed data in order to do the final analysis. These can be found on: https://clolesen.com/datascience2020.html. See below which files are used for which purposes.
- fitness_task1.csv
- fitness_task4.csv
- trans_data_task1.csv
- trans_data_task4.csv
- run_trans_data_task1.csv
- run_trans_data_task4.csv
- agent_trans_data_task1.csv
- agent_trans_data_task4.csv
- all_avg_data_task1.csv
- all_avg_data_task4.csv
- cor_data_task1.csv
- cor_data_task4.csv


#### Script for running the simulation ####
The simualtion was run in MABE using the script XXXXXXXXXXXXX. Requires MABE to be installed and has a long run time. 
### NB: This script was made by DUDE et al.
This script produced the following files
- file.csv
- file.csv


#### Scripts for extracting information and measures from the simulation ####
The information from the simultion was extracted using XXXXXXXXXXXX. This script produced the following files
- file.csv
- file.csv

This script used the following files:
- file.csv
- file.csv


#### Script for extracting surprisal measures ####
Here the different surprisal measures were calculated, using CountStates.py. We produced different files for the two tasks and depending on the surprisal calculation (relative to all states, to states within the run, or to states within the specific agent). It produced the following files:
- trans_data_task1.csv
- trans_data_task4.csv
- run_trans_data_task1.csv
- run_trans_data_task4.csv
- agent_trans_data_task1.csv
- agent_trans_data_task4.csv

From the files:
- file.csv
- file.csv

#### Generating Averaged data ####
In _____, the trial-level data was averaged to make the generation level data used later. This script produced:
- all_avg_data_task1.csv
- all_avg_data_task4.csv

From the files:
- fitness_task1.csv
- fitness_task4.csv
- trans_data_task1.csv
- trans_data_task4.csv
- run_trans_data_task1.csv
- run_trans_data_task4.csv
- agent_trans_data_task1.csv
- agent_trans_data_task4.csv


#### Scripts for analysing how the agents changed across generations ####
### These scripts were also used to replicate the findings of Albantakis et al. (2014).
In AvgAnalysis.R, the replication analysis was made on the generation level of Phi, number of concepts and fitness, and the analysis of the surprisal measures on the generation level. This script used the following files:
- all_avg_data_task1.csv
- all_avg_data_task4.csv


#### Generating cross-correlations ####
Here cross-correlations were calculated and stored in a new csv. This was done using timeseries.R. 

It produced the following files:
- cor_data_task1.csv
- cor_data_task4.csv.
- run_cordata_task1.csv
- run_cordata_task4.csv.
- agent_cordata_task1.csv
- agent_cordata_task4.csv.

Using the following files
- trans_data_task1.csv
- trans_data_task4.csv
- run_trans_data_task1.csv
- run_trans_data_task4.csv
- agent_trans_data_task1.csv
- agent_trans_data_task4.csv

#### Scripts for investigating the cross-correlation between Phi and surprisal ####
In CorAnalysis.R, the cross-correlation between Phi and surprisal conditional on the previous states were investigated. This produced the correlation strength densities across timelags and made the examples found in the Appendix.
This used the files:
- cor_data_task1.csv
- cor_data_task4.csv.
- run_cordata_task1.csv
- run_cordata_task4.csv.
- agent_cordata_task1.csv
- agent_cordata_task4.csv.




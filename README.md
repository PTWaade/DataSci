# DataSci
Repository for Data Science exam at Aarhus University, CogSci, 2020.

This README outlines the role of the different scripts of the analysis.

Note that running the simulation takes days on normal laptops and output huge amounts of data. The scripts extracting information and measures from the simulation data takes hours to run and also output big csvs. The data we extracted from the simulation can be found on: https://clolesen.com/datascience2020.html?fbclid=IwAR3yZaVcQdWyHv5dTEIyKRuT1A9hAxtEkm7Mf-FegpPkOS6Vr0sm3lQOf-g

#### List of all csv files used in the analysis ####
A lot of data was simulated and manipulated in different ways. Not all data manipulations were used in the final analyses. the following is a list of the csv files used in the final analyses. They can be found on the website above.
- file.csv
- file.csv
- file.csv
- file.csv
- file.csv
- file.csv
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


#### Scripts for analysing how the agents changed across generations ####
### These scripts were also used to replicate the findings of dude et al. (YYYY).
In AvgAnalysis.R, Phi, number of concepts and fitness was averaged across all runs to plot their change across generations. The data came from script XXXXX.py. The analysis of this script used the following files:
- file.csv
- file.csv


#### Scripts for investigating the correlation between Phi and surprisal ####
The timeseries.R script calculated the cross-correlation between Phi and different measures of surprisal. The data came from script XXX. These cross-correlations were then used in CorAnalysis.R. The analysis of this script used the following files:
- file.csv
- file.csv

In CorAnalysis.R, the cross-correlation between Phi and surprisal conditional on the previous states were plotted at different timelags. The analysis of this script uses the following files:
- cor_data_task1.csv
- cor_data_task4.csv.




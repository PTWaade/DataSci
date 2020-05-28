"""
- Import libraries
- Set up directories
- Define hyperparameters
- load data
- Set up and run loop
- Create data frame
"""
###############################
#------ Import Libraries ------
###############################

import numpy as np
import random
from ipywidgets import interact, interact_manual
import ipywidgets as widgets
from IPython import display
import time
from pathlib import Path
import pandas as pd
import os
import copy
import networkx as nx
import pickle
import sys
from scipy import stats


#####################
#------ PATH--------
####################

path = '/Users/christoffer/Documents/CogSci/MA/DataSci/Exam_script/' #Change to appropriate path

#################################
#------ Set Up Directories ------
#################################

# Change directory in order to load actual agency
os.chdir("..")
actual_agency_path = path + "actual_agency"
sys.path.append(actual_agency_path)

from pyanimats import *
from pyTPM import *
import actual_agency as agency
import pyphi
from pyphi import actual, config, Direction

#------------------------
#--- Configure pyphi ----
# ------------------------

# Don't check if the system can be in the current state. Justified, as system was observed in current state
pyphi.config.VALIDATE_SUBSYSTEM_STATES = False
# Only considers bipartitions on the level of mechabisms (small phi). MIPs of bipartition has high correlation with MIPs of all possible partitions, R = 0.921
# From Evaluating Approximations and Heuristic Measures of Integrated Information
pyphi.config.PARTITION_TYPE = 'BI'
# Assumes that the MIP is always found by cutting only 1 node. On the level of system (Big Phi). Greatly reduces computation, correlates well with true phi
# From Evaluating Approximations and Heuristic Measures of Integrated Information
pyphi.config.CUT_ONE_APPROXIMATION = True 

# Turn progress bares off
pyphi.config.PROGRESS_BARS = False

# Change directory back to Example directory
os.chdir(path) 

#%load_ext autoreload
#%autoreload 2
#%aimport pyTPM, pyanimats, actual_agency

#############################################
#------ Hyperparameters for Simulation ------
#############################################
# defining parameters to use 
generations = 30000
n_agents = int(generations/500.+1)
n_runs = 2
n_trials = 128
n_timeSteps = 35


#################################
#------ Loading MABE Data ------
#################################

# LOD data
with open('MABE_data/LOD_data.pkl','rb') as f:
    LOD_data = pickle.load(f)

# Activity data
with open('MABE_data/activity.pkl','rb') as f:
    activity = pickle.load(f)

# Genome data
with open('MABE_data/genome.pkl','rb') as f:
    all_genomes = pickle.load(f)


######################
#------ FITNESS ------
######################

# Calculate fitness
for n in range(n_runs):
    LOD_data[n]['fitness'] = (LOD_data[n]['correct_AVE'] 
                                /(LOD_data[n]['correct_AVE']+LOD_data[n]['incorrect_AVE']))

# Make CSV with fitness data
fit_list = []
run_list = []
agent_list = []
for run in range(n_runs):
    for agent in range(n_agents): #len(LOD_data[n]['fitness'][:])
        fit_list.append(LOD_data[n]['fitness'][agent])
        agent_list.append(agent)
        run_list.append(run)

df = pd.DataFrame({
    'run': run_list,
    'agent': agent_list,
    'fitness': fit_list
    })
df.to_csv('results_data/fitness.csv', sep=',')


###########################
#------ IIT ANALYSIS ------
###########################

# Extract brain activity data from pickle
brain_activity = []
for r in range(n_runs):
    print(f"Run: {r}")
    brain_activity.append(agency.getBrainActivity(activity[r], n_agents, n_trials, world_height=34))


#Loop through run, agents, trials and time steps. Store data in .csv each run.
for run in range(n_runs): #Loop through runs

    # Define empty lists for data
    Phi_list = []
    n_concepts_list = []
    concept_phis_list = []
    run_list = []
    agent_list = []
    trial_list = []
    timeStep_list = []
    S1_list = []
    S2_list = []
    M1_list = []
    M2_list = []
    H1_list = []
    H2_list = []
    H3_list = []
    H4_list = []


    for agent in range(n_agents): #For each run loop through agents
        
        print(f"Run: {run} - Agent: {agent}")

        #Create an empty dictionary to store agent states in
        #The same system state always has the same Phi, number of concepts and concept phis
        #Thus there is no reason to calculate phi multiple times, once per state is sufficient
        state_dict = {}

        #Define the genome of agent i on run j
        #The genome contains all information on the agent, e.g. what kind of network it is
        genome = agency.get_genome(all_genomes, run, agent)

        #Define agent i's:
        # - transition probability matrix (TPM)
        # - the types of gates in the TPM, here they're deterministic
        # - connectivity matrix (CM)
        TPM, TPM_gates, cm = genome2TPM(genome, n_nodes=8, n_sensors=2, n_motors=2, 
            gate_type='deterministic',states_convention='loli',
            remove_sensor_motor_effects=True)

        #Define the network from the TPM and CM
        network = pyphi.network.Network(np.array(TPM), cm=np.array(cm), 
            node_labels=('S1','S2','M1','M2','H1','H2','H3','H4'), purview_cache=None)

        for trial in range(n_trials): #For each agent loop through its trials
            #Loop through timesteps of trials.
            # - First trial is discarded as all nodes are initilized as 0, which is not always a reachable system state
            # - Last trial is discarded as two node states in the simulatein are defined as -1, signaling the trial has ended.
            for timeStep in range(1,n_timeSteps-1):
                
                # Find states of the nodes in the network
                state = tuple(np.array(brain_activity[run][agent][trial][timeStep]).astype(int))

                #If the state is not in the state dictionary, create the state's entry
                if state not in state_dict.keys():

                    #Find major complex, i.e. the complex with the highest Phi. This complex has potential for consciousness
                    major_complex = pyphi.compute.network.major_complex(network,state)

                    # NUMBER OF CONCEPTES
                    n_concepts = len(major_complex.ces) #Calculate value
                    n_concepts_list.append(n_concepts) #Append to list
                    
                    
                    # phi of concepts
                    concept_phis = major_complex.ces.phis #Calculate values
                    concept_phis = '-'.join([str(elem) for elem in concept_phis]) #Make values into one string
                    concept_phis_list.append(concept_phis) #Append to list
                    
                    # Phi of complex
                    Phi = major_complex.phi #Calculate value
                    Phi_list.append(Phi) #Append to list

                    # Create and entry in the state dictionary
                    state_dict[state] = [n_concepts, concept_phis, Phi]
                
                #If the state is in the state dictionary, get values by indexing
                else:
                    values = state_dict.get(state)
                    n_concepts_list.append(values[0])
                    concept_phis_list.append(values[1])
                    Phi_list.append(values[2])
                    
                # Append loop indexes, i.e. runs agents, trials and timesteps
                run_list.append(run)
                agent_list.append(agent)
                trial_list.append(trial)
                timeStep_list.append(timeStep)

                # Append node states. S is sensory, M is motor and H is hidden
                S1_list.append(state[0])
                S2_list.append(state[1])                    
                M1_list.append(state[2])
                M2_list.append(state[3])
                H1_list.append(state[4])
                H2_list.append(state[5])
                H3_list.append(state[6])
                H4_list.append(state[7])
            # end time step loop
        # end trial loop
    # end agent loop

    # create data frame from lists                
    data = pd.DataFrame({
    'run': run_list,
    'agent': agent_list,
    'trial': trial_list,
    'timeStep': timeStep_list,
    'Phi': Phi_list,
    'n_concepts': n_concepts_list,
    'concept_phis': concept_phis_list,
    'S1': S1_list,
    'S2': S2_list,
    'M1': M1_list,
    'M2': M2_list,
    'H1': H1_list,
    'H2': H2_list,
    'H3': H3_list,
    'H4': H4_list
    })

    #Write out above data frame as a .csv
    data.to_csv('results_data/activity_results_data_run' + str(run) + '.csv', sep=',')

# end runs loop

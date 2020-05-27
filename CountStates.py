### Preparation ###
import collections as col
import numpy as np
import csv

# Set which file to read
in_filename = "activity_results_data_ALL_task5.csv"

# Set the start token
start_tok = "Init"

# Decide whether to count states seperated by run and by agent
#sep = "RunAgent"
#sep = "Run"
sep = "None"

### Count states ###
# Make empty counters for counting states
statefreq_lone = col.Counter()
statefreq_cond = col.Counter()
statefreq_cond2 = col.Counter()
blnkt_statefreq_lone = col.Counter()
blnkt_statefreq_cond = col.Counter()
blnkt_statefreq_cond2 = col.Counter()
int_statefreq_lone = col.Counter()
int_statefreq_cond = col.Counter()
int_statefreq_cond2 = col.Counter()

# Open csv file
with open(in_filename) as csvfile:
    stateslist = csv.reader(csvfile)
    # Skip the header
    next(stateslist, None)

    # The previous two states is the init token on the first trial
    prev_state = start_tok
    prev_blnkt_state = start_tok
    prev_int_state = start_tok

    prev2_state = start_tok
    prev2_blnkt_state = start_tok
    prev2_int_state = start_tok

    # Go through each row
    for row in stateslist:
        #Get the run and agent
        run = row[2]
        agent = row[3]
        
        # Create the prefix used here
        if sep == "RunAgent":
            prefix = "run" + run + "_agent" + agent + ": "
        elif sep == "Run":
            prefix = "run" + run + ": "
        else:
            prefix = ""
        
        # Get the state from the list of integers
        state = "".join([str(integer) for integer in row[-8:]])
        blnkt_state = "".join([str(integer) for integer in row[-8:-4]])
        int_state = "".join([str(integer) for integer in row[-4:]])

        # Combine previous and current state
        state_cond = prev_state + " -> " + state
        blnkt_state_cond = prev_blnkt_state + " -> " + blnkt_state
        int_state_cond = prev_int_state + " -> " + int_state

        state_cond2 = prev2_state + " -> " + prev_state + " -> " + state
        blnkt_state_cond2 = prev2_blnkt_state + " -> " + prev_blnkt_state + " -> " + blnkt_state
        int_state_cond2 = prev2_int_state + " -> " + prev_int_state + " -> " + int_state

        # Count the state
        statefreq_lone[prefix + state] += 1
        statefreq_cond[prefix + state_cond] += 1
        statefreq_cond2[prefix + state_cond2] += 1
        blnkt_statefreq_lone[prefix + blnkt_state] += 1
        blnkt_statefreq_cond[prefix + blnkt_state_cond] += 1
        blnkt_statefreq_cond2[prefix + blnkt_state_cond2] += 1
        int_statefreq_lone[prefix + int_state] += 1
        int_statefreq_cond[prefix + int_state_cond] += 1
        int_statefreq_cond2[prefix + int_state_cond2] += 1

        # If it's the last timepoint of a trial
        if row[4] == '33':
            # Set the previous trial for the next round to be the init token
            prev_state = start_tok
            prev_blnkt_state = start_tok
            prev_int_state = start_tok

            prev2_state = start_tok
            prev2_blnkt_state = start_tok
            prev2_int_state = start_tok
        # Otherwise
        else:
            # Set the current state to be the previous state on the next round
            prev2_state = prev_state
            prev2_blnkt_state = prev_blnkt_state
            prev2_int_state = prev_int_state
            
            prev_state = state
            prev_blnkt_state = blnkt_state
            prev_int_state = int_state



### Calculate Surprisal of States ###
# Get the total number of states
if sep == "RunAgent":
    #The number of states in a given agent in a given run
    states_nr = 33*128
elif sep == "Run":
    #The number of states in a given run
    states_nr = 33*128*121
else:
    #All states
    states_nr = 33*128*121*50

# Set up empty dictionaries for storing surprise sizes
statesurprise_lone = {}
statesurprise_cond = {}
statesurprise_cond2 = {}
blnkt_statesurprise_lone = {}
blnkt_statesurprise_cond = {}
blnkt_statesurprise_cond2 = {}
int_statesurprise_lone = {}
int_statesurprise_cond = {}
int_statesurprise_cond2 = {}

# Go through each state
for key, freq in statefreq_lone.most_common():
    # And calculate the surprisal -log(Prob(state))
    statesurprise_lone[key] = round(- np.log(freq/states_nr), ndigits=5)

# Go through each state
for key, freq in statefreq_cond.most_common():
    # And calculate the surprisal -log(Prob(state))
    statesurprise_cond[key] = round(- np.log(freq/states_nr), ndigits=5)

# Go through each state
for key, freq in statefreq_cond2.most_common():
    # And calculate the surprisal -log(Prob(state))
    statesurprise_cond2[key] = round(- np.log(freq/states_nr), ndigits=5)

# Go through each state
for key, freq in blnkt_statefreq_lone.most_common():
    # And calculate the surprisal -log(Prob(state))
    blnkt_statesurprise_lone[key] = round(- np.log(freq/states_nr), ndigits=5)

# Go through each state
for key, freq in blnkt_statefreq_cond.most_common():
    # And calculate the surprisal -log(Prob(state))
    blnkt_statesurprise_cond[key] = round(- np.log(freq/states_nr), ndigits=5)

# Go through each state
for key, freq in blnkt_statefreq_cond2.most_common():
    # And calculate the surprisal -log(Prob(state))
    blnkt_statesurprise_cond2[key] = round(- np.log(freq/states_nr), ndigits=5)

# Go through each state
for key, freq in int_statefreq_lone.most_common():
    # And calculate the surprisal -log(Prob(state))
    int_statesurprise_lone[key] = round(- np.log(freq/states_nr), ndigits=5)

# Go through each state
for key, freq in int_statefreq_cond.most_common():
    # And calculate the surprisal -log(Prob(state))
    int_statesurprise_cond[key] = round(- np.log(freq/states_nr), ndigits=5)

# Go through each state
for key, freq in int_statefreq_cond2.most_common():
    # And calculate the surprisal -log(Prob(state))
    int_statesurprise_cond2[key] = round(- np.log(freq/states_nr), ndigits=5)



### Add surprisal to dataset ###
# Open the input_file in read mode and output_file in write mode
with open(in_filename, 'r') as read_obj, \
        open('NEW_' + in_filename, 'w', newline='') as write_obj:
        
    # Create a csv.reader object from the input file object
    data_in = csv.reader(read_obj)
    # Create a csv.writer object from the output file object
    data_out = csv.writer(write_obj)

    # Get the header of the input dataframe
    header = next(data_in)
    # Add two more column names to the header
    header.append(prefix + "surprise_lone")
    header.append(prefix + "surprise_cond")
    header.append(prefix + "surprise_cond2")
    header.append(prefix + "blnkt_surprise_lone")
    header.append(prefix + "blnkt_surprise_cond")
    header.append(prefix + "blnkt_surprise_cond2")
    header.append(prefix + "int_surprise_lone")
    header.append(prefix + "int_surprise_cond")
    header.append(prefix + "int_surprise_cond2")

    # And put it in the output file
    data_out.writerow(header)

    # The previous state is the init token on the first trial
    prev_state = start_tok
    prev_blnkt_state = start_tok
    prev_int_state = start_tok

    prev2_state = start_tok
    prev2_blnkt_state = start_tok
    prev2_int_state = start_tok

    # Read each row of the input csv file as list
    for row in data_in:
        #Get the run and agent
        run = row[2]
        agent = row[3]
        
        # Create the prefix used here
        if sep == "RunAgent":
            prefix = "run" + run + "_agent" + agent + ": "
        elif sep == "Run":
            prefix = "run" + run + ": "
        else:
            prefix = ""

        # Get the input state from the list of integers
        state = "".join([str(integer) for integer in row[-8:]])
        blnkt_state = "".join([str(integer) for integer in row[-8:-4]])
        int_state = "".join([str(integer) for integer in row[-4:]])

        # Combine previous and current state
        state_cond = prev_state + " -> " + state
        blnkt_state_cond = prev_blnkt_state + " -> " + blnkt_state
        int_state_cond = prev_int_state + " -> " + int_state

        state_cond2 = prev2_state + " -> " + prev_state + " -> " + state
        blnkt_state_cond2 = prev2_blnkt_state + " -> " + prev_blnkt_state + " -> " + blnkt_state
        int_state_cond2 = prev2_int_state + " -> " + prev_int_state + " -> " + int_state

        # Get surprises from dictionaries
        surprise_lone = statesurprise_lone[prefix + state]
        surprise_cond = statesurprise_cond[prefix + state_cond]
        surprise_cond2 = statesurprise_cond2[prefix + state_cond2]
        blnkt_surprise_lone = blnkt_statesurprise_lone[prefix + blnkt_state]
        blnkt_surprise_cond = blnkt_statesurprise_cond[prefix + blnkt_state_cond]
        blnkt_surprise_cond2 = blnkt_statesurprise_cond2[prefix + blnkt_state_cond2]
        int_surprise_lone = int_statesurprise_lone[prefix + int_state]
        int_surprise_cond = int_statesurprise_cond[prefix + int_state_cond]
        int_surprise_cond2 = int_statesurprise_cond2[prefix + int_state_cond2]

        # Append the surprises to the row
        row.append(surprise_lone)
        row.append(surprise_cond)
        row.append(surprise_cond2)
        row.append(blnkt_surprise_lone)
        row.append(blnkt_surprise_cond)
        row.append(blnkt_surprise_cond2)
        row.append(int_surprise_lone)
        row.append(int_surprise_cond)
        row.append(int_surprise_cond2)
        
        # Add the updated row to the output file
        data_out.writerow(row)

        # If it's the last timepoint of a trial
        if row[4] == '33':
            # Set the previous trial for the next round to be the init token
            prev_state = start_tok
            prev_blnkt_state = start_tok
            prev_int_state = start_tok

            prev2_state = start_tok
            prev2_blnkt_state = start_tok
            prev2_int_state = start_tok

        # Otherwise
        else:
            # Set the current state to be the previous state on the next round
            prev2_state = prev_state
            prev2_blnkt_state = prev_blnkt_state
            prev2_int_state = prev_int_state
            
            prev_state = state
            prev_blnkt_state = blnkt_state
            prev_int_state = int_state

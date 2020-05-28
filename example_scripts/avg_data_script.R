library(data.table)
library(rstudioapi)
#Set WD to the folder containing the script
setwd(setwd(dirname(getActiveDocumentContext()$path)))

# Combining the files from the IIT analysis
files = list.files(pattern = "activity_results_data*", path = "results_data", full.names = T)
data = do.call(rbind, lapply(files, function(x) fread(x)))
data = data[,2:16]

#Write the full data frame for surprisal analysis
write.csv(data, file = "trans_data.csv")

# Import function that averrages that data and outputs a data frame with these
source("avg_function.R")

# Make average data with surprisal relative to all LODs
avg_none = make_avg_data("None")
write.csv(avg_none, file = "results_data/none_average_data.csv")

# Make average data with surprisal relative to each LOD
avg_run = make_avg_data("Run")
write.csv(avg_run, file = "results_data/run_average_data.csv")

# Make average data with surprisal relative to the specific animat
avg_agent = make_avg_data("Agent")
write.csv(avg_agent, file = "results_data/agent_average_data.csv")

# When all data are saved seperatly - join into one.
# the first two
data = fread("results_data/none_average_data.csv")[,2:27]
run_data = fread("results_data/run_average_data.csv")[,2:21]
new_data = merge(data, run_data, by = c("run", "agent"))

#remove data for freeing up memory
rm(data)
rm(run_data)

# Adding the last data
agent_data = fread("results_data/agent_average_data.csv")[,2:21]
new_data = merge(new_data, agent_data, by = c("run", "agent"))

# Write csv with all average data
write.csv(new_data, file = "results_data/all_avg_data.csv")

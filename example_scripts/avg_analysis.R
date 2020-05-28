library(data.table)
library(tidyverse)
library(ggpubr)
library(rstudioapi)
#Set WD to folder containing scripts
setwd(setwd(paste0(dirname(getActiveDocumentContext()$path), "/results_data")))

#### IMPORT DATA AND OTHER VARIABLES ####
avg_data = fread("all_avg_data.csv")
fit_data = fread("fitness.csv")

colors = scale_color_manual(values=c("black", "#3498DB", "#E53935", "#27AE60"))
fills = scale_fill_manual(values=c("black", "#3498DB", "#E53935", "#27AE60"))


#### AVERAGING FUNCTIONS ####
#Make function for averaging over the last 6 timepoints
avrgTimePoints <- function(avg_dataX, ntimepoints = 6){
  #Make empty dataframe for filling in the new values
  avg_dataX_new <- avg_dataX[FALSE,]
  #Go through each of the sub-dataframes
  for (df_sub in split(avg_dataX, avg_dataX$run)) {
    df_sub <- as.data.frame(df_sub)
    #Make a new empty version of it for filling in datapoints
    df_sub_new <- df_sub[FALSE,]
    #Go through each row and all columns except the last 2 (task and generation)
    for (r in 1:nrow(df_sub)){
      for (c in 4:(ncol(df_sub))){
        #For those datapoints that have less that 6 
        if (r>ntimepoints){
          #Average from beginning
          df_sub_new[r,c] <- mean(df_sub[(r-ntimepoints):r,c])
        #From others
        } else {
          #Average over last 6 datapoints
          df_sub_new[r,c] <- mean(df_sub[0:r,c])
        }
        #Also save the task and generation data
        df_sub_new[r, 1:3] <- df_sub[r, 1:3]
      }
    }
    #Add the sub-dataframe to the output
    avg_dataX_new <- rbind(avg_dataX_new, df_sub_new)
  }
  return(avg_dataX_new)
}

#A function for averaging over runs and getting summary statistics. It ends with adding to the inputted avg_LOD.
avrgRuns <- function(avg_LOD, avg_dataX, fit_dataX, task = "Not Set") {
  #Go through each agent one by one
  for (a in unique(avg_dataX$agent)){
    #Make a subset with data for that agent across all runs
    d = subset(avg_dataX, agent == a)
    
    #Summarize fitness measures
    fit = mean(fit_dataX[fit_dataX$agent==a,]$fitness)
    fit_se = sd(fit_dataX[fit_dataX$agent==a,]$fitness)/sqrt(length(d$agent))
    
    #Summarize IIT measures
    phi = mean(d$Phi_mean)
    phi_se = sd(d$Phi_mean)/sqrt(length(d$agent))
    phi_max = mean(d$Phi_max)
    phi_max_se = sd(d$Phi_max)/sqrt(length(d$agent))
    
    nCon = mean(d$nCon_mean)
    nCon_se = sd(d$nCon_mean)/sqrt(length(d$agent))
    nCon_max = mean(d$nCon_max)
    nCon_max_se = sd(d$nCon_max)/sqrt(length(d$agent))
    
    #Summarise surprisal measures
    sur = mean(d$surprise_cond_mean)
    sur_se = sd(d$surprise_cond_mean)/sqrt(length(d$agent))
    sur2 = mean(d$surprise_cond2_mean)
    sur2_se = sd(d$surprise_cond2_mean)/sqrt(length(d$agent))
    int_sur = mean(d$int_surprise_cond_mean)
    int_sur_se = sd(d$int_surprise_cond_mean)/sqrt(length(d$agent))
    int_sur2 = mean(d$int_surprise_cond2_mean)
    int_sur2_se = sd(d$int_surprise_cond2_mean)/sqrt(length(d$agent))
    blnkt_sur = mean(d$blnkt_surprise_cond_mean)
    blnkt_sur_se = sd(d$blnkt_surprise_cond_mean)/sqrt(length(d$agent))
    blnkt_sur2 = mean(d$blnkt_surprise_cond2_mean)
    blnkt_sur2_se = sd(d$blnkt_surprise_cond2_mean)/sqrt(length(d$agent))
    
    Runsur = mean(d$Runsurprise_cond_mean)
    Runsur_se = sd(d$Runsurprise_cond_mean)/sqrt(length(d$agent))
    Runsur2 = mean(d$Runsurprise_cond2_mean)
    Runsur2_se = sd(d$Runsurprise_cond2_mean)/sqrt(length(d$agent))
    Runint_sur = mean(d$Runint_surprise_cond_mean)
    Runint_sur_se = sd(d$Runint_surprise_cond_mean)/sqrt(length(d$agent))
    Runint_sur2 = mean(d$Runint_surprise_cond2_mean)
    Runint_sur2_se = sd(d$Runint_surprise_cond2_mean)/sqrt(length(d$agent))
    Runblnkt_sur = mean(d$Runblnkt_surprise_cond_mean)
    Runblnkt_sur_se = sd(d$Runblnkt_surprise_cond_mean)/sqrt(length(d$agent))
    Runblnkt_sur2 = mean(d$Runblnkt_surprise_cond2_mean)
    Runblnkt_sur2_se = sd(d$Runblnkt_surprise_cond2_mean)/sqrt(length(d$agent))
    
    RunAgentsur = mean(d$RunAgentsurprise_cond_mean)
    RunAgentsur_se = sd(d$RunAgentsurprise_cond_mean)/sqrt(length(d$agent))
    RunAgentsur2 = mean(d$RunAgentsurprise_cond2_mean)
    RunAgentsur2_se = sd(d$RunAgentsurprise_cond2_mean)/sqrt(length(d$agent))
    RunAgentint_sur = mean(d$RunAgentint_surprise_cond_mean)
    RunAgentint_sur_se = sd(d$RunAgentint_surprise_cond_mean)/sqrt(length(d$agent))
    RunAgentint_sur2 = mean(d$RunAgentint_surprise_cond2_mean)
    RunAgentint_sur2_se = sd(d$RunAgentint_surprise_cond2_mean)/sqrt(length(d$agent))
    RunAgentblnkt_sur = mean(d$RunAgentblnkt_surprise_cond_mean)
    RunAgentblnkt_sur_se = sd(d$RunAgentblnkt_surprise_cond_mean)/sqrt(length(d$agent))
    RunAgentblnkt_sur2 = mean(d$RunAgentblnkt_surprise_cond2_mean)
    RunAgentblnkt_sur2_se = sd(d$RunAgentblnkt_surprise_cond2_mean)/sqrt(length(d$agent))
    
    #Add it all to the new dataframe
    avg_LOD = rbind(avg_LOD, data.frame(Fitness = fit, Fitness_se = fit_se,
                                        Phi=phi, Phi_se = phi_se, 
                                        Phi_max=phi_max, Phi_max_se=phi_max_se,
                                        nCon=nCon, nCon_se = nCon_se,
                                        nCon_max=nCon_max, nCon_max_se=nCon_max_se,
                                        
                                        Surprise=sur, Suprise_se=sur_se,
                                        Surprise2=sur2, Suprise2_se=sur2_se,
                                        int_Surprise=int_sur, int_Suprise_se=int_sur_se,
                                        int_Surprise2=int_sur2, int_Suprise2_se=int_sur2_se,
                                        blnkt_Surprise=blnkt_sur, blnkt_Suprise_se=blnkt_sur_se,
                                        blnkt_Surprise2=blnkt_sur2, blnkt_Suprise2_se=blnkt_sur2_se,
                                        
                                        Run_Surprise=Runsur, Run_Suprise_se=Runsur_se,
                                        Run_Surprise2=Runsur2, Run_Suprise2_se=Runsur2_se,
                                        Run_int_Surprise=Runint_sur, Run_int_Suprise_se=Runint_sur_se,
                                        Run_int_Surprise2=Runint_sur2, Run_int_Suprise2_se=Runint_sur2_se,
                                        Run_blnkt_Surprise=Runblnkt_sur, Run_blnkt_Suprise_se=Runblnkt_sur_se,
                                        Run_blnkt_Surprise2=Runblnkt_sur2, Run_blnkt_Suprise2_se=Runblnkt_sur2_se,
                                        
                                        Agent_Surprise=RunAgentsur, Agent_Suprise_se=RunAgentsur_se,
                                        Agent_Surprise2=RunAgentsur2, Agent_Suprise2_se=RunAgentsur2_se,
                                        Agent_int_Surprise=RunAgentint_sur, Agent_int_Suprise_se=RunAgentint_sur_se,
                                        Agent_int_Surprise2=RunAgentint_sur2, Agent_int_Suprise2_se=RunAgentint_sur2_se,
                                        Agent_blnkt_Surprise=RunAgentblnkt_sur, Agent_blnkt_Suprise_se=RunAgentblnkt_sur_se,
                                        Agent_blnkt_Surprise2=RunAgentblnkt_sur2, Agent_blnkt_Suprise2_se=RunAgentblnkt_sur2_se,
                                        
                                        Task=task, Generation = a))
  }
  return(avg_LOD)
}

#### PLOT DATA ####
#Do averaging on datasets
avg_data <- avrgTimePoints(avg_data,ntimepoints = 6)

#Make dataframe with the final averages and summary statistics for all tasks etc
avg_LOD <- data.frame()
avg_LOD <- avrgRuns(avg_LOD, avg_data, fit_data, task = "Example task")


#### IIT PLOTS ####
#Phi plot
avg_phi_plot = ggplot(avg_LOD, aes(x = Generation, y = Phi)) +
  geom_ribbon(aes(ymin = Phi - Phi_se, ymax = Phi + Phi_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  theme_classic() +
  theme(legend.position = "none") +
  labs(y = "Average Phi", title = " ") +
  geom_vline(xintercept = 45, color = "darkgray", linetype="dashed") +
  geom_vline(xintercept = 81, color = "darkgray", linetype="dashed") +
  colors + fills

#Concept plot
avg_con_plot = ggplot(avg_LOD, aes(x = Generation, y = nCon)) +
  geom_ribbon(aes(ymin = nCon - nCon_se, ymax = nCon + nCon_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  theme_classic() +
  theme(legend.position = "none") +
  labs(y = "Average number of concepts", title = " ") +
  geom_vline(xintercept = 45, color = "darkgray", linetype="dashed") +
  geom_vline(xintercept = 81, color = "darkgray", linetype="dashed") +
  colors + fills

# Evolution of phi for all animats seperatly 
# Task 1
all_agents_plot = ggplot(avg_data, aes(x = agent, y = Phi_mean))+
  geom_line() +
  theme_minimal() +
  facet_wrap(~run, ncol = 5) +
  labs(y = "Average Phi", x = "Generation", title = "Task 1: Average Phi over time for all agents seperatly")




#### SURPRISAL PLOTS ####

# Relative to all animats on all LOD
# System state
avg_sur_plot = ggplot(avg_LOD, aes(x = Generation, y = Surprise)) +
  geom_ribbon(aes(ymin = Surprise - Suprise_se, ymax = Surprise + Suprise_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  theme_classic()+
  theme(legend.position = "none") +
  labs(y = "Surprisal - All runs", title = "Full system state",  x = "") +
  geom_vline(xintercept = 45, color = "darkgray", linetype="dashed") +
  geom_vline(xintercept = 81, color = "darkgray", linetype="dashed") +
  colors + fills

# Blanket state
avg_blnkt_sur_plot = ggplot(avg_LOD, aes(x = Generation, y = blnkt_Surprise)) +
  geom_ribbon(aes(ymin = blnkt_Surprise - blnkt_Suprise_se, ymax = blnkt_Surprise + blnkt_Suprise_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  theme_classic()+
  theme(legend.position = "none") +
  labs(y = "",title = "Blanket states",  x = "") +
  geom_vline(xintercept = 45, color = "darkgray", linetype="dashed") +
  geom_vline(xintercept = 81, color = "darkgray", linetype="dashed") +
  colors + fills


# Internal states
avg_int_sur_plot = ggplot(avg_LOD, aes(x = Generation, y = int_Surprise)) +
  geom_ribbon(aes(ymin = int_Surprise - int_Suprise_se, ymax = int_Surprise + int_Suprise_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  theme_classic()+
  theme(legend.position = "none") +
  labs(y = "", title = "Internal states", x = "") +
  geom_vline(xintercept = 45, color = "darkgray", linetype="dashed") +
  geom_vline(xintercept = 81, color = "darkgray", linetype="dashed") +
  colors + fills



# Relative to all animats on each LOD
# Systems states
Run_avg_sur_plot = ggplot(avg_LOD, aes(x = Generation, y = Run_Surprise)) +
  geom_ribbon(aes(ymin = Run_Surprise - Run_Suprise_se, ymax = Run_Surprise + Run_Suprise_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  theme_classic()+
  theme(legend.position = "none") +
  labs(y = "Surprisal - LOD", x = "", title = " ") +
  geom_vline(xintercept = 45, color = "darkgray", linetype="dashed") +
  geom_vline(xintercept = 81, color = "darkgray", linetype="dashed") +
  colors + fills

# Blanket states
Run_avg_blnkt_sur_plot = ggplot(avg_LOD, aes(x = Generation, y = Run_blnkt_Surprise)) +
  geom_ribbon(aes(ymin = Run_blnkt_Surprise - Run_blnkt_Suprise_se, ymax = Run_blnkt_Surprise + Run_blnkt_Suprise_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  theme_classic()+
  theme(legend.position = "none") +
  labs(y = "", x = "", title = " ") +
  geom_vline(xintercept = 45, color = "darkgray", linetype="dashed") +
  geom_vline(xintercept = 81, color = "darkgray", linetype="dashed") +
  colors + fills

# Internal states
Run_avg_int_sur_plot = ggplot(avg_LOD, aes(x = Generation, y = Run_int_Surprise)) +
  geom_ribbon(aes(ymin = Run_int_Surprise - Run_int_Suprise_se, ymax = Run_int_Surprise + Run_int_Suprise_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  theme_classic()+
  theme(legend.position = "none") +
  labs(y = "", x = "", title = " ") +
  geom_vline(xintercept = 45, color = "darkgray", linetype="dashed") +
  geom_vline(xintercept = 81, color = "darkgray", linetype="dashed") +
  colors + fills


# Relative to the specific animat
# System states
Agent_avg_sur_plot = ggplot(avg_LOD, aes(x = Generation, y = Agent_Surprise)) +
  geom_ribbon(aes(ymin = Agent_Surprise - Agent_Suprise_se, ymax = Agent_Surprise + Agent_Suprise_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  theme_classic()+
  theme(legend.position = "none") +
  labs(y = "Surprisal - Animat", title = " ") +
  geom_vline(xintercept = 45, color = "darkgray", linetype="dashed") +
  geom_vline(xintercept = 81, color = "darkgray", linetype="dashed") +
  colors + fills

# Blanket states
Agent_avg_blnkt_sur_plot = ggplot(avg_LOD, aes(x = Generation, y = Agent_blnkt_Surprise)) +
  geom_ribbon(aes(ymin = Agent_blnkt_Surprise - Agent_blnkt_Suprise_se, ymax = Agent_blnkt_Surprise + Agent_blnkt_Suprise_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  theme_classic()+
  theme(legend.position = "none") +
  labs(y = "", title = " ") +
  geom_vline(xintercept = 45, color = "darkgray", linetype="dashed") +
  geom_vline(xintercept = 81, color = "darkgray", linetype="dashed") +
  colors + fills

# Internal states
Agent_avg_int_sur_plot = ggplot(avg_LOD, aes(x = Generation, y = Agent_int_Surprise)) +
  geom_ribbon(aes(ymin = Agent_int_Surprise - Agent_int_Suprise_se, ymax = Agent_int_Surprise + Agent_int_Suprise_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  theme_classic()+
  theme(legend.position = "none") +
  labs(y = "", title = " ") +
  geom_vline(xintercept = 45, color = "darkgray", linetype="dashed") +
  geom_vline(xintercept = 81, color = "darkgray", linetype="dashed") +
  colors + fills


#### FITNESS PLOT ####

fitness_plot = ggplot(avg_LOD, aes(x = Generation, y = Fitness)) +
  geom_ribbon(aes(ymin = Fitness - Fitness_se, ymax = Fitness + Fitness_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  theme_classic() +
  theme(legend.position = "none") +
  labs(y = "Fitness", title = " ") +
  geom_vline(xintercept = 45, color = "darkgray", linetype="dashed") +
  geom_vline(xintercept = 81, color = "darkgray", linetype="dashed") +
  colors + fills


#### SAVING FINAL PLOTS ####

# ALL Surpisal plots
ggsave(
  "surprisal_plot.jpg",
  ggarrange(
  avg_sur_plot,
  avg_blnkt_sur_plot,
  avg_int_sur_plot,
  
  Run_avg_sur_plot,
  Run_avg_blnkt_sur_plot,
  Run_avg_int_sur_plot,
  
  Agent_avg_sur_plot,
  Agent_avg_blnkt_sur_plot,
  Agent_avg_int_sur_plot,
  
  labels = "auto",
  common.legend = T, align = "v"
  ), width = 7.5, height = 6
)


ggsave(
  "replicate_plot.jpg",
  ggarrange(
  avg_phi_plot,
  avg_con_plot,
  fitness_plot,
  labels = "auto",
  ncol = 3, common.legend = T
), width = 7.5, height = 3
)

ggsave(
  "all_agents_plot.jpg",
  all_agents_plot, width = 7.5, height = 14
)


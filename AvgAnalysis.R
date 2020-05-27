library(data.table)
library(gganimate)
library(tidyverse)
library(ggpubr)


#### IMPORT DATA AND OTHER VARIABLES####
# trans_data1 = fread("trans_data_task1.csv")
# trans_data4 = fread("trans_data_task4.csv")

# cor_data1 = fread("cor_data_task1.csv", stringsAsFactors = F)
# cor_data4 = fread("cor_data_task4.csv", stringsAsFactors = F)

avg_data1 = fread("avg_data_task1.csv")
avg_data4 = fread("avg_data_task4.csv")

fit_data1 = fread("fitness_task1.csv")
fit_data4 = fread("fitness_task4.csv")

colors = scale_color_manual(values=c("#E69F00", "#009e73", "#CC79A7", "#0072B2"))
fills = scale_fill_manual(values=c("#E69F00", "#009e73", "#CC79A7", "#0072B2"))

#### GENERATION SCALE ANALYSIS ####

#seven fittest in task 4
tail(sort(subset(fit_data4, agent==120)$fitness),7)
best_runs6 = subset(fit_data4, agent==120 & fitness>0.9687500)$run
avg_data4_best = subset(avg_data4, run %in% best_runs6)
fit_data4_best <- fit_data4[run %in% best_runs6,]

#perfect fit in task 1
best_perfect =  subset(fit_data1, agent==120 & fitness==1)$run
avg_data1_best = subset(avg_data1, run %in% best_perfect)
fit_data1_perfect <- fit_data1[run %in% best_perfect,]

#### Averaging function####
#Make function for averaging over the last 6 timepoints
avrgTimePoints <- function(avg_dataX, ntimepoints = 6){
  #Make empty dataframe for filling in the new values
  avg_dataX_new <- avg_dataX[FALSE,]
  #Go through each of the sub-dataframes
  for (df_sub in split(avg_dataX, avg_dataX$run)) {
    df_sub <- as.data.frame(df_sub)
    #Make a new empty version of it for filling in datapoints
    df_sub_new <- df_sub[FALSE,]
    #If its a fitness dataset
    if (ncol(df_sub) == 5){
      #Go through each row and all columns except the first 3
      for (r in 1:nrow(df_sub)){
        for (c in 5:(ncol(df_sub))){
          #For those datapoints that have less that 6 
          if (r>ntimepoints){
            #Average from beginning
            df_sub_new[r,c] <- mean(df_sub[(r-ntimepoints):r,c])
            #From others
          } else {
            #Average over last 6 datapoints
            df_sub_new[r,c] <- mean(df_sub[0:r,c])
          }
          #Also save the first the columns
          df_sub_new[r, 1:4] <- df_sub[r, 1:4]
        }
      }
    #If its an avg dataset  
    } else {
      #Go through each row and all columns except the first 3
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
          #Also save the first the columns
          df_sub_new[r, 1:3] <- df_sub[r, 1:3]
        }
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

#Do averaging on datasets
avg_data1 <- avrgTimePoints(avg_data1,ntimepoints = 6)
avg_data4 <- avrgTimePoints(avg_data4,ntimepoints = 6)
avg_data1_best <- avrgTimePoints(avg_data1_best,ntimepoints = 6)
avg_data4_best <- avrgTimePoints(avg_data4_best,ntimepoints = 6)

fit_data1 <- avrgTimePoints(fit_data1, ntimepoints = 6)
fit_data4 <- avrgTimePoints(fit_data4, ntimepoints = 6)
fit_data1_perfect <- avrgTimePoints(fit_data1_perfect, ntimepoints = 6)
fit_data4_best <- avrgTimePoints(fit_data4_best, ntimepoints = 6)

#Make dataframe with the final averages and summary statistics for all tasks etc
avg_LOD <- data.frame()
avg_LOD <- avrgRuns(avg_LOD, avg_data1, fit_data1, task = "1")
avg_LOD <- avrgRuns(avg_LOD, avg_data4, fit_data4, task = "4")
avg_LOD <- avrgRuns(avg_LOD, avg_data1_best, fit_data1_perfect, task = "1 - Perfect")
avg_LOD <- avrgRuns(avg_LOD, avg_data4_best, fit_data4_best, task = "4 - Best")

#Phi plot
avg_phi_plot = ggplot(subset(avg_LOD, Task!="1 - Perfect"), aes(x = Generation, y = Phi)) +
  geom_ribbon(aes(ymin = Phi - Phi_se, ymax = Phi + Phi_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  #geom_smooth(method = "lm", aes(fill = Task), color = "black", size = 2, se=F) +
  #geom_smooth(method = "lm", aes(color = Task), se=F) +
  theme_classic() +
  theme(legend.position = "none") +
  labs(y = "Φ") +
  colors + fills

avg_phi_max_plot = ggplot(avg_LOD, aes(x = Generation, y = Phi_max)) +
  #geom_ribbon(aes(ymin = Phi_max - Phi_max_se, ymax = Phi_max + Phi_max_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(fill = Task), color = "#222222",size = 1.1) +
  geom_line(aes(color = Task)) +
  geom_smooth(method = "lm", aes(fill = Task), color = "black", size = 2, se=F) +
  geom_smooth(method = "lm", aes(color = Task), se=F) +
  theme_classic() +
  #theme(legend.position = "none") +
  labs(y = "Φ max") +
  colors + fills


#### Surprise Plots ####
avg_sur_plot = ggplot(avg_LOD, aes(x = Generation, y = Surprise)) +
  geom_ribbon(aes(ymin = Surprise - Suprise_se, ymax = Surprise + Suprise_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  #ylim(6.4,8.5) +
  theme_classic()+
  labs(y = "Surprisal 1") +
  colors + fills

avg_sur2_plot = ggplot(avg_LOD, aes(x = Generation, y = Surprise2)) +
  geom_ribbon(aes(ymin = Surprise2 - Suprise2_se, ymax = Surprise2 + Suprise2_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  #ylim(6.8,10) +
  theme_classic()+
  labs(y = "Surprisal 2") +
  colors + fills

avg_blnkt_sur_plot = ggplot(avg_LOD, aes(x = Generation, y = blnkt_Surprise)) +
  geom_ribbon(aes(ymin = blnkt_Surprise - blnkt_Suprise_se, ymax = blnkt_Surprise + blnkt_Suprise_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  #ylim(6.4,8.5) +
  theme_classic()+
  labs(y = "Blanket Surprisal 1") +
  colors + fills

avg_blnkt_sur2_plot = ggplot(avg_LOD, aes(x = Generation, y = blnkt_Surprise2)) +
  geom_ribbon(aes(ymin = blnkt_Surprise2 - blnkt_Suprise2_se, ymax = blnkt_Surprise2 + blnkt_Suprise2_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  #ylim(6.8,10) +
  theme_classic()+
  labs(y = "Blanket Surprisal 2") +
  colors + fills

#Surprise Plots
avg_int_sur_plot = ggplot(avg_LOD, aes(x = Generation, y = int_Surprise)) +
  geom_ribbon(aes(ymin = int_Surprise - int_Suprise_se, ymax = int_Surprise + int_Suprise_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  #ylim(6.4,8.5) +
  theme_classic()+
  labs(y = "Internal Surprisal 1") +
  colors + fills

avg_int_sur2_plot = ggplot(avg_LOD, aes(x = Generation, y = int_Surprise2)) +
  geom_ribbon(aes(ymin = int_Surprise2 - int_Suprise2_se, ymax = int_Surprise2 + int_Suprise2_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  #ylim(6.8,10) +
  theme_classic()+
  labs(y = "Internal Surprisal 2") +
  colors + fills


## Run ##

Run_avg_sur_plot = ggplot(avg_LOD, aes(x = Generation, y = Run_Surprise)) +
  geom_ribbon(aes(ymin = Run_Surprise - Run_Suprise_se, ymax = Run_Surprise + Run_Suprise_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  #ylim(6.4,8.5) +
  theme_classic()+
  labs(y = "Run_Surprisal 1") +
  colors + fills

Run_avg_sur2_plot = ggplot(avg_LOD, aes(x = Generation, y = Run_Surprise2)) +
  geom_ribbon(aes(ymin = Run_Surprise2 - Run_Suprise2_se, ymax = Run_Surprise2 + Run_Suprise2_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  #ylim(6.8,10) +
  theme_classic()+
  labs(y = "Run_Surprisal 2") +
  colors + fills

Run_avg_blnkt_sur_plot = ggplot(avg_LOD, aes(x = Generation, y = Run_blnkt_Surprise)) +
  geom_ribbon(aes(ymin = Run_blnkt_Surprise - Run_blnkt_Suprise_se, ymax = Run_blnkt_Surprise + Run_blnkt_Suprise_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  #ylim(6.4,8.5) +
  theme_classic()+
  labs(y = "Run_Blanket Surprisal 1") +
  colors + fills

Run_avg_blnkt_sur2_plot = ggplot(avg_LOD, aes(x = Generation, y = Run_blnkt_Surprise2)) +
  geom_ribbon(aes(ymin = Run_blnkt_Surprise2 - Run_blnkt_Suprise2_se, ymax = Run_blnkt_Surprise2 + Run_blnkt_Suprise2_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  #ylim(6.8,10) +
  theme_classic()+
  labs(y = "Run_Blanket Surprisal 2") +
  colors + fills

#Surprise Plots
Run_avg_int_sur_plot = ggplot(avg_LOD, aes(x = Generation, y = Run_int_Surprise)) +
  geom_ribbon(aes(ymin = Run_int_Surprise - Run_int_Suprise_se, ymax = Run_int_Surprise + Run_int_Suprise_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  #ylim(6.4,8.5) +
  theme_classic()+
  labs(y = "Run_Internal Surprisal 1") +
  colors + fills

Run_avg_int_sur2_plot = ggplot(avg_LOD, aes(x = Generation, y = Run_int_Surprise2)) +
  geom_ribbon(aes(ymin = Run_int_Surprise2 - Run_int_Suprise2_se, ymax = Run_int_Surprise2 + Run_int_Suprise2_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  #ylim(6.8,10) +
  theme_classic()+
  labs(y = "Run_Internal Surprisal 2") +
  colors + fills


## Agent ##

Agent_avg_sur_plot = ggplot(avg_LOD, aes(x = Generation, y = Agent_Surprise)) +
  geom_ribbon(aes(ymin = Agent_Surprise - Agent_Suprise_se, ymax = Agent_Surprise + Agent_Suprise_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  #ylim(6.4,8.5) +
  theme_classic()+
  labs(y = "Agent_Surprisal 1") +
  colors + fills

Agent_avg_sur2_plot = ggplot(avg_LOD, aes(x = Generation, y = Agent_Surprise2)) +
  geom_ribbon(aes(ymin = Agent_Surprise2 - Agent_Suprise2_se, ymax = Agent_Surprise2 + Agent_Suprise2_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  #ylim(6.8,10) +
  theme_classic()+
  labs(y = "Agent_Surprisal 2") +
  colors + fills

Agent_avg_blnkt_sur_plot = ggplot(avg_LOD, aes(x = Generation, y = Agent_blnkt_Surprise)) +
  geom_ribbon(aes(ymin = Agent_blnkt_Surprise - Agent_blnkt_Suprise_se, ymax = Agent_blnkt_Surprise + Agent_blnkt_Suprise_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  #ylim(6.4,8.5) +
  theme_classic()+
  labs(y = "Agent_Blanket Surprisal 1") +
  colors + fills

Agent_avg_blnkt_sur2_plot = ggplot(avg_LOD, aes(x = Generation, y = Agent_blnkt_Surprise2)) +
  geom_ribbon(aes(ymin = Agent_blnkt_Surprise2 - Agent_blnkt_Suprise2_se, ymax = Agent_blnkt_Surprise2 + Agent_blnkt_Suprise2_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  #ylim(6.8,10) +
  theme_classic()+
  labs(y = "Agent_Blanket Surprisal 2") +
  colors + fills

#Surprise Plots
Agent_avg_int_sur_plot = ggplot(avg_LOD, aes(x = Generation, y = Agent_int_Surprise)) +
  geom_ribbon(aes(ymin = Agent_int_Surprise - Agent_int_Suprise_se, ymax = Agent_int_Surprise + Agent_int_Suprise_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  #ylim(6.4,8.5) +
  theme_classic()+
  labs(y = "Agent_Internal Surprisal 1") +
  colors + fills

Agent_avg_int_sur2_plot = ggplot(avg_LOD, aes(x = Generation, y = Agent_int_Surprise2)) +
  geom_ribbon(aes(ymin = Agent_int_Surprise2 - Agent_int_Suprise2_se, ymax = Agent_int_Surprise2 + Agent_int_Suprise2_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  #ylim(6.8,10) +
  theme_classic()+
  labs(y = "Agent_Internal Surprisal 2") +
  colors + fills


#### Showing plots ####
# ALL Surpise plots
ggarrange(
  #avg_sur_plot,
  #avg_sur2_plot,
  #avg_blnkt_sur_plot,
  #avg_blnkt_sur2_plot,
  #avg_int_sur_plot,
  #avg_int_sur2_plot,
  
  #Run_avg_sur_plot,
  #Run_avg_sur2_plot,
  #Run_avg_blnkt_sur_plot,
  #Run_avg_blnkt_sur2_plot,
  #Run_avg_int_sur_plot,
  #Run_avg_int_sur2_plot,
  
  #Agent_avg_sur_plot,
  Agent_avg_sur2_plot,
  #Agent_avg_blnkt_sur_plot,
  Agent_avg_blnkt_sur2_plot,
  #Agent_avg_int_sur_plot,
  Agent_avg_int_sur2_plot,
  
  labels = "auto"
)

ggplot(avg_LOD, aes(x = Generation, y = Fitness)) +
  geom_ribbon(aes(ymin = Fitness - Fitness_se, ymax = Fitness + Fitness_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task)) +
  #geom_smooth(method = "lm", aes(fill = Task), color = "black", size = 2, se=F) +
  #geom_smooth(method = "lm", aes(color = Task), se=F) +
  theme_classic() +
  #theme(legend.position = "none") +
  labs(y = "Fitness") +
  colors + fills

ggplot(avg_LOD, aes(x = Generation, y = Fitness)) +
  geom_ribbon(aes(ymin = Fitness - Fitness_se, ymax = Fitness + Fitness_se, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(color = Task), linetype="dashed") +
  geom_ribbon(aes(ymin = Run_Surprise/10 - Run_Suprise_se/10, ymax = Run_Surprise/10 + Run_Suprise_se/10, fill = Task), color = F, alpha = 0.3) +
  geom_line(aes(y = Run_Surprise/10, color = Task)) +
  #geom_smooth(method = "lm", aes(fill = Task), color = "black", size = 2, se=F) +
  #geom_smooth(method = "lm", aes(color = Task), se=F) +
  theme_classic() +
  #theme(legend.position = "none") +
  labs(y = "Fitness") +
  colors + fills

  
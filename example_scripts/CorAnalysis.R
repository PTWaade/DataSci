#### Preparation ####
#Load packages
pacman::p_load(data.table, ggplot2, tidyr, rstudioapi)

#Sets working directory to the folder where this script is located.
setwd(paste0(dirname(getActiveDocumentContext()$path), "/results_data"))

#Get data
cor <- fread("none_cordata.csv")
run_cor <- fread("run_cordata.csv")
agent_cor <- fread("agent_cordata.csv")

#Remove trials with no phi
cor <- cor[!(is.na(cor$CCPhi_SC)),]
run_cor <- run_cor[!(is.na(run_cor$CCPhi_SC)),]
agent_cor <- agent_cor[!(is.na(agent_cor$CCPhi_SC)),]

#Make surprise characters
cor$CCPhi_SC = as.character(cor$CCPhi_SC)
run_cor$CCPhi_SC = as.character(run_cor$CCPhi_SC)
agent_cor$CCPhi_SC = as.character(agent_cor$CCPhi_SC)

#### Setting up Functions ####
#Function for unpacking all the correlation data
unpack_cor = function(corX) {
  #Make empty array for populating 
  all_corX = array(0,c(length(corX$CCPhi_SC),65))
  #Go through each row
  for(r in 1:length(corX$CCPhi_SC)){
    #Get out a string with the values
    string = corX[r,5][[1]]
    #Split it into the values
    values = as.numeric(strsplit(string, ";")[[1]])
    #Save it 
    all_corX[r,] = values
  }
  #Change names so that they go from negative half to positive half, and the middle is 0 lag
  colnames(all_corX) <- as.integer(-ncol(all_corX)/2):as.integer(ncol(all_corX)/2)
  return(all_corX)
}

#Function for printing all rose with correlation at lack 0 in a certain interval
printExamples = function(cordata, lowerbound, upperbound) {
  for (r in 1:length(cordata$CCPhi_SC)) {
    values <- as.numeric(strsplit(cordata[r,5][[1]], ";")[[1]])
    if (!is.na(values[33])) {
      if (values[33] < upperbound & values[33] > lowerbound){
        print(paste("RunAgentTrial",cordata[r,2], cordata[r,3], cordata[r,4]))
        print(values[31:34])
      }
    }
  }
}

#Function for making plot data
makePlotD = function(d_unpacked, firstTaskNR,
                         d_unpacked2 = NA, secondTaskNR = NA,
                         mergedata = FALSE,
                         lagmin, lagmax) {
  #---- create first data frame ----
  n_lags_all = length(d_unpacked[1,]) #Total number of lags
  n_lags = length(lagmin:lagmax) #Number of lags to plot
  lag0 = median(1:n_lags_all) #Number of lag 0. Assumes symmetry in the lenght of pos and neg lags
  minimum = lag0 + lagmin #If above is true, this find the lag specified by lagmin
  maximum = lag0 + lagmax #If above is true, this find the lag specified by lagmax
  
  #Subset unpacked data to specified lags
  d_sub = d_unpacked[ ,minimum:maximum]
  
  #Make it into a data frame
  dplot = as.data.frame(d_sub)
  
  #Add characters to lag names. This is because some data wrangling is annoying when names are numbers
  colnames(dplot) <- gsub("-", "Neg", paste("Lag", colnames(dplot), sep = ""))
  
  #Make data into long format
  dplot = gather(dplot, key = "lag", "cor", 1:n_lags)
  
  #Make lag names numbers again
  dplot$lag = gsub("LagNeg", "-", dplot$lag)
  dplot$lag = gsub("Lag", "", dplot$lag)
  
  #Make variables numeric for plot
  dplot$lag = as.numeric(dplot$lag) #Numeric instead of factor for right order
  dplot$cor = as.numeric(dplot$cor)
  
  #Add task as factor
  dplot$task = firstTaskNR
  dplot$task = as.factor(dplot$task)
  
  if (mergedata == TRUE){
    n_lags_all = length(d_unpacked[1,]) #Total number of lags
    n_lags = length(lagmin:lagmax) #Number of lags to plot
    lag0 = median(1:n_lags_all) #Number of lag 0. Assumes symmetry in the lenght of pos and neg lags
    minimum = lag0 + lagmin #If above is true, this find the lag specified by lagmin
    maximum = lag0 + lagmax #If above is true, this find the lag specified by lagmax
    
    #Subset unpacked data to specified lags
    d_sub2 = d_unpacked2[ ,minimum:maximum]
    
    #Make it into a data frame
    dplot2 = as.data.frame(d_sub2)
    
    #Add characters to lag names. This is because some data wrangling is annoying when names are numbers
    colnames(dplot2) <- gsub("-", "Neg", paste("Lag", colnames(dplot2), sep = ""))
    
    #Make data into long format
    dplot2 = gather(dplot2, key = "lag", "cor", 1:n_lags)
    
    #Make lag names numbers again
    dplot2$lag = gsub("LagNeg", "-", dplot2$lag)
    dplot2$lag = gsub("Lag", "", dplot2$lag)
    
    #Make variables numeric for plot
    dplot2$lag = as.numeric(dplot2$lag) #Numeric instead of factor for right order
    dplot2$cor = as.numeric(dplot2$cor)
    
    #Add task as factor
    dplot2$task = secondTaskNR
    dplot2$task = as.factor(dplot2$task)
    
    d_merged = rbind(dplot, dplot2)
    
    return(d_merged)
    
  } else{
    return(dplot)
  }
}

#Function for plotting unpacked data
#Scalesize can be "max", "free", "free_x" and "free_y".
plot_fun = function(d_unpacked, 
                    taskcolors,
                         lagmin, lagmax,
                         scalesize = "max", width) {
  
  dplot = d_unpacked[lagmin <= d_unpacked$lag & d_unpacked$lag <= lagmax,]
  
  #create labels
  #lag_labels = paste("Lag:", as.character(unique(dplot$lag)), sep = " ")
  #names(lag_labels) = seq(lagmin, lagmax, 1)
  
  #Plot with free scale size?
  
  if (scalesize != "max"){
    #Plot
    ggplot(dplot, aes(cor, col = task))+
      geom_vline(xintercept = 0, col = "lightgrey")+
      geom_density(show_guide=FALSE)+
      stat_density(geom="line",position="identity", size = width)+
      facet_wrap(~lag, scales = scalesize)+
      labs(title = "Cross-correlation between Phi and surprisal across time lags",
           x = "Correlation Strength", y = "Density")+
      theme_classic()+
      scale_color_manual(values= taskcolors,
                         guide = guide_legend(override.aes = list(
                           linetype = c("solid", "solid"), shape = c(1,1))))
    
  } else {
    #Plot
    ggplot(dplot, aes(cor, col = task))+
      geom_vline(xintercept = 0, col = "lightgrey")+
      geom_density(show_guide=FALSE)+
      stat_density(geom="line",position="identity", size = width)+
      facet_wrap(~lag)+
      labs(title = "Cross-correlation between Phi and surprisal across time lags",
           x = "Correlation Strength", y = "Density")+
      theme_classic()+
      scale_color_manual(values= taskcolors,
                         guide = guide_legend(override.aes = list(
                           linetype = c("solid", "solid"), shape = c(1,1))))
  }
}

#### Unpacking data ####
# #Unpack full datasets
cor_unpack <- unpack_cor(cor)
run_cor_unpack <- unpack_cor(run_cor)
agent_cor_unpack <- unpack_cor(agent_cor)

#### Plot data ####
### Make data frame for ggplot
#d_unpacked is for task 1, d_unpacked2 should be for task 4
#Note that we here split by surprisal type instead of task, due to this being a toy example with only one task.
dplot = makePlotD(d_unpacked = run_cor_unpack, firstTaskNR = "run",
          d_unpacked2 = agent_cor_unpack, secondTaskNR = "agent",
          mergedata = TRUE, 
          lagmin = -32, lagmax = 32) 

### Plot the data
#Scalesize can be "max", "free", "free_x" and "free_y". Max uses global maximum for scale limits.
plot_fun(d_unpacked = dplot, lagmin = -6, lagmax = 5, #Change this to make a subset of the density plots
         scalesize = "max", taskcolor = c("black", "#3498DB"), width = 0.8)

####Find examples from data ####
#First find examples in cor
printExamples(cor, -0.0001, 0.001)
printExamples(cor, -0.99, -0.9)
printExamples(cor, 0.8, 0.95)

#Load in transition data
tran = fread("none_trans_data.csv")
#run_tran = fread("run_trans_data.csv")
#agent_tran = fread("agent_trans_data.csv")

#And plot the phi and surprise fluctuating across the trials on a specific generation of a specific run
ggplot(subset(tran, run == 1 & agent == 13), aes(x = timeStep, y = Phi)) +
  geom_line() +
  theme_minimal() +
  geom_line(aes(y = Nonesurprise_cond/10), color = "red") +
  facet_wrap(~trial)



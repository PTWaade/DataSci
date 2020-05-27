#### Preparation ####
#Load packages
pacman::p_load(data.table, ggplot2, tidyr, rstudioapi)

#Sets working directory to the folder where this script is located.
setwd(dirname(getActiveDocumentContext()$path))



#Get data
#cor1 = fread("cor_data_task1.csv")
#cor4 = fread("cor_data_task4.csv")
#agent_cor1 <- fread("agent_cordata_task1.csv")
#agent_cor4 <- fread("agent_cordata_task4.csv")
#run_cor1 <- fread("run_cordata_task1.csv")
#run_cor4 <- fread("run_cordata_task4.csv")
#run_cor1 <- fread("run_cordata_task1.csv")
#run_cor4 = dcor
run_cor4 <- fread("cor_data_task1.csv")
run_cor1 <- fread("cor_data_task4.csv")



#Rename
#names(run_cor4)[names(run_cor4) == "CCPhi_RunSurCond"] <- "CCPhi_SC"
#names(run_cor1)[names(run_cor1) == "CCPhi_RunSurLone"] <- "CCPhi_SC"

#Remove trials with no phi
#cor1 = cor1[!(is.na(cor1$CCPhi_SC)),]
#cor4 = cor4[!(is.na(cor4$CCPhi_SC)),]
#agent_cor1 <- agent_cor1[!(is.na(agent_cor1$CCPhi_SC)),]
#agent_cor4 <- agent_cor4[!(is.na(agent_cor4$CCPhi_SC)),]
#run_cor1 <- run_cor1[!(is.na(run_cor1$CCPhi_SC)),]
run_cor4 <- run_cor4[!(is.na(run_cor4$CCPhi_SC)),]
run_cor1 <- run_cor1[!(is.na(run_cor1$CCPhi_SC)),]

#Make surprise characters
run_cor4$CCPhi_SC = as.character(run_cor4$CCPhi_SC)
run_cor1$CCPhi_SC = as.character(run_cor1$CCPhi_SC)

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
#all_cor1 <- unpack_cor(cor1)
#all_cor4 <- unpack_cor(cor4)
#agent_all_cor1 <- unpack_cor(agent_cor1)
#agent_all_cor4 <- unpack_cor(agent_cor4)
#run_all_cor1 <- unpack_cor(run_cor1)
cor1_unpack <- unpack_cor(run_cor1)
cor4_unpack <- unpack_cor(run_cor4)

#remove old data frames
rm(run_cor1, run_cor4)


# #Unpack late datasets
#all_cor1_late <- unpack_cor(cor1[cor1$agent > 60])
#all_cor4_late <- unpack_cor(cor4[cor4$agent > 60])
# #Unpack early datasets
#all_cor1_early <- unpack_cor(cor1[cor1$agent < 30])
#all_cor4_early <- unpack_cor(cor4[cor4$agent < 30])


#### Plot data ####

### Make data frame for ggplot
#d_unpacked is for task 1, d_unpacked2 should be for task 4
dplot = makePlotD(d_unpacked = cor1_unpack, firstTaskNR = 1,
          d_unpacked2 = cor4_unpack, secondTaskNR = 4,
          mergedata = TRUE,
          lagmin = -32, lagmax = 32)

rm(cor1_unpack, cor4_unpack)

### Plot the data
#Scalesize can be "max", "free", "free_x" and "free_y". Max uses global maximum for scale limits.
plot_fun(d_unpacked = dplot, lagmin = -6, lagmax = 5, scalesize = "max", taskcolor = c("black", "#3498DB"), width = 0.8)





par(mfrow=c(4,3))
for (i in -5:5) plot(density(run_all_cor4[,as.character(i)], na.rm = T), main=paste("Lag = ", i, sep = ""), xlab = "Correlation")


####Find examples from data ####
#First find examples in cor
printExamples(cor4[cor4$agent > 60], -0.0001, 0.0001)
#Run 0 Agent 98 Trial 81
#Run 8 Agent 100 Trial 68
printExamples(cor4[cor4$agent > 60], -0.99, -0.95)
#Run 20 Agent 74 Trial 44 and 46
#Run 26 Agent 102 Trial 42 and 106
printExamples(cor4[cor4$agent > 60], 0.90, 0.95)
# Run 33 Agent 114 Trial 16, 48, 80 and 112
# Run 32 Agent 99 Trial 29
#Run 40 Agent 108 Trial 110

#Load in transition data
#tran1 = fread("trans_data_task1.csv")
#tran4 = fread("trans_data_task4.csv")


# #And plot the phi and surprise relations
# ggplot(subset(tran4, run == 26 & agent == 102), aes(x = timeStep, y = Phi)) +
#   geom_line() +
#   theme_minimal() +
#   geom_line(aes(y = Runsurprise_cond/10), color = "red") +
#   facet_wrap(~trial)



make_avg_data = function(type) {
  if (type == "None") data = fread("results_data/none_trans_data.csv")
  if (type == "Agent") data = fread("results_data/agent_trans_data.csv")
  if (type == "Run") data = fread("results_data/run_trans_data.csv")
  
  i = 1
  sum_data_list = list()
  for (r in unique(data$run)) {
    
    for (a in unique(data$agent)){
      cat("\rRun:",r,"- Agent:",a)
      d = data[data$run==r & data$agent==a,] 
      
      if (type == "None"){
        Phi_mean = mean(d$Phi)
        Phi_max = max(d$Phi)
        
        nCon_mean = mean(d$n_concepts)
        nCon_max = max(d$n_concepts)
        
        ConPhi_list = c()
        for (x in d$concept_phis){
          ConPhi = as.numeric(strsplit(x, "-")[[1]])
          ConPhi_list = c(ConPhi_list,ConPhi)
        }
        
        ConPhi_mean = mean(ConPhi_list)
        ConPhi_max = max(ConPhi_list)
        
        surprise_lone_mean = mean(d$Nonesurprise_lone)
        surprise_lone_max = max(d$Nonesurprise_lone)
        
        surprise_cond_mean = mean(d$Nonesurprise_cond)
        surprise_cond_max = max(d$Nonesurprise_cond)
        
        surprise_cond2_mean = mean(d$Nonesurprise_cond2)
        surprise_cond2_max = max(d$Nonesurprise_cond2)
        
        blnkt_surprise_lone_mean = mean(d$Noneblnkt_surprise_lone)
        blnkt_surprise_lone_max = max(d$Noneblnkt_surprise_lone)
        
        blnkt_surprise_cond_mean = mean(d$Noneblnkt_surprise_cond)
        blnkt_surprise_cond_max = max(d$Noneblnkt_surprise_cond)
        
        blnkt_surprise_cond2_mean = mean(d$Noneblnkt_surprise_cond2)
        blnkt_surprise_cond2_max = max(d$Noneblnkt_surprise_cond2)
        
        int_surprise_lone_mean = mean(d$Noneint_surprise_lone)
        int_surprise_lone_max = max(d$Noneint_surprise_lone)
        
        int_surprise_cond_mean = mean(d$Noneint_surprise_cond)
        int_surprise_cond_max = max(d$Noneint_surprise_cond)
        
        int_surprise_cond2_mean = mean(d$Noneint_surprise_cond2)
        int_surprise_cond2_max = max(d$Noneint_surprise_cond2)
        
        sum_data_list[[i]] = data.frame(run = r, agent = a,
                                        Phi_mean = Phi_mean, Phi_max = Phi_max,
                                        nCon_mean=nCon_mean, nCon_max=nCon_max,
                                        ConPhi_mean=ConPhi_mean, ConPhi_max=ConPhi_max,
                                        surprise_lone_mean = surprise_lone_mean, surprise_lone_max = surprise_lone_max,
                                        surprise_cond_mean = surprise_cond_mean, surprise_cond_max = surprise_cond_max,
                                        surprise_cond2_mean = surprise_cond2_mean, surprise_cond2_max = surprise_cond2_max,
                                        blnkt_surprise_lone_mean = blnkt_surprise_lone_mean, blnkt_surprise_lone_max = blnkt_surprise_lone_max,
                                        blnkt_surprise_cond_mean = blnkt_surprise_cond_mean, blnkt_surprise_cond_max = blnkt_surprise_cond_max,
                                        blnkt_surprise_cond2_mean = blnkt_surprise_cond2_mean, blnkt_surprise_cond2_max = blnkt_surprise_cond2_max,
                                        int_surprise_lone_mean = int_surprise_lone_mean, int_surprise_lone_max = int_surprise_lone_max,
                                        int_surprise_cond_mean = int_surprise_cond_mean, int_surprise_cond_max = int_surprise_cond_max,
                                        int_surprise_cond2_mean = int_surprise_cond2_mean, int_surprise_cond2_max = int_surprise_cond2_max
        )
        
      }
      
      if (type == "Agent"){
        RunAgentsurprise_lone_mean = mean(d$RunAgentsurprise_lone)
        RunAgentsurprise_lone_max = max(d$RunAgentsurprise_lone)
        
        RunAgentsurprise_cond_mean = mean(d$RunAgentsurprise_cond)
        RunAgentsurprise_cond_max = max(d$RunAgentsurprise_cond)
        
        RunAgentsurprise_cond2_mean = mean(d$RunAgentsurprise_cond2)
        RunAgentsurprise_cond2_max = max(d$RunAgentsurprise_cond2)
        
        RunAgentblnkt_surprise_lone_mean = mean(d$RunAgentblnkt_surprise_lone)
        RunAgentblnkt_surprise_lone_max = max(d$RunAgentblnkt_surprise_lone)
        
        RunAgentblnkt_surprise_cond_mean = mean(d$RunAgentblnkt_surprise_cond)
        RunAgentblnkt_surprise_cond_max = max(d$RunAgentblnkt_surprise_cond)
        
        RunAgentblnkt_surprise_cond2_mean = mean(d$RunAgentblnkt_surprise_cond2)
        RunAgentblnkt_surprise_cond2_max = max(d$RunAgentblnkt_surprise_cond2)
        
        RunAgentint_surprise_lone_mean = mean(d$RunAgentint_surprise_lone)
        RunAgentint_surprise_lone_max = max(d$RunAgentint_surprise_lone)
        
        RunAgentint_surprise_cond_mean = mean(d$RunAgentint_surprise_cond)
        RunAgentint_surprise_cond_max = max(d$RunAgentint_surprise_cond)
        
        RunAgentint_surprise_cond2_mean = mean(d$RunAgentint_surprise_cond2)
        RunAgentint_surprise_cond2_max = max(d$RunAgentint_surprise_cond2)
        
        sum_data_list[[i]] = data.frame(run = r, agent = a,
                                        RunAgentsurprise_lone_mean = RunAgentsurprise_lone_mean, RunAgentsurprise_lone_max = RunAgentsurprise_lone_max,
                                        RunAgentsurprise_cond_mean = RunAgentsurprise_cond_mean, RunAgentsurprise_cond_max = RunAgentsurprise_cond_max,
                                        RunAgentsurprise_cond2_mean = RunAgentsurprise_cond2_mean, RunAgentsurprise_cond2_max = RunAgentsurprise_cond2_max,
                                        RunAgentblnkt_surprise_lone_mean = RunAgentblnkt_surprise_lone_mean, RunAgentblnkt_surprise_lone_max = RunAgentblnkt_surprise_lone_max,
                                        RunAgentblnkt_surprise_cond_mean = RunAgentblnkt_surprise_cond_mean, RunAgentblnkt_surprise_cond_max = RunAgentblnkt_surprise_cond_max,
                                        RunAgentblnkt_surprise_cond2_mean = RunAgentblnkt_surprise_cond2_mean, RunAgentblnkt_surprise_cond2_max = RunAgentblnkt_surprise_cond2_max,
                                        RunAgentint_surprise_lone_mean = RunAgentint_surprise_lone_mean, RunAgentint_surprise_lone_max = RunAgentint_surprise_lone_max,
                                        RunAgentint_surprise_cond_mean = RunAgentint_surprise_cond_mean, RunAgentint_surprise_cond_max = RunAgentint_surprise_cond_max,
                                        RunAgentint_surprise_cond2_mean = RunAgentint_surprise_cond2_mean, RunAgentint_surprise_cond2_max = RunAgentint_surprise_cond2_max
                                        
        )
      }
      
      if (type == "Run"){
        Runsurprise_lone_mean = mean(d$Runsurprise_lone)
        Runsurprise_lone_max = max(d$Runsurprise_lone)
        
        Runsurprise_cond_mean = mean(d$Runsurprise_cond)
        Runsurprise_cond_max = max(d$Runsurprise_cond)
        
        Runsurprise_cond2_mean = mean(d$Runsurprise_cond2)
        Runsurprise_cond2_max = max(d$Runsurprise_cond2)
        
        Runblnkt_surprise_lone_mean = mean(d$Runblnkt_surprise_lone)
        Runblnkt_surprise_lone_max = max(d$Runblnkt_surprise_lone)
        
        Runblnkt_surprise_cond_mean = mean(d$Runblnkt_surprise_cond)
        Runblnkt_surprise_cond_max = max(d$Runblnkt_surprise_cond)
        
        Runblnkt_surprise_cond2_mean = mean(d$Runblnkt_surprise_cond2)
        Runblnkt_surprise_cond2_max = max(d$Runblnkt_surprise_cond2)
        
        Runint_surprise_lone_mean = mean(d$Runint_surprise_lone)
        Runint_surprise_lone_max = max(d$Runint_surprise_lone)
        
        Runint_surprise_cond_mean = mean(d$Runint_surprise_cond)
        Runint_surprise_cond_max = max(d$Runint_surprise_cond)
        
        Runint_surprise_cond2_mean = mean(d$Runint_surprise_cond2)
        Runint_surprise_cond2_max = max(d$Runint_surprise_cond2)
        
        sum_data_list[[i]] = data.frame(run = r, agent = a,
                                        Runsurprise_lone_mean = Runsurprise_lone_mean, Runsurprise_lone_max = Runsurprise_lone_max,
                                        Runsurprise_cond_mean = Runsurprise_cond_mean, Runsurprise_cond_max = Runsurprise_cond_max,
                                        Runsurprise_cond2_mean = Runsurprise_cond2_mean, Runsurprise_cond2_max = Runsurprise_cond2_max,
                                        Runblnkt_surprise_lone_mean = Runblnkt_surprise_lone_mean, Runblnkt_surprise_lone_max = Runblnkt_surprise_lone_max,
                                        Runblnkt_surprise_cond_mean = Runblnkt_surprise_cond_mean, Runblnkt_surprise_cond_max = Runblnkt_surprise_cond_max,
                                        Runblnkt_surprise_cond2_mean = Runblnkt_surprise_cond2_mean, Runblnkt_surprise_cond2_max = Runblnkt_surprise_cond2_max,
                                        Runint_surprise_lone_mean = Runint_surprise_lone_mean, Runint_surprise_lone_max = Runint_surprise_lone_max,
                                        Runint_surprise_cond_mean = Runint_surprise_cond_mean, Runint_surprise_cond_max = Runint_surprise_cond_max,
                                        Runint_surprise_cond2_mean = Runint_surprise_cond2_mean, Runint_surprise_cond2_max = Runint_surprise_cond2_max
        )
      }
      
      i = i + 1
    } # end loop through agents
  } # end loop through runs
  
  sum_data = do.call("rbind", sum_data_list)
  
  if (type == "None"){
    sum_data$ConPhi_mean[is.nan(sum_data$ConPhi_mean)] = 0
    sum_data$ConPhi_max[is.infinite(sum_data$ConPhi_max)] = 0
  }
  
  return(sum_data)
}
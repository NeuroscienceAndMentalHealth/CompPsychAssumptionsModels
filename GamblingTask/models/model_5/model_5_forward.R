gamble_model5<-function(task,params){
  
  N=nrow(params)
  
  #remove any repeated trials (removes second instance)
  gamble_dups_idx<-which(duplicated(task%>%select(id,trial_nr),fromLast=TRUE))
  if (length(gamble_dups_idx)>0){
    task<-task[-c(gamble_dups_idx),]
  } 
  
  task<-task%>%
    filter(!is.na(trial_nr))
  
  gamble=matrix(data=NA,nrow=N,ncol=nrow(task[task$id==1,]))
  
  
  for (i in 1:N) {
    task_temp = task[task$id==i,]
    params_temp = params[params$id==i,]
    for (t in 1:nrow(task_temp)) {
      
      if (task_temp$safe[t] < 0){ 
        evSafe   = - (params_temp$gamble_lambda * pow(abs(task_temp$safe[t]), params_temp$gamble_rho_l)); 
        evGamble = - 0.5 * params_temp$gamble_lambda * pow(abs(task_temp$risky_loss[t]), params_temp$gamble_rho_l); 
      }
      if (task_temp$safe[t] == 0){ 
        evSafe   = pow(task_temp$safe[t], params_temp$gamble_rho_g);
        evGamble = 0.5 * (pow(task_temp$risky_gain[t], params_temp$gamble_rho_g) - params_temp$gamble_lambda * pow(abs(task_temp$risky_loss[t]), params_temp$gamble_rho_l));
      }
      if (task_temp$safe[t] > 0) { 
        evSafe   = pow(task_temp$safe[t], params_temp$gamble_rho_g);
        evGamble = 0.5 * pow(task_temp$risky_gain[t], params_temp$gamble_rho_g);
      }
      pGamble  = boot::inv.logit(params_temp$gamble_tau * (evGamble - evSafe));
      gamble[i, t] = sample(0:1,1,prob=c(1-pGamble,pGamble))
    }
    task$chose_risky[task$id==i]=gamble[i,]
  }
  return(task)
}
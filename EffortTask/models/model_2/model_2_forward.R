effort_model2<-function(task,params){
  
  Ns=nrow(params)
  
  task<-task%>%
    filter(phase=='main')
  
  #remove any repeated trials (removes first instance)
  effort_dups_idx<-which(duplicated(task%>%select(id,trial_nr),fromLast=TRUE))
  if (length(effort_dups_idx)>0){
    task<-task[-c(effort_dups_idx),]
  }
  
  y=matrix(data=NA,nrow=Ns,ncol=nrow(task[task$id==1,]))
  
  
  for (i_subj in 1:Ns) {
    task_temp = task[task$id==i_subj,]
    params_temp = params[params$id==i_subj,]
    for (i_trial in 1:nrow(task_temp)) {
      p_accept=boot::inv.logit(  params_temp$effort_thetaI
                                 + task_temp$reward[i_trial]*params_temp$effort_thetaR
                                 + task_temp$difficulty[i_trial]*params_temp$effort_thetaE)
      y[i_subj,i_trial] = sample(0:1,1,prob=c(1-p_accept,p_accept))
    }
    task$accepted[task$id==i_subj]=y[i_subj,]
  }
  return(task)
}
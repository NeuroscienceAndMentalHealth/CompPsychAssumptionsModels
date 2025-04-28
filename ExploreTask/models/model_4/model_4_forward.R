explore_model4<-function(task,params){
  
  N=nrow(params)
  
  task<-gershman_preproc(task)
  
  choice=matrix(data=NA,nrow=N,ncol=nrow(task[task$id==1,]))
  
  for (i in 1:N) {
    task_temp = task[task$id==i,]
    params_temp = params[params$id==i,]
    for (t in 1:nrow(task_temp)) {
      prob=boot::inv.logit(params_temp$explore_theta_V * task_temp$kalman_value_difference[t] + params_temp$explore_theta_RU * task_temp$kalman_sigma_difference[t])
      choice[i,t] = sample(0:1,1,prob=c(1-prob,prob));
    }
    task$choice[task$id==i]=choice[i,]
  }
  return(task)
}
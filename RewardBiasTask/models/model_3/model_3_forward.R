rbias_model3<-function(task,params){
  
  N=nrow(params)

  #remove any repeated trials (removes first instance)
  rbias_dups_idx<-which(duplicated(task%>%select(id,trial_nr),fromLast=TRUE))
  if (length(rbias_dups_idx)>0){
    task<-task[-c(rbias_dups_idx),]
  }
  
  choice=matrix(data=NA,nrow=N,ncol=nrow(task[task$id==1,]))
  
  accuracy<-matrix(c(1,0,0,1),nrow=2) #little helper matrix
  
  task=task%>%
    mutate(congruence = case_when(
      coherence==0.8 & target_right==0 ~ 1,
      coherence==0.6 & target_right==0 ~ 2,
      coherence==0.5 ~ 3, 
      coherence==0.6 & target_right==1 ~ 4,
      coherence==0.8 & target_right==1 ~ 5))
  for (i in 1:N) {
    task_temp = task[task$id==i,]
    params_temp = params[params$id==i,]
    if (nrow(task_temp)<1){
      break
    }
    for (t in 1:nrow(task_temp)) {
      if(task_temp$congruence[t]==9){ #essentially turns this off
        correct<-c(0,0)
      } else {
        correct<-accuracy[task_temp$target_right[t]+1,]
      }
      v = matrix(0, nrow=2, ncol=5)
      v[1,]<-params_temp$rbias_initV
      v[2,]<-1-params_temp$rbias_initV
      tempv = v[,task_temp$congruence[t]]
             	 choice[i,t] = sample(1:2,1,prob=softmax(tempv + params_temp$rbias_instruction_sensitivity * correct))
		           v[choice[i,t],task_temp$congruence[t]] = v[choice[i,t],task_temp$congruence[t]]+ params_temp$rbias_alpha * (params_temp$rbias_reward_sensitivity[i]* task_temp$reward[t]-v[choice[i,t],task_temp$congruence[t]])
             
    }
    task$said_right[task$id==i]=choice[i,]
  }
  return(task)
}     
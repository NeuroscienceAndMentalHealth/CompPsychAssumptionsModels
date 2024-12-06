samples = 100 #imagine we have 100 posterior samples of a parameter
nsub=20 #20 participants
r = 0.5 #correlation of the WITHIN-PARTICIPANT samples at t1 and t2 (can change this)

library('MASS')

out<-data.frame(matrix(rep(0,nsub*2),ncol=2)) #empty data frame which stores the participant-level means
colnames(out)<-c('t1','t2')

for (i in 1:nsub){

  data = mvrnorm(n=samples, mu=c(0, 0), Sigma=matrix(c(1, r, r, 1), nrow=2), empirical=TRUE) #this generates data that should be perfectly correlated at 0.5 - imagine this like a series of posterior estimates of a parameter at t1 and t2
  
  out[i,1]<-mean(data[,1]) #what we do outside stan - i.e. just get the posterior mean for time 1
  out[i,2]<-mean(data[,2]) #and for time 2
  
  # Assess that it works
}

cor.test(data[,1],data[,2]) #just to show that the samples generated for the last participant are indeed perfectly correlated at the value we set r to above

cor.test(out[,1],out[,2]) #this number varies a lot - as just taking the mean from the long list of 100 'posterior samples' for our 20 participants 
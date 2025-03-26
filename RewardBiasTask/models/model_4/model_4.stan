/* Pizzagalli model 4 : 'Stimulus action w/sep sensitivities'
*  author: Alex Pike
*  email: alex.pike@ucl.ac.uk
*/


data {
     int<lower=1> N; 				//Number of subjects (strictly positive int)
     int<lower=1> T;  				//Number of trials (strictly positive int)
     int<lower=1> levels;     //Number of levels of congruence: set to 5

     array[N,T] int<lower=1,upper=2> choice; 		 // Array of ints containing the choice made for each trial and participant (i.e. whether they chose left or right) — (rows: participants, columns: trials)
     array[N,T] int<lower=0,upper=1> accuracy; //For whether they actually responded correctly (even if unrewarded)
     array[N,T] int<lower=-1,upper=1> rwd;		//Matrix of integers containing the reward received on a given trial (1 or 0) — (rows: participants, columns : trials)
     array[N,T] int<lower=1,upper=levels> congruence; //The congruence of the stimuli: should be integers from 1 to levels

     matrix[2,levels] Vinits;		//Matrix of reals containing the initial q-values for left and right for each congruence level - not used in this model;

}

transformed data {
     array[N,T] vector [2] correct;
     for (i in 1:N){
       for (t in 1:T){
              if (choice[i,t]==1){
                      if (accuracy[i,t]==1){
                         correct[i,t]=[1, 0]';
                     } else if (accuracy[i,t]==0){
                         correct[i,t]=[0, 1]';
                     }
               } else if (choice[i,t]==2){
                      if (accuracy[i,t]==1){
                         correct[i,t]=[0,1]';
                    } else if (accuracy[i,t]==0){
                         correct[i,t]=[1,0]';
                    }
               }
       }
     }
}

parameters {
     vector[5] mu;
     vector<lower=0>[5] sigma;

     vector[N] alpha_raw;
     vector[N] init_raw; //array of initial values - (rows: participants, columns: actions, 3rd dimension: congruence levels)
     vector[N] reward_sensitivity_raw;
     vector[N] punish_sensitivity_raw;
     vector[N] instruction_sensitivity_raw;
}

transformed parameters {
     vector<lower=0,upper=1>[N] alpha;
     vector[N] initV;
     vector <lower=0>[N] reward_sensitivity;
     vector <lower=0>[N] punish_sensitivity;
     vector <lower=0>[N] instruction_sensitivity;
     
     for (i in 1:N) {
       alpha[i] = Phi_approx(mu[1]+ sigma[1]*alpha_raw[i]); //non-centered parameterisation of learning rate
       initV[i] = mu[2] + sigma[2]*init_raw[i];
       reward_sensitivity[i] = Phi_approx(mu[3] + sigma[3]*reward_sensitivity_raw[i])*5;
       punish_sensitivity[i] = Phi_approx(mu[4] + sigma[4]*punish_sensitivity_raw[i])*5;
       instruction_sensitivity[i] = Phi_approx(mu[5] + sigma[5]*instruction_sensitivity_raw[i])*5;
     }
}

model {

     mu ~ normal(0,1);
     sigma ~ cauchy(0,2.5);

     alpha_raw ~ std_normal();
     init_raw ~ std_normal();
     reward_sensitivity_raw ~ std_normal();
     punish_sensitivity_raw ~ std_normal();
     instruction_sensitivity_raw ~ std_normal();
     


     for (i in 1:N) {
             matrix [2,levels] v;
             real sens;

             v = [rep_row_vector(initV[i],levels),rep_row_vector(1-initV[i],levels)];

             for (t in 1:T) {
               vector [2] tempv;
               tempv = [v[1,congruence[i,t]],v[2,congruence[i,t]]]';
             	 choice[i,t] ~ categorical_logit(tempv + instruction_sensitivity[i] * correct[i,t]);
             	 //essentially this 2-reward part indexes the 1st inv temp for that participant if they were rewarded, and the second if not
             	 if (rwd[i,t]==1){
             	   sens=reward_sensitivity[i];
             	 } else {
             	   sens=punish_sensitivity[i];
             	 }
		           v[choice[i,t],congruence[i,t]] = v[choice[i,t],congruence[i,t]]+ alpha[i] * (sens * rwd[i,t]-v[choice[i,t],congruence[i,t]]); 
             }
             
     }
}
generated quantities {
      vector [N] log_lik;

        for (i in 1:N) {
                  matrix [2,levels] v;
                  real sens;
    
                  v = [rep_row_vector(initV[i],levels),rep_row_vector(1-initV[i],levels)];
                  log_lik[i] = 0;

                  for (t in 1:T) {
                    vector [2] tempv;
                    tempv = [v[1,congruence[i,t]],v[2,congruence[i,t]]]';
                    log_lik[i] += categorical_logit_lpmf( choice[i,t] | (tempv + instruction_sensitivity[i] * correct[i,t]));
                    if (rwd[i,t]==1){
                 	    sens=reward_sensitivity[i];
                 	  } else {
                 	    sens=punish_sensitivity[i];
                 	  }
                    v[choice[i,t],congruence[i,t]] = v[choice[i,t],congruence[i,t]]+ alpha[i] * (sens * rwd[i,t]-v[choice[i,t],congruence[i,t]]);
                  }
        }
}

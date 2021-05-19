/* Pizzagalli model 4 : 'Stimulus action w/sep sensitivities'
*  author: Alex Pike
*  email: alex.pike@ucl.ac.uk
*/


data {
     int<lower=1> N; 				//Number of subjects (strictly positive int)
     int<lower=1> T;  				//Number of trials (strictly positive int)
     int<lower=1> levels;     //Number of levels of congruence: set to 5

     int<lower=1,upper=2> choice[N,T]; 		 // Array of ints containing the choice made for each trial and participant (i.e. whether they chose left or right) — (rows: participants, columns: trials)
     int<lower=0,upper=1> accuracy[N,T]; //For whether they actually responded correctly (even if unrewarded)
     int<lower=0,upper=1> rwd[N,T];		//Matrix of integers containing the reward received on a given trial (1 or 0) — (rows: participants, columns : trials)
     int<lower=1,upper=levels> congruence[N,T]; //The congruence of the stimuli: should be integers from 1 to levels

     matrix[2,levels] Vinits;		//Matrix of reals containing the initial q-values for left and right for each congruence level - not used in this model;

}

parameters {
     real<lower=0> k_tau;
     real<lower=0> theta_tau;
     real mu_alpha;
     real<lower=0> sigma_alpha;
     real mu_inits[2,levels];
     real<lower=0> sigma_inits[2,levels];
     real k_instruction_sens;
     real<lower=0> theta_instruction_sens;

     matrix<lower=0, upper=6>[N,2] tau;
     vector[N] alpha_raw;
     real inits_raw [N, 2,levels]; //array of initial values - (rows: participants, columns: actions, 3rd dimension: congruence levels)
     vector<lower=0>[N] instruction_sens_raw;
}

transformed parameters {
     matrix<lower=0>[N,2] inv_temp;
     vector<lower=0,upper=1>[N] alpha;
     real initV [N, 2, levels];
     vector<lower=0>[N] instruction_sens;
     

     inv_temp = 1 ./ tau;
     instruction_sens = 1 ./ instruction_sens_raw;
     for (i in 1:N) {
       alpha[i] = Phi_approx(mu_alpha + sigma_alpha*alpha_raw[i]); //non-centered parameterisation of learning rate
       for (a in 1:2){
         for (b in 1:levels){
            initV[i,a,b] = mu_inits[a,b] + sigma_inits[a,b]*inits_raw[i,a,b];
         }
       }
     }
}

model {
     k_tau ~ normal(0.8,20);
     theta_tau ~ normal(1,20);
     k_instruction_sens ~ normal(0.8,20);
     theta_instruction_sens ~ normal(1,20);
     
     mu_alpha ~ normal(0,3);
     sigma_alpha ~ cauchy(0,5);
     for (a in 1:2){
       for (b in 1:levels){
         mu_inits[a,b] ~ normal(0,1);
         sigma_inits[a,b] ~ cauchy(0,5);
         inits_raw[,a,b] ~ std_normal();
       }
     }


     tau[,1] ~ gamma(k_tau,theta_tau);
     tau[,2] ~ gamma(k_tau,theta_tau);
     instruction_sens_raw ~ gamma(k_instruction_sens,theta_instruction_sens);
     
     alpha_raw ~ std_normal();

     
     

     for (i in 1:N) {
             real v [2,levels];

             v = initV[i,,];

             for (t in 1:T) {
               vector [2] tempv;
               tempv = [v[1,congruence[i,t]],v[2,congruence[i,t]]]';
             	 choice[i,t] ~ categorical_logit(tempv + instruction_sens[i] * accuracy[i,t]);
             	 //essentially this 2-reward part indexes the 1st inv temp for that participant if they were rewarded, and the second if not
		           v[choice[i,t],congruence[i,t]] = v[choice[i,t],congruence[i,t]]+ alpha[i] * (inv_temp[i,(2-rwd[i,t])] * rwd[i,t]-v[choice[i,t],congruence[i,t]]); 
             }
             
     }
}
generated quantities {
      real log_lik[N];

        for (i in 1:N) {
                  real v [2,levels];

                  v = initV[i,,];
                  log_lik[i] = 0;

                  for (t in 1:T) {
                    vector [2] tempv;
                    tempv = [v[1,congruence[i,t]],v[2,congruence[i,t]]]';
                    log_lik[i] += categorical_logit_lpmf( choice[i,t] | (tempv + instruction_sens[i] * accuracy[i,t]));
                    v[choice[i,t],congruence[i,t]] = v[choice[i,t],congruence[i,t]]+ alpha[i] * (inv_temp[i,2-rwd[i,t]] * rwd[i,t]-v[choice[i,t],congruence[i,t]]);
                  }
        }
}

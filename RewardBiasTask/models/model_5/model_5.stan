/* Pizzagalli model 5 : 'action only'
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
     real<lower=0, upper=20> theta_tau;
     real mu_alpha;
     real<lower=0> sigma_alpha;
     real mu_inits;
     real<lower=0> sigma_inits;
     real k_instruction_sens;
     real<lower=0> theta_instruction_sens;

     vector<lower=0, upper=6>[N] tau;
     vector[N] alpha_raw;
     matrix[N,2] inits_raw; //matrix of initial values - (rows: participants, columns: stimuli)
     vector<lower=0, upper=6>[N] instruction_sens_raw;
}

transformed parameters {
     vector<lower=0>[N] inv_temp;
     vector<lower=0,upper=1>[N] alpha;
     matrix[N,2] initV;
     vector<lower=0>[N] instruction_sens;
     

     inv_temp = 1 ./ tau;
     instruction_sens = 1 ./ instruction_sens_raw;
     for (i in 1:N) {
       alpha[i] = Phi_approx(mu_alpha + sigma_alpha*alpha_raw[i]); //non-centered parameterisation of learning rate
       initV[i] = mu_inits + sigma_inits*inits_raw[i];
     }
}

model {
     k_tau ~ normal(0.8,20);
     theta_tau ~ normal(1,20);
     k_instruction_sens ~ normal(0.8,20);
     theta_instruction_sens ~ normal(1,20);
     
     mu_alpha ~ normal(0,3);
     sigma_alpha ~ cauchy(0,5);
     mu_inits ~ normal(0,1);
     sigma_inits ~ cauchy(0,5);


     tau ~ gamma(k_tau,theta_tau);
     instruction_sens_raw ~ gamma(k_instruction_sens,theta_instruction_sens);
     
     alpha_raw ~ std_normal();
     inits_raw[,1] ~ std_normal(); //need two as a matrix, not vector
     inits_raw[,2] ~ std_normal();
     
     

     for (i in 1:N) {
             vector[2] v;

             v = initV[i]';

             for (t in 1:T) {
             		choice[i,t] ~ categorical_logit(v + instruction_sens[i] * accuracy[i,t]);
		            v[choice[i,t]] = v[choice[i,t]]- alpha[i] * (inv_temp[i] * rwd[i,t]-v[choice[i,t]]);
             }
             
     }
}
generated quantities {
      real log_lik[N];

        for (i in 1:N) {
                  vector[2] v;

                  v = initV[i]';
                  log_lik[i] = 0;

                  for (t in 1:T) {
                    log_lik[i] += categorical_logit_lpmf( choice[i,t] | (v + instruction_sens[i] * accuracy[i,t]));
                    v[choice[i,t]] = v[choice[i,t]]- alpha[i] * (inv_temp[i] * rwd[i,t]-v[choice[i,t]]);
                  }
        }
}

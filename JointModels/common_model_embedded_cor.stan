/* Common model with embedded correlations : 'Learning rate, initialisation value, temperature'
*  author: Alex Pike
*  email: alex.pike@ucl.ac.uk
*/


data {
  // common values
     int<lower=1> N; 				  //Number of participants (strictly positive int)
  
  // Pizzagalli task
     int<lower=1> reward_T;  				//Number of trials (strictly positive int)
     int<lower=1> levels;     //Number of levels of congruence: set to 5

     int<lower=1,upper=2> reward_choice[N,reward_T]; 		 // Array of ints containing the choice made for each trial and participant (i.e. whether they chose left or right) — (rows: participants, columns: trials)
     int<lower=0,upper=1> accuracy[N,reward_T]; //For whether they actually responded correctly (even if unrewarded)
     int<lower=-1,upper=1> rwd[N,reward_T];		//Matrix of integers containing the reward received on a given trial (1 or 0) — (rows: participants, columns : trials)
     int<lower=1,upper=levels> congruence[N,reward_T]; //The congruence of the stimuli: should be integers from 1 to levels

     //matrix[2,levels] Vinits;		//Matrix of reals containing the initial q-values for left and right for each congruence level - not used in this model;

  // Bandit task
     int<lower=1> bandit_T;  				//Number of trials (strictly positive int)
     int<lower=1, upper=bandit_T> Tsubj[N]; 		//Number of trials per subject (1D array of ints) — contains the max number of trials per subject
     int<lower=2> No; 				//Number of choice options in total (int) — set to 4
     int<lower=2> Nopt;				//Number of choice options per trial (int) — set to 4

     matrix[N,bandit_T] rew;		//Matrix of reals containing the reward received on a given trial (1 or 0) — (rows: participants, columns : trials)
     matrix[N,bandit_T] pun;		//Matrix of reals containing the penalty received on a given trial (-1 or 0) — (rows: participants, columns : trials)
     vector[No] Vinits;		//Vector or reals containing the initial q-values (set to [0, 0, 0, 0] for now);

     int<lower=1,upper=No> unchosen[No,No-1]; // Preset matrix that maps lists unchosen options from chosen one — set to [2, 3, 4; 1, 3, 4; 1, 2, 4; 1, 2, 3]
     int<lower=1,upper=No> bandit_choice[N,bandit_T]; 		 // Array of ints containing the reward_choice made for each trial and participant (i.e. option chosen out of 4) — (rows: participants, columns: trials)
}
}

transformed data {
     vector[2] correct[N,reward_T];
     for (i in 1:N){
       for (t in 1:reward_T){
              if (reward_choice[i,t]==1){
                      if (accuracy[i,t]==1){
                         correct[i,t]=[1, 0]';
                     } else if (accuracy[i,t]==0){
                         correct[i,t]=[0, 1]';
                     }
               } else if (reward_choice[i,t]==2){
                      if (accuracy[i,t]==1){
                         correct[i,t]=[0,1]';
                    } else if (accuracy[i,t]==0){
                         correct[i,t]=[1,0]';
                    }
               }
       }
     }
     
     vector[No] initV;
     initV = Vinits;
}

parameters {
     real mu_alpha;
     real<lower=0> sigma_alpha;
     real mu_init;
     real<lower=0> sigma_init;
     real<lower=0> k_reward_sensitivity;
     real<lower=0> theta_reward_sensitivity;
     real<lower=0> k_instruction_sensitivity;
     real<lower=0> theta_instruction_sensitivity;

     vector[N] alpha_raw;
     vector[N] init_raw; //array of initial values - (rows: participants, columns: actions, 3rd dimension: congruence levels)
     vector<lower=0>[N] reward_sensitivity_raw;
     vector<lower=0>[N] instruction_sensitivity_raw;
}

transformed parameters {
     vector<lower=0>[N] reward_sensitivity;
     vector<lower=0,upper=1>[N] alpha;
     vector[N] initV;
     vector<lower=0>[N] instruction_sensitivity;
     

     reward_sensitivity = 1 ./ reward_sensitivity_raw;
     instruction_sensitivity = 1 ./ instruction_sensitivity_raw;
     for (i in 1:N) {
       alpha[i] = Phi_approx(mu_alpha + sigma_alpha*alpha_raw[i]); //non-centered parameterisation of learning rate
       initV[i] = mu_init + sigma_init*init_raw[i];
     }
}

model {
     mu_alpha ~ normal(0,3);
     sigma_alpha ~ cauchy(0,5);
     mu_init ~ normal(0,3);
     sigma_init ~ cauchy(0,5);
     
     k_reward_sensitivity ~ normal(0.8,20);
     theta_reward_sensitivity ~ normal(1,20);
     k_instruction_sensitivity ~ normal(0.8,20);
     theta_instruction_sensitivity ~ normal(1,20);

     alpha_raw ~ std_normal();
     init_raw ~ std_normal();
     
     reward_sensitivity_raw ~ gamma(k_reward_sensitivity,theta_reward_sensitivity);
     instruction_sensitivity_raw ~ gamma(k_instruction_sensitivity,theta_instruction_sensitivity);


     for (i in 1:N) {
             matrix [2,levels] v;

             v = [rep_row_vector(initV[i],levels),rep_row_vector(1-initV[i],levels)];

             for (t in 1:reward_T) {
               vector [2] tempv;
               tempv = [v[1,congruence[i,t]],v[2,congruence[i,t]]]';
             	 reward_choice[i,t] ~ categorical_logit(tempv + instruction_sensitivity[i] * correct[i,t]);
		           v[reward_choice[i,t],congruence[i,t]] = v[reward_choice[i,t],congruence[i,t]]+ alpha[i] * (reward_sensitivity[i]* rwd[i,t]-v[reward_choice[i,t],congruence[i,t]]);
             }
             
     }
}
generated quantities {
      real log_lik[N];

        for (i in 1:N) {
                  matrix [2,levels] v;

                  v = [rep_row_vector(initV[i],levels),rep_row_vector(1-initV[i],levels)];
                  log_lik[i] = 0;

                  for (t in 1:reward_T) {
                    vector [2] tempv;
                    tempv = [v[1,congruence[i,t]],v[2,congruence[i,t]]]';
                    log_lik[i] = log_lik[i] + categorical_logit_lpmf( reward_choice[i,t] | (tempv + instruction_sensitivity[i] * correct[i,t]));
                    v[reward_choice[i,t],congruence[i,t]] = v[reward_choice[i,t],congruence[i,t]]+ alpha[i] * (reward_sensitivity[i] * rwd[i,t]-v[reward_choice[i,t],congruence[i,t]]);
                  }
        }
}

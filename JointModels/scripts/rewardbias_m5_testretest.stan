/* Pizzagalli model 5 : 'action only'
*  author: Alex Pike
*  email: alex.pike@ucl.ac.uk
*/

data {
  int<lower=1> N; 				//Number of subjects (strictly positive int)
  int<lower=1> T;  				//Number of trials (strictly positive int)
  int<lower=1> levels;     //Number of levels of congruence: set to 5
  int<lower=1> N_time;          // Number of timepoints
  array[N, N_time, T] int<lower=1,upper=2> choice; 		 // Array of ints containing the choice made for each trial and participant (i.e. whether they chose left or right) — (rows: participants, columns: trials)
  array[N, N_time, T] int<lower=0,upper=1> accuracy; //For whether they actually responded correctly (even if unrewarded)
  array[N, N_time, T] int<lower=-1,upper=1> rwd;		//Matrix of integers containing the reward received on a given trial (1 or 0) — (rows: participants, columns : trials)
  array[N, N_time, T] int<lower=1,upper=levels> congruence; //The congruence of the stimuli: should be integers from 1 to levels
  
  matrix[2,levels] Vinits;		//Matrix of reals containing the initial q-values for left and right for each congruence level - not used in this model;
}

transformed data {
     array[N, N_time, T] vector[2] correct;
     for (time in 1:N_time){
       for (i in 1:N){
         for (t in 1:T){
                if (choice[i, time, t]==1){
                        if (accuracy[i, time, t]==1){
                           correct[i, time,t]=[1, 0]';
                       } else if (accuracy[i, time, t]==0){
                           correct[i, time, t]=[0, 1]';
                       }
                 } else if (choice[i, time, t]==2){
                        if (accuracy[i, time, t]==1){
                           correct[i, time, t]=[0,1]';
                      } else if (accuracy[i, time, t]==0){
                           correct[i, time, t]=[1,0]';
                      }
                 }
         }
       }
     }
}

parameters {
  // Group-level correlation matrix (cholesky factor for faster computation)
  cholesky_factor_corr[N_time] L_R_alpha;
  cholesky_factor_corr[N_time] L_R_init;
  cholesky_factor_corr[N_time] L_R_i_sens; 
  cholesky_factor_corr[N_time] L_R_r_sens; 
  
  // Group-level parameter means
  vector[N_time] mu_alpha;
  vector[N_time] mu_init;
  vector[N_time] mu_i_sens;
  vector[N_time] mu_r_sens;
  
  // Group-level parameter SDs
  vector<lower=0>[N_time] sd_alpha;
  vector<lower=0>[N_time] sd_init;
  vector<lower=0>[N_time] sd_i_sens;
  vector<lower=0>[N_time] sd_r_sens;
  
  // Individual-level parameters (before being transformed)
  matrix[N_time,N] alpha_pr; 
  matrix[N_time,N] init_pr; 
  matrix[N_time,N] i_sens_pr;
  matrix[N_time,N] r_sens_pr;
}

transformed parameters {
  //Individual-level parameter off-sets (for non-centred parameterization)
  matrix[N_time,N] alpha_tilde;
  matrix[N_time,N] init_tilde;
  matrix[N_time,N] i_sens_tilde;
  matrix[N_time,N] r_sens_tilde;
  
  //Individual_level parameters
  matrix<lower=0, upper=1> [N,N_time] alpha;
  matrix [N,N_time] initV; 
  matrix<lower=0, upper=3> [N,N_time] i_sens;
  matrix<lower=0> [N,N_time] r_sens; 
  
  //Construct individual offsets (for non-centred parameterization)
  alpha_tilde = diag_pre_multiply(sd_alpha, L_R_alpha) * alpha_pr;
  init_tilde = diag_pre_multiply(sd_init, L_R_init) * init_pr;
  i_sens_tilde = diag_pre_multiply(sd_i_sens, L_R_i_sens) * i_sens_pr;
  r_sens_tilde = diag_pre_multiply(sd_r_sens, L_R_r_sens) * r_sens_pr;
  
  for (time in 1:N_time){
    for (i in 1:N) {
      alpha[i, time] = Phi_approx(mu_alpha[time] + alpha_tilde[time,i]);
      initV[i, time] = mu_init[time] + init_tilde[time,i];
      i_sens[i, time] = Phi_approx(mu_i_sens[time] + i_sens_tilde[time,i]) * 3;
      r_sens[i, time] = Phi_approx(mu_r_sens[time] + r_sens_tilde[time,i]) * 5;
    } // end of subj loop
  }
}

model {
  //Prior on cholesky factors of correlation matrix
  L_R_alpha ~ lkj_corr_cholesky(1);
  L_R_init ~ lkj_corr_cholesky(1);
  L_R_i_sens ~ lkj_corr_cholesky(1);
  L_R_r_sens ~ lkj_corr_cholesky(1);
  
  // Priors on group-level means
  mu_alpha  ~ normal(0, 1);
  mu_init  ~ normal(0, 1);
  mu_i_sens  ~ normal(0, 1);
  mu_r_sens  ~ normal(0, 1);
  
  //Priors on group level SDs
  sd_alpha ~ normal(0, 1);
  sd_init ~ normal(0, 1);
  sd_i_sens ~ normal(0, 1);
  sd_r_sens ~ normal(0, 1);
  
  // Individual parameters for non-centered parameterization
  to_vector(alpha_pr) ~ normal(0, 1);
  to_vector(init_pr) ~ normal(0, 1);
  to_vector(i_sens_pr) ~ normal(0, 1);
  to_vector(r_sens_pr) ~ normal(0, 1);
  
  for (time in 1:N_time){
    for (i in 1:N) {
      vector [2] v;
      
      v = [initV[i, time],(1-initV[i, time])]';
      
      for (t in 1:T) {
       	choice[i, time, t] ~ categorical_logit(v + i_sens[i, time] * correct[i, time, t]);
        v[choice[i, time, t]] = v[choice[i, time, t]]+ alpha[i, time] * (r_sens[i, time] * rwd[i, time, t]-v[choice[i, time, t]]);
      }
           
    }
  }
}
generated quantities {
  // test-retest correlations
  corr_matrix[N_time] R_alpha;
  corr_matrix[N_time] R_init;
  corr_matrix[N_time] R_i_sens;
  corr_matrix[N_time] R_r_sens;
  
  // Reconstruct correlation matrices from cholesky factor
  R_alpha = L_R_alpha * L_R_alpha';
  R_init = L_R_init * L_R_init';
  R_i_sens = L_R_i_sens * L_R_i_sens';
  R_r_sens = L_R_r_sens * L_R_r_sens';

  array[N, N_time] real log_lik;
  
  for (time in 1:N_time){
    for (i in 1:N) {
      vector [2] v;
      
      v = [initV[i, time],(1-initV[i, time])]';
      log_lik[i, time] = 0;
      
      for (t in 1:T) {
        log_lik[i, time] += categorical_logit_lpmf( choice[i, time, t] | (v + i_sens[i, time] * correct[i, time, t]));
        v[choice[i, time, t]] = v[choice[i, time, t]]+ alpha[i, time] * (r_sens[i, time] * rwd[i, time, t]-v[choice[i, time, t]]);
      }
    }
  }
}

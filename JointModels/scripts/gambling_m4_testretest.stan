// modified from Alex's DDM model

data {
  int<lower=1> N; // number of subjects
  int<lower=1> T; // number of trials 
  int<lower=1> N_time;          // Number of timepoints
  array[N, N_time] int<lower=1, upper=T> T_subj; // # of trials within subjects, timepoints
  array[N, N_time, T] real<lower=0> gain;
  array[N, N_time, T] real<lower=0> loss;  // absolute loss amount
  array[N, N_time, T] real cert;
  array[N, N_time, T] int<lower=-1, upper=1> gamble;
}
transformed data {
}
parameters {
  // Group-level correlation matrix (cholesky factor for faster computation)
  cholesky_factor_corr[N_time] L_R_rho;
  cholesky_factor_corr[N_time] L_R_lambda;
  cholesky_factor_corr[N_time] L_R_tau;  
  
  // Group-level parameter means
  vector[N_time] rho_mean;
  vector[N_time] lambda_mean;
  vector[N_time] tau_mean;
  
  // EDIT Group-level parameter SDs
  vector<lower=0>[N_time] rho_sd;
  vector<lower=0>[N_time] lambda_sd;
  vector<lower=0>[N_time] tau_sd;
  
  // EDIT Individual-level parameters (before being transformed)
  matrix[N_time,N] rho_pr; 
  matrix[N_time,N] lambda_pr; 
  matrix[N_time,N] tau_pr; 
}
transformed parameters {
  //Individual-level parameter off-sets (for non-centred parameterization)
  matrix[N_time,N] rho_tilde;
  matrix[N_time,N] lambda_tilde;
  matrix[N_time,N] tau_tilde;
  
  //Individual_level parameters
  matrix<lower=0, upper=2> [N,N_time] rho; 
  matrix<lower=0, upper=5> [N,N_time] lambda;
  matrix<lower=0, upper=30> [N,N_time] tau; 
  
  
  //Construct individual offsets (for non-centred parameterization)
  rho_tilde     = diag_pre_multiply(rho_sd, L_R_rho)       * rho_pr;
  lambda_tilde  = diag_pre_multiply(lambda_sd, L_R_lambda) * lambda_pr;
  tau_tilde     = diag_pre_multiply(tau_sd, L_R_tau)       * tau_pr;
  
  for (time in 1:N_time){
    for (i in 1:N) {
      rho[i, time]    = Phi_approx(rho_mean[time]     + rho_tilde[time,i]) *2;
      lambda[i, time] = Phi_approx(lambda_mean[time]  + lambda_tilde[time,i]) *5;
      tau[i, time]    = Phi_approx(tau_mean[time]     + tau_tilde[time,i]) *30;
    } // end of subj loop
  }
}
model {
  // ra_prospect: Original model in Soko-Hessner et al 2009 PNAS
  // hyper parameters
  
  //Prior on cholesky factors of correlation matrix
  L_R_rho    ~ lkj_corr_cholesky(1);
  L_R_lambda ~ lkj_corr_cholesky(1);
  L_R_tau    ~ lkj_corr_cholesky(1);
  
  // Priors on group-level means
  rho_mean     ~ normal(0, 1);
  lambda_mean  ~ normal(0, 1);
  tau_mean     ~ normal(0, 1);
  
  //Priors on group level SDs
  rho_sd    ~ normal(0, 1);
  lambda_sd ~ normal(0, 1);
  tau_sd    ~ normal(0, 1);
  
  // Individual parameters for non-centered parameterization
  to_vector(rho_pr)     ~ normal(0, 1);
  to_vector(lambda_pr)  ~ normal(0, 1);
  to_vector(tau_pr)     ~ normal(0, 1);
  
  // Begin time loop
  for (time in 1:N_time){
    for (i in 1:N) { // Begin subject loop
      for (t in 1:T_subj[i, time]) {
        real evSafe = 0.0;    // evSafe, evGamble, pGamble can be a scalar to save memory and increase speed.
        real evGamble = 0.0;  // they are left as arrays as an example for RL models.
        real pGamble = 0.0;
        
        if (cert[i, time, t] < 0){ // If loss trials only (sure option is negative)
          evSafe   = - (lambda[i, time] * pow(abs(cert[i, time, t]), 1.0)); // applies risk and loss aversion to negative sure option and negate
          evGamble = - 0.5 * lambda[i, time] * pow(abs(loss[i, time, t]), 1.0); //Gain is always Zero
        }
        if (cert[i, time, t] == 0){ // mixed gamble trials (sure option is exactly 0)
          evSafe   = pow(cert[i, time, t], 1.0); // could replace by 0;
          evGamble = 0.5 * (pow(gain[i, time, t], 1.0) - lambda[i, time] * pow(abs(loss[i, time, t]), 1.0));
        }
        if (cert[i, time, t] > 0) { // Gain only trials (sure option is positive)
          evSafe   = pow(cert[i, time, t], 1.0);
          evGamble = 0.5 * pow(gain[i, time, t], 1.0); //Loss is always 0
        }

        pGamble = inv_logit(tau[i, time] * (evGamble - evSafe));
        gamble[i, time, t] ~ bernoulli(pGamble);
      } //end of trial loop
    } // end of subject loop
  } // end of time loop
}
generated quantities {
  // test-retest correlations
  corr_matrix[N_time] R_rho;
  corr_matrix[N_time] R_lambda;
  corr_matrix[N_time] R_tau;
  
  // For log likelihood calculation
  array[N, N_time, T] int post_pred;
  array[N, N_time, T] real log_lik;
  
  // Reconstruct correlation matrices from cholesky factor
  R_rho     = L_R_rho    * L_R_rho';
  R_lambda  = L_R_lambda * L_R_lambda';
  R_tau     = L_R_tau    * L_R_tau';
  
  // EDIT initialize LL and post_pred arrays to -1
  for (i in 1:N) {
    post_pred[i,,] = rep_array(-1, N_time, T);
    log_lik[i,,] = rep_array(-1.0, N_time, T);
  }
  
  
  // For posterior predictive check
  //real y_pred[N, T];
  
  // Set all posterior predictions to 0 (avoids NULL values)
 // for (i in 1:N) {
 //   for (t in 1:T) {
 //     y_pred[i, t] = -1;
//    }
//  }
  

  
  { // local section, this saves time and space
  for (time in 1:N_time){
    for (i in 1:N) {
      
      for (t in 1:T_subj[i, time]) {
        real evSafe = 0.0;    // evSafe, evGamble, pGamble can be a scalar to save memory and increase speed.
        real evGamble = 0.0;  // they are left as arrays as an example for RL models.
        real pGamble = 0.0;
        //log_lik[i, time, t] = 0;
        
        if (cert[i, time, t] < 0){ // If loss trials only (sure option is negative)
          evSafe   = - (lambda[i, time] * pow(abs(cert[i, time, t]), 1.0)); // applies risk and loss aversion to negative sure option and negate
          evGamble = - 0.5 * lambda[i, time] * pow(abs(loss[i, time, t]), 1.0); //Gain is always Zero
        }
        if (cert[i, time, t] == 0){ // mixed gamble trials (sure option is exactly 0)
          evSafe   = pow(cert[i, time, t], 1.0); // could replace by 0;
          evGamble = 0.5 * (pow(gain[i, time, t], 1.0) - lambda[i, time] * pow(abs(loss[i, time, t]), 1.0));
        }
        if (cert[i, time, t] > 0) { // Gain only trials (sure option is positive)
          evSafe   = pow(cert[i, time, t], 1.0);
          evGamble = 0.5 * pow(gain[i, time, t], 1.0); //Loss is always 0
        }
        pGamble    = inv_logit(tau[i, time] * (evGamble - evSafe));
        
        log_lik[i, time, t] = bernoulli_lpmf(gamble[i, time, t] | pGamble);
        
        // EDIT generate posterior prediction for current trial
        post_pred[i, time, t] = bernoulli_rng(pGamble);
        
        // generate posterior prediction for current trial
        //y_pred[i, t] = bernoulli_rng(pGamble);
      }
    }
  }
  }
}



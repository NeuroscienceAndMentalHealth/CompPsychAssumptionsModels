/* AGT model 2 : Lin R + Lin E + Intercept
*  Author : vincent.valton@ucl.ac.uk
*/

data{
    int<lower=1> Ns; // number of subjects (strictly positive int)
    int<lower=0> Nx; // maximum number of trials (int)
    int<lower=1> Ni; // number of predictors (ignore for now and set to 1)
    int<lower=1> N_time; // number of timepoints - normally two for test-retest, could in theory have more
    array[Ns, N_time, Nx] int<lower=0,upper=1> y; // Responses (accept/refuse = 1 or 0) — 2D array of ints (rows: participant, columns: trials)
    array[Ns, N_time, Nx] real<lower=0> x_rwd; // Matrix of reals containing the reward level for each participant and trial — (rows: participant, column: trial)
    array[Ns, N_time, Nx] real<lower=0> x_eff;   // Matrix of reals containing the effort level for each participant and trial — (rows: participant, column: trial)
}

parameters{
	// Group-level correlation matrix (cholesky factor for faster computation)
  cholesky_factor_corr[N_time] L_R_I;
  cholesky_factor_corr[N_time] L_R_R;
  cholesky_factor_corr[N_time] L_R_E; 
  
  // Group-level parameter means
  vector[N_time] mu_I;
  vector[N_time] mu_R;
  vector[N_time] mu_E;
  
  // Group-level parameter SDs
  vector<lower=0>[N_time] sd_I;
  vector<lower=0>[N_time] sd_R;
  vector<lower=0>[N_time] sd_E;
  
  // Individual-level parameters (before being transformed)
  matrix[N_time,Ns] I_pr; 
  matrix[N_time,Ns] R_pr; 
  matrix[N_time,Ns] E_pr;
}

transformed parameters{
  //Individual-level parameter off-sets (for non-centred parameterization)
  matrix[N_time,Ns] I_tilde;
  matrix[N_time,Ns] R_tilde;
  matrix[N_time,Ns] E_tilde;
  
  //Individual_level parameters
  matrix [Ns,N_time] thetaI;
  matrix [Ns,N_time] thetaR; 
  matrix [Ns,N_time] thetaE; 
  
  //Construct individual offsets (for non-centred parameterization)
  I_tilde = diag_pre_multiply(sd_I, L_R_I) * I_pr;
  R_tilde = diag_pre_multiply(sd_R, L_R_R) * R_pr;
  E_tilde = diag_pre_multiply(sd_E, L_R_E) * E_pr;
  
  for (time in 1:N_time){
    for (i in 1:Ns) {
      thetaI[i, time] = mu_I[time]  + I_tilde[time,i];
      thetaR[i, time] = mu_R[time]  + R_tilde[time,i];
      thetaE[i, time] = mu_E[time]  + E_tilde[time,i];
    } // end of subj loop
  }
}

model{

	//Prior on cholesky factors of correlation matrix
  L_R_I ~ lkj_corr_cholesky(1);
  L_R_R ~ lkj_corr_cholesky(1);
  L_R_E ~ lkj_corr_cholesky(1);
  
  // Priors on group-level means
  mu_I  ~ normal(0, 10);
  mu_R  ~ normal(0, 10);
  mu_E  ~ normal(0, 10);
  
  //Priors on group level SDs
  sd_I ~ normal(0, 2.5);
  sd_R ~ normal(0, 2.5);
  sd_E ~ normal(0, 2.5);
  
  // Individual parameters for non-centered parameterization
  to_vector(I_pr) ~ normal(0, 1);
  to_vector(R_pr) ~ normal(0, 1);
  to_vector(E_pr) ~ normal(0, 1);
  
  // being loop
  for (time in 1:N_time){
  	for (i_subj in 1:Ns) {
  		for (i_trial in 1:Nx) {
  			y[i_subj, time, i_trial] ~ bernoulli_logit(thetaI[i_subj, time]
                                           + x_rwd[i_subj, time, i_trial]*thetaR[i_subj, time]
                                           + x_eff[i_subj, time, i_trial]*thetaE[i_subj, time]);
  		} // time loop
  	} // subj loop
  } // trial loop
}
generated quantities{
  // test-retest correlations
  corr_matrix[N_time] R_I;
  corr_matrix[N_time] R_R;
  corr_matrix[N_time] R_E;
  
  
  // Reconstruct correlation matrices from cholesky factor
  R_I = L_R_I * L_R_I';
  R_R = L_R_R * L_R_R';
  R_E = L_R_E * L_R_E';
  
  // initlaise log like array
  array[Ns, N_time] real log_lik;
  
  for (time in 1:N_time){
    for (i_subj in 1:Ns){
      log_lik[i_subj, time] = 0;
    }
  }
  
  for (time in 1:N_time){
    for (i_subj in 1:Ns){
      for (i_trial in 1:Nx) {
        log_lik[i_subj, time]=log_lik[i_subj,time]+bernoulli_logit_lpmf(y[i_subj, time, i_trial]|thetaI[i_subj, time]
                                                                               + x_rwd[i_subj, time, i_trial]*thetaR[i_subj, time]
                                                                               + x_eff[i_subj, time, i_trial]*thetaE[i_subj, time]);
      }
    }
  }
}

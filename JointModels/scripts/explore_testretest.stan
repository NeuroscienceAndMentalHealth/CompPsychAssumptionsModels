/* RL Explore model 4 : 'value difference (V) + sigma difference (RU) Only : UCB model only (Upper Confidence Bound : Changes Intercept)'
*  author: Vincent Valton
*  email: vincent.valton@ucl.ac.uk
*/


data {
  int<lower=1> N; 				            //Number of subjects (strictly positive int)
  int<lower=1> T;  				          //Number of trials (strictly positive int)
  int<lower=1> N_time;          // Number of timepoints
  array[N, N_time] int<lower=1, upper=T> Tsubj; 		//Number of trials per subject (1D array of ints) — contains the max number of trials per subject
  
  // V, RU and VTU need to be transformed: mutate_at(vars(starts_with('kalman')), scale) %>% # z transform
  array[N, N_time, T] real V;		  //Matrix of z-transformed reals containing the Kalman Value Difference (i.e. V) on that trial — (rows: participants, columns : trials)
  array[N, N_time, T] real RU;		//Matrix of z-transformed reals containing the Kalman Sigma Difference (i.e. RU) on that trial — (rows: participants, columns : trials)
  array[N, N_time, T] real VTU;		//Matrix of z-transformed reals containing the Kalman Value Difference (i.e. V/TU) on that trial — (rows: participants, columns : trials)
  
  array[N, N_time, T] int choice;   // Array of ints containing the choice made for each trial and participant (i.e. option chosen out of 2 : 0 or 1) — (rows: participants, columns: trials)
}

parameters {
  // Group-level correlation matrix (cholesky factor for faster computation)
  cholesky_factor_corr[N_time] L_R_V;
  cholesky_factor_corr[N_time] L_R_RU;  
  
  // Group-level parameter means
  vector[N_time] mu_V;
  vector[N_time] mu_RU;
  
  // EDIT Group-level parameter SDs
  vector<lower=0>[N_time] sd_V;
  vector<lower=0>[N_time] sd_RU;
  
  // EDIT Individual-level parameters (before being transformed)
  matrix[N_time,N] V_pr; 
  matrix[N_time,N] RU_pr; 
}

transformed parameters {
  //Individual-level parameter off-sets (for non-centred parameterization)
  matrix[N_time,N] V_tilde;
  matrix[N_time,N] RU_tilde;
  
  //Individual_level parameters
  matrix [N,N_time] theta_V;
  matrix [N,N_time] theta_RU; 
  
  //Construct individual offsets (for non-centred parameterization)
  V_tilde  = diag_pre_multiply(sd_V, L_R_V) * V_pr;
  RU_tilde     = diag_pre_multiply(sd_RU, L_R_RU) * RU_pr;
  
  for (time in 1:N_time){
    for (i in 1:N) {
      theta_V[i, time] = mu_V[time]  + V_tilde[time,i];
      theta_RU[i, time] = mu_RU[time] + RU_tilde[time,i];
    } // end of subj loop
  }
}

model {
  // hyper parameters
  
  //Prior on cholesky factors of correlation matrix
  L_R_V ~ lkj_corr_cholesky(1);
  L_R_RU    ~ lkj_corr_cholesky(1);
  
  // Priors on group-level means
  mu_V ~ normal(0, 1);
  mu_RU ~ normal(0, 1);
  
  //Priors on group level SDs
  sd_V ~ normal(0, 1);
  sd_RU ~ normal(0, 1);
  
  // Individual parameters for non-centered parameterization
  to_vector(V_pr) ~ normal(0, 1);
  to_vector(RU_pr) ~ normal(0, 1);
  
  for (time in 1:N_time){
    for (i in 1:N) {
      for (t in 1:(Tsubj[i, time])) {
       	choice[i, time, t] ~ bernoulli(Phi(  (theta_V[i, time] * V[i, time, t]) + (theta_RU[i, time] * RU[i, time, t]) ));
      }
    }
  }
}

generated quantities {
  
  // test-retest correlations
  corr_matrix[N_time] R_V;
  corr_matrix[N_time] R_RU;
  
  
  // Reconstruct correlation matrices from cholesky factor
  R_V = L_R_V * L_R_V';
  R_RU = L_R_RU * L_R_RU';

  array [N, N_time] real log_lik;
  
    for (time in 1:N_time){
      for (i in 1:N) {
        for (t in 1:(Tsubj[i, time])) {
          log_lik[i, time] = log_lik[i, time] + bernoulli_lpmf( choice[i, time, t] | Phi(  (theta_V[i, time] * V[i, time, t]) + (theta_RU[i, time] * RU[i, time, t]) ) );
        }
      }
    }
}

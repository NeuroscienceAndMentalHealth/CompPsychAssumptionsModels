/* RL 4arm-bandit model 5 : 'RL (Full) -- \tau + \alpha_Pos{Chosen} + \alpha_Neg{Chosen} + \alpha_Pos{Unchosen} + \alpha_Neg{Unchosen}'
*  author: Vincent Valton
*  email: vincent.valton@ucl.ac.uk
*/

data {
    int<lower=1> N; 				//Number of subjects (strictly positive int)
    int<lower=1> T;  				//Number of trials (strictly positive int)
    int<lower=1> N_time; // number of timepoints - normally two for test-retest, could in theory have more
    array[N, N_time] int<lower=1, upper=T> Tsubj; 		//Number of trials per subject (1D array of ints) — contains the max number of trials per subject
    int<lower=2> No; 				//Number of choice options in total (int) — set to 4
    int<lower=2> Nopt;				//Number of choice options per trial (int) — set to 4
    array[N, N_time, T] real rwd;		//Matrix of reals containing the reward received on a given trial (1 or 0) — (rows: participants, columns : trials)
    array[N, N_time, T] real plt;		//Matrix of reals containing the penalty received on a given trial (-1 or 0) — (rows: participants, columns : trials)
    vector[No] Vinits;		//Vector or reals containing the initial q-values (set to [0, 0, 0, 0] for now);
    array[No,No-1] int<lower=1,upper=No> unchosen; // Preset matrix that maps lists unchosen options from chosen one — set to [2, 3, 4; 1, 3, 4; 1, 2, 4; 1, 2, 3]
    array[N, N_time, T] int<lower=1,upper=No> choice; 		 // Array of ints containing the choice made for each trial and participant (i.e. option chosen out of 4) — (rows: participants, columns: trials)
}

transformed data {

    vector[No] initV;
    initV = Vinits;
}

parameters {
  
    // Group-level correlation matrix (cholesky factor for faster computation)
    cholesky_factor_corr[N_time] L_R_lrPosC;
    cholesky_factor_corr[N_time] L_R_lrPosU;
    cholesky_factor_corr[N_time] L_R_lrNegC;
    cholesky_factor_corr[N_time] L_R_lrNegU;
    cholesky_factor_corr[N_time] L_R_tau;
    
    // Group-level parameter means
    vector[N_time] lrPosC_mean;
    vector[N_time] lrPosU_mean;
    vector[N_time] lrNegC_mean;
    vector[N_time] lrNegU_mean;
    vector[N_time] tau_mean;
    
    // Group-level parameter SDs
    vector<lower=0>[N_time] lrPosC_sd;
    vector<lower=0>[N_time] lrPosU_sd;
    vector<lower=0>[N_time] lrNegC_sd;
    vector<lower=0>[N_time] lrNegU_sd;
    vector<lower=0>[N_time] tau_sd;
    
    // Individual-level parameters (before being transformed)
    matrix[N_time, N] lrPosC_pr;
    matrix[N_time, N] lrPosU_pr;
    matrix[N_time, N] lrNegC_pr;
    matrix[N_time, N] lrNegU_pr;
    matrix[N_time, N] tau_pr;
}

transformed parameters {
    
    // Individual-level parameter offsets
    matrix[N_time, N] lrPosC_tilde;
    matrix[N_time, N] lrPosU_tilde;
    matrix[N_time, N] lrNegC_tilde;
    matrix[N_time, N] lrNegU_tilde;
    matrix[N_time, N] tau_tilde;
      
    // Individual-level parameters 
    matrix<lower=0, upper=1>[ N, N_time] lrPosC;
    matrix<lower=0, upper=1>[N, N_time] lrPosU;
    matrix<lower=0, upper=1>[N, N_time] lrNegC;
    matrix<lower=0, upper=1>[N, N_time] lrNegU;
    matrix<lower=0>[N, N_time] tau;
    
    // Construct individual offsets (for non-centered parameterization)
    lrPosC_tilde = diag_pre_multiply(lrPosC_sd, L_R_lrPosC) * lrPosC_pr;
    lrPosU_tilde = diag_pre_multiply(lrPosU_sd, L_R_lrPosU) * lrPosU_pr;
    lrNegC_tilde = diag_pre_multiply(lrNegC_sd, L_R_lrNegC) * lrNegC_pr;
    lrNegU_tilde = diag_pre_multiply(lrNegU_sd, L_R_lrNegU) * lrNegU_pr;
    tau_tilde = diag_pre_multiply(tau_sd, L_R_tau) * tau_pr;
  
    // Compute individual-level parameters from non-centered parameterization
    for (time in 1:N_time){
      for (i in 1:N) {
        lrPosC[i, time] = Phi_approx(lrPosC_mean[time]  + lrPosC_tilde[time,i]);
        lrPosU[i, time] = Phi_approx(lrPosU_mean[time]  + lrPosU_tilde[time,i]);
        lrNegC[i, time] = Phi_approx(lrNegC_mean[time]  + lrNegC_tilde[time,i]);
        lrNegU[i, time] = Phi_approx(lrNegU_mean[time]  + lrNegU_tilde[time,i]);
        tau[i, time]    = exp(tau_mean[time] + tau_tilde[time,i]);
      }
    }
}

model {
     lrPosC_mean ~ normal(0,1);
     lrPosC_sd ~ normal(0,1);

     lrPosU_mean ~ normal(0,1);
     lrPosU_sd ~ normal(0,1);

     lrNegC_mean ~ normal(0,1);
     lrNegC_sd ~ normal(0,1);

     lrNegU_mean ~ normal(0,1);
     lrNegU_sd ~ normal(0,1);

     tau_mean ~ normal(0,1);
     tau_sd ~ normal(0,0.2);

     // Individual parameters for non-centered parameterization
     to_vector(lrPosC_pr) ~ normal(0, 1);
     to_vector(lrPosU_pr) ~ normal(0, 1);
     to_vector(lrNegC_pr) ~ normal(0, 1);
     to_vector(lrNegU_pr) ~ normal(0, 1);
     to_vector(tau_pr) ~ normal(0, 1);

     for (i in 1:N) {
       for (time in 1:N_time){
             vector[No] v_rwd;
             vector[No] v_plt;
             vector[No] v;
             vector[No] peR;
             vector[No] peP;

             v = initV;
             v_rwd = initV;
             v_plt = initV;
             
               for (t in 1:(Tsubj[i, time])) {
               		choice[i, time, t] ~ categorical_logit( tau[i, time] * v );
                            // Calculate PE for chosen option
                            peR[choice[i, time, t]] = rwd[i, time, t] - v_rwd[choice[i, time, t]];
                            peP[choice[i, time, t]] = -abs(plt[i, time, t]) - v_plt[choice[i, time, t]];
  
                            // Update values for chosen option based on sign of PE
                            if (peR[choice[i, time, t]] > 0) { //Positive PE use lrPos for Chosen
                                  v_rwd[choice[i, time, t]] = v_rwd[choice[i, time, t]] + lrPosC[i, time] * peR[choice[i, time, t]];
                            }
                            else { //Negative PE use lrNeg for Unchosen
                                  v_rwd[choice[i, time, t]] = v_rwd[choice[i, time, t]] + lrNegC[i, time] * peR[choice[i, time, t]];
                            }
                            if (peP[choice[i, time, t]] > 0) { //Positive PE use lrPos for Unchosen
                                  v_plt[choice[i, time, t]] = v_plt[choice[i, time, t]] + lrPosC[i, time] * peP[choice[i, time, t]];
                            }
                            else { //Negative PE use lrNeg for Unchosen
                                  v_plt[choice[i, time, t]] = v_plt[choice[i, time, t]] + lrNegC[i, time] * peP[choice[i, time, t]];
                            }
  
                            // Calculate PE for all unchosen options & update values
                            for (i_unchosen in 1:(No-1)) {
                                  peR[unchosen[choice[i, time, t],i_unchosen]] = 0.0 - v_rwd[unchosen[choice[i, time, t],i_unchosen]];
                                  peP[unchosen[choice[i, time, t],i_unchosen]] = 0.0 - v_plt[unchosen[choice[i, time, t],i_unchosen]];
  
                                  //update corresponding v_rwd & v_plt
                                  if (peR[unchosen[choice[i, time, t],i_unchosen]] > 0) { //Positive PE use lrPos for Unchosen
                                        v_rwd[unchosen[choice[i, time, t],i_unchosen]] = v_rwd[unchosen[choice[i, time, t],i_unchosen]] + lrPosU[i, time] * peR[unchosen[choice[i, time, t],i_unchosen]];
                                  }
                                  else { //Negative PE use lrNeg for Unchosen
                                        v_rwd[unchosen[choice[i, time, t],i_unchosen]] = v_rwd[unchosen[choice[i, time, t],i_unchosen]] + lrNegU[i, time] * peR[unchosen[choice[i, time, t],i_unchosen]];
                                  }
                                  if (peP[unchosen[choice[i, time, t],i_unchosen]] > 0) { //Positive PE use lrPos for Unchosen
                                        v_plt[unchosen[choice[i, time, t],i_unchosen]] = v_plt[unchosen[choice[i, time, t],i_unchosen]] + lrPosU[i, time] * peP[unchosen[choice[i, time, t],i_unchosen]];
                                  }
                                  else { //Negative PE use lrNeg for Unchosen
                                        v_plt[unchosen[choice[i, time, t],i_unchosen]] = v_plt[unchosen[choice[i, time, t],i_unchosen]] + lrNegU[i, time] * peP[unchosen[choice[i, time, t],i_unchosen]];
                                  }
                            }
  
                          // update value of all options (not just chosen)
                          v = v_rwd + v_plt;
               }
         }
     }
}
generated quantities {
  
      // test-retest correlations
      corr_matrix[N_time] R_lrPosC;
      corr_matrix[N_time] R_lrPosU;
      corr_matrix[N_time] R_lrNegC;
      corr_matrix[N_time] R_lrNegU;
      corr_matrix[N_time] R_tau;
      
      
      // Reconstruct correlation matrices from cholesky factor
      R_lrPosC = L_R_lrPosC * L_R_lrPosC';
      R_lrPosU = L_R_lrPosU * L_R_lrPosU';
      R_lrNegC = L_R_lrNegC * L_R_lrNegC';
      R_lrNegU = L_R_lrNegU * L_R_lrNegU';
      R_tau = L_R_tau * L_R_tau';
      
      array[N, N_time] real log_lik;

        for (i in 1:N) {
          for (time in 1:N_time){
          
                  vector[No] v_rwd;
                  vector[No] v_plt;
                  vector[No] v;
                  vector[No] peR;
                  vector[No] peP;

                  v = initV;
                  v_rwd = initV;
                  v_plt = initV;
                  log_lik[i, time] = 0;

                  for (t in 1:(Tsubj[i, time])) {
                    log_lik[i, time] = log_lik[i, time] + categorical_logit_lpmf( choice[i, time, t] | tau[i, time] * v );
                            // Calculate PE for chosen option
                            peR[choice[i, time, t]] = rwd[i, time, t] - v_rwd[choice[i, time, t]];
                            peP[choice[i, time, t]] = -abs(plt[i, time, t]) - v_plt[choice[i, time, t]];

                            // Update values for chosen option based on sign of PE
                            if (peR[choice[i, time, t]] > 0) { //Positive PE use lrPos for Chosen
                                  v_rwd[choice[i, time, t]] = v_rwd[choice[i, time, t]] + lrPosC[i, time] * peR[choice[i, time, t]];
                            }
                            else { //Negative PE use lrNeg for Unchosen
                                  v_rwd[choice[i, time, t]] = v_rwd[choice[i, time, t]] + lrNegC[i, time] * peR[choice[i, time, t]];
                            }
                            if (peP[choice[i, time, t]] > 0) { //Positive PE use lrPos for Unchosen
                                  v_plt[choice[i, time, t]] = v_plt[choice[i, time, t]] + lrPosC[i, time] * peP[choice[i, time, t]];
                            }
                            else { //Negative PE use lrNeg for Unchosen
                                  v_plt[choice[i, time, t]] = v_plt[choice[i, time, t]] + lrNegC[i, time] * peP[choice[i, time, t]];
                            }

                            // Calculate PE for all unchosen options & update values
                            for (i_unchosen in 1:(No-1)) {
                                  peR[unchosen[choice[i, time, t],i_unchosen]] = 0.0 - v_rwd[unchosen[choice[i, time, t],i_unchosen]];
                                  peP[unchosen[choice[i, time, t],i_unchosen]] = 0.0 - v_plt[unchosen[choice[i, time, t],i_unchosen]];

                                  //update corresponding v_rwd & v_plt
                                  if (peR[unchosen[choice[i, time, t],i_unchosen]] > 0) { //Positive PE use lrPos for Unchosen
                                          v_rwd[unchosen[choice[i, time, t],i_unchosen]] = v_rwd[unchosen[choice[i, time, t],i_unchosen]] + lrPosU[i, time] * peR[unchosen[choice[i, time, t],i_unchosen]];
                                  }
                                  else { //Negative PE use lrNeg for Unchosen
                                          v_rwd[unchosen[choice[i, time, t],i_unchosen]] = v_rwd[unchosen[choice[i, time, t],i_unchosen]] + lrNegU[i, time] * peR[unchosen[choice[i, time, t],i_unchosen]];
                                  }
                                  if (peP[unchosen[choice[i, time, t],i_unchosen]] > 0) { //Positive PE use lrPos for Unchosen
                                          v_plt[unchosen[choice[i, time, t],i_unchosen]] = v_plt[unchosen[choice[i, time, t],i_unchosen]] + lrPosU[i, time] * peP[unchosen[choice[i, time, t],i_unchosen]];
                                  }
                                  else { //Negative PE use lrNeg for Unchosen
                                          v_plt[unchosen[choice[i, time, t],i_unchosen]] = v_plt[unchosen[choice[i, time, t],i_unchosen]] + lrNegU[i, time] * peP[unchosen[choice[i, time, t],i_unchosen]];
                                  }
                            }

                    // update value of all options (not just chosen)
                    v = v_rwd + v_plt;
                }
          }
        }
}

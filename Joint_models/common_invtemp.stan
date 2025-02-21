/* Common model with embedded correlations based on separate winning models: gambling task and bandit task'
*  author: Alex Pike
*  email: alex.pike@york.ac.uk
*/

data {
  // common values
     int<lower=1> N; 				  //Number of participants (strictly positive int)
  
  // Gambling task
    int<lower=1> gamble_T; // Max number of trials (strictly positive int)
    array [N] int <lower=1, upper=gamble_T> gamble_Tsubj; // Max number of trials per participants (1D array of ints)
    array [N, gamble_T] int<lower=-1, upper=1> gamble; // 2D array of ints containing whether participant gambled or not (1, 0 respectively) on a given trial — (Row: participant, column: Trial)
    array [N, gamble_T] int cert; // 2D Array of reals containing value for the sure option for each participant and trial — (Row: participant, columns: trials)
    array [N, gamble_T] real<lower=0> gain; // 2D Array of reals containing value for the gain in the gamble for each participant and trial — (Row: participant, columns: trials)
    array [N, gamble_T] real<lower=0> loss; // 2D Array of reals containing value for the loss in the gamble for each participant and trial — (Row: participant, columns: trials)


  // Bandit task
     int<lower=1> bandit_T;  				//Number of trials (strictly positive int)
     array [N] int <lower=1, upper=bandit_T> bandit_Tsubj; 		//Number of trials per subject (1D array of ints) — contains the max number of trials per subject
     int<lower=2> No; 				//Number of reward_choice options in total (int) — set to 4
     int<lower=2> Nopt;				//Number of reward_choice options per trial (int) — set to 4

     matrix[N,bandit_T] rwd;		//Matrix of reals containing the reward received on a given trial (1 or 0) — (rows: participants, columns : trials)
     matrix[N,bandit_T] plt;		//Matrix of reals containing the penalty received on a given trial (-1 or 0) — (rows: participants, columns : trials)
     vector[No] Vinits;		//Vector or reals containing the initial q-values (set to [0, 0, 0, 0] for now);

     array [No, No-1] int<lower=1,upper=No> unchosen; // Preset matrix that maps lists unchosen options from chosen one — set to [2, 3, 4; 1, 3, 4; 1, 2, 4; 1, 2, 3]
     array [N, bandit_T] int<lower=1,upper=No> bandit_choice; // Array of ints containing the reward_reward_choice made for each trial and participant (i.e. option chosen out of 4) — (rows: participants, columns: trials)
}


transformed data {
     //Declare vars
     // Bandit 
     vector[No] initV; //Bandit
     initV = Vinits;

}

parameters {
  
    // Correlation
    // Group-level correlation matrix (cholesky factor for faster computation) - you don't need to know what this means, but you do need to define it! 
      cholesky_factor_corr[2] L_R_invtemp; 
  
     // Gambling
      real mu_p;
      real<lower=0> sigma;
      vector[N] lambda_nc;

     // Bandit
     real<lower=0> a_lrPosC;
     real<lower=0> b_lrPosC;
     real<lower=0> a_lrPosU;
     real<lower=0> b_lrPosU;
     real<lower=0> a_lrNegC;
     real<lower=0> b_lrNegC;
     real<lower=0> a_lrNegU;
     real<lower=0> b_lrNegU;

     vector<lower=0, upper=1>[N] lrPosC;
     vector<lower=0, upper=1>[N] lrPosU;
     vector<lower=0, upper=1>[N] lrNegC;
     vector<lower=0, upper=1>[N] lrNegU;

     // inv temp params
     vector [2] invtemp_mean;
     vector<lower=0> [2] invtemp_sd;
     matrix [2,N] invtemp_pr;
}

transformed parameters {
  //declare 
  vector<lower=0, upper=5>[N] lambda;
  
  // Individual-level parameter offsets
  matrix[2,N] invtemp_i_tilde;
  
  // Individual-level parameters 
  matrix[N,2] invtemp_i;

  // Construct individual offsets (for non-centered parameterization)
  invtemp_i_tilde = diag_pre_multiply(invtemp_sd, L_R_invtemp) *invtemp_pr;

  // Compute individual-level parameters from non-centered parameterization
  for (i in 1:N) {
    // Mean in task 1
    invtemp_i[i,1] = exp(invtemp_mean[1] + invtemp_i_tilde[1,i]); //as you can see, here we just add the group mean and group sd for time 1 to the z-scored individual difference bit for that participant, exp transform to make positive
    // Mean in task 2
    invtemp_i[i,2] = exp(invtemp_mean[2] + invtemp_i_tilde[2,i]);
    // Gamble lambda
    lambda[i] = Phi_approx(mu_p + sigma * lambda_nc[i]) * 5;
  }
}

model {
  // Prior on cholesky factor of correlation matrix - again, no need to know what this means, but you need this for each parameter you want to estimate a correlation matrix for. lkj_corr_cholesky is a particular type of prior stan lets you define
  L_R_invtemp    ~ lkj_corr_cholesky(1);
  
  // Inv temp priors
  invtemp_mean ~ normal (0, 1.0); //CHECK
  invtemp_sd ~ cauchy (0, 2.5); 
  to_vector(invtemp_pr) ~ normal(0, 1.0); 
  
  // Gamble pars
  mu_p  ~ normal(0, 1.0);
  sigma ~ cauchy (0, 2.5);

  lambda_nc ~ normal(0, 1.0);

  // Bandit pars
   a_lrPosC ~ normal(1,5);
   b_lrPosC ~ normal(1,5);
  
   a_lrPosU ~ normal(1,5);
   b_lrPosU ~ normal(1,5);
  
   a_lrNegC ~ normal(1,5);
   b_lrNegC ~ normal(1,5);
  
   a_lrNegU ~ normal(1,5);
   b_lrNegU ~ normal(1,5);
  
   lrPosC  ~ beta(a_lrPosC,b_lrPosC);
   lrPosU  ~ beta(a_lrPosU,b_lrPosU);
   lrNegC  ~ beta(a_lrNegC,b_lrNegC);
   lrNegU  ~ beta(a_lrNegU,b_lrNegU);

  for (i in 1:N) {
    
    //bandit var declarations
     vector[No] v_rwd;
     vector[No] v_plt;
     vector[No] v;
     vector[No] peR;
     vector[No] peP;

    
    //Gamble
    for (t in 1:gamble_Tsubj[i]) {
      real evSafe;
      real evGamble;
      real pGamble;

      if (cert[i,t] < 0){ // If loss trials only (sure option is negative)
        evSafe   = - (lambda[i] * pow(abs(cert[i, t]), 1.0)); // applies risk and loss aversion to negative sure option and negate
        evGamble = - 0.5 * lambda[i] * pow(abs(loss[i, t]), 1.0); //Gain is always Zero
      }
      if (cert[i,t] == 0){ // mixed gamble trials (sure option is exactly 0)
        evSafe   = pow(cert[i, t], 1.0); // could replace by 0;
        evGamble = 0.5 * (pow(gain[i, t], 1.0) - lambda[i] * pow(abs(loss[i, t]), 1.0));
      }
      if (cert[i,t] > 0) { // Gain only trials (sure option is positive)
        evSafe   = pow(cert[i, t], 1.0);
        evGamble = 0.5 * pow(gain[i, t], 1.0); //Loss is always 0
      }
      pGamble  = inv_logit(invtemp_i[i,1] * (evGamble - evSafe));
      gamble[i, t] ~ bernoulli(pGamble);
    }
    
    // Bandit
    v = initV;
    v_rwd = initV;
    v_plt = initV;
    for (t in 1:(bandit_Tsubj[i])) {
            bandit_choice[i,t] ~ categorical_logit(invtemp_i[i,2] * v );
            // Calculate PE for chosen option
            peR[bandit_choice[i,t]] = rwd[i,t] - v_rwd[bandit_choice[i,t]];
            peP[bandit_choice[i,t]] = -abs(plt[i,t]) - v_plt[bandit_choice[i,t]];

            // Update values for chosen option based on sign of PE
            if (peR[bandit_choice[i,t]] > 0) { //Positive PE use lrPos for Chosen
                  v_rwd[bandit_choice[i,t]] = v_rwd[bandit_choice[i,t]] + lrPosC[i] * peR[bandit_choice[i,t]];
            }
            else { //Negative PE use lrNeg for Unchosen
                  v_rwd[bandit_choice[i,t]] = v_rwd[bandit_choice[i,t]] + lrNegC[i] * peR[bandit_choice[i,t]];
            }
            if (peP[bandit_choice[i,t]] > 0) { //Positive PE use lrPos for Unchosen
                  v_plt[bandit_choice[i,t]] = v_plt[bandit_choice[i,t]] + lrPosC[i] * peP[bandit_choice[i,t]];
            }
            else { //Negative PE use lrNeg for Unchosen
                  v_plt[bandit_choice[i,t]] = v_plt[bandit_choice[i,t]] + lrNegC[i] * peP[bandit_choice[i,t]];
            }

            // Calculate PE for all unchosen options & update values
            for (i_unchosen in 1:(No-1)) {
                  peR[unchosen[bandit_choice[i,t],i_unchosen]] = 0.0 - v_rwd[unchosen[bandit_choice[i,t],i_unchosen]];
                  peP[unchosen[bandit_choice[i,t],i_unchosen]] = 0.0 - v_plt[unchosen[bandit_choice[i,t],i_unchosen]];

                  //update corresponding v_rwd & v_plt
                  if (peR[unchosen[bandit_choice[i,t],i_unchosen]] > 0) { //Positive PE use lrPos for Unchosen
                        v_rwd[unchosen[bandit_choice[i,t],i_unchosen]] = v_rwd[unchosen[bandit_choice[i,t],i_unchosen]] + lrPosU[i] * peR[unchosen[bandit_choice[i,t],i_unchosen]];
                  }
                  else { //Negative PE use lrNeg for Unchosen
                        v_rwd[unchosen[bandit_choice[i,t],i_unchosen]] = v_rwd[unchosen[bandit_choice[i,t],i_unchosen]] + lrNegU[i] * peR[unchosen[bandit_choice[i,t],i_unchosen]];
                  }
                  if (peP[unchosen[bandit_choice[i,t],i_unchosen]] > 0) { //Positive PE use lrPos for Unchosen
                        v_plt[unchosen[bandit_choice[i,t],i_unchosen]] = v_plt[unchosen[bandit_choice[i,t],i_unchosen]] + lrPosU[i] * peP[unchosen[bandit_choice[i,t],i_unchosen]];
                  }
                  else { //Negative PE use lrNeg for Unchosen
                        v_plt[unchosen[bandit_choice[i,t],i_unchosen]] = v_plt[unchosen[bandit_choice[i,t],i_unchosen]] + lrNegU[i] * peP[unchosen[bandit_choice[i,t],i_unchosen]];
                  }
            }

          // update value of all options (not just chosen)
          v = v_rwd + v_plt;
    }
  }
}

generated quantities {
    // test-retest correlations
    corr_matrix[2] R_invtemp;
    
    // Reconstruct correlation matrix from cholesky factor
    R_invtemp = L_R_invtemp * L_R_invtemp';
    
    
}

/* Common model with embedded correlations based on separate winning models: gambling task and explore task'
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


  // Gershman task 
     int<lower=1> explore_T;  				          //Number of trials (strictly positive int)
     array [N] int <lower=1, upper=explore_T> explore_Tsubj; 		//Number of trials per subject (1D array of ints) — contains the max number of trials per subject

     // V, RU and VTU need to be transformed: mutate_at(vars(starts_with('kalman')), scale) %>% # z transform
     matrix[N,explore_T] V;		  //Matrix of z-transformed reals containing the Kalman Value Difference (i.e. V) on that trial — (rows: participants, columns : trials)
     matrix[N,explore_T] RU;		//Matrix of z-transformed reals containing the Kalman Sigma Difference (i.e. RU) on that trial — (rows: participants, columns : trials)
     matrix[N,explore_T] VTU;		//Matrix of z-transformed reals containing the Kalman Value Difference (i.e. V/TU) on that trial — (rows: participants, columns : trials)

     array [N, explore_T] int<lower=0,upper=1> explore_choice;    // Array of ints containing the choice made for each trial and participant (i.e. option chosen out of 2 : 0 or 1) — (rows: participants, columns: trials)
}

parameters {
  
    // Correlation
    // Group-level correlation matrix (cholesky factor for faster computation) - you don't need to know what this means, but you do need to define it! 
      cholesky_factor_corr[2] L_R_invtemp; 
  
     // Gambling
      vector[3] mu_p;
      vector<lower=0>[3] sigma;
      vector[N] rho_g_nc;
      vector[N] rho_l_nc;
      vector[N] lambda_nc;

     // Explore
     // Group-level parameters
      real mu_explore;
      real<lower=0> sd_explore;

      vector[N] nc_theta_RU;

     // inv temp params
     vector [2] invtemp_mean;
     vector <lower=0> [2] invtemp_sd;
     matrix [2,N] invtemp_pr;
}

transformed parameters {
  //declare 
  vector<lower=0,upper=2>[N] rho_g;
  vector<lower=0,upper=2>[N] rho_l;
  vector<lower=0, upper=5>[N] lambda;
  vector[N] theta_RU;
  
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
    // Gamble rho_g
    rho_g[i] = Phi_approx(mu_p[1] + sigma[1] * rho_g_nc[i]) * 2;
    // Gamble rho_l
    rho_l[i] = Phi_approx(mu_p[2] + sigma[2] * rho_l_nc[i]) * 2;
    // Gamble lambda
    lambda[i] = Phi_approx(mu_p[3] + sigma[3] * lambda_nc[i]) * 5;
  }
  
    // Explore theta_RU (doesn't need to be in loop)
    theta_RU = mu_explore + sd_explore * nc_theta_RU;
}

model {
  // Prior on cholesky factor of correlation matrix - again, no need to know what this means, but you need this for each parameter you want to estimate a correlation matrix for. lkj_corr_cholesky is a particular type of prior stan lets you define
  L_R_invtemp    ~ lkj_corr_cholesky(1);
  
  // Inv temp priors
  invtemp_mean ~ normal (0,1); //CHECK
  invtemp_sd ~ normal (0,0.2); 
  to_vector(invtemp_pr) ~ normal(0,1); 
  
  // Gamble pars
  mu_p  ~ normal(0, 1.0);
  sigma ~ normal (0, 0.2);

  rho_g_nc ~ normal(0, 1.0);
  rho_l_nc ~ normal(0, 1.0);
  lambda_nc ~ normal(0, 1.0);

  // Explore pars
  mu_explore ~ normal(0,5);
  sd_explore ~ cauchy(0,2.5);

  to_vector(nc_theta_RU) ~ normal(0,1);

  for (i in 1:N) {
    
    //Gamble
    for (t in 1:gamble_Tsubj[i]) {
      real evSafe;
      real evGamble;
      real pGamble;

      if (cert[i,t] < 0){ // If loss trials only (sure option is negative)
        evSafe   = - (lambda[i] * pow(abs(cert[i, t]), rho_l[i])); // applies risk and loss aversion to negative sure option and negate
        evGamble = - 0.5 * lambda[i] * pow(abs(loss[i, t]), rho_l[i]); //Gain is always Zero
      }
      if (cert[i,t] == 0){ // mixed gamble trials (sure option is exactly 0)
        evSafe   = pow(cert[i, t], rho_g[i]); // could replace by 0;
        evGamble = 0.5 * (pow(gain[i, t], rho_g[i]) - lambda[i] * pow(abs(loss[i, t]), rho_l[i]));
      }
      if (cert[i,t] > 0) { // Gain only trials (sure option is positive)
        evSafe   = pow(cert[i, t], rho_g[i]);
        evGamble = 0.5 * pow(gain[i, t], rho_g[i]); //Loss is always 0
      }
      pGamble  = inv_logit(invtemp_i[i,1] * (evGamble - evSafe));
      gamble[i, t] ~ bernoulli(pGamble);
    }
    
    // Explore
       for (t in 1:(explore_Tsubj[i])) {
       		explore_choice[i,t] ~ bernoulli(Phi(  (-1*invtemp_i[i,2] * V[i,t]) + (theta_RU[i] * RU[i,t]) ));
       }
  }
}

generated quantities {
    // test-retest correlations
    corr_matrix[2] R_invtemp;
    
    // Reconstruct correlation matrix from cholesky factor
    R_invtemp = L_R_invtemp * L_R_invtemp';
    
    
}

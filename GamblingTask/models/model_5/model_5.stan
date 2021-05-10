/* Gambling model 5 :
*  Author : vincent.valton@ucl.ac.uk
* Prospect Theory model: 'Sokol Hesner' model with Temperature, Risk aversion (rho) for gain and loss separately, and Loss aversion (lambda)
*/

data {
  int<lower=1> N;// Number of subjects (strictly positive int)
  int<lower=1> T; // Max number of trials (strictly positive int)
  int<lower=1, upper=T> Tsubj[N]; // Max number of trials per participants (1D array of ints)
  int<lower=-1, upper=1> gamble[N, T]; // 2D array of ints containing whether participant gambled or not (1, 0 respectively) on a given trial — (Row: participant, column: Trial)
  real cert[N, T]; // 2D Array of reals containing value for the sure option for each participant and trial — (Row: participant, columns: trials)
  real<lower=0> gain[N, T]; // 2D Array of reals containing value for the gain in the gamble for each participant and trial — (Row: participant, columns: trials)
  real<lower=0> loss[N, T]; // 2D Array of reals containing value for the loss in the gamble for each participant and trial — (Row: participant, columns: trials)

}

parameters {
  vector[4] mu_p;
  vector<lower=0>[4] sigma;
  vector[N] rho_g_nc; //risk aversion in Gains domain
  vector[N] rho_l_nc; //risk aversion in Loss domain
  vector[N] lambda_nc;
  vector[N] tau_nc;
}

transformed parameters {
  vector<lower=0, upper=2>[N] rho_g;
  vector<lower=0, upper=2>[N] rho_l;
  vector<lower=0, upper=5>[N] lambda;
  vector<lower=0>[N] tau;

  for (i in 1:N) {
    rho_g[i]    = Phi_approx(mu_p[1] + sigma[1] * rho_g_nc[i]) * 2;
    rho_l[i]    = Phi_approx(mu_p[2] + sigma[2] * rho_l_nc[i]) * 2;
    lambda[i]   = Phi_approx(mu_p[3] + sigma[3] * lambda_nc[i]) * 5;
  }
  tau = exp(mu_p[4] + sigma[4] * tau_nc);
}

model {
  mu_p  ~ normal(0, 1.0);
  sigma ~ normal(0, 0.2);

  rho_g_nc  ~ normal(0, 1.0);
  rho_l_nc  ~ normal(0, 1.0);
  lambda_nc ~ normal(0, 1.0);
  tau_nc    ~ normal(0, 1.0);

  for (i in 1:N) {
    for (t in 1:Tsubj[i]) {
      real evSafe;
      real evGamble;
      real pGamble;

      if (cert[i,t] < 0){ // If loss trials only (sure option is negative)
        evSafe   = - (lambda[i] * pow(fabs(cert[i, t]), rho_l[i])); // applies risk and loss aversion to negative sure option and negate
        evGamble = - 0.5 * lambda[i] * pow(fabs(loss[i, t]), rho_l[i]); //Gain is always Zero
      }
      if (cert[i,t] == 0){ // mixed gamble trials (sure option is exactly 0)
        evSafe   = pow(cert[i, t], rho_g[i]); // could replace by 0;
        evGamble = 0.5 * (pow(gain[i, t], rho_g[i]) - lambda[i] * pow(fabs(loss[i, t]), rho_l[i]));
      }
      if (cert[i,t] > 0) { // Gain only trials (sure option is positive)
        evSafe   = pow(cert[i, t], rho_g[i]);
        evGamble = 0.5 * pow(gain[i, t], rho_g[i]); //Loss is always 0
      }
      pGamble  = inv_logit(tau[i] * (evGamble - evSafe));
      gamble[i, t] ~ bernoulli(pGamble);
    }
  }
}

generated quantities {
  real log_lik[N];
  // For posterior predictive check
  real y_pred[N, T];

  // Set all posterior predictions to 0 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      y_pred[i, t] = -1;
    }
  }

  { // local section, this saves time and space
    for (i in 1:N) {
      log_lik[i] = 0;
      for (t in 1:Tsubj[i]) {
        real evSafe;    // evSafe, evGamble, pGamble can be a scalar to save memory and increase speed.
        real evGamble;  // they are left as arrays as an example for RL models.
        real pGamble;

        if (cert[i,t] < 0){ // If loss trials only (sure option is negative)
          evSafe   = - (lambda[i] * pow(fabs(cert[i, t]), rho_l[i])); // applies risk and loss aversion to negative sure option and negate
          evGamble = - 0.5 * lambda[i] * pow(fabs(loss[i, t]), rho_l[i]); //Gain is always Zero
        }
        if (cert[i,t] == 0){ // mixed gamble trials (sure option is exactly 0)
          evSafe   = pow(cert[i, t], rho_g[i]); // could replace by 0;
          evGamble = 0.5 * (pow(gain[i, t], rho_g[i]) - lambda[i] * pow(fabs(loss[i, t]), rho_l[i]));
        }
        if (cert[i,t] > 0) { // Gain only trials (sure option is positive)
          evSafe   = pow(cert[i, t], rho_g[i]);
          evGamble = 0.5 * pow(gain[i, t], rho_g[i]); //Loss is always 0
        }
        pGamble    = inv_logit(tau[i] * (evGamble - evSafe));
        log_lik[i] = log_lik[i] + bernoulli_lpmf(gamble[i, t] | pGamble);

        // generate posterior prediction for current trial
        y_pred[i, t] = bernoulli_rng(pGamble);
      }
    }
  }
}

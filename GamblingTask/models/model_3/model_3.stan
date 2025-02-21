/* Gambling model 3 :
*  Author : vincent.valton@ucl.ac.uk
* 'no RA' model : Temperature + Loss aversion parameter (lambda), no risk aversion
*/


data {
  int<lower=1> N;// Number of subjects (strictly positive int)
  int<lower=1> T; // Max number of trials (strictly positive int)
  array [N] int <lower=1, upper=T> Tsubj; // Max number of trials per participants (1D array of ints)
  array [N, T] int<lower=-1, upper=1> gamble; // 2D array of ints containing whether participant gambled or not (1, 0 respectively) on a given trial — (Row: participant, column: Trial)
  array [N, T] int cert; // 2D Array of reals containing value for the sure option for each participant and trial — (Row: participant, columns: trials)
  array [N, T] real<lower=0> gain; // 2D Array of reals containing value for the gain in the gamble for each participant and trial — (Row: participant, columns: trials)
  array [N, T] real<lower=0> loss; // 2D Array of reals containing value for the loss in the gamble for each participant and trial — (Row: participant, columns: trials)

}

parameters {
  vector[2] mu_p;
  vector<lower=0>[2] sigma;
  vector[N] lambda_nc;
  vector[N] tau_nc;
}

transformed parameters {
  vector<lower=0, upper=5>[N] lambda;
  vector<lower=0>[N] tau;

  for (i in 1:N) {
    lambda[i] = Phi_approx(mu_p[1] + sigma[1] * lambda_nc[i]) * 5;
  }
  tau = exp(mu_p[2] + sigma[2] * tau_nc);
}

model {
  mu_p  ~ normal(0, 1.0);
  sigma ~ normal(0, 0.2);

  lambda_nc ~ normal(0, 1.0);
  tau_nc    ~ normal(0, 1.0);

  for (i in 1:N) {
    for (t in 1:Tsubj[i]) {
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
      pGamble  = inv_logit(tau[i] * (evGamble - evSafe));
      gamble[i, t] ~ bernoulli(pGamble);
    }
  }
}

generated quantities {
  vector [N] log_lik;

  array [N,T] real y_pred;

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

        pGamble    = inv_logit(tau[i] * (evGamble - evSafe));
        log_lik[i] = log_lik[i] + bernoulli_lpmf(gamble[i, t] | pGamble);

        // generate posterior prediction for current trial
        y_pred[i, t] = bernoulli_rng(pGamble);
      }
    }
  }
}

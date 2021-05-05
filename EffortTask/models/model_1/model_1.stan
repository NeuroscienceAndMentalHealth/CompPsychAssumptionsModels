/* AGT model 1 : Intercept Only
*  Author : vincent.valton@ucl.ac.uk
*/

data{
    int<lower=1> Ns; // number of subjects (strictly positive int)
    int<lower=0> Nx; // maximum number of trials (int)
    int<lower=1> Ni; // number of predictors (ignore for now and set to 1)
    int<lower=0,upper=1> y[Ns,Nx]; // Responses (accept/refuse = 1 or 0) — 2D array of ints (rows: participant, columns: trials)

    matrix<lower=0>[Ns,Nx] x_rwd; // Matrix of reals containing the reward level for each participant and trial — (rows: participant, column: trial)
    matrix<lower=0>[Ns,Nx] x_eff;   // Matrix of reals containing the effort level for each participant and trial — (rows: participant, column: trial)
}

parameters{
	// Group-level parameters
	real muI;
	real<lower=0> sdI;

	// Subject-level parameters
	vector[Ns] nc_thetaI;
}

transformed parameters{
  vector[Ns] thetaI;

  thetaI = muI + sdI * nc_thetaI;
}

model{
	muI ~ normal(0,10);
	sdI ~ cauchy(0,2.5);

	nc_thetaI ~ normal(0,1);

	for (i_subj in 1:Ns) {
		for (i_trial in 1:Nx) {
			y[i_subj,i_trial] ~ bernoulli_logit( thetaI[i_subj] );
		}
	}
}
generated quantities{
  real log_lik[Ns];

  for (i_subj in 1:Ns){
    log_lik[i_subj]=0;
    for (i_trial in 1:Nx) {
      log_lik[i_subj]=log_lik[i_subj]+bernoulli_logit_lpmf(y[i_subj,i_trial]|thetaI[i_subj]);
    }
  }
}

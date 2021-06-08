/* RL Explore model 4 : 'value difference (V) + sigma difference (RU) Only : UCB model only (Upper Confidence Bound : Changes Intercept)'
*  author: Vincent Valton
*  email: vincent.valton@ucl.ac.uk
*/


data {
     int<lower=1> N; 				            //Number of subjects (strictly positive int)
     int<lower=1> T;  				          //Number of trials (strictly positive int)
     int<lower=1, upper=T> Tsubj[N]; 		//Number of trials per subject (1D array of ints) — contains the max number of trials per subject

     // V, RU and VTU need to be transformed: mutate_at(vars(starts_with('kalman')), scale) %>% # z transform
     matrix[N,T] V;		  //Matrix of z-transformed reals containing the Kalman Value Difference (i.e. V) on that trial — (rows: participants, columns : trials)
     matrix[N,T] RU;		//Matrix of z-transformed reals containing the Kalman Sigma Difference (i.e. RU) on that trial — (rows: participants, columns : trials)
     matrix[N,T] VTU;		//Matrix of z-transformed reals containing the Kalman Value Difference (i.e. V/TU) on that trial — (rows: participants, columns : trials)

     int choice[N,T];   // Array of ints containing the choice made for each trial and participant (i.e. option chosen out of 2 : 0 or 1) — (rows: participants, columns: trials)
}

parameters {
  // Group-level parameters
 vector[2] mus;
 vector<lower=0>[2] sds;

 // Subject-level parameters
 vector[N] nc_theta_V;
 vector[N] nc_theta_RU;
}

transformed parameters {
     vector[N] theta_V;
     vector[N] theta_RU;

     theta_V = mus[1] + sds[1] * nc_theta_V;
     theta_RU = mus[2] + sds[2] * nc_theta_RU;
}

model {
    mus ~ normal(0,5);
    sds ~ cauchy(0,2.5);

    nc_theta_V ~ normal(0,1);
    nc_theta_RU ~ normal(0,1);

     for (i in 1:N) {
             for (t in 1:(Tsubj[i])) {
             		choice[i,t] ~ bernoulli(Phi(  (theta_V[i] * V[i,t]) + (theta_RU[i] * RU[i,t]) ));
             }
     }
}
generated quantities {
      real log_lik[N];

        for (i in 1:N) {
                  log_lik[i] = 0;

                  for (t in 1:(Tsubj[i])) {
                    log_lik[i] = log_lik[i] + bernoulli_lpmf( choice[i,t] | Phi(  (theta_V[i] * V[i,t]) + (theta_RU[i] * RU[i,t]) ) );
                  }
        }
}

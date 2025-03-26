/* RL Explore model 2 : 'Value Difference Only'
*  author: Vincent Valton
*  email: vincent.valton@ucl.ac.uk
*/


data {
     int<lower=1> N; 				            //Number of subjects (strictly positive int)
     int<lower=1> T;  				          //Number of trials (strictly positive int)
     array [N] int<lower=1, upper=T> Tsubj; 		//Number of trials per subject (1D array of ints) — contains the max number of trials per subject

     // V, RU and VTU need to be transformed: mutate_at(vars(starts_with('kalman')), scale) %>% # z transform
     matrix[N,T] V;		  //Matrix of z-transformed reals containing the Kalman Value Difference (i.e. V) on that trial — (rows: participants, columns : trials)
     matrix[N,T] RU;		//Matrix of z-transformed reals containing the Kalman Sigma Difference (i.e. RU) on that trial — (rows: participants, columns : trials)
     matrix[N,T] VTU;		//Matrix of z-transformed reals containing the Kalman Value Difference (i.e. V/TU) on that trial — (rows: participants, columns : trials)

     array[N,T] int choice;   // Array of ints containing the choice made for each trial and participant (i.e. option chosen out of 2 : 0 or 1) — (rows: participants, columns: trials)
}

parameters {
  // Group-level parameters
 real muV;
 real<lower=0> sdV;

 // Subject-level parameters
 vector[N] nc_theta_V;
}

transformed parameters {
     vector[N] theta_V;

     theta_V = muV + sdV * nc_theta_V;
}

model {
    muV ~ normal(0,5);
    sdV ~ cauchy(0,2.5);

    nc_theta_V ~ normal(0,1);

     for (i in 1:N) {
             for (t in 1:(Tsubj[i])) {
             		choice[i,t] ~ bernoulli(Phi(  (theta_V[i] * V[i,t])));
             }
     }
}
generated quantities {
      array [N] real log_lik;

        for (i in 1:N) {
                  log_lik[i]=0;

                  for (t in 1:(Tsubj[i])) {
                    log_lik[i] = log_lik[i] + bernoulli_lpmf( choice[i,t] | Phi(  (theta_V[i] * V[i,t])) );
                  }
        }
}

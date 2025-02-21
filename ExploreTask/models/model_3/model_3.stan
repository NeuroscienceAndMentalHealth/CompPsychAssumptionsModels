/* RL Explore model 3 : 'Weighted value difference (v/TU) Only : Thompson Sampling - Changes slope'
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
 real mu_vTU;
 real<lower=0> sd_vTU;

 // Subject-level parameters
 vector[N] nc_theta_vTU;
}

transformed parameters {
     vector[N] theta_vTU;

     theta_vTU = mu_vTU + sd_vTU * nc_theta_vTU;
}

model {
    mu_vTU ~ normal(0,5);
    sd_vTU ~ cauchy(0,2.5);

    nc_theta_vTU ~ normal(0,1);

     for (i in 1:N) {
             for (t in 1:(Tsubj[i])) {
             		choice[i,t] ~ bernoulli(Phi(  (theta_vTU[i] * VTU[i,t])  ));
             }
     }
}
generated quantities {
      array [N] real log_lik;

        for (i in 1:N) {
          
  
                  log_lik[i]=0;
  
                  for (t in 1:(Tsubj[i])) {
                    log_lik[i] = log_lik[i] + bernoulli_lpmf( choice[i,t] | Phi(  (theta_vTU[i] * VTU[i,t])  ) );
                  }
        }
}

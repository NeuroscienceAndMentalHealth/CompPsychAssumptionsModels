/* RL Explore model 1 : 'NULL RL -- \tau'
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
     real<lower=0> k_tau;
     real<lower=0, upper=20> theta_tau;

     vector<lower=0, upper=6>[N] tau;
}

transformed parameters {
     vector<lower=0>[N] inv_temp;

     inv_temp = 1 ./ tau;
}

model {
     k_tau ~ normal(0.8,20);
     theta_tau ~ normal(1,20);

     tau ~ gamma(k_tau,theta_tau);

     for (i in 1:N) {
             vector[2] v = [0.5,0.5]';

             for (t in 1:(Tsubj[i])) {
             		choice[i,t] ~ bernoulli_logit( inv_temp[i] * v );
             }
     }
}
generated quantities {
      array [N] real log_lik;

        for (i in 1:N) {
                  vector[2] v = [0.5, 0.5]';
                  log_lik[i] = 0;

                  for (t in 1:(Tsubj[i])) {
                    log_lik[i] = log_lik[i] + bernoulli_logit_lpmf( choice[i,t] | inv_temp[i] * v );
                  }
        }
}

/* Pizzagalli model 1 : 'NULL'
*  author: Alex Pike
*  email: alex.pike@ucl.ac.uk
*/

data {
     int<lower=1> N; 				//Number of subjects (strictly positive int)
     int<lower=1> T;  				//Number of trials (strictly positive int)
     int<lower=1> levels;     //Number of levels of congruence: set to 5

     int<lower=1,upper=2> choice[N,T]; 		 // Array of ints containing the choice made for each trial and participant (i.e. whether they chose left or right) — (rows: participants, columns: trials)
     int<lower=0,upper=1> accuracy[N,T]; //For whether they actually responded correctly (even if unrewarded)
     int<lower=-1,upper=1> rwd[N,T];		//Matrix of integers containing the reward received on a given trial (1 or 0) — (rows: participants, columns : trials)
     int<lower=1,upper=levels> congruence[N,T]; //The congruence of the stimuli: should be integers from 1 to levels

     matrix[2,levels] Vinits;		//Matrix of reals containing the initial q-values for left and right for each congruence level - set to 0 in this model;
}

transformed data {
     matrix[2,levels] initV;
     initV = Vinits;
}

parameters {
     real<lower=0> k_tau;
     real<lower=0> theta_tau;

     vector<lower=0>[N] tau;
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
             matrix [2,levels] v;

             v = initV;

             for (t in 1:T) {
                vector [2] tempv;
                tempv = [v[1,congruence[i,t]],v[2,congruence[i,t]]]';
             		choice[i,t] ~ categorical_logit( inv_temp[i] * tempv );
             }
     }
}
generated quantities {
      real log_lik[N];

        for (i in 1:N) {
                  matrix [2,levels] v;

                  v = initV;
                  log_lik[i] = 0;

                  for (t in 1:T) {
                    vector [2] tempv;
                    tempv = [v[1,congruence[i,t]],v[2,congruence[i,t]]]';
                    log_lik[i] = log_lik[i] + categorical_logit_lpmf( choice[i,t] | inv_temp[i] * tempv );
                  }
        }
}

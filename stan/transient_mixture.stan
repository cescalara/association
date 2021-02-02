/**
 * Model a mixture of 2 Gaussians
 * with time-varying weights.
 */

data {

  int<lower=1> N_obs;
  real y[N_obs];
  real t[N_obs];
}

transformed data {

  simplex[2] w[N_obs];

  for (i in 1:N_obs) {

    w[i][1] = t[i];

    w[i][2] = 1 - w[i][1];
    
  }
  
}

parameters {

  ordered[2] mu;
  real<lower=0> sigma;

}

model {

  /* likelihood */
  for (i in 1:N_obs) {
    target += log_mix( w[i][1], normal_lpdf(y[i] | mu[1], sigma), normal_lpdf(y[i] | mu[2], sigma) );
  }

}

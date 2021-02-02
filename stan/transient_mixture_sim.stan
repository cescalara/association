/**
 * Simulate draws from a mixture of 2 Gaussians
 * wuth time-varying weights.
 */

data {

  real mu[2];
  real sigma;

  int<lower=1> N_obs;
  vector[N_obs] t;  
}

transformed data {

  simplex[2] w[N_obs];

  for (i in 1:N_obs) {

    w[i][1] = t[i];

    w[i][2] = 1 - w[i][1];
          
  }
  
}

generated quantities {

  int<lower=1> lambda[N_obs];
  real y[N_obs];

  for (i in 1:N_obs) {
    lambda[i] = categorical_rng(w[i]);
    y[i] = normal_rng(mu[lambda[i]], sigma);
  }
  
}

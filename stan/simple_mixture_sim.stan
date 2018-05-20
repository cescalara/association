/**
 * Simulate draws from a mixture of 2 Gaussians
 */

data {

  real mu[2];
  real sigma;

  simplex[2] w;
  int<lower=1> N_obs;
  
}

generated quantities {

  int<lower=1> lambda[N_obs];
  real y[N_obs];

  for (i in 1:N_obs) {
    lambda[i] = categorical_rng(w);
    y[i] = normal_rng(mu[lambda[i]], sigma);
  }
  
}

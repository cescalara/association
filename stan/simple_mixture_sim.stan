/**
 * Simulate draws from a mixture of 2 Gaussians
 */

data {

  array[2] real mu;
  real sigma;

  simplex[2] w;
  int<lower=1> N_obs;
  
}

generated quantities {

  array[N_obs] int<lower=1> lambda;
  array[N_obs] real y;

  for (i in 1:N_obs) {
    lambda[i] = categorical_rng(w);
    y[i] = normal_rng(mu[lambda[i]], sigma);
  }
  
}

/**
 * Model a mixture of 2 Gaussians
 */

data {

  int<lower=1> N_obs;
  real y[N_obs];
  
}

parameters {

  simplex[2] w;
  ordered[2] mu;
  real<lower=0> sigma;

}

model {

  /* priors */
  // sigma ~ lognormal(4, 5);
  //  mu ~ normal(20, 10);

  /* likelihood */
  for (i in 1:N_obs) {
    target += log_mix(w[1], normal_lpdf(y[i] | mu[1], sigma), normal_lpdf(y[i] | mu[2], sigma) );
  }

}

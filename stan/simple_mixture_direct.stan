/**
 * Model a mixture of 2 Gaussians
 * Add in the necessary code to calculate 
 * the marginal posterior of the discrete 
 * labels, lambda
 */

data {

  int<lower=1> N_obs;
  array[N_obs] real y;
  
}

parameters {

  simplex[2] w;
  positive_ordered[2] mu;
  real<lower=0> sigma;

}

transformed parameters {

  array[N_obs] vector[2] lp;
  vector[N_obs] lpp;
  vector[2] log_w = log(w);
  

  /* likelihood */
  for (i in 1:N_obs) {
    lp[i] = log_w;
    for (k in 1:2) {
      lp[i, k] += normal_lpdf(y[i] | mu[k], sigma);
    }
  }

}

model {

  /* priors */
  sigma ~ lognormal(4, 5);
  mu ~ normal(20, 10);

  for (i in 1:N_obs) {
    target += log_sum_exp(lp[i]);
  }

}

generated quantities {

  array[N_obs] int<lower=1, upper=2> lambda;
  array[N_obs] real log_prob;
  
  for (i in 1:N_obs) {
    
    lambda[i] = categorical_logit_rng(lp[i]);
  }
  
}

/**
 * Model a mixture of 2 Gaussians
 * with time-varying weights.
 * Add in the necessary code to calculate 
 * the marginal posterior of the discrete 
 * labels, lambda
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

transformed parameters {

  vector[2] lp[N_obs];
  vector[N_obs] lpp;
  vector[2] log_w[N_obs] = log(w);
  

  /* likelihood */
  for (i in 1:N_obs) {
    lp[i] = log_w[i];
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

  int<lower=1, upper=2> lambda[N_obs];
  real log_prob[N_obs];
  
  for (i in 1:N_obs) {
    
    lambda[i] = categorical_logit_rng(lp[i]);
  }
}

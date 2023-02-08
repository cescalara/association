/**
 * Model a mixture of 2 Gaussians
 * Add in the necessary code to calculate 
 * the marginal posterior of the discrete 
 * labels, lambda
 * 
 * Parallel version using map_rect
 */

functions {

vector lp_reduce(vector global, vector local, array[] real real_data, array[] int int_data) {

  int N = int_data[1]; 
  array[N] real y;
  vector[2] log_w;
  vector[2] mu; 
  real sigma;
  array[N] vector[2] lp;
  array[N] real results;

  y = real_data;
  log_w = global[1:2];
  mu = global[3:4];
  sigma = global[5];
  
  for (i in 1:N) {

    for (k in 1:2) {

      lp[i][k] = log_w[k];
      lp[i][k] += normal_lpdf(y[i] | mu[k], sigma);

    }

  }

  for (i in 1:N) {
    results[i] = log_sum_exp(lp[i]);
  }

  return [sum(results)]';
}

}

data {

  int<lower=1> N_obs;
  array[N_obs] real y;
  int<lower=1> N_shards;
  int<lower=1> M;
  
}

transformed data {
 
  array[N_shards, 1] int int_data;
  array[N_shards, M] real real_data;

  int<lower=1> start;
  int insert_start;
  int<lower=1> end;
  int<lower=1> insert_len;

  for (i in 1:N_shards) {

    start = M * (i-1) + 1;
    insert_start = 1;
    
    if ((i != N_shards) || (N_obs%M == 0)) {

      end = i * M;
      insert_len = M;

    }
    else {

      end = (start - 1) + N_obs%M;
      insert_len = N_obs%M;

    }

    int_data[i, 1] = insert_len;
    real_data[i, insert_start:insert_len] = to_array_1d(y[start:end]);

  }

}

parameters {

  simplex[2] w;
  ordered[2] mu;
  real<lower=0> sigma;

}

transformed parameters {

  vector[2] log_w = log(w);
  vector[5] global_pars;
  array[N_shards] vector[0] local_pars;

  global_pars[1:2] = log_w;
  global_pars[3:4] = mu;
  global_pars[5] = sigma; 

}

model {

  /* priors */
  sigma ~ lognormal(4, 5);
  mu ~ normal(20, 10);

  target += sum(map_rect(lp_reduce, global_pars, local_pars, real_data, int_data));

}

generated quantities {

  array[N_obs] int<lower=1, upper=2> lambda;
  array[N_obs] vector[2] lp;
  
  for (i in 1:N_obs) {

    for (k in 1:2) {

      lp[i][k] = log_w[k];
      lp[i][k] += normal_lpdf(y[i] | mu[k], sigma);

    }

    lambda[i] = categorical_logit_rng(lp[i]);

  }
    
}

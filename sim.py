import numpy as np
from scipy.special import loggamma, logsumexp
from random import randint, random
import json


# coefficients of multinomial distribution
def log_mult_coeff(N_pop, N_pos, N_neg):
    num = loggamma(N_pop+1)
    den = loggamma(N_pop-N_pos-N_neg+1) + loggamma(N_pos+1) + loggamma(N_neg+1)
    return num - den 

# uniform prior
def log_covid_prior(p):
    return 0.

# log likelihood (multinomial) of N_pos, N_neg covid counts (not observed)
def log_likelihood(p_flu, p_covid, N_pop, N_pos, N_neg):
    c = log_mult_coeff(N_pop, N_pos, N_neg)
    N_not_tested = N_pop - N_pos - N_neg
    l1 = N_not_tested * ( np.log(1-p_flu)    + np.log(1-p_covid) )
    l2 = N_neg        * ( np.log(1-p_covid)  + np.log(p_flu) )
    l3 = N_pos        * ( np.log(p_covid) )
    out = c + (l1 + l2 + l3)
    return out

# log likelihood given that number of symtomps is at least the number of tests
# (marginal multinomial)
def log_likelihood_given_data(p_flu, p_covid,
                              N_pop,
                              N_pos_observed,
                              N_neg_observed):

    log_likes = []
    for n_pos in range(N_pos_observed,N_pop+1):
        for n_neg in range(N_neg_observed,N_pop+1):
            if (N_pop - n_pos - n_neg) >= 0:
                log_likes.append(log_likelihood(p_flu, 
                                           p_covid, 
                                           N_pop, 
                                           n_pos,
                                           n_neg))
        
    out = logsumexp(log_likes)
    return out


# conditioning on flu proportion
def flu_conditional(flu_mean, p_covid,
                 N_pop, 
                 N_pos_observed,
                 N_neg_observed):
        
    log_like = log_likelihood_given_data(flu_mean, 
                                         p_covid, 
                                         N_pop, 
                                         N_pos_observed, 
                                         N_neg_observed)
            
    return log_like + log_covid_prior(p_covid)

# posterior inference over COVID-19 proportion
def posterior(N_pop, N_pos_observed, N_neg_observed, flu_mean):
    log_post = [ flu_conditional(flu_mean, p, N_pop, N_pos_observed, N_neg_observed ) for p in PI_SUPPORT ]

    m = np.exp(log_post)
    m = np.array(m)/max(m)
    return m


if __name__ == '__main__':
    EXPERIMENT_NAME = 'simulation'

    # set this to True to reproduce NYC example in post
    NYC_EXAMPLE = False 
    EXAMPLE = 4

    if NYC_EXAMPLE:
        # use smaller grid for NYC example as it's slow
        PI_SUPPORT = np.linspace(0.0001,0.99,25)
    else:
        PI_SUPPORT = np.linspace(0.0001,0.99,100)

    if NYC_EXAMPLE:
        # NYC approximate numbers
        # 26k tested, 6k positive. Scale down by factor of 3000
        N_pop          = 1333 * 1 # 8 million (~NYC pop) / 3000
        N_pos_observed = 1 * 1  # 6000/3000 = 2
        N_neg_observed = 3 * 2 #  floor((26000 - 6000)/3000) = 6 # flooring to over-estimate COVID-19 proportion
    elif EXAMPLE == 1:
        N_pop          = 100
        N_pos_observed = 0
        N_neg_observed = 10
    elif EXAMPLE == 2:
        N_pop          = 100
        N_pos_observed = 10
        N_neg_observed = 0
    elif EXAMPLE == 3:
        N_pop          = 100
        N_pos_observed = 10
        N_neg_observed = 1
    elif EXAMPLE == 4:
        N_pop          = 100
        N_pos_observed = 10
        N_neg_observed = 2
    elif EXAMPLE == 5:
        N_pop          = 100
        N_pos_observed = 40
        N_neg_observed = 10
    else:
        print('Set either NYC_EXAMPLE to True or set EXAMPLE number between 1 to 5 inclusive')
        exit()

    if NYC_EXAMPLE:
        real_flu_prop  = 0.006 # ~0.6% have flu in a period of 1 month
        small_flu_prop = 0.0000001 # near 0 flu rate for comparison
    else:
        real_flu_prop  = 0.10 # toy examples use 10% of flu proportion to emphasize differences
        small_flu_prop = 0.0000001 # near 0 flu rate for comparison

    #post = posterior(N_pop, N_pos_observed, N_neg_observed, log_flu_prior, flu_mean = small_flu_prop)
    print('Running with flu...')
    post_with_flu    = posterior(N_pop, N_pos_observed, N_neg_observed, flu_mean = real_flu_prop)
    print('Running without flu...')
    post_without_flu = posterior(N_pop, N_pos_observed, N_neg_observed, flu_mean = small_flu_prop)

    with open('%s_with_flu.json' % EXPERIMENT_NAME, 'w') as fout:
        data = { 'range' : list(PI_SUPPORT), 'post' : list(post_with_flu) }
        fout.write(json.dumps(data))

    with open('%s_without_flu.json' % EXPERIMENT_NAME, 'w') as fout:
        data = { 'range' : list(PI_SUPPORT), 'post' : list(post_without_flu) }
        fout.write(json.dumps(data))

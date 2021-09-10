#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:55:39 2019

corr_plumes.py

Generates couple of correlated time series having the same statistical feature  
of real plumes detected in open space.

This function creates two stimuli with a given average values of 
concentration aver(C), with both distributions of whiff and blanks having a 
power law decay with a power of -3/2, with a distribution with a concentration 
values as an exponential C. The two stimuli are correlated .


see also stats_for_plumes.py

@author: mp525
"""



import numpy as np
from scipy.stats import norm
import stats_for_plumes as stats


def main(corr_plumes_params):
    
    # ***************************************************************************
    # PARAMS FOR STIMULI GENERATION

    t2sim           = corr_plumes_params['t2sim']  # [s] time duration of whole simulated stimulus
    sample_rate     = corr_plumes_params['sample_rate']  # [Hz] num of samples per each sec
    n_sample2       = corr_plumes_params['n_sample2']  # [ms] num of samples with constant concentration
    tot_n_samples   = int(t2sim*sample_rate) # [] duration of whole simulated stimulus in number of samples
    
    #  PARAMS FOR WHIFF AND BLANK DISTRIOBUTIONS
    g               = corr_plumes_params['g']  # -1/2      # for a power law of -3/2
    whiff_min       = corr_plumes_params['whiff_min']  # 3e-3      # [s]
    whiff_max       = corr_plumes_params['whiff_max']  # 3        # [s] 3, 50,150
    
    blank_min       = corr_plumes_params['blank_min']  # 3e-3      # [s]
    blank_max       = corr_plumes_params['blank_max']  # 25        # [s]  25, 35
    
    # PARAMS FOR CONCENTRATION DISTRIBUTIONS
    # fit of average concentration at 75 m, Mylne and Mason 1991, Fig.10 
    b1              = corr_plumes_params['b_conc']  # -(np.log10(1-.997) + np.log10(1-.5))/10.7
    a1              = corr_plumes_params['a_conc']  # -0.3*b1 - np.log10(1-.5)

    rho_c           = corr_plumes_params['rho_c']  # 
    rho_t           = corr_plumes_params['rho_t']  # 
    
    quenched        = corr_plumes_params['quenched']
    seed_num        = corr_plumes_params['seed_num']
    
    
    # *********************************************************
    # GENERATE CORRELATED BLANKS AND WHIFFS
    if not(np.isnan(seed_num)):
        np.random.seed(seed=seed_num)
    
    # WARNING: 
    ratio_min = 2 #130*(whiff_max*blank_max)**0.1/(25*3)
    # I looked for a 'ratio_min' to have almost the double of needed sample points
    #    print(int(t2sim*sample_rate/ratio_min))

    n1 = np.random.normal(size=int(t2sim*sample_rate/ratio_min)) 
    n2 = np.random.normal(size=int(t2sim*sample_rate/ratio_min))
    n1c = rho_t*n1 + np.sqrt(1-rho_t**2)*n2        
    u1 = norm.cdf(n1)
    u1c = norm.cdf(n1c)
    
    t_bls = stats.rnd_pow_law(blank_min, blank_max, g, u1)
    t_bls_cor = stats.rnd_pow_law(blank_min, blank_max, g, u1c)

    n1 = np.random.normal(size=int(t2sim*sample_rate/ratio_min))
    n2 = np.random.normal(size=int(t2sim*sample_rate/ratio_min))
    n1c = rho_t*n1 + np.sqrt(1-rho_t**2)*n2        
    u1 = norm.cdf(n1)
    u1c = norm.cdf(n1c)
    
    t_whs = stats.rnd_pow_law(whiff_min, whiff_max, g, u1)
    t_whs_cor = stats.rnd_pow_law(whiff_min, whiff_max, g, u1c)
    
    # *********************************************************
    if np.minimum(np.sum(t_bls)+np.sum(t_whs), 
                  np.sum(t_bls_cor)+np.sum(t_whs_cor)) < t2sim*1.05:
        print('Start over whiffs and blanks generation: '+ 
            'the shorter stimulus is too short!')
        
        
        # *********************************************************
        # GENERATE CORRELATED BLANKS AND WHIFFS (slow, secure way)
        t_bl_tot    = 0
        t_bls       = np.zeros(1)
        t_bls_cor   = np.zeros(1)    
        
        while t_bl_tot < t2sim:
            n1 = np.random.normal()
            n2 = np.random.normal()
            n1c = rho_t*n1 + np.sqrt(1-rho_t**2)*n2        
            u1 = norm.cdf(n1)
            u1c = norm.cdf(n1c)
            
            tmp_t = stats.rnd_pow_law(blank_min, blank_max, g, u1)
            tmp_tc = stats.rnd_pow_law(blank_min, blank_max, g, u1c)
            t_bls = np.append(t_bls, tmp_t)
            t_bls_cor = np.append(t_bls_cor, tmp_tc)
            t_bl_tot = np.min([np.sum(t_bls), np.sum(t_bls_cor)])
            
        t_wh_tot    = 0
        t_whs       = np.zeros(1)
        t_whs_cor   = np.zeros(1)    
        
        while t_wh_tot < t2sim:
            n1 = np.random.normal()
            n2 = np.random.normal()
            n1c = rho_t*n1 + np.sqrt(1-rho_t**2)*n2        
            u1 = norm.cdf(n1)
            u1c = norm.cdf(n1c)
            
            tmp_t = stats.rnd_pow_law(whiff_min, whiff_max, g, u1)
            tmp_tc = stats.rnd_pow_law(whiff_min, whiff_max, g, u1c)
            t_whs = np.append(t_whs, tmp_t)
            t_whs_cor = np.append(t_whs_cor, tmp_tc)
            t_wh_tot = np.min([np.sum(t_whs), np.sum(t_whs_cor)])

    
    # *********************************************************
    # GENERATE CORRELATED CONCENTRATION VALUES
    concs       = np.zeros(tot_n_samples)
    concs_cor   = np.zeros(tot_n_samples)    
    
    n1 = np.random.normal(size=int(tot_n_samples/n_sample2))
    n2 = np.random.normal(size=int(tot_n_samples/n_sample2))
    n1c = rho_c*n1 + np.sqrt(1-rho_c**2)*n2        
    u1 = norm.cdf(n1)
    u1c = norm.cdf(n1c)
    
    tmp_c = stats.rnd_mylne_75m(a1, b1, u1)
    tmp_cc = stats.rnd_mylne_75m(a1, b1, u1c)
    for nn in range(int(tot_n_samples/n_sample2)): 
        concs[nn*n_sample2:(nn+1)*n_sample2]        =  tmp_c[nn]
        concs_cor[nn*n_sample2:(nn+1)*n_sample2]    =  tmp_cc[nn]
    
    
    
    # *********************************************************
    # SET TO ZERO DURING BLANKS THE CONCENTRATION VALUES
    nn1      = 0
    nn2      = 0
    exc_cnt     = 0
    
    while nn1 < tot_n_samples:    
        t_bl1        = t_bls[exc_cnt] # draw a blank 
        t_wh1        = t_whs[exc_cnt] # draw a whiff 
        
        t_bl2        = t_bls_cor[exc_cnt] # draw a blank 
        t_wh2        = t_whs_cor[exc_cnt] # draw a whiff 
        
        if quenched:    
            if nn1>nn2:
                t_bl = min(t_bl1, t_bl2)
                t_wh = min(t_wh1, t_wh2)
                t_bl_cor = max(t_bl1, t_bl2)
                t_wh_cor = max(t_wh1, t_wh2)
            else:
                t_bl = max(t_bl1, t_bl2)
                t_wh = max(t_wh1, t_wh2)
                t_bl_cor = min(t_bl1, t_bl2)
                t_wh_cor = min(t_wh1, t_wh2)
        else:
            t_bl = t_bl1
            t_wh = t_wh1
            t_bl_cor = t_bl2
            t_wh_cor = t_wh2

            
        #******************************************************
        # Reintroduce the values of blanks and whiffs in their correct series
        t_bls[exc_cnt] = t_bl# draw a blank 
        t_whs[exc_cnt] = t_wh # draw a whiff 
        
        t_bls_cor[exc_cnt] = t_bl_cor # draw a blank 
        t_whs_cor[exc_cnt] = t_wh_cor # draw a whiff 
        #******************************************************
                
        concs[nn1:nn1 + int(t_bl*sample_rate)] = 0
        concs_cor[nn2:nn2 + int(t_bl_cor*sample_rate)] = 0                
                
        nn1     = nn1 + int(t_bl*sample_rate + t_wh*sample_rate)
        nn2     = nn2 + int(t_bl_cor*sample_rate + t_wh_cor*sample_rate)
        exc_cnt = exc_cnt + 1

    #   Complete the second series 
    while nn2 < tot_n_samples:    
        t_bl_cor    = t_bls_cor[exc_cnt] # draw a blank 
        t_wh_cor    = t_whs_cor[exc_cnt] # draw a whiff 
        concs_cor[nn2:nn2 + int(t_bl_cor*sample_rate)] = 0     
        nn2         = nn2 + int(t_bl_cor*sample_rate + t_wh_cor*sample_rate)
        exc_cnt     = exc_cnt + 1
           
    # ********************************
                
    t_dyn = np.transpose(np.array((t_bls, t_whs)))
    t_dyn = t_dyn.reshape(np.size(t_bls)*2,1)
    t_dyn = np.cumsum(t_dyn)
    
    t_dyn_cor = np.transpose(np.array((t_bls_cor, t_whs_cor)))
    t_dyn_cor = t_dyn_cor.reshape(np.size(t_bls)*2,1)
    t_dyn_cor = np.cumsum(t_dyn_cor)
    
    t_dyn_cor = t_dyn_cor[t_dyn<t2sim]
    t_dyn = t_dyn[t_dyn<t2sim]
    
    return concs, concs_cor, t_dyn, t_dyn_cor,

  
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:55:39 2019

corr_plumes.py

Generates couple of correlated time series having the same statistical feature  
of real plumes detected in open space.

see also stats_for_plumes.py

@author: mp525
"""

# This function is corr_stimuli_generator.py, it creates two stimuli with a 
#   given average values of concentration aver(C), with both distributions of 
#   whiff and blanks having a power law decay with a power of -3/2, with a 
#   distribution with a concentration values as an exponential C. The two 
#   stimuli are correlated .


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import timeit
import stats_for_plumes as stats


# *****************************************************************
# STANDARD FIGURE PARAMS
fs = 20
lw = 2
plt.rc('text', usetex=True)  # laTex in the polot
#plt.rc('font', family='serif')
fig_size = [20, 10]
fig_position = 700,10
title_fs = 20 # font size of ticks
label_fs = 20 # font size of labels
black   = 'xkcd:black'
blue    = 'xkcd:blue'
red     = 'xkcd:red'
green   = 'xkcd:green'
purple  = 'xkcd:purple'
orange  = 'xkcd:orange'
# *****************************************************************



# ******************************************************************* 
def main(*stim_params):
    
    # ***************************************************************************
    # PARAMS FOR STIMULI GENERATION
#    stim_params = [duration, sample_rate, n_sample2, 
#               g, whiff_min, whiff_max, blank_min, blank_max, a_conc, b_conc,
#               rho_c, rho_t, ]
    t2sim           = stim_params[0]  # [s] time duration of whole simulated stimulus
    sample_rate     = stim_params[1]  # [Hz] num of samples per each sec
    n_sample2       = stim_params[2]  # [ms] num of samples with constant concentration
    tot_n_samples   = int(t2sim*sample_rate) # [] duration of whole simulated stimulus in number of samples
    
    #  PARAMS FOR WHIFF AND BLANK DISTRIOBUTIONS
    g               = stim_params[3]  # -1/2      # for a power law of -3/2
    whiff_min       = stim_params[4]  # 3e-3      # [s]
    whiff_max       = stim_params[5]  # 3        # [s] 3, 50,150
    
    blank_min       = stim_params[6]  # 3e-3      # [s]
    blank_max       = stim_params[7]  # 25        # [s]  25, 35
    
    # PARAMS FOR CONCENTRATION DISTRIBUTIONS
    # fit of average concentration at 75 m, Mylne and Mason 1991, Fig.10 
    b1              = stim_params[8]  # -(np.log10(1-.997) + np.log10(1-.5))/10.7
    a1              = stim_params[9]  # -0.3*b1 - np.log10(1-.5)

    rho_c           = stim_params[10]  # 
    rho_t           = stim_params[11]  # 
    
    quenched        = stim_params[12]
    seed_num        = stim_params[13]
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

  

if __name__ == '__main__':
    
    quenched        = True      # if True Tbl and Twh are chosen to compensate the distance between stimuli
    
    n_loop          = 1
    n_repet_loop    = 1
    fig_plumes      = 1
    fig_save        = 0
    
    fld_output      = 'open_field_stimuli/images/'
    
    # *******************************************************************
    # PARAMS FOR STIMULUS GENERATION
    t2sim           = 200      # [s] time duration of whole simulated stimulus
    pts_ms          = 10
    sample_rate     = pts_ms*1000       # [Hz] num of samples per each sec
    n_sample2       = 5                 # [ms] num of samples with constant concentration
    
    tot_n_samples   = int(t2sim*sample_rate) # [] duration of whole simulated stimulus in number of samples
    
    # *******************************************************************
    #  PARAMS FOR WHIFF AND BLANK DISTRIOBUTIONS
    g               = -1/2  # -1/2 for a power law of -3/2, 1 for uniform distribution
    whiff_min       = 3e-3  # [s]
    whiff_max       = 3     # [s] 3, 50,150
    
    blank_min       = 3e-3   # [s]
    blank_max       = 25     # [s]  25
    
    # *******************************************************************
    # PARAMS FOR CONCENTRATION DISTRIBUTIONS
    # fit of average concentration at 75 m, Mylne and Mason 1991, Fig.10 
    b_conc = -(np.log10(1-.997) + np.log10(1-.5))/10.7
    a_conc = -0.3*b_conc - np.log10(1-.5)
    
    # *******************************************************************
    # PARAMS FOR CORRELATED STIMULI
    rho_c       = 1     # corr. between normal distribution to generate concentration        
    rho_t_exp   = 5      # 0,1,3,5, correlation between normal distribution to generate whiffs and blanks
    rho_t       = 1-10**-rho_t_exp
    
    # *******************************************************************
    # CALCULATE THE THEORETICAL MEAN WHIFF, MEAN BLANK DURATIONS AND INTERMITTENCY
    pdf_wh, logbins, wh_mean = stats.whiffs_blanks_pdf(whiff_min, whiff_max, g)
    pdf_bl, logbins, bl_mean = stats.whiffs_blanks_pdf(blank_min, blank_max, g)
    
    print('main Stim. params:')
    print('durs:%d s'%t2sim)
    print('rho_{wh,bl}:1-10^{%d}' %round(np.log10(1-rho_t)))
    print('rho_{conc}: %.3f' %rho_c)
    
    seed_num    = 0#np.nan
    # *******************************************************************
    # arguments for the generation of stimuli function
    stim_params = [t2sim, sample_rate, n_sample2, g, whiff_min, whiff_max, 
           blank_min, blank_max, a_conc, b_conc,rho_c, rho_t, quenched, seed_num, ]   
    
    
    # *******************************************************************
    # REPEATED STIMULI GENERATION
    
    y_avg           = -np.ones(n_loop)*2
    w_avg           = -np.ones(n_loop)*2
    cor_stim        = -np.ones(n_loop)*2
    overlap_stim    = -np.ones(n_loop)*2
    cor_whiff       = -np.ones(n_loop)*2
    interm_est      = -np.ones(n_loop)*2
    interm_est2      = -np.ones(n_loop)*2
    perc_dif        = -np.ones(n_repet_loop)
    
    start_seed      = np.random.randint(1, 1000, n_repet_loop)
    print('seed: %d'%start_seed)
    for pp in range(n_repet_loop):
        tic = timeit.default_timer()    
        for rr in range(n_loop):
            seed_num    = rr#np.nan
            stim_params[-1] = seed_num+start_seed[pp]
#            print(stim_params[-1])
            out_y, out_w, t_dyn, t_dyn_cor, = main(*stim_params)
            
            y_avg[rr] = np.mean(out_y)
            w_avg[rr] = np.mean(out_w)
            interm_est[rr] = np.sum(out_y>0)/(t2sim*sample_rate)
            interm_est2[rr] = np.sum(out_w>0)/(t2sim*sample_rate)
    
            if (np.sum(out_y)!=0) & (np.sum(out_w)!=0):
                cor_stim[rr] = np.corrcoef(out_y, out_w)[1,0]
                overlap_stim[rr] = stats.overlap(out_y, out_w)
                nonzero_concs1  = out_y[(out_y>0) & (out_w>0)]
                nonzero_concs2  = out_w[(out_y>0) & (out_w>0)]
                if np.size(nonzero_concs1)>0 & np.size(nonzero_concs2)>0:
                    cor_whiff[rr]   = np.corrcoef(nonzero_concs1, nonzero_concs2)[0, 1] 
    
        perc_dif[pp] = (np.mean(y_avg) - np.mean(w_avg))/np.mean(y_avg)
        cor_stim = cor_stim[cor_stim>=-1.0]
        overlap_stim = overlap_stim[overlap_stim>=-1.0]
    
        toc = timeit.default_timer()
        print('Tot time, %d repetition: %.3fs' %(n_loop, (toc-tic)))
        
            
        interm_th = wh_mean/(wh_mean+bl_mean)
        print('Theor. Interm.: %.2g'%interm_th)
        print('')
        print('Estimated values:')
        print('stim., Mean: %.2f'%np.mean(y_avg) +', std: %.2g' %(np.std(y_avg)))
        print('stim.2, Mean: %.2f'%np.mean(w_avg) +', std: %.2g' %(np.std(w_avg)))
        print('percent difference: %.2g'%(perc_dif[pp]))
        print('')
        print('Interm., Mean: %.2g'%np.mean(interm_est) +', std: %.2g' %(np.std(interm_est)))
        print('Interm. 2, Mean: %.2g'%np.mean(interm_est2)+', std: %.2g' %(np.std(interm_est2)))
        print('Corr blank+whiff, Mean: %.2g' %(np.mean(cor_stim))+
                     ', std: %.2g' %(np.std(cor_stim)))
        print('Overlap, Mean: %.2g' %(np.mean(overlap_stim))+
                     ', std: %.2g' %(np.std(overlap_stim)))
        print('Corr whiff, Mean: %.2g' %(np.mean(cor_whiff))+
                     ', std: %.2g' %(np.std(cor_whiff)))
        
#    print(perc_dif)
#    print('Mean perc_dif: %.2f'%np.mean(perc_dif))
    
    #********************************************************************
    # Stimuli FIGURE
    if fig_plumes:
        #%%
        t2plot = np.linspace(0, t2sim, np.size(out_y))
        overlap_t = (out_y>0) & (out_w>0)
        
        t2plot_lim = 0, t2sim
        t2plot_lim2 = 0, .5
        
        rs = 3
        cs = 1
        fig = plt.figure(figsize=(10, 5), )    
        
        ax_st0  = plt.subplot(rs,cs, 1)
        ax_st   = plt.subplot(rs,cs, 2)
        ax_st2  = plt.subplot(rs,cs, 3)
        
        # PLOT 
        ax_st0.plot(t2plot, overlap_t, '.', color=blue, label='Overlap Stimuli')  
        
        ax_st.plot(t2plot, out_y, color=green, label='Stimulus 1')  
        ax_st.plot(t2plot, out_w, color=purple, label='Stimulus 2')  
        
        # SETTINGS
        ax_st0.set_xlim(t2plot_lim)
        ax_st0.set_ylabel('Overlap', fontsize=label_fs)
        
        ll, bb, ww, hh = ax_st0.get_position().bounds
        ax_st0.set_position([ll, bb+.04, ww, hh])
        
        ax_st.set_xlim(t2plot_lim)
        ax_st.set_ylabel('Concentration', fontsize=label_fs)
        
        ll, bb, ww, hh = ax_st.get_position().bounds
        ax_st.set_position([ll, bb+.04, ww, hh])
        
        ax_st2.plot(t2plot, out_y, color=green, label='Stimulus 1')  
        ax_st2.plot(t2plot, out_w, color=purple, label='Stimulus 2')  
        
        ax_st2.set_xlim(t2plot_lim2)
        ax_st2.set_xlabel('Time   (s)', fontsize=label_fs)
        ax_st2.set_ylabel('Concentration', fontsize=label_fs)
        ll, bb, ww, hh = ax_st2.get_position().bounds
        ax_st2.set_position([ll, bb+.04, ww, hh])
        
        plt.show()
        
        
        if fig_save:
            fig.savefig(fld_output + '/stimuli_timecourse_dur20s_cor_%d.png'%rho_t_exp)
      
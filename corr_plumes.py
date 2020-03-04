#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:55:39 2019

corr_plumes.py

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
from scipy.integrate import quad



# *****************************************************************
# STANDARD FIGURE PARAMS
fig_save = False
fig_size = [20, 10]
fig_position = 700,10
#plt.rc('font', family='serif')
plt.rc('text', usetex=True)  # laTex in the polot
#plt.ion() # plt.ioff() # to avoid showing the plot every time
lw      = 2
fs      = 20
title_fs = 25
black   = 'xkcd:black'
blue    = 'xkcd:blue'
red     = 'xkcd:red'
green   = 'xkcd:green'
purple  = 'xkcd:purple'
orange  = 'xkcd:orange'
cmap    = plt.get_cmap('rainbow')
# *****************************************************************

def overlap(a,b):
    a = (a>0)*1.0
    b = (b>0)*1.0
    return np.sum(a*b)*2.0/(np.sum(a)+np.sum(b))

def rnd_pow_law(a, b, g, r):
    """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b"""
    ag, bg = a**g, b**g
    return (ag + (bg - ag)*r)**(1./g)

# concentration values drawn from fit of Mylne and Mason 1991 fit of fig.10 
def rnd_mylne_75m(a1, b1, r):
    """Mylne and Mason 1991 fit of fig.10 """
    y = ((1-np.heaviside(r-.5, 0.5)) * .3*r/.5 + 
         np.heaviside(r-.5, 0.5)* (-(a1 + np.log10(1-r))/b1))
    return y

def whiffs_blanks_pdf(dur_min, dur_max, g):
    logbins  = np.logspace(np.log10(dur_min),np.log10(dur_max))
    pdf_th  = pdf_pow_law(logbins, dur_min, dur_max, g)      # theoretical values of the pdf
    dur_mean = quad(lambda x: x*pdf_pow_law(x, dur_min, dur_max, g), dur_min, dur_max) # integrate the pdf to check it sum to 1
    return pdf_th, logbins, dur_mean[0]

def pdf_pow_law(x, a, b, g):
    ag, bg = a**g, b**g
    return g * x**(g-1) / (bg - ag)

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
    tot_n_samples   = 1+int(t2sim*sample_rate) # [] duration of whole simulated stimulus in number of samples
    
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
    
    t_bls = rnd_pow_law(blank_min, blank_max, g, u1)
    t_bls_cor = rnd_pow_law(blank_min, blank_max, g, u1c)

    n1 = np.random.normal(size=int(t2sim*sample_rate/ratio_min))
    n2 = np.random.normal(size=int(t2sim*sample_rate/ratio_min))
    n1c = rho_t*n1 + np.sqrt(1-rho_t**2)*n2        
    u1 = norm.cdf(n1)
    u1c = norm.cdf(n1c)
    
    t_whs = rnd_pow_law(whiff_min, whiff_max, g, u1)
    t_whs_cor = rnd_pow_law(whiff_min, whiff_max, g, u1c)
    
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
            
            tmp_t = rnd_pow_law(blank_min, blank_max, g, u1)
            tmp_tc = rnd_pow_law(blank_min, blank_max, g, u1c)
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
            
            tmp_t = rnd_pow_law(whiff_min, whiff_max, g, u1)
            tmp_tc = rnd_pow_law(whiff_min, whiff_max, g, u1c)
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
    
    tmp_c = rnd_mylne_75m(a1, b1, u1)
    tmp_cc = rnd_mylne_75m(a1, b1, u1c)
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
    
    n_repet         = 5
    fig_plumes      = 1
    
    # *******************************************************************
    # PARAMS FOR STIMULUS GENERATION
    t2sim           = 21.0    # [s] time duration of whole simulated stimulus
    sample_rate     = 1000 # [Hz] num of samples per each sec
    n_sample2       = 5    # [ms] num of samples with constant concentration
    
    tot_n_samples   = int(t2sim*sample_rate) # [] duration of whole simulated stimulus in number of samples
    
    # *******************************************************************
    #  PARAMS FOR WHIFF AND BLANK DISTRIOBUTIONS
    g               = -1/2      # 1    # -1/2 for a power law of -3/2, 1 for uniform distribution
    whiff_min       = 3e-3      # [s]
    whiff_max       = 3    # [s] 3, 50,150
    
    blank_min       = 3e-3      # [s]
    blank_max       = .03     # [s]  25
    
    # *******************************************************************
    # PARAMS FOR CONCENTRATION DISTRIBUTIONS
    # fit of average concentration at 75 m, Mylne and Mason 1991, Fig.10 
    b_conc = -(np.log10(1-.997) + np.log10(1-.5))/10.7
    a_conc = -0.3*b_conc - np.log10(1-.5)
    tic = timeit.default_timer()    
    
    rho_c       = .1     # correlation between normal distribution to generate concentration        
    rho_t_exp   = 0      # correlation between normal distribution to generate whiffs and blanks
    rho_t       = 1-10**-rho_t_exp
    
    seed_num    = np.nan
    # *******************************************************************
    # arguments for the generation of stimuli function
    stim_params = [t2sim, sample_rate, n_sample2, g, whiff_min, whiff_max, 
           blank_min, blank_max, a_conc, b_conc,rho_c, rho_t, quenched, seed_num, ]
    
    
    # *******************************************************************
    # REPEATED STIMULI GENERATION
    
    y_avg           = -np.ones(n_repet)*2
    w_avg           = -np.ones(n_repet)*2
    cor_stim        = -np.ones(n_repet)*2
    overlap_stim    = -np.ones(n_repet)*2
    cor_whiff       = -np.ones(n_repet)*2
    interm_est      = -np.ones(n_repet)*2
    interm_est2      = -np.ones(n_repet)*2
    
    for rr in range(n_repet):
        out_y, out_w, t_dyn, t_dyn_cor, = main(*stim_params)
        
        y_avg[rr] = np.mean(out_y)
        w_avg[rr] = np.mean(out_w)
        interm_est[rr] = np.sum(out_y>0)/(t2sim*sample_rate)
        interm_est2[rr] = np.sum(out_w>0)/(t2sim*sample_rate)

        if (np.sum(out_y)!=0) & (np.sum(out_w)!=0):
            cor_stim[rr] = np.corrcoef(out_y, out_w)[1,0]
            overlap_stim[rr] = overlap(out_y, out_w)
            nonzero_concs1  = out_y[(out_y>0) & (out_w>0)]
            nonzero_concs2  = out_w[(out_y>0) & (out_w>0)]
            if np.size(nonzero_concs1)>0 & np.size(nonzero_concs2)>0:
                cor_whiff[rr]   = np.corrcoef(nonzero_concs1, nonzero_concs2)[0, 1] 

    print('percent difference: %.2g'%((np.mean(y_avg) - np.mean(w_avg))/np.mean(y_avg)))
    
    cor_stim = cor_stim[cor_stim>=-1.0]
    overlap_stim = overlap_stim[overlap_stim>=-1.0]

    toc = timeit.default_timer()
    print('Tot time, averaged over %d repetition: %.3f' %(n_repet, (toc-tic)))
    # CALCULATE THE THEORETICAL MEAN WHIFF, MEAN BLANK DURATIONS AND INTERMITTENCY
    pdf_wh, logbins, wh_mean = whiffs_blanks_pdf(whiff_min, whiff_max, g)
    pdf_bl, logbins, bl_mean = whiffs_blanks_pdf(blank_min, blank_max, g)
    
    interm_th = wh_mean/(wh_mean+bl_mean)
    print('Theoretical Intermittency: %.2g'%interm_th)
    print('Estimated Intermittency: %.2g'%np.mean(interm_est))
    print('Estimated Intermittency 2: %.2g'%np.mean(interm_est2))
    
    #%%
    if fig_plumes:
        rs = 2
        cs = 3
        fig = plt.figure(figsize=(24,11), )    
        
        ax_st = plt.subplot(rs,cs,1)
        ax_st.plot(out_y, label='Stimulus 1')  
        ax_st.plot(out_w, label='Stimulus 2')  
        ax_st.set_title(r'Sample: $ \rho_{whiff}$:' + '%.2f' %cor_whiff[-1] +
                            r'$; \rho_{stim}$:' + '%.2f' %cor_stim[-1] +
                            '; overlap:' + '%.2f' %overlap_stim[-1], fontsize=22)
        ax_st.set_xlabel('Time   (s)', fontsize=20)
        ax_st.set_ylabel('Concentration', fontsize=20)
        ax_st.legend(fontsize=20)
        ax_st.set_xticks(np.linspace(0,t2sim*sample_rate,5))
        ax_st.set_xticklabels(np.linspace(0,t2sim,5))
        
        ax_ph = plt.subplot(rs,cs,2)
        ax_ph.plot(out_y, out_w, 'o')
        ax_ph.set_xlabel('Stimulus 1', fontsize=20)
        ax_ph.set_ylabel('Stimulus 2', fontsize=20)
        
        ax_cor = plt.subplot(rs,cs,3)
        ax_cor.hist(interm_est, bins=50, label='interm est', color=orange, alpha=.5, density=True,)  
        ax_cor.hist(interm_est2, bins=50, label='interm est 2', color=purple, alpha=.5, density=True,)  
        ax_cor.hist(cor_stim, bins=50, label='corr blank+whiff', color=blue, alpha=.5, density=True,)  
        ax_cor.hist(overlap_stim, bins=50, label='overlap', color=red, alpha=.5, density=True,)  
        n_tmp, _, _=ax_cor.hist(cor_whiff, bins=50, label='corr whiff', color=green, alpha=.5, density=True,)  
        ax_cor.plot([interm_th, interm_th], [0, np.max(n_tmp)], label='theor. interm', color=purple,)
        ax_cor.set_title(r'Corr. Params:  g:%.1f' %g+r'; $\rho_{wh,bl}:1-10^{%d}$' %round(np.log10(1-rho_t)) +
                                  r'; $\rho^N_{conc}$: ' + '%.3f' %rho_c, fontsize=22)
        ax_cor.set_ylabel('probab distr funct', fontsize=20)
        ax_cor.set_xlabel('correlation', fontsize=20)
        ax_cor.text(-.1, np.max(n_tmp)*.7, 'Corr blank+whiff, Mean: %.2g' %(np.mean(cor_stim))+
                     ', std: %.2g' %(np.std(cor_stim)), color=blue, fontsize=20)
        ax_cor.text(-.1, np.max(n_tmp)*.5, 'Overlap, Mean: %.2g' %(np.mean(overlap_stim))+
                     ', std: %.2g' %(np.std(overlap_stim)), color=red, fontsize=20)
        ax_cor.text(-.1, np.max(n_tmp)*.3, 'Corr whiff, Mean: %.2g' %(np.mean(cor_whiff))+
                     ', std: %.2g' %(np.std(cor_whiff)), color=green, fontsize=20)
        ax_cor.set_xlim((-.2,1))
        ax_cor.legend(fontsize=15)
        
        ax_cumsum = plt.subplot(rs,1,2)
        ax_cumsum.plot(t_dyn, t_dyn-t_dyn_cor, '.', label=r'$\Delta T_{start}$ (s)')
        ax_cumsum.set_ylabel(r'$\Delta T_{start} $(s)', fontsize=20)
        ax_cumsum.set_xlabel(r'Time (s)', fontsize=20)
        ax_cumsum.legend(fontsize=20)
        plt.show()
    #    fig.savefig('open_field_stimuli/rnd_corr_plumes_rhoT%d_rhoC%d.png'%(rho_t_exp,100*rho_c))

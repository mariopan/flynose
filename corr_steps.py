#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:49:28 2019
rnd_corr_stimuli.py

This function creates two correlated stimuli with a fixed value of 
concentration aver(C); durations of both whiff and blanks are rvs. 
The correlation between the two stimuli is zero on average, but it spans a big
range, from -1 to 1 in some cases.

See also plume.py, corr_plumes.py

@author: mp525
"""


import numpy as np
import matplotlib.pyplot as plt
import timeit


# *****************************************************************
# STANDARD FIGURE PARAMS
fig_save = False
fig_size = [20, 10]
fig_position = 700,10
#plt.rc('font', family='serif')
plt.rc('text', usetex=True)  # laTex in the polot
plt.ion() # plt.ioff() # to avoid showing the plot every time
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


def rise_func(t, last_stim, c, tau_on):
    rf = last_stim+(c-last_stim)*(1.0-np.exp(-t/tau_on))
    return rf

def decay_func(t, last_stim, c, tau_off):
    df = c - (c-last_stim)* np.exp(-t/tau_off)
    return df


tic = timeit.default_timer()    

corr_wanted = .8

def main(t2sim=2500, pts_ms = 1,seed_num=0):
    
    t_extra_onset   = 1000  # exta time to reach eaounh time points
    t_extra_offset  =  500  # exta time to reach eaounh time points
    t_tot0           = t2sim + t_extra_onset + t_extra_offset 
    
    t_open_valve    = 100          # ms, 1/lambda, pulse mean interval 
    n_switch_ip     = int(t_tot0/t_open_valve) # approximate number of valvees
    
    tau_on          = 20     # rising/onset timescale 
    tau_off         = 60     # decay/offset timescale
    
    conc_y          = .2 # max value of conc. for stimuli to ORN_y
    conc_w          = .2  # max value of conc. for stimuli to ORN_w
    
    if not(np.isnan(seed_num)):
        np.random.seed(seed_num)  
    
    # nagel times:
    t_switch = np.linspace(t_open_valve*np.random.random(), 
                        int(t_tot0),n_switch_ip)
    switch_time_all  = (t_switch*pts_ms).astype(int)   # valve in time [integers]
    rnd_flip_y = np.random.randint(2, size=n_switch_ip)     # random flip of 0s and 1s
    rnd_flip_y = ((np.round(rnd_flip_y)-.5)*2).astype(int)  # random flip of 1s and -1s
    switch_time_y = switch_time_all[rnd_flip_y == 1]
    n_switch_y = np.size(switch_time_y)  # number of times the valve change state        
    rand_uni = np.random.uniform(0,1,n_switch_ip)           # random numers between 0 and 1
    rand_shf = np.round(rand_uni + corr_wanted/2)           # random flip of 0s and 1s, prob.(1) = 0.5+crr/2
    rand_shf = ((np.round(rand_shf)-.5)*2).astype(int)      # random flip of 1s and -1s, prob.(1) = 0.5+crr/2
    rnd_flip_w = rand_shf*rnd_flip_y                        # rnd flip of 1s and -1s, slightly different
    
    switch_time_w = switch_time_all[rnd_flip_w == 1]
    
    n_switch_w = np.size(switch_time_w)  # number of times the valve change state
    max_switch_time = np.max([switch_time_y[-1], switch_time_w[-1]])
    
    t_tot = max_switch_time             # [ms] tot time of simulation
    n_time_pts = t_tot*pts_ms + 1       # number of time points
   
    # define vectors for the simulation
    t = np.linspace(0, t_tot, n_time_pts)   # time vector# store solution
    stim_y = np.zeros_like(t)       # stimulus vector  
    stim_w = np.zeros_like(t)       # stimulus vector  
    valve_y = np.zeros_like(t)       # valve vector
    valve_w = np.zeros_like(t)       # valve vector
    
    
    # ------------------------------------------------------------
    # STIMULATION FOR Y
    # record initial conditions
    switcher = 1                   # variable to change valve's value
    stim_y[0] = 0.0 
    valve_y[0:switch_time_y[0]] = -1  
    last_stim = stim_y[0]
    for ii in range(1,n_switch_y):
        t_start = switch_time_y[ii-1]
        t_stop = switch_time_y[ii]
        valve_y[t_start:t_stop] = switcher
        
        if switcher == 1:
            stim_y[t_start:t_stop] = rise_func(t[t_start:t_stop]-t[t_start], 
                  last_stim, conc_y*(1+0.0*np.random.normal()), tau_on)
        elif switcher == -1:
            stim_y[t_start:t_stop] = decay_func(t[t_start:t_stop]-t[t_start], 
                  last_stim, 0.0, tau_off)
        last_stim = stim_y[t_stop-1]
        switcher = -1 * switcher
    
    
    # ------------------------------------------------------------
    # STIMULATION FOR W
    # record initial conditions
    switcher = 1#np.sign(corr_wanted)                   # variable to change valve's value
    stim_w[0] = 0.0
    valve_w[0:switch_time_w[0]] = -1  
    # STIMULATION FOR W
    last_stim = stim_w[0]
    for ii in range(1,n_switch_w):
        t_start = switch_time_w[ii-1]
        t_stop = switch_time_w[ii]
        valve_w[t_start:t_stop] = switcher
        
        if switcher == 1:
            stim_w[t_start:t_stop] = rise_func(t[t_start:t_stop]-t[t_start], last_stim, conc_w*(1+0.0*np.random.normal()), tau_on)
        elif switcher == -1:
            stim_w[t_start:t_stop] = decay_func(t[t_start:t_stop]-t[t_start], last_stim, 0.0, tau_off)
        last_stim = stim_w[t_stop-1]
        switcher = -1 * switcher

    out_y = stim_y[t_extra_onset*pts_ms:1+(t_extra_onset+t2sim)*pts_ms]
    out_w = stim_w[t_extra_onset*pts_ms:1+(t_extra_onset+t2sim)*pts_ms]
    if out_w.size< t2sim*pts_ms:
        print('size of output vectors: %d '%(out_w.size))
        print('oh oh')
           
    return [out_y, out_w]
    
if __name__ == '__main__':
    print('run directly')
    tic = timeit.default_timer()    
    n_repet = 100
    t2sim = 3000
    pts_ms = 10
    corr_samples = np.zeros(n_repet)
    for rr in range(n_repet):
        rnd_seed = np.random.randint(0, 3551)
        out_y, out_w = main(t2sim=t2sim, pts_ms= pts_ms, seed_num=rnd_seed)
        corr_samples[rr] = np.corrcoef(out_y, out_w)[1,0]
    
    
    toc = timeit.default_timer()
    print('Tot time, averaged over %d repetition: %.3f' %(n_repet, (toc-tic)))
    
    fig = plt.figure(figsize=(24,6), )    
    
    ax_st = plt.subplot(1,3,1)
    ax_st.plot(out_y, label='stimulus 1')  
    ax_st.plot(out_w, label='stimulus 2')  
    ax_st.text(1000, .1, 'corr stim: %f' %(corr_samples[-1]))
    ax_st.set_title('Sample of stimuli ', fontsize=title_fs)
    ax_st.set_xlabel('Time   (ms)', fontsize=fs)
    ax_st.set_ylabel('Concentration', fontsize=fs)
    ax_st.legend(fontsize=20)
    ax_st.set_xticks(np.linspace(0,t2sim*pts_ms,5))
    ax_st.set_xticklabels(np.linspace(0,t2sim,5))
    
    ax_ph = plt.subplot(1,3,2)
    ax_ph.plot(out_y, out_w, 'o')
    ax_ph.set_xlabel('Stimulus 1', fontsize=fs)
    ax_ph.set_ylabel('Stimulus 2', fontsize=fs)
    
    ax_cor = plt.subplot(1,3,3)
    ax_cor.hist(corr_samples, 30)  
    ax_cor.set_title('Stimuli correlation distribution', fontsize=title_fs)
    ax_cor.set_ylabel('Probab distr funct', fontsize=fs)
    ax_cor.set_xlabel('Correlation', fontsize=fs)
    
    fig.savefig('open_field_stimuli/rnd_corr_step_stimuli.png')
    

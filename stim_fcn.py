#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 09:51:12 2021
Generate odorant stimuli of several kinds	

see also corr_plumes.py

@author: mario
"""
import numpy as np
from scipy import signal
import corr_plumes

# stimulus
def main(stim_params, verbose=False):
    
    tmp_ks = \
        ['stim_type', 'stim_dur', 'pts_ms', 't_tot', 
         't_on', 'concs', 'conc0', 'od_noise', 'od_filter_frq']    
    [stim_type, stim_dur, pts_ms, t_tot, t_on, concs, conc0, od_noise, od_filter_frq] = [
        stim_params[x] for x in tmp_ks]  
    
    # Stimulus params    
    n_od            = len(concs)
    t_off           = t_on+stim_dur
    
    n2sim           = int(pts_ms*t_tot) + 1       # number of time points
    
    
    # Create an order 3 lowpass butterworth filter:
    extra_t         = pts_ms*200                
    rand_ts         =  np.random.randn(n2sim+extra_t, n_od)*od_noise
    filter_ord      = 3
    b, a            = signal.butter(filter_ord, od_filter_frq)    
    filt_ts         = np.zeros_like(rand_ts)
    filt_ts         = signal.filtfilt(b, a, rand_ts.T).T    

    u_od        = np.ones((n2sim+extra_t, n_od)) * conc0*(1 + filt_ts*np.sqrt(1/pts_ms))    
    u_od        = u_od[extra_t:, :]
    
    if stim_type == 'ext':
        if verbose:
            print('ext stimuli, from Kim et al. 2011')
        
        stim_data_name = stim_params['stim_data_name'] 
        ex_stim = np.loadtxt(stim_data_name+'.dat')
     
        # Sims params
        t_tot           = ex_stim[-1,0]*1e3 # [ms] t_tot depends on data
        n2sim           = int(t_tot*pts_ms)+1 
        # n_ex_stim       = np.size(ex_stim, axis=0)#pts_ms*t_tot + 1      # number of time points
        stim_params['t_tot'] = t_tot
        
        u_od            = np.zeros((n2sim, 2))
        u_od[:, 0]      = .00004*ex_stim[:,1]
        u_od[:, 1]      = .00004*(ex_stim[0,1]+ex_stim[-1,1])/2
        
        
    elif stim_type == 'pl':
        # real plumes stimuli
        if verbose:
            print('u_od is extracted from real plumes')
    
        # PARAMS FOR GENERATION OF PLUMES
        
        plume_params    = stim_params['plume_params']        
        
        quenched        = True          # if True Tbl and Twh are chosen to compensate the distance between stimuli
        t2sim_s         = (t_tot-t_on[0])/1000  # [s]
        sample_rate     = 1000*pts_ms   # [Hz] num of samples per each sec
        n_sample2       = 5             # [ms] num of samples with constant concentration

        #  PARAMS FOR WHIFF AND BLANK DISTRIOBUTIONS
        g               = plume_params['g']            # -1/2 for a power law of -3/2, 1 for uniform distribution
        whiff_min       = plume_params['whiff_min']    # 3e-3 [s]
        whiff_max       = plume_params['whiff_max']    # [s] 3, 50,150
        
        blank_min       = plume_params['blank_min']      # 3e-3 [s]
        blank_max       = plume_params['blank_max']       # [s]  25, 35
        
        # PARAMS FOR CONCENTRATION DISTRIBUTIONS
        # fit of average concentration at 75 m, Mylne and Mason 1991, Fig.10 
        b_conc          = -(np.log10(1-.997) + np.log10(1-.5))/10.7
        a_conc          = -0.3*b_conc - np.log10(1-.5)
        
        rho_c           = plume_params['rho_c']      # correlation between normal distribution to generate whiffs and blanks
        rho_t_exp       = plume_params['rho_t_exp']     # correlation between normal distribution to generate concentration        
        rho_t           = 1-10**-rho_t_exp
        
        stim_seed       = plume_params['stim_seed']
        
        # arguments for the generation of stimuli function
        corr_plumes_in = [t2sim_s, sample_rate, n_sample2, g, whiff_min, whiff_max, 
               blank_min, blank_max, a_conc, b_conc,rho_c, rho_t, quenched, stim_seed]
        
        # PLUME GENERATION
        out_corr_plumes = corr_plumes.main(*corr_plumes_in)
        
        for nn in range(n_od):
            stim_on         = t_on[nn]*pts_ms   # [num. of samples]
            u_od[stim_on:, nn] = out_corr_plumes[nn]*concs[nn] + .0*u_od[stim_on:, nn]

             
    elif (stim_type == 'rs'):
        # baseline stimuli
        if verbose:
            print('u_od is constant')
        
    elif stim_type == 'ts':
        # Single pulse with a triangular shape   
        if verbose:
            print('u_od is triangular pulse')        
        for nn in range(n_od):
            t_peak          = t_on[nn] + stim_dur[nn]/2     # [ms]
            stim_peak       = int(t_peak*pts_ms)
            
            stim_on         = t_on[nn]*pts_ms   # [num. of samples]
            stim_off        = t_off[nn]*pts_ms 
            
            u_od[stim_on:stim_peak, nn] = np.linspace(conc0, concs[nn], stim_peak-stim_on)
            u_od[stim_peak:stim_off, nn] = np.linspace(concs[nn], conc0, stim_off-stim_peak)
        
    elif (stim_type == 'ss'):
        # Single Step Stimuli with a slow dynamics emulating what seen in exps
        if verbose:
            print('u_od is single step')        
        n2sim           = int(t_tot*pts_ms)+1 
        tau_on          = 50
        for nn in range(n_od):
            stim_on         = t_on[nn]*pts_ms   # [num. of samples]
            stim_off        = t_off[nn]*pts_ms    
            
            # stimulus onset
            t_tmp           = \
                np.linspace(0, t_off[nn]-t_on[nn], stim_off-stim_on)
            
            u_od[stim_on:stim_off, nn] += \
                + (concs[nn] - conc0)*(1 - np.exp(-t_tmp/tau_on)) # conc0
            
            # stimulus offset
            t_tmp           = \
                np.linspace(0, t_tot-t_off[nn], n2sim-stim_off)    
            
            u_od[stim_off:, nn]  += \
                (u_od[stim_off-1, nn]-conc0)*np.exp(-t_tmp/tau_on)
 
    u_od[u_od<0] = 0
    return u_od
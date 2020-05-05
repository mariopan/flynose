#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# script name: synapse_dynORNPN.py

"""
Created on Thu Jan 1 2019

Simulation for the dynamics of a general synapse with Rall-Seynosky-Desthexe
script name: flynose.py
@author: mario

"""

import numpy as np
import scipy.stats as spst
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import timeit

import pickle        
from os import path
from os import mkdir
from shutil import copyfile

import sys
sys.path.insert(0, '/flynose/')

import corr_steps
import corr_plumes
import sdf_krofczik
import stats_for_plumes as stats


# *****************************************************************
# FUNCTIONS

def depalo_eq2(z,t,u,u2,orn_params,):
    ax = orn_params[0]
    cx = orn_params[1]
    bx = orn_params[2]
    by = orn_params[3]
    dy = orn_params[4]
    b = orn_params[5]
    d = orn_params[6]
    ar = orn_params[7]
    n = orn_params[8]
    nsi_value = orn_params[9]
    
    r = z[0]
    x = z[1]
    y = z[2]
    
    s = z[3] # r2
    q = z[4] # x2
    w = z[5] # y2
    
    drdt = b*u**n*(1-r) - d*r
    dsdt = b*u2**n*(1-s) - d*s
    
    dydt = ar*r - cx*x*(1+dy*y) - by*y - nsi_value*w*y 
    dxdt = ax*y - bx*x
    
    dwdt = ar*s - cx*q*(1+dy*w) - by*w - nsi_value*y*w
    dqdt = ax*w - bx*q
    dzdt = [drdt,dxdt,dydt,dsdt,dqdt,dwdt]
    return dzdt

def rect_func(b, x):
    ot = b[0]/(1 + np.exp(-b[1]*(x-b[2])))
    return ot


def pn2ln_v_ex(x0,t, s, ln_params, ):
#    ln_params = np.array([tau_s_ln, c_ln, g_ln, a_s_ln, vrev_ln, vrest_ln])
    c = ln_params[1]
    g = ln_params[2]
    vrev = ln_params[4]
    vrest = ln_params[5]
    
    # PN -> LN equations:
    # ORN -> PN equations:
    dt = t[1]-t[0]
    b = -(1 + g*s)/c
    a = (vrest + g*s*vrev)/c
    y = (x0 + a/b)*np.exp(b*dt)-a/b
    #dvdt = ((vrest-v) + g*s*(vrev-v) )/c
    return y

def pn2ln_s_ex(x0,t, u_pn, ln_params, ):
    #    ln_params = np.array([tau_s_ln, c_ln, g_ln, a_s_ln, vrev_ln, vrest_ln])
    tau_s = ln_params[0]
    a_s = ln_params[3]
    
    # PN -> LN equation of s:
    b = (-1-a_s*u_pn)/tau_s
    a = a_s*u_pn/tau_s
    dt = t[1]-t[0]
    y = (x0 + a/b)*np.exp(b*dt)-a/b
#    dsdt = (a_s*u_pn*(1-s) - s)/tau_s       
    return y


def x_ln_fun_ex(x0,t,u_ln, tau_x, a_x,):
    b = (-a_x*u_ln-1)/tau_x
    a = a_x*u_ln/tau_x
    dt = t[1]-t[0]
    y = (x0 + a/b)*np.exp(b*dt)-a/b
    return y

def orn2pn_s_ex(x0,t, u_orn, x_pn,x_ln,pn_params,):
    #    pn_params  = np.array([tau_s_pn, c_pn, g_pn, a_s_pn, vrev_pn, vrest_pn])
    tau_s = pn_params[0]
    a_s = pn_params[3]
    
    # ORN -> PN equations:
    b = (-1-a_s*u_orn*(1-x_pn)*(1-x_ln))/tau_s
    a = a_s*u_orn*(1-x_pn)*(1-x_ln)/tau_s
    dt = t[1]-t[0]
    y = (x0 + a/b)*np.exp(b*dt)-a/b
    return y

def orn2pn_v_ex(x0,t, s, pn_params,):
#    pn_params  = np.array([tau_s_pn, c_pn, g_pn, a_s_pn, vrev_pn, vrest_pn])
    c = pn_params[1]
    g = pn_params[2]
    vrev = pn_params[4]
    vrest = pn_params[5]
    
    # ORN -> PN equations:
    dt = t[1]-t[0]
    b = -(1 + g*s)/c
    a = (vrest + g*s*vrev)/c
    y = (x0 + a/b)*np.exp(b*dt)-a/b
#    dvdt = (vrest + g*s*vrev)/c  - v*(1 + g*s)/c
    return y

def x_adapt_ex(x0,t,u_orn, tau, a_ad,):
    b = (-a_ad*u_orn-1)/tau
    a = a_ad*u_orn/tau
    dt = t[1]-t[0]
    y = (x0 + a/b)*np.exp(b*dt)-a/b
    return y

#%%*****************************************************************

# *****************************************************************
# STANDARD FIGURE PARAMS
lw = 2
fs = 13
plt.rc('text', usetex=True)  # laTex in the polot
#plt.rc('font', family='serif')
fig_size = [12, 12]
fig_position = 1300,10
title_fs = 20 # font size of ticks
label_fs = 20 # font size of labels
panel_fs = 30 # font size of panel's letter
black   = 'xkcd:black'
blue    = 'xkcd:blue'
red     = 'xkcd:red'
green   = 'xkcd:green'
purple  = 'xkcd:purple'
orange  = 'xkcd:orange'
cmap    = plt.get_cmap('rainbow')
data_save = True
# *****************************************************************


def main(params2an, fig_opts, verbose=False, fld_analysis='', stim_seed=0):
    
    orn_fig     = fig_opts[0]
    al_dyn      = 1
    al_fig      = fig_opts[1]
    stimulus    = params2an[7] # 'ss'   # 'rs'   #  'pl'  # 'ts'
    fig_save    = fig_opts[3]
    # *****************************************************************
    # STIMULUS GENERATION
    
    nsi_value       = params2an[0] # 0.3     #.2 0.1 0.05 0.00
    ln_spike_height = params2an[1] # .0        # .3
    
    # Stimuli params 
    dur2an          = params2an[2] # 250
    delays2an        = params2an[3] # 0
    if verbose:
        print('Stimulus: ' + stimulus)
        print('Stimuli duration:      ' + '%.0f'%dur2an + ' ms')
        print('Delay between stimuli:    ' + '%.0f'%delays2an + ' ms')
     
    id_colors2u     = [1, 5]
    col_glo         = np.zeros((2,4))
    col_glo[0,:]    = cmap(int(255/(id_colors2u[0]+1.3)))
    col_glo[1,:]    = cmap(int(255/(id_colors2u[1]+1.3)))
    
    
    # average sdf parameters
    tau_sdf         = 20
    dt_sdf          = 5   
    
    # Sims params
    t_on            = 300      # [ms]
    t2simulate      = np.maximum(t_on+params2an[2],1200) # [ms]
    pts_ms          = 5             # simulated points per ms
    n2sim           = pts_ms*t2simulate + 1      # number of time points
    t               = np.linspace(0, t2simulate, n2sim) # time points
    
    # Stimulus params
    stim_on         = t_on*pts_ms 
    t_off           = t_on + dur2an     # [ms]
    stim_off        = t_off*pts_ms    
    t_on2           = t_on+delays2an     # [ms]
    stim_on2        = t_on2*pts_ms
    t_off2          = t_on2 + dur2an    # [ms]
    stim_off2       = t_off2*pts_ms

    cov_hom         = 0.4 # Covariance value homotypic ORNs
    nu_pn_noise     = 200 # Hz  - PNs Noise into each PNs
    nu_ln_noise     = 0 # Hz    - LNs Noise into each PNs
    
        
    # initialize output vectors
    num_glo         = 2     # number of glomeruli
    u_od            = np.zeros((n2sim, num_glo))
    cor_stim        = np.nan
    overlap_stim    = np.nan
    cor_whiff       = np.nan
    
    ext_stimulus = False # flag indicating an external stimulus is provided        
    if stimulus == 'ss':
        # Single Step Stimuli
        
        peak1           = np.array([params2an[4]]) #[.25, .5, 1, 2, 4] # np.concatenate((np.linspace(0.25,.9, 5),))
        peak2           = np.array([params2an[4]*params2an[5]]) #[.25, .5, 1, 2, 4] # np.concatenate((np.linspace(0.25,.9, 5),))
        
        tau_on          = 50
        t_tmp           = np.linspace(0, t_off-t_on, stim_off-stim_on)    
        t_tmp2          = np.linspace(0, t_off2-t_on2, stim_off2-stim_on2)               
            
        u_od[stim_on:stim_off, 0] = peak1*(1-np.exp(-t_tmp/tau_on))
        u_od[stim_on2:stim_off2, 1] = peak2*(1-np.exp(-t_tmp2/tau_on))
        
        t_tmp           = np.linspace(0, t2simulate-t_off, 1+t2simulate*pts_ms-stim_off)    
        t_tmp2          = np.linspace(0, t2simulate-t_off2, 1+t2simulate*pts_ms-stim_off2)               
        
        u_od[stim_off:, 0] = u_od[stim_off-1, 0]*np.exp(-t_tmp/tau_on)
        u_od[stim_off2:, 1] = u_od[stim_off2-1, 1]*np.exp(-t_tmp2/tau_on)
        
        
    elif stimulus == 'ts':
        # Single Step Stimuli
        dur2an          = params2an[2] # 250
        delays2an        = params2an[3] # 0
        if verbose:
            print('Stimulus: Triangle Step')
            print('Stimuli duration:      ' + '%.0f'%dur2an + ' ms')
            print('Delay between stimuli:    ' + '%.0f'%delays2an + ' ms')
        
        peak1           = np.array([params2an[4]]) #[.25, .5, 1, 2, 4] # np.concatenate((np.linspace(0.25,.9, 5),))
        peak2           = np.array([params2an[4]*params2an[5]]) #[.25, .5, 1, 2, 4] # np.concatenate((np.linspace(0.25,.9, 5),))
        t_peak          = t_on + dur2an/2     # [ms]
        stim_peak       = int(t_peak*pts_ms)
        
        
        t_peak2         = t_on2 + dur2an/2     # [ms]
        stim_peak2      = int(t_peak2*pts_ms)
        
        t_tmp           = np.linspace(0, t_off-t_on, stim_off-stim_on)    
        t_tmp2          = np.linspace(0, t_off2-t_on2, stim_off2-stim_on2)               
            
        u_od[stim_on:stim_peak, 0] = np.linspace(0, peak1, stim_peak-stim_on)
        u_od[stim_peak:stim_off, 0] = np.linspace(peak1, 0, stim_off-stim_peak)
                
        u_od[stim_on2:stim_peak2, 1] = np.linspace(0, peak2, stim_peak2-stim_on2)
        u_od[stim_peak2:stim_off2, 1] = np.linspace(peak2, 0, stim_off2-stim_peak2)
        
    elif stimulus == 'rs':
        
        # Random pulses stimuli 
        mmm,ppp = corr_steps.main(t2sim=(t2simulate-t_on),pts_ms=pts_ms,seed_num=stim_seed)
        if np.size(mmm)<(n2sim-stim_on):
            print('oh oh %d'%(np.size(mmm)))
            print('oh oh %d'%(n2sim-stim_on))
            mmm,ppp = corr_steps.main(t2sim=(t2simulate-t_on),pts_ms=pts_ms,seed_num=stim_seed)
        cor_stim = np.corrcoef(mmm, ppp)[1,0]
        if verbose:
            print('Stimulus: Random Steps')
            print('corr. input y and w: ' + '%f'%(cor_stim) + ', seed: %g'%(stim_seed))
        u_od[stim_on:, 0] = mmm
        u_od[stim_on:, 1] = ppp
    elif stimulus == 'pl':
        if verbose:
            print('Stimulus: Plumes')
        
        # *******************************************************************
        # PARAMS FOR GENERATION OF PLUMES
        quenched        = True          # if True Tbl and Twh are chosen to compensate the distance between stimuli
        t2sim_s         = (t2simulate-t_on)/1000  # [s]
        sample_rate     = 1000*pts_ms   # [Hz] num of samples per each sec
        n_sample2       = 5             # [ms] num of samples with constant concentration
        # tot_n_samples   = int(t2simulate*sample_rate) # [] duration of whole simulated stimulus in number of samples
        
        # *******************************************************************
        #  PARAMS FOR WHIFF AND BLANK DISTRIOBUTIONS
        g               = -1/2# 1    # -1/2 for a power law of -3/2, 1 for uniform distribution
        whiff_min       = 3e-3      # [s]
        whiff_max       = params2an[8]        # [s] 3, 50,150
        
        blank_min       = 3e-3      # [s]
        blank_max       = params2an[9]       # [s]  25, 35
        
        # *******************************************************************
        # PARAMS FOR CONCENTRATION DISTRIBUTIONS
        # fit of average concentration at 75 m, Mylne and Mason 1991, Fig.10 
        b_conc          = -(np.log10(1-.997) + np.log10(1-.5))/10.7
        a_conc          = -0.3*b_conc - np.log10(1-.5)
        
        rho_c           = 1      # correlation between normal distribution to generate whiffs and blanks
        rho_t_exp       = params2an[6]     # correlation between normal distribution to generate concentration        
        rho_t           = 1-10**-rho_t_exp
        
        # CALCULATE THE THEORETICAL MEAN WHIFF, MEAN BLANK DURATIONS AND INTERMITTENCY
        pdf_wh, logbins, wh_mean = stats.whiffs_blanks_pdf(whiff_min, whiff_max, g)
        pdf_bl, logbins, bl_mean = stats.whiffs_blanks_pdf(blank_min, blank_max, g)
        
        interm_th = wh_mean/(wh_mean+bl_mean)
        if verbose:
            print('Theoretical Intermittency: %.2g'%interm_th)
        
        # ******************************************************************* 
        # arguments for the generation of stimuli function
        #np.random.seed()
        stim_params = [t2sim_s, sample_rate, n_sample2, g, whiff_min, whiff_max, 
               blank_min, blank_max, a_conc, b_conc,rho_c, rho_t, quenched, stim_seed]
        print(stim_params)
        
        # *******************************************************************
        # PLUME GENERATION
        out_y, out_w, _, _= corr_plumes.main(*stim_params)
        u_od[stim_on:, 0] = out_y*params2an[4]
        u_od[stim_on:, 1] = out_w*params2an[4]*params2an[5]
               
        cor_stim        = -2
        overlap_stim    = -2
        cor_whiff       = -2
        
        interm_est_1 = np.sum(out_y>0)/(t2sim_s*sample_rate)
        interm_est_2 = np.sum(out_w>0)/(t2sim_s*sample_rate)

        if (np.sum(out_y)!=0) & (np.sum(out_w)!=0):
            cor_stim        = np.corrcoef(out_y, out_w)[1,0]
            overlap_stim    = stats.overlap(out_y, out_w)
            nonzero_concs1  = out_y[(out_y>0) & (out_w>0)]
            nonzero_concs2  = out_w[(out_y>0) & (out_w>0)]
            cor_whiff       = np.corrcoef(nonzero_concs1, nonzero_concs2)[0, 1] # np.corrcoef(concs1, concs2)[0, 1]
    else:
        ext_stimulus = True # flag indicating an external stimulus is provided
        stim_data_name = params2an[10]+params2an[7]+'.dat'
        ex_stim = np.loadtxt(stim_data_name)
     
        # Sims params
        t2simulate      = ex_stim[-1,0]*1e3 # [ms] t2simulate depends on data
        pts_ms          = 5             # simulated points per ms
        n2sim           = np.size(ex_stim, axis=0)#pts_ms*t2simulate + 1      # number of time points
        t               = np.linspace(0, t2simulate, n2sim) # time points
        # t = ex_stim[:,0]*1e3         
        
        # Stimulus params
        t_on            = 0      # [ms]
        stim_on         = t_on*pts_ms 
        t_off           = t_on + dur2an     # [ms]
        stim_off        = t_off*pts_ms    
        t_on2           = t_on+delays2an     # [ms]
        stim_on2        = t_on2*pts_ms
        t_off2          = t_on2 + dur2an    # [ms]
        stim_off2       = t_off2*pts_ms
        
        u_od            = np.zeros((n2sim, num_glo))
        u_od[:, 0]      = .01*ex_stim[:,1]
        u_od[:, 1]      = .01*(ex_stim[0,1]+ex_stim[-1,1])/2
    
        
    od_avg1 = np.mean(u_od[stim_on:, 0])
    od_avg2 = np.mean(u_od[stim_on:, 1])    
    
    # *****************************************************************
    # CONNECTIVITY PARAMETERS
    
    # *****************************************************************
    # NETWORK PARAMETERS 
    num_orns_pn         = 18    # number of ORNs per each PN in each glomerulus
    num_orns_glo        = 40    # number of ORNs per each glomerulus
    num_orns_tot        = num_orns_glo*num_glo  # total number of ORNs 
    num_pns_glo         = 5     # number of PNs per each glomerulus
    num_lns_glo         = 3     # number of LNs per each glomerulus
    num_pns_tot         = num_pns_glo*num_glo # number of total PNs
    num_lns_tot         = num_lns_glo*num_glo # number of total LNs 
    
    # *****************************************************************
    # ORN PARAMETERS 
    # rectification params
    c_rect              = 1
    a_rect              = 3.3 
    nu_max_rect         = 250
    B0                  = [nu_max_rect, a_rect, c_rect]
    
    # Spiking machine params
    ax                  = 0.25  
    bx                  = 0.002     
    cx                  = 0.0028     # 0.004
    by                  = 0.2       # 0.12
    dy                  = 1           
    ar                  = 1
    
    # Transduction params
    n                   = 1                 # 1
    b                   = 0.01      #*100# 1.75
    d                   = 0.009     #*100# 1.1
    orn_params          = np.array([ax, cx, bx, by,dy,b,d,ar,n,nsi_value,])

    #**************************************
    # ORN, PN and LN PARAMETERS
    spike_length        = int(4*pts_ms)
    t_ref               = 2*pts_ms  # ms; refractory period 
    thr                 = 1         # [mV] firing threshold
    
    orn_spike_height    = .3
    pn_spike_height     = .3
    
    # *****************************************************************
    # GENERATION OF THE CONNECTIVITY MATRIX
    
    # Each ORN belongs to ONLY one of the glomeruli
    ids_orn_glo = np.zeros((num_orns_tot,), dtype=int)
    ids_orn_glo[:num_orns_glo] = 0
    ids_orn_glo[num_orns_glo:] = 1
    
    # Correlation is high only on the same glomerulus ORNs
    mv_mean     = np.zeros(num_orns_tot)
    mv_cov      = np.zeros((num_orns_tot,num_orns_tot))
    mv_cov_tmp  = ((1-cov_hom)*np.identity(num_orns_glo) +
                 cov_hom*np.ones((num_orns_glo, num_orns_glo))) # diagonal covariance
    mv_cov[num_orns_glo:, num_orns_glo:] = mv_cov_tmp
    mv_cov[:num_orns_glo, :num_orns_glo] = mv_cov_tmp
        
    # Each PN belongs to ONLY one of the glomeruli
    ids_pn_glo = np.zeros(num_pns_tot, dtype=int)
    ids_pn_glo[:num_pns_glo] = 0
    ids_pn_glo[num_pns_glo:] = 1
    # Each LN belongs to ONLY one of the glomeruli
    ids_ln_glo = np.zeros(num_lns_tot, dtype=int)
    ids_ln_glo[:num_lns_glo] = 0
    ids_ln_glo[num_lns_glo:] = 1
    
    # Each PN is connected randomly with a sub-sample of ORNs
    ids_orn_pn          = np.zeros((num_pns_tot, num_orns_pn), dtype=int)
    
    # Connectivity matrices between ORNs and PNs 
    orn_pn_mat          = np.zeros((num_orns_tot, num_pns_tot))
   
    for pp in range(num_pns_tot):
        rnd_ids         = np.random.permutation(num_orns_glo) 
        tmp_ids = rnd_ids[:num_orns_pn] + num_orns_glo*ids_pn_glo[pp]
        ids_orn_pn[pp,:] = tmp_ids
        orn_pn_mat[tmp_ids, pp] = orn_spike_height
    
    # Connectivity matrices between ORNs and PNs and LNs
    tmp_ones            = np.ones((num_pns_glo, num_lns_glo))
    pn_ln_mat           = np.zeros((num_pns_tot, num_lns_tot))
    pn_ln_mat[num_pns_glo:, num_lns_glo:] = tmp_ones*pn_spike_height
    pn_ln_mat[:num_pns_glo, :num_lns_glo] = tmp_ones*pn_spike_height
    
    tmp_ones            = np.ones((num_lns_glo, num_pns_glo))
    ln_pn_mat           = np.zeros((num_lns_tot, num_pns_tot))
    ln_pn_mat[num_lns_glo:, :num_pns_glo] = tmp_ones*ln_spike_height
    ln_pn_mat[:num_lns_glo, num_pns_glo:] = tmp_ones*ln_spike_height
    
    # *****************************************************************
    # GENERATE ORN RESPONSE TO ODOR INPUT 
    num_spike_orn       = np.zeros((n2sim, num_glo))
    r_orn               = np.zeros((n2sim, num_glo))
    x_orn               = np.zeros((n2sim, num_glo))
    y_orn               = np.zeros((n2sim, num_glo))
    nu_orn              = np.zeros((n2sim, num_glo))   
    
    # initial conditions
    z_orn0          = np.ones((num_glo, 3))*[0, 0, 0]
    r_orn[0,:]        = z_orn0[:, 0]
    x_orn[0,:]        = z_orn0[:, 1]
    y_orn[0,:]        = z_orn0[:, 2]
    
    for tt in range(1, n2sim-t_ref-1):
        # span for next time step
        tspan = [t[tt-1],t[tt]]
        
        z0_unid = np.zeros(6)
        z0_unid[0:3] = z_orn0[0,:]
        z0_unid[3:6] = z_orn0[1,:]
        z_orn = odeint(depalo_eq2, z0_unid, tspan,
                       args=(u_od[tt, 0], u_od[tt, 1], orn_params,))
        for gg in range(num_glo):
            z_orn0[gg,0] = z_orn[1][0+gg*3]
            z_orn0[gg,1] = z_orn[1][1+gg*3]
            z_orn0[gg,2] = z_orn[1][2+gg*3]
        
            r_orn[tt,gg] = z_orn[1][0+gg*3]
            x_orn[tt,gg] = z_orn[1][1+gg*3]
            y_orn[tt,gg] = z_orn[1][2+gg*3]
           
            nu_orn[tt,gg] = rect_func(B0, y_orn[tt,gg])
    
    orn_avg1 = np.mean(nu_orn[stim_on:stim_off, 0])
    orn_avg2 = np.mean(nu_orn[stim_on2:stim_off2, 1])
    orn_peak1 = np.max(nu_orn[stim_on:stim_off, 0])
    orn_peak2 = np.max(nu_orn[stim_on2:stim_off2, 1])

          
    # *****************************************************************
    # Transform the average nu_orn into a spiking 
    # matrix n2sim by num_orns_tot of correlated spikes:
    num_spike_orn   = np.zeros((n2sim, num_orns_tot))
    u_orn           = np.zeros((n2sim, num_pns_tot))
    
    # generate a matrix n2sim by num_orns_tot of correlated spikes:
    rnd     = np.random.multivariate_normal(mv_mean, mv_cov, n2sim)
    rnd     = spst.norm.cdf(rnd)
    
    nu_tmp  = np.concatenate((np.tile(nu_orn[:,0], (num_orns_glo,1)), 
                np.tile(nu_orn[:,1], (num_orns_glo,1)))).transpose()
    t_zeros = np.zeros((1, num_pns_tot))
    num_spike_orn = (rnd < nu_tmp/(pts_ms*1e3))*1.0
    orn_spike_all = num_spike_orn.dot(orn_pn_mat) 
    u_orn = u_orn + orn_spike_all
    for tt in range(spike_length-1):
        orn_spike_all = np.concatenate((t_zeros, orn_spike_all[:-1,:]))
        u_orn = u_orn + orn_spike_all
        
    # *****************************************************************
    # Per each PNs, add a noise signal coming from other PNs, LNs, ...
    rnd_pn  = np.random.random_sample((n2sim,num_pns_tot))
    rnd_ln  = np.random.random_sample((n2sim,num_pns_tot))    
    pn_spike_all = (rnd_pn < nu_pn_noise/(pts_ms*1e3))*pn_spike_height
    ln_spike_all = (rnd_ln < nu_ln_noise/(pts_ms*1e3))*ln_spike_height
    u_orn = u_orn - ln_spike_all + pn_spike_all
    for tt in range(spike_length-1):
        pn_spike_all = np.concatenate((t_zeros, pn_spike_all[:-1,:]))
        ln_spike_all = np.concatenate((t_zeros, ln_spike_all[:-1,:]))
        u_orn = u_orn - ln_spike_all + pn_spike_all
    pn_spike_all = None
    ln_spike_all = None
    orn_spike_all = None
    t_zeros = None
    


    #******************************************************************
    # Calculate the SDF of ORN spike matrix
    orn_spike_matrix = np.asarray(np.where(num_spike_orn))
    orn_spike_matrix[0,:] = orn_spike_matrix[0,:]/pts_ms
    orn_spike_matrix = np.transpose(orn_spike_matrix)
    orn_sdf_norm, orn_sdf_time = sdf_krofczik.main(spike_mat = orn_spike_matrix, 
                                               tau_sdf=tau_sdf, dt_sdf=dt_sdf)  # (Hz, ms)
    orn_sdf_norm = orn_sdf_norm*1e3
    
    # SAVE SDF OF ORN FIRING RATE
    if data_save & al_dyn==0:
        name_data = ['/ORNrate' +
                    '_stim_' + params2an[7] +
                    '_nsi_%.1f'%(params2an[0]) +
                    '_lnspH_%.2f'%(params2an[1]) +
                    '_dur2an_%d'%(params2an[2]) +
                    '_delays2an_%d'%(params2an[3]) +
                    '_peak_%.2f'%(params2an[4]) +
                    '_peakratio_%.1f'%(params2an[5]) + # 
                    '.pickle'] #'_rho_%.1f'%(params2an[6]) +
        if ext_stimulus:
            name_data = ['/ORNrate' +
                    '_stim_' + params2an[7] +
                    '_nsi_%.1f'%(params2an[0]) +
                    '_lnspH_%.2f'%(params2an[1]) +
                    '.pickle']
            
        output_names = ['t', 'u_od', 'orn_sdf_norm', 'orn_sdf_time', ]        
        
        params2an_names = ['nsi_value', 'ln_spike_height', 'dur2an', 'delays2an', 
                           'peak', 'peak_ratio', 'rho', 'stim_type', ]

        with open(fld_analysis+name_data[0], 'wb') as f:
            pickle.dump([params2an, t, u_od, orn_sdf_norm, orn_sdf_time, 
                         params2an_names, output_names], f)

    # *****************************************************************
    # FIGURE ORN 

    if orn_fig:  
        t2plot = -200, 1000 #t2simulate #-t_on, t2simulate-t_on
        rs = 4 # number of rows
        cs = 1 # number of cols

        if stimulus == 'pl':
            t2plot = 0, 1000#2000, 4000
            rs = 2 # number of rows
            cs = 2 # number of cols
            
        if ext_stimulus:
            t2plot = 0, t2simulate
            
        panels_id = ['a.', 'b.', 'c.', 'd.']
        fig_orn = plt.figure(figsize=[8.5, 8])
#        fig_orn.canvas.manager.window.wm_geometry("+%d+%d" % fig_position )
        fig_orn.tight_layout()
        
        ax_orn1 = plt.subplot(rs, cs, 1)
        ax_orn2 = ax_orn1.twinx()
        ax_orn3 = plt.subplot(rs, cs, 2)
        ax_orn4 = ax_orn3.twinx()
        ax_orn_sc = plt.subplot(rs, cs, 3)
        ax_orn_fr = plt.subplot(rs, cs, 4)
        
        
        ax_orn1.plot(t-t_on, u_od[:,0], color= green,linewidth=lw+1, 
                     label=r'Glom %d'%(1))
#        ax_orn1.plot(t-t_on, u_od[:,1], '--', color=purple,linewidth=lw, 
#                     label=r'Glom %d'%(2)) # before it was multiplied by 100
        ax_orn2.plot(t-t_on, r_orn[:,0], color=col_glo[1,:]/2,linewidth=lw+1,
                     label=r'r, glom: %d'%(1))
#        ax_orn2.plot(t-t_on, r_orn[:,1],'--', color=col_glo[1,:]/2,linewidth=lw,
#                     label=r'r, glom: %d'%(2))
        ax_orn3.plot(t-t_on, x_orn[:,0], color=green,linewidth=lw+1,
                     label=r'Od, glom : %d'%(0))
#        ax_orn3.plot(t-t_on, x_orn[:,1], '--',color=purple,linewidth=lw, 
#                     label=r'Od, glom : %d'%(1))
        ax_orn4.plot(t-t_on, y_orn[:,0], color=col_glo[1,:]/2, linewidth=lw+1, 
                     label=r'Od, glom : %d'%(0))
#        ax_orn4.plot(t-t_on, y_orn[:,1], '--',color=col_glo[1,:]/2, linewidth=lw, 
#                     label=r'Od, glom : %d'%(1))
        ax_orn_fr.plot(orn_sdf_time-t_on, np.mean(orn_sdf_norm[:,:num_orns_glo], axis=1), 
                     color=green,  linewidth=lw+1,label='sdf glo 1')
        ax_orn_fr.plot(orn_sdf_time-t_on, np.mean(orn_sdf_norm[:,num_orns_glo:], axis=1), 
                     color=purple, linewidth=lw,label='sdf glo 2')
        spikes_orn_0 = np.argwhere(num_spike_orn[:,:num_orns_glo])
        spikes_orn_1 = np.argwhere(num_spike_orn[:,num_orns_glo:])
        
        ax_orn_sc.scatter(spikes_orn_0[:,0]/pts_ms-t_on, 
                        spikes_orn_0[:,1], color=purple, s=10)
        ax_orn_sc.scatter(spikes_orn_1[:,0]/pts_ms-t_on, 
                        num_orns_glo+spikes_orn_1[:,1], color=green, s=10)


        ax_orn1.tick_params(axis='both', which='major', labelsize=label_fs-5)
        ax_orn2.tick_params(axis='both', which='major', labelsize=label_fs-5)
        ax_orn3.tick_params(axis='both', which='major', labelsize=label_fs-5)
        ax_orn4.tick_params(axis='both', which='major', labelsize=label_fs-5)
        ax_orn_fr.tick_params(axis='both', which='major', labelsize=label_fs-5)
        ax_orn_sc.tick_params(axis='both', which='major', labelsize=label_fs-5)
        
        ax_orn1.yaxis.label.set_color(green)
        ax_orn1.set_ylabel('Odor \n concentration', fontsize=label_fs)
        ax_orn2.yaxis.label.set_color(col_glo[0,:]/2)
        ax_orn2.set_ylabel(r'r ', fontsize=label_fs)
        ax_orn3.yaxis.label.set_color(green)
        ax_orn3.set_ylabel(r'y ', fontsize=label_fs)
        ax_orn4.yaxis.label.set_color(col_glo[1,:]/2)
        ax_orn4.set_ylabel(r'x ', fontsize=label_fs)
        ax_orn_fr.set_ylabel('firing rates (Hz)', fontsize=label_fs)
        ax_orn_fr.set_xlabel('Time  (ms)', fontsize=label_fs) 
        ax_orn_sc.set_ylabel('Neuron id', fontsize=label_fs)

        ax_orn1.text(-.15, 1.25, panels_id[0], transform=ax_orn1.transAxes, color=blue,
                          fontsize=panel_fs, fontweight='bold', va='top', ha='right')
        ax_orn3.text(-.15, 1.25, panels_id[1], transform=ax_orn3.transAxes, color=blue,
                          fontsize=panel_fs, fontweight='bold', va='top', ha='right')
        ax_orn_sc.text(-.15, 1.25, panels_id[2], transform=ax_orn_sc.transAxes, color=blue,
                          fontsize=panel_fs, fontweight='bold', va='top', ha='right')
        ax_orn_fr.text(-.15, 1.25, panels_id[3], transform=ax_orn_fr.transAxes, color=blue,
                          fontsize=panel_fs, fontweight='bold', va='top', ha='right')
        
        ax_orn1.spines['top'].set_color('none')
        ax_orn2.spines['top'].set_color('none')
        ax_orn3.spines['top'].set_color('none')
        ax_orn4.spines['top'].set_color('none')
        ax_orn_sc.spines['right'].set_color('none')
        ax_orn_sc.spines['top'].set_color('none')
        ax_orn_fr.spines['right'].set_color('none')
        ax_orn_fr.spines['top'].set_color('none')
        
        ll, bb, ww, hh = ax_orn1.get_position().bounds
        ww_new = ww - 0.04
        bb_plus = 0.015
        ll_new = ll+.05
        ax_orn1.set_position([ll_new, bb+2*bb_plus, ww_new, hh])
        ll, bb, ww, hh = ax_orn2.get_position().bounds
        ax_orn2.set_position([ll_new, bb+2*bb_plus, ww_new, hh])
        ll, bb, ww, hh = ax_orn3.get_position().bounds
        ax_orn3.set_position([ll_new, bb+1.5*bb_plus, ww_new, hh])
        ll, bb, ww, hh = ax_orn4.get_position().bounds
        ax_orn4.set_position([ll_new, bb+1.5*bb_plus, ww_new, hh])
        ll, bb, ww, hh = ax_orn_sc.get_position().bounds
        ax_orn_sc.set_position([ll_new, bb+bb_plus, ww_new, hh])
        ll, bb, ww, hh = ax_orn_fr.get_position().bounds
        ax_orn_fr.set_position([ll_new, bb-bb_plus, ww_new, hh])
        
        ax_orn1.set_xlim((t2plot))
        ax_orn2.set_xlim((t2plot))
        ax_orn3.set_xlim((t2plot))
        ax_orn4.set_xlim((t2plot))
        ax_orn_sc.set_xlim((t2plot))
        ax_orn_fr.set_xlim((t2plot))
        
        if fig_save:
            if ext_stimulus:
                fig_name_orn = '/ORN_' + params2an[7] + '.png'
            else:
                fig_name_orn = ['/ORN_t1_' + '%d'%(t_on-t_on) + '-' + '%d'%(t_off-t_on) + 
                            '_t2_' + '%d'%(t_on2-t_on) + '-' + '%d'%(t_off2-t_on) + 
                            '_peak1_'  + '%0.2f'%(peak1) +
                            '_peak2_'  + '%0.2f'%(peak2) +
                            '.png']
            
            fig_orn.savefig(fld_analysis + fig_name_orn)
    # ******************************************************************
 
    
    # *****************************************************************
    # AL SIMULATION 
    # *****************************************************************
      
                                
    # *****************************************************************
    # PN and LN PARAMETERS and OUTPUT VECTORS


    #**************************************
    # PN PARAMETERS
    c_pn                = .5        # 0.5
    a_s_pn              = 2.5       # 2.5
    g_pn                = 1         # IF neuron capacitance
    
    vrest_pn            = -6.5      # -4.5 [mV] resting potential
    vrev_pn             = 15.0      # [mV] reversal potential
    
    tau_s_pn            = 10        # [ms]
    a_x_pn              = 2.         # ORN input coeff for adaptation variable x_pn
    tau_x_pn            = 600    # [ms] time scale for dynamics of adaptation variable x_pn
    x_pn0               = 0.48*np.ones(num_pns_tot)     # 0.27
    
    pn_params  = np.array([tau_s_pn, c_pn, g_pn, a_s_pn, vrev_pn, vrest_pn])
    
    #**************************************
    # LN PARAMETERS
    c_ln                = .5        # 0.5
    a_s_ln              = 2.5       # 2.5
    g_ln                = 1         # IF neuron capacitance
    
    vrest_ln            = -3.0      # -1.5 [mV] resting potential
    vrev_ln             = 15.0      # [mV] reversal potential
    
    tau_s_ln            = 10        # [ms]
    a_x_ln              = 10.         # ORN input coeff for adaptation variable x_ln
    tau_x_ln            = 600    # [ms] time scale for dynamics of adaptation variable x_ln
    x_ln0               = 0.025*np.ones(num_pns_tot) # 0.2
    ln_params = np.array([tau_s_ln, c_ln, g_ln, a_s_ln, vrev_ln, vrest_ln])
    #**************************************
    
    # INITIALIZE LN to PN output vectors
    x_pn            = np.zeros((n2sim, num_pns_tot))
    u_pn            = np.zeros((n2sim, num_lns_tot))
    u_ln            = np.zeros((n2sim, num_pns_tot))
    x_ln            = np.zeros((n2sim, num_pns_tot))
    
    # INITIALIZE PN output vectors
    num_spike_pn    = np.zeros((n2sim, num_pns_tot))
    
    # INITIALIZE LN output vectors
    s_ln            = np.zeros((n2sim, num_lns_tot))
    v_ln            = np.zeros((n2sim, num_lns_tot))
    num_spike_ln    = np.zeros((n2sim, num_lns_tot))  
    
    # PN and LN params initial conditions
    x_pn[0, :]      = x_pn0
    s_pn            = np.zeros((n2sim, num_pns_tot))
    v_pn            = np.ones((n2sim, num_pns_tot))*vrest_pn
    pn_ref_cnt      = np.zeros(num_pns_tot) # Refractory period counter starts from 0
    
    x_ln[0, :]      = x_ln0
    s_ln            = np.zeros((n2sim, num_lns_tot))
    v_ln            = np.ones((n2sim, num_lns_tot))*vrest_ln
    ln_ref_cnt      = np.zeros(num_lns_tot) # initially the ref period cnter is equal to 0
            
    
    
    if al_dyn:
        # *****************************************************************
        # solve ODE for PN and LN
        for tt in range(1, n2sim-t_ref-1):
            # span for next time step
            tspan = [t[tt-1],t[tt]]
            
            pp_rnd = np.arange(num_pns_tot) # np.random.permutation(num_pns_tot)
            
            # ******************************************************************
            # Vectorized and fast UPDATE PNS 
            # ******************************************************************
            # adaptation variable of PN neuron
            x_pn[tt, pp_rnd] = x_adapt_ex(x_pn[tt-1,pp_rnd],tspan, 
                    u_orn[tt, pp_rnd], tau_x_pn, a_x_pn, )        
        
            # Inhibitory input to PNs
            x_ln[tt, pp_rnd] = x_ln_fun_ex(x_ln[tt-1, pp_rnd],tspan, 
                    u_ln[tt, pp_rnd], tau_x_ln, a_x_ln, )
        
            # *********************************
            # ORN -> PN synapses
            
            # *********************************
            # For those PNs whose ref_cnt is different from zero:
            pn_ref_0 = pn_ref_cnt==0
            s_pn[tt, pn_ref_0] = orn2pn_s_ex(s_pn[tt-1, pn_ref_0],tspan, 
                u_orn[tt, pn_ref_0], x_pn[tt-1, pn_ref_0], x_ln[tt-1, pn_ref_0], pn_params, )
            v_pn[tt, pn_ref_0] = orn2pn_v_ex(v_pn[tt-1, pn_ref_0],tspan, 
                    s_pn[tt-1, pn_ref_0], pn_params, )
            
            # *********************************
            # For those PNs whose ref_cnt is different from zero:
            pn_ref_no0 = pn_ref_cnt!=0
            pn_ref_cnt[pn_ref_no0] = pn_ref_cnt[pn_ref_no0] - 1  # Refractory period count down
            
            # For those PNs whose Voltage is above threshold AND whose ref_cnt is equal to zero:
            pn_above_thr = (v_pn[tt, :] >= thr) & (pn_ref_cnt==0)
            num_spike_pn[tt, pn_above_thr] = num_spike_pn[tt, pn_above_thr] + 1
            u_pn[tt:tt+spike_length, :] = (u_pn[tt:tt+spike_length, :] + 
                    np.sum(pn_ln_mat[pn_above_thr,:], axis=0))
            pn_ref_cnt[pn_above_thr] = t_ref
            
            # *********************************
            # PN -> LN synapses        
            # *********************************
            # For those LNs whose ref_cnt is different from zero:
            ln_ref_0 = ln_ref_cnt==0
            s_ln[tt, ln_ref_0] = pn2ln_s_ex(s_ln[tt-1, ln_ref_0], tspan, 
                        u_pn[tt, ln_ref_0], ln_params, )
            v_ln[tt, ln_ref_0] = pn2ln_v_ex(v_ln[tt-1, ln_ref_0], tspan, 
                        s_ln[tt-1, ln_ref_0], ln_params, )
            
            # *********************************
            # For those LNs whose ref_cnt is different from zero:
            ln_ref_no0 = ln_ref_cnt!=0
            ln_ref_cnt[ln_ref_no0] = ln_ref_cnt[ln_ref_no0] - 1  # Refractory period count down
            
            # For those LNs whose Voltage is above threshold AND whose ref_cnt is equal to zero:
            ln_above_thr = (v_ln[tt, :] >= thr) & (ln_ref_cnt==0)
            num_spike_ln[tt, ln_above_thr] = num_spike_ln[tt, ln_above_thr] + 1
            u_ln[tt:tt+spike_length, :] = (u_ln[tt:tt+spike_length, :] + 
                        np.sum(ln_pn_mat[ln_above_thr,:], axis=0))
            ln_ref_cnt[ln_above_thr] = t_ref
            # ******************************************************************
            
        # *****************************************************************
        # Calculate the SDF of the PNs and LNs
        pn_spike_matrix = np.asarray(np.where(num_spike_pn))
        pn_spike_matrix[0,:] = pn_spike_matrix[0,:]/pts_ms
        pn_spike_matrix = np.transpose(pn_spike_matrix)
        pn_sdf_norm, pn_sdf_time = sdf_krofczik.main(spike_mat = 
                            pn_spike_matrix, tau_sdf=tau_sdf, dt_sdf=dt_sdf)  # (Hz, ms)
        pn_sdf_norm = pn_sdf_norm*1e3
    
        ln_spike_matrix = np.asarray(np.where(num_spike_ln))
        ln_spike_matrix[0,:] = ln_spike_matrix[0,:]/pts_ms
        ln_spike_matrix = np.transpose(ln_spike_matrix)
        ln_sdf_norm, ln_sdf_time = sdf_krofczik.main(spike_mat = 
                            ln_spike_matrix, tau_sdf=tau_sdf, dt_sdf=dt_sdf)  # (Hz, ms)
        ln_sdf_norm = ln_sdf_norm*1e3
    
        
  
        # *************************************************************************
        # COLLECT AND SAVE DATA
        id_stim = np.flatnonzero((pn_sdf_time>t_on) & (pn_sdf_time<t_off))
        id_stim2 = np.flatnonzero((pn_sdf_time>t_on2) & (pn_sdf_time<t_off2))
        pn_sdf_dt = pn_sdf_time[1]-pn_sdf_time[0]
        
        pn_avg1  = np.mean(pn_sdf_norm[id_stim, :num_pns_glo])
        pn_m50_1 = np.sum(np.mean(pn_sdf_norm[id_stim, :num_pns_glo], axis=1)>50)*pn_sdf_dt
        pn_m100_1 = np.sum(np.mean(pn_sdf_norm[id_stim, :num_pns_glo], axis=1)>100)*pn_sdf_dt
        pn_m150_1 = np.sum(np.mean(pn_sdf_norm[id_stim, :num_pns_glo], axis=1)>150)*pn_sdf_dt
        
        pn_avg2  = np.mean(pn_sdf_norm[id_stim, num_pns_glo:])
        pn_m50_2 = np.sum(np.mean(pn_sdf_norm[id_stim2, num_pns_glo:], axis=1)>100)*pn_sdf_dt
        pn_m100_2 = np.sum(np.mean(pn_sdf_norm[id_stim2, num_pns_glo:], axis=1)>100)*pn_sdf_dt
        pn_m150_2 = np.sum(np.mean(pn_sdf_norm[id_stim2, num_pns_glo:], axis=1)>100)*pn_sdf_dt
            
        id_post_stim = np.flatnonzero((pn_sdf_time>t_on) & (pn_sdf_time<t_off+100))
        id_post_stim2 = np.flatnonzero((pn_sdf_time>t_on2) & (pn_sdf_time<t_off2+100))
        pn_peak1  = np.max(np.mean(pn_sdf_norm[id_post_stim, :num_pns_glo], axis=1)) # using average PN
        pn_peak2  = np.max(np.mean(pn_sdf_norm[id_post_stim2, num_pns_glo:], axis=1)) # using average PN
        
        
        # SAVE SDF OF conc, ORN, PN and LN FIRING RATE
        if data_save:
            name_data = ['/ORNALrate' +
                        '_stim_' + params2an[7] +
                        '_nsi_%.1f'%(params2an[0]) +
                        '_lnspH_%.2f'%(params2an[1]) +
                        '_dur2an_%d'%(params2an[2]) +
                        '_delays2an_%d'%(params2an[3]) +
                        '_peak_%.2f'%(params2an[4]) +
                        '_peakratio_%.1f'%(params2an[5]) + # 
                        '.pickle'] #'_rho_%.1f'%(params2an[6]) +  
                                
            output_names = ['t', 'u_od', 'orn_sdf_norm', 'orn_sdf_time', 
                            'pn_sdf_norm', 'pn_sdf_time', 
                            'ln_sdf_norm', 'ln_sdf_time', ]
            
            params2an_names = ['nsi_value', 'ln_spike_height', 'dur2an', 'delays2an', 
                               'peak', 'peak_ratio', 'rho', 'stim_type', ]
    
            with open(fld_analysis+name_data[0], 'wb') as f:
                pickle.dump([params2an, t, u_od, orn_sdf_norm, orn_sdf_time, 
                             pn_sdf_norm, pn_sdf_time, 
                             ln_sdf_norm, ln_sdf_time, 
                             params2an_names, output_names], f)
                
        if data_save:
            name_data = ['/ORNPNLN' +
                        '_stim_' + params2an[7] +
                        '_nsi_%.1f'%(params2an[0]) +
                        '_lnspH_%.2f'%(params2an[1]) +
                        '_dur2an_%d'%(params2an[2]) +
                        '_peak_%.1f'%(params2an[4]) +
                        '_rho_%d'%(params2an[6])]  
    
            if params2an[8]<10:
                name_data = [name_data[0] +
                             '_wmax_%.1g'%(params2an[8])]
            else:
                name_data = [name_data[0] +
                             '_wmax_%.2g'%(params2an[8])]
    
            if params2an[9]>10:
                name_data = [name_data[0] +
                        '_bmax_%.2g'%(params2an[9]) +
                        '.pickle']
            else:
                name_data = [name_data[0] +
                        '_bmax_%.1g'%(params2an[9]) +
                        '.pickle']
                    
                
            output_names = ['cor_stim', 'overlap_stim', 'cor_whiff', 
                             'interm_th', 'interm_est_1', 'interm_est_2', 'od_avg1', 
                             'od_avg2', 'orn_avg1', 'orn_avg2', 'pn_avg1', 'pn_avg2', 
                             'pn_m50_1', 'pn_m100_1', 'pn_m150_1', 
                             'pn_m50_2', 'pn_m100_2', 'pn_m150_2', ]
            
            
            params2an_names = ['nsi_value', 'ln_spike_height', 'dur2an', 'delays2an', 
                               'peak', 'peak_ratio', 'rho', 'stim_type', 'w_max', 'b_max']
            if stimulus == 'pl':
                with open(fld_analysis+name_data[0], 'wb') as f:
                    pickle.dump([params2an, cor_stim, overlap_stim, cor_whiff, 
                                 interm_th, interm_est_1, interm_est_2, od_avg1, od_avg2, orn_avg1, 
                                 orn_avg2, pn_avg1, pn_avg2, pn_m50_1, pn_m100_1, 
                                 pn_m150_1, pn_m50_2, pn_m100_2, pn_m150_2, 
                                 params2an_names, output_names], f)


    # %******************************************
    # FIGURE ORN, PN, LN
    if al_dyn & al_fig:
        # ******************************************
        # FIGURE ORN, PN, LN

        t2plot = -100, t2simulate #000-t_on, t2simulate
        rs = 4 # number of rows
        cs = 1 # number of cols
        fig_size = [7, 8] 
        
        if stimulus == 'pl':
            #lw = 1.1
            t2plot = 0, 4000
            rs = 2 # number of rows
            cs = 2 # number of cols
            fig_size = [10, 5]

        
        fig_pn = plt.figure(figsize=fig_size)
        
        ax_conc = plt.subplot(rs, cs, 1)
        ax_orn = plt.subplot(rs, cs, 2)
        ax_pn = plt.subplot(rs, cs, 3)
        ax_ln = plt.subplot(rs, cs, 4)
        
        ax_conc.plot(t-t_on, 100*u_od[:,0], color=green, linewidth=lw+2, 
                      label='glom : '+'%d'%(1))
        ax_conc.plot(t-t_on, 100*u_od[:,1], '--',color=purple, linewidth=lw+1, 
                      label='glom : '+'%d'%(2))
         
        ax_orn.plot(orn_sdf_time-t_on, np.mean(orn_sdf_norm[:,:num_orns_glo], axis=1), 
                     color=green, linewidth=lw+1,label='sdf glo 1')
        ax_orn.plot(orn_sdf_time-t_on, np.mean(orn_sdf_norm[:,num_orns_glo:], axis=1), 
                     '--',color=purple, linewidth=lw,label='sdf glo 2')
        
        for pp in range(num_pns_tot):
            if pp >= num_pns_glo:
                ax_pn.plot(pn_sdf_time-t_on, pn_sdf_norm[:,pp], '--',color=purple, 
                              linewidth=lw, label='PN : '+'%d'%(pp))
            else:
                ax_pn.plot(pn_sdf_time-t_on, pn_sdf_norm[:,pp], color=green, 
                              linewidth=lw+1, label='PN : '+'%d'%(pp))
        
        for ll in range(num_lns_tot):
            if ll >= num_lns_glo:
                ax_ln.plot(ln_sdf_time-t_on, ln_sdf_norm[:,ll], '--',color=purple, 
                              linewidth=lw, label='LN : '+'%d'%(ll))
            else:
                ax_ln.plot(ln_sdf_time-t_on, ln_sdf_norm[:,ll], color=green,
                              linewidth=lw+1, label='LN : '+'%d'%(ll))      
        ax_conc.set_xlim(t2plot)
        ax_orn.set_xlim(t2plot)
        ax_pn.set_xlim(t2plot)
        ax_ln.set_xlim(t2plot)
        
        ax_orn.set_ylim((0, 130))
        ax_pn.set_ylim((0, 150))
        ax_ln.set_ylim((0, 200))

        ax_conc.tick_params(axis='both', labelsize=label_fs)
        ax_orn.tick_params(axis='both', labelsize=label_fs)
        ax_pn.tick_params(axis='both', labelsize=label_fs)
        ax_ln.tick_params(axis='both', labelsize=label_fs)
        
        ax_conc.set_xticklabels('')
        ax_orn.set_xticklabels('')
        ax_pn.set_xticklabels('')
        
        ax_conc.set_ylabel('Input ORN ', fontsize=label_fs)
        ax_orn.set_ylabel(r' ORN  (Hz)', fontsize=label_fs)
        ax_pn.set_ylabel(r' PN  (Hz)', fontsize=label_fs)
        ax_ln.set_ylabel(r' LN  (Hz)', fontsize=label_fs)
        ax_ln.set_xlabel('Time  (ms)', fontsize=label_fs)
        if stimulus == 'pl':
            ax_orn.set_ylim((0, 150))
            ax_pn.set_ylim((0, 180))
            ax_ln.set_ylim((0, 250))
            ax_pn.set_xticks(np.linspace(0, t2plot[1], 6))
            ax_ln.set_xticks(np.linspace(0, t2plot[1], 6))
            ax_pn.set_xticklabels(np.linspace(0, t2plot[1], 6)/1e3)
            ax_ln.set_xticklabels(np.linspace(0, t2plot[1], 6)/1e3)
            ax_pn.set_xlabel('Time  (ms)', fontsize=label_fs)
            ax_conc.text(-.15, 1.15, 'a.', transform=ax_conc.transAxes,
                color=blue, fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            ax_orn.text(-.15, 1.15, 'b.', transform=ax_orn.transAxes,
                color=blue, fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            ax_pn.text(-.15, 1.15, 'c.', transform=ax_pn.transAxes,
                color=blue, fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            ax_ln.text(-.15, 1.15, 'd.', transform=ax_ln.transAxes,
                color=blue, fontsize=panel_fs, fontweight='bold', va='top', ha='right')

            
        ax_conc.spines['right'].set_color('none')
        ax_conc.spines['top'].set_color('none')
        ax_orn.spines['right'].set_color('none')
        ax_orn.spines['top'].set_color('none')
        ax_pn.spines['right'].set_color('none')
        ax_pn.spines['top'].set_color('none')
        ax_ln.spines['right'].set_color('none')
        ax_ln.spines['top'].set_color('none')
        
        if (stimulus == 'pl'):
            dx = 0
        else:
            dx = 0.05
        dy = 0.05
            
        ll, bb, ww, hh = ax_conc.get_position().bounds
        ax_conc.set_position([ll+dx, bb+dy, ww, hh])
        ll, bb, ww, hh = ax_pn.get_position().bounds
        ax_pn.set_position([ll+dx, bb+dy, ww, hh])
        ll, bb, ww, hh = ax_orn.get_position().bounds
        ax_orn.set_position([ll+.05, bb+dy, ww, hh])
        ll, bb, ww, hh = ax_ln.get_position().bounds
        ax_ln.set_position([ll+.05, bb+dy, ww, hh])
    
        if fig_save:
            if stimulus == 'ts':
                fig_pn.savefig(fld_analysis+  '/ORNPNLN' +
                            '_stim_' + params2an[7] +
                            '_nsi_%.1f'%(params2an[0]) +
                            '_lnspH_' + '%.2f'%(params2an[1]) +
                            '_dur2an_' + '%d'%(params2an[2]) +
                            '_delay2an_' + '%d'%(params2an[3]) +
                            '_peak_' + '%.1f'%(params2an[4]) +
                            '_peakratio_' + '%.1f'%(params2an[5]) +
                            '.png')
            elif stimulus == 'pl':
                fig_pn.savefig(fld_analysis+  '/ORNPNLN' +
                            '_stim_' + params2an[7] +
                            '_nsi_%.1f'%(params2an[0]) +
                            '_lnspH_' + '%.2f'%(params2an[1]) +
                            '_dur2an_' + '%d'%(params2an[2]) +
                            '_peak_' + '%.1f'%(params2an[4]) +
                            '_peakratio_' + '%.1f'%(params2an[5]) +    
                            '_rho_' + '%d'%(params2an[6]) +
                            '_wmax_' + '%.1g'%(params2an[8]) +
                            '_bmax_' + '%.1g'%(params2an[9]) +
                            '.png')
        if fig_opts[2]==False:
            plt.close()
            
    # *************************************************************************
    
    orn_stim = [orn_avg1, orn_avg2, orn_peak1, orn_peak2, ]
    if al_dyn:
        pn_stim = [pn_avg1, pn_avg2, pn_peak1, pn_peak2,]
    else:
        pn_stim = np.zeros(4)
    
    return  [orn_stim, pn_stim, ]


if __name__ == '__main__':
    print('run directly')
    stim_data_fld = ''
    stim_seed = 0   # if =np.nan() no traceable random
    
    #***********************************************
    # Trials and errors
    fld_analysis    = '../NSI_analysis/trialserrors'
    inh_conds       = ['nsi', ] #'ln', 'noin'
    stim_type       = 'ss' # 'ts'  # 'ts' # 'ss' # 'pl'
    stim_dur        = 100
    ln_spike_h      = 0.4
    nsi_str         = 0.3
    delays2an       = 0
    peak_ratio      = 1
    peak            = 1 
    
    # real plumes params
    b_max           = 3   # 3, 50, 150
    w_max           = 3   # 3, 50, 150
    rho             = 0   #[0, 1, 3, 5]: 

    orn_fig         = 1
    al_fig          = 1
    fig_ui          = 1        
    fig_save        = 0
    
    fig_opts = [orn_fig, al_fig, fig_ui, fig_save]
    print('conc: %.1f, stim_dur:%dms, spike LN: %.1f, NSI strength: %.1f'
          %(peak, stim_dur,ln_spike_h,nsi_str))
    

    if path.isdir(fld_analysis):
        print('OLD analysis fld: ' + fld_analysis)    
    else:
        print('NEW analysis fld: ' + fld_analysis)    
        mkdir(fld_analysis)
    copyfile('flynose.py', fld_analysis+'/flynose.copy.py') 
    
    
    pn_avg_dif  = 0
    pn_avg      = 0
    pn_peak     = 0

    params2an = [0, .0, stim_dur, delays2an, peak, 
                 peak_ratio, rho, stim_type,w_max,b_max]
    if len(stim_type)>2:
        params2an.append(stim_data_fld)
    tic = timeit.default_timer()
    for inh_cond in inh_conds:
        if inh_cond == 'nsi':
            params2an[0:2] = [nsi_str, .0, ]
        elif inh_cond == 'noin':
            params2an[0:2] = [0, 0, ]
        elif inh_cond == 'ln':
            params2an[0:2] = [.0, ln_spike_h,]
        
        #    params2an = [nsi_value, ln_spike_height, dur2an, delays2an, peak, peak_ratio]
        plt.ion()      # ioff() # to avoid showing the plot every time     
        
        [orn_stim, pn_stim,] = main(params2an, fig_opts, verbose = False, 
            fld_analysis = fld_analysis, stim_seed=stim_seed)
        pn_avg_dif = (pn_stim[0]-pn_stim[1])
        pn_avg = (pn_stim[0]+pn_stim[1])/2
        pn_peak = (pn_stim[2]+pn_stim[3])/2        
        
        print(inh_cond+' inh, peak:%.1f, avg:%.1f, avg dif:%.1f'
              %(pn_peak, pn_avg, pn_avg_dif))
        
        toc = timeit.default_timer()
    print('time to run %d sims: %.1fs'%(np.size(inh_conds),toc-tic))
    print('')
                        
else:
    print('run from import')

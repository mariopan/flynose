#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# script name: synapse_dynORNPN.py

"""
Created on Thu Jan 1 2019

Simulation for the dynamics of a general synapse with Rall-Seynosky-Desthexe
script name: system_ORNPNLN_corr.py
@author: mario

"""

import numpy as np
import scipy.stats as spst
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import timeit
from os import getcwd as pwd
from scipy.integrate import quad

import pickle        
from os import path
from os import mkdir
from shutil import copyfile

import sys
sys.path.insert(0, '/flynose/')

import flynose.corr_steps as corr_steps
import flynose.corr_plumes as corr_plumes
import flynose.sdf_krofczik as sdf_krofczik
#from scipy.optimize import curve_fit

# *****************************************************************
# FUNCTIONS
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

def olsen_orn_pn(nu_orn, sigma, nu_max):
    nu_pn = nu_max * np.power(nu_orn, 1.5)/(np.power(nu_orn, 1.5) + np.power(sigma,1.5))
    return nu_pn

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

def depalo_eq(z,t,u,orn_params,):
    ax = orn_params[0]
    cx = orn_params[1]
    bx = orn_params[2]
    by = orn_params[3]
    dy = orn_params[4]
    b = orn_params[5]
    d = orn_params[6]
    ar = orn_params[7]
    n = orn_params[8]
    
    r = z[0]
    x = z[1]
    y = z[2]
    drdt = b*u**n*(1-r) - d*r
    dxdt = ax*y - bx*x
    dydt = ar*r - cx*x*(1+dy*y) - by*y 
    dzdt = [drdt,dxdt,dydt]
    return dzdt

def rect_func(b, x):
    ot = b[0]/(1 + np.exp(-b[1]*(x-b[2])))
    return ot

def running_sum(x, N):
    out = np.zeros_like(x, dtype=np.float64)
    dim_len = x.shape[0]
    for ii in range(dim_len):
        if N%2 == 0:
            a, b = ii - (N-1)//2, ii + (N-1)//2 + 2
        else:
            a, b = ii - (N-1)//2, ii + (N-1)//2 + 1

        #cap indices to min and max indices
        a = max(0, a)
        b = min(dim_len, b)
        out[ii] = np.sum(x[a:b])
    return out

def pn2ln_s(z,t, u_pn, ln_params, ):

    #    ln_params = np.array([tau_s_ln, c_ln, g_ln, a_s_ln, vrev_ln, vrest_ln])
    tau_s = ln_params[0]
    a_s = ln_params[3]
    
    # PN -> LN equation of s:
    s = z[0] 
    dsdt = (a_s*u_pn*(1-s) - s)/tau_s       
    return dsdt

def pn2ln_v(z,t, s, ln_params, ):
#    ln_params = np.array([tau_s_ln, c_ln, g_ln, a_s_ln, vrev_ln, vrest_ln])
    c = ln_params[1]
    g = ln_params[2]
    vrev = ln_params[4]
    vrest = ln_params[5]
    
    # PN -> LN equations:
    v = z[0] 
    dvdt = ((vrest-v) + g*s*(vrev-v) )/c
    return dvdt

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

def pn2ln(z,t, u_pn, ln_params, ):
#    ln_params = np.array([tau_s_ln, c_ln, g_ln, a_s_ln, vrev_ln, vrest_ln])
    tau_s = ln_params[0]
    c = ln_params[1]
    g = ln_params[2]
    a_s = ln_params[3]
    vrev = ln_params[4]
    vrest = ln_params[5]
    
    # PN -> LN equations:
    s = z[0] 
    v = z[1] 
    dsdt = (a_s*u_pn*(1-s) - s)/tau_s   
    dvdt = ((vrest-v) + g*s*(vrev-v) )/c
    
    dzdt = [dsdt, dvdt]
    return dzdt

def x_ln_fun(z,t, u_ln, tau_x, a_x,):
    x = z[0] 
    dxdt = (a_x*u_ln*(1-x) - x)/tau_x
    return dxdt

def x_ln_fun_ex(x0,t,u_ln, tau_x, a_x,):
    b = (-a_x*u_ln-1)/tau_x
    a = a_x*u_ln/tau_x
    dt = t[1]-t[0]
    y = (x0 + a/b)*np.exp(b*dt)-a/b
    return y


def orn2pn_s(z,t, u_orn, x_pn,x_ln,pn_params,):
#    pn_params  = np.array([tau_s_pn, c_pn, g_pn, a_s_pn, vrev_pn, vrest_pn])
    tau_s = pn_params[0]
    a_s = pn_params[3]
    
    # ORN -> PN equations:
    s = z[0] 
    dsdt = (a_s*u_orn*(1-s)*(1-x_pn)*(1-x_ln) - s)/tau_s
    return dsdt


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


def orn2pn_v(z,t, s, pn_params,):
#    pn_params  = np.array([tau_s_pn, c_pn, g_pn, a_s_pn, vrev_pn, vrest_pn])
    c       = pn_params[1]
    g       = pn_params[2]
    vrev    = pn_params[4]
    vrest   = pn_params[5]
    
    # ORN -> PN equations:
    v = z[0]  # v_PN
    dvdt = ((vrest-v) + g*s*(vrev-v) )/c
    return dvdt

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

def orn2pn(z,t, u_orn, x_pn,x_ln,pn_params,):
#    pn_params  = np.array([tau_s_pn, c_pn, g_pn, a_s_pn, vrev_pn, vrest_pn])
    tau_s = pn_params[0]
    c = pn_params[1]
    g = pn_params[2]
    a_s = pn_params[3]
    vrev = pn_params[4]
    vrest = pn_params[5]
    
    # ORN -> PN equations:
    s = z[0] 
    v = z[1]  # v_PN
    dsdt = (a_s*u_orn*(1-s)*(1-x_pn)*(1-x_ln) - s)/tau_s
    dvdt = ((vrest-v) + g*s*(vrev-v) )/c
    dzdt = [dsdt, dvdt]
    return dzdt

def x_adapt(z,t, u_orn, tau, a,):
    # PN adaptation variable 
    x = z[0]    
    dxdt = (a*u_orn*(1-x) - x)/tau
    return dxdt

def x_adapt_ex(x0,t,u_orn, tau, a_ad,):
    b = (-a_ad*u_orn-1)/tau
    a = a_ad*u_orn/tau
    dt = t[1]-t[0]
    y = (x0 + a/b)*np.exp(b*dt)-a/b
    return y


def poisson_isi(rate, n_spikes):
    isi = -np.log(1.0 - np.random.random_sample(n_spikes)) / rate
    return isi

def freq_eff(fr, t_ref, pts_ms):
    t_ref_sec = t_ref/1000/pts_ms # t_ref in second
    fr_eff = fr /(1+ fr*t_ref_sec) # effective firing rates (considering refractary period)
    return fr_eff  
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
    al_dyn      = 0
    al_fig      = fig_opts[1]
    stimulus    = params2an[7] # 'ss'   # 'rs'   #  'pl'  # 'ts'
    
    # *****************************************************************
    # STIMULUS GENERATION
    
    nsi             = True #  False #   flag to choose whether the simulation is with or w/o NSIs
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
    
    t2simulate      = np.maximum(params2an[2],1200) # [ms]
    t2average       = 20      # [ms] time window for moving average in the plot
    t_on            = 300      # [ms]
    # average sdf parameters
    tau_sdf         = 20
    dt_sdf          = 5   
    
    pts_ms          = 5             # simulated points per ms
    n2sim           = pts_ms*t2simulate + 1      # number of time points
    t               = np.linspace(0, t2simulate, n2sim) # time points
    
    t_off           = t_on + dur2an     # [ms]
    stim_off        = t_off*pts_ms    
    t_on2           = t_on+delays2an     # [ms]
    stim_on2        = t_on2*pts_ms
    t_off2          = t_on2 + dur2an    # [ms]
    stim_off2       = t_off2*pts_ms

    cov_hom         = 0.4 # Covariance value homotypic ORNs
    nu_pn_noise     = 200 # Hz  - PNs Noise into each PNs
    nu_ln_noise     = 0 # Hz    - LNs Noise into each PNs
    
    # STEP stimulus 1st glomerulus
    stim_on         = t_on*pts_ms 
    # WARNING: useful only for correlation measures
    # spon_time       = np.arange(50*pts_ms, stim_on-10*pts_ms)
    # stim_time       = np.arange(stim_on+10*pts_ms, stim_on+200*pts_ms)
        
    # initialize output vectors
    num_glo         = 2     # number of glomeruli
    u_od            = np.zeros((n2sim, num_glo))
    cor_stim        = np.nan
    overlap_stim    = np.nan
    cor_whiff       = np.nan
    
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
        pdf_wh, logbins, wh_mean = whiffs_blanks_pdf(whiff_min, whiff_max, g)
        pdf_bl, logbins, bl_mean = whiffs_blanks_pdf(blank_min, blank_max, g)
        
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
        
        od_avg1_tmp = np.mean(out_y)#stim_off
        od_avg2_tmp = np.mean(out_w)#stim_off
        print('od_avg1 tmp:%.2f'%(od_avg1_tmp))
        print('od_avg2 tmp:%.2f'%(od_avg2_tmp))
        
        cor_stim        = -2
        overlap_stim    = -2
        cor_whiff       = -2
        
        interm_est_1 = np.sum(out_y>0)/(t2sim_s*sample_rate)
        interm_est_2 = np.sum(out_w>0)/(t2sim_s*sample_rate)
#        interm_est[rr,1] = np.sum(out_w>0)/(t2sim*sample_rate)
        if (np.sum(out_y)!=0) & (np.sum(out_w)!=0):
            cor_stim        = np.corrcoef(out_y, out_w)[1,0]
            overlap_stim    = overlap(out_y, out_w)
            nonzero_concs1  = out_y[(out_y>0) & (out_w>0)]
            nonzero_concs2  = out_w[(out_y>0) & (out_w>0)]
            cor_whiff       = np.corrcoef(nonzero_concs1, nonzero_concs2)[0, 1] # np.corrcoef(concs1, concs2)[0, 1]
    
    od_avg1 = np.mean(u_od[stim_on:, 0])#stim_off
    od_avg2 = np.mean(u_od[stim_on:, 1])#stim_off
    print('od_avg1:%.2f'%(od_avg1))
    print('od_avg2:%.2f'%(od_avg2))
    
    
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
    
    #    # Correlation calculus params
    #    wind_len    = 2*pts_ms    # window length for the sliding window average
    #    ss_len      = 1*pts_ms    # subsample length
    
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
    nu_orn_run          = np.zeros((n2sim, num_glo))
    
    
    # initial conditions
    z_orn0          = np.ones((num_glo, 3))*[0, 0, 0]
    r_orn[0,:]        = z_orn0[:, 0]
    x_orn[0,:]        = z_orn0[:, 1]
    y_orn[0,:]        = z_orn0[:, 2]
    
    for tt in range(1, n2sim-t_ref-1):
        # span for next time step
        tspan = [t[tt-1],t[tt]]
        if nsi == False:
            for gg in range(num_glo):
                z_orn = odeint(depalo_eq, z_orn0[gg,:], tspan,
                               args=(u_od[tt, gg],orn_params,))
                z_orn0[gg,:] = z_orn[1]
                r_orn[tt,gg] = z_orn[1][0]
                x_orn[tt,gg] = z_orn[1][1]
                y_orn[tt,gg] = z_orn[1][2]
                   
                nu_orn[tt,gg] = rect_func(B0, y_orn[tt,gg])
        elif nsi == True:
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
    out_orn_run     = np.zeros((n2sim, num_pns_tot))
    
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

        output_names = ['t', 'u_od', 'orn_sdf_norm', 'orn_sdf_time', ]        
        
        params2an_names = ['nsi_value', 'ln_spike_height', 'dur2an', 'delays2an', 
                           'peak', 'peak_ratio', 'rho', 'stim_type', ]

        with open(fld_analysis+name_data[0], 'wb') as f:
            pickle.dump([params2an, t, u_od, orn_sdf_norm, orn_sdf_time, 
                         params2an_names, output_names], f)

    # *****************************************************************
    # FIGURE ORN 
    for gg in range(num_glo):
        nu_orn_run[:,gg] = (running_sum(nu_orn[:,gg], int(pts_ms*t2average))
            /(t2average*pts_ms))
    for pp in range(num_pns_tot):
        out_orn_run[:,pp] = (running_sum(u_orn[:,pp], int(pts_ms*t2average))
            /(t2average*pts_ms))

    if orn_fig:  
        t2plot = -200, 1000 #t2simulate #-t_on, t2simulate-t_on
        if stimulus == 'pl':
#            lw = 1.1
            t2plot = 2000, 4000
        rs = 4 # number of rows
        cs = 1 # number of cols
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
        ax_orn3.set_ylabel(r'x ', fontsize=label_fs)
        ax_orn4.yaxis.label.set_color(col_glo[1,:]/2)
        ax_orn4.set_ylabel(r'y ', fontsize=label_fs)
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
            fig_orn.savefig(fld_analysis + 
                            '/ORN_t1_' + '%d'%(t_on-t_on) + '-' + '%d'%(t_off-t_on) + 
                            '_t2_' + '%d'%(t_on2-t_on) + '-' + '%d'%(t_off2-t_on) + 
                            '_peak1_'  + '%0.2f'%(peak1) +
                            '_peak2_'  + '%0.2f'%(peak2) +
                            '.png')
    
             
    #    %%*****************************************************************
            
    #    # *****************************************************************
    #    # CORRELATION BETWEEN ORN RESPONSES
    #    orns2an_glo1    = np.array([0,1,2,3,4])
    #    orns2an_glo2    = np.array([ 40, 41,42,43,44])
    #    orns2an         = np.concatenate((orns2an_glo1,orns2an_glo2))
    #    num_orns2an     = orns2an.size
    #    num_orn_g1_2an  = int(num_orns2an/2)
    #        
    #    conds = ['spon', 'stim']
    #    cor_orn_m = np.zeros((np.size(conds), num_glo))
    #    
    #    for stim_cond in conds:
    #        cor_orn = np.nan*np.ones((num_orns2an, num_orns2an))                    
    #        for oo in range(num_orns2an-1):
    #            if stim_cond == 'spon':
    #                orn0 = num_spike_orn[spon_time,oo]  
    #            else:
    #                orn0 = num_spike_orn[stim_time,oo] 
    #            orn0 = running_sum(orn0, wind_len)/wind_len
    #            orn0 = np.array(orn0[::ss_len])
    #            if sum(orn0) != 0:
    #                for oo1 in range(oo+1, num_orns2an):
    #                    if stim_cond == 'spon':
    #                        orn1 = num_spike_orn[spon_time, oo1]
    #                    elif stim_cond == 'stim':
    #                        orn1 = num_spike_orn[stim_time, oo1]                     
    #                    orn1 = running_sum(orn1, wind_len)/wind_len
    #                    orn1 = np.array(orn1[::ss_len])              
    #                    if sum(orn1) != 0:
    #                        cor_orn[oo,oo1] = np.corrcoef(orn1, orn0)[0, 1]
    #                        cor_orn[oo1,oo] = cor_orn[oo,oo1] 
    #                
    #        cor_orn_g1 = np.arctanh(cor_orn[:num_orn_g1_2an, :num_orn_g1_2an])
    #        cor_orn_g2 = np.arctanh(cor_orn[num_orn_g1_2an:,num_orn_g1_2an:])
    #        
    #        if stim_cond == 'spon':
    #            cor_orn_m[0, 0] = np.tanh(np.nanmean(cor_orn_g1))
    #            cor_orn_m[0, 1] = np.tanh(np.nanmean(cor_orn_g2))
    #            if verbose:
    #                print('cor ORNs g1 spontan:' + '%.2f'%cor_orn_m[0, 0])
    #                print('cor ORNs g2 spontan:' + '%.2f'%cor_orn_m[0, 1])
    #        elif stim_cond == 'stim':
    #            cor_orn_m[1, 0] = np.tanh(np.nanmean(cor_orn_g1))
    #            cor_orn_m[1, 1] = np.tanh(np.nanmean(cor_orn_g2))
    #            if verbose:
    #                print('cor ORNs g1 stimul:' + '%.2f'%cor_orn_m[1, 0])
    #                print('cor ORNs g2 un-stim:' + '%.2f'%cor_orn_m[1, 1])
    #   # ******************************************
        
#    # %%******************************************
#    
#    # FIGURE corr of ORNs
#    fig = plt.figure(figsize=(15,8), )
#    ax1 = plt.subplot(2,3,1)
#    sh1 = ax1.imshow(cor_orn_g1)
#    plt.colorbar(sh1, ax=ax1, fraction=0.046, pad=0.04)
#    if stim_cond == 'spon':
#        ax1.set_title('spontaneous activity')
#        ax1.set_title('spontaneous activity')
#    else:
#        ax1.set_title('stimulated')
#            
#    cor_tmp = cor_orn_g1[~np.isnan(cor_orn_g1)].flatten()
#    if ~np.any(np.isinf(cor_tmp)):
#        ax5 = plt.subplot(2,3,2)
#        ax5.hist(cor_tmp)
#        ax6 = plt.subplot(2,3,3)
#        ax6.plot(cor_tmp)
#    else:
#        print('some inf value')
#    
#    ax4 = plt.subplot(2,3,4)
#    sh2 = ax4.imshow(cor_orn_g2)
#    plt.colorbar(sh2, ax=ax4, fraction=0.046, pad=0.04)
#    cor_tmp = cor_orn_g2[~np.isnan(cor_orn_g2)].flatten()
#    if ~np.any(np.isinf(cor_tmp)):
#        ax5 = plt.subplot(2,3,5)
#        ax5.hist(cor_tmp)
#        ax6 = plt.subplot(2,3,6)
#        ax6.plot(cor_tmp)
#    else:
#        print('some inf value')
#    
#    # *****************************************************************         
                                
    # *****************************************************************
    # PN and LN PARAMETERS and OUTPUT VECTORS
#def pn_ln_sim(net_params, stim_params):
#    n2sim       = stim_params[0]
#    t           = stim_params[1]
#    u_orn       = stim_params[2]
#    
#    num_pns_tot = net_params[0]
#    num_lns_tot = net_params[1]
#    pn_params   = net_params[2]
#    ln_params   = net_params[3]
#    cmn_params  = net_params[4]

    # *****************************************************************
    # AL SIMULATION 
    # *****************************************************************

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
    nu_pn_run       = np.zeros((n2sim, num_pns_tot))
    
    # INITIALIZE LN output vectors
    s_ln            = np.zeros((n2sim, num_lns_tot))
    v_ln            = np.zeros((n2sim, num_lns_tot))
    num_spike_ln    = np.zeros((n2sim, num_lns_tot))  
    nu_ln_run       = np.zeros((n2sim, num_lns_tot))
    
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
    
        # *****************************************************************
        # Calculate the running average of the PNs and LNs activities
        for pp in range(num_pns_tot):
            nu_pn_run[:,pp] = (running_sum(num_spike_pn[:,pp], int(pts_ms*t2average))
                /t2average*1e3)
        for ll in range(num_lns_tot):
            nu_ln_run[:,ll] = (running_sum(num_spike_ln[:,ll], int(pts_ms*t2average))
                /t2average*1e3)
        
  
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
    #    pn_peak1  = np.max(np.mean(nu_pn_run[stim_on:stim_off+100*pts_ms, :num_pns_glo], axis=1)) # using average PN
    #    pn_peak2  = np.max(np.mean(nu_pn_run[stim_on:stim_off+100*pts_ms, num_pns_glo:], axis=1)) # using average PN
        
    #    pn_avg1  = np.mean(nu_pn_run[stim_on:stim_off, :num_pns_glo]) # using average PN
    #    pn_m50_1  =np.sum(np.mean(nu_pn_run[stim_on:stim_off, :num_pns_glo], axis=1)>50)
    #    pn_m100_1  =np.sum(np.mean(nu_pn_run[stim_on:stim_off, :num_pns_glo], axis=1)>100)
    #    pn_m150_1  =np.sum(np.mean(nu_pn_run[stim_on:stim_off, :num_pns_glo], axis=1)>150)
        
    #    pn_avg2  = np.mean(nu_pn_run[stim_on:stim_off, num_pns_glo:]) 
    #    pn_m50_2  =np.sum(np.mean(nu_pn_run[stim_on:stim_off, num_pns_glo:], axis=1)>50)
    #    pn_m100_2  =np.sum(np.mean(nu_pn_run[stim_on:stim_off, num_pns_glo:], axis=1)>100)
    #    pn_m150_2  =np.sum(np.mean(nu_pn_run[stim_on:stim_off, num_pns_glo:], axis=1)>150)
        
        
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
        # %%******************************************
        # FIGURE ORN, PN, LN

        t2plot = -100, t2simulate #000-t_on, t2simulate
        if stimulus == 'pl':
            #lw = 1.1
            t2plot = 2000, 4000
        rs = 4 # number of rows
        cs = 1 # number of cols
        
        fig_pn = plt.figure(figsize=[7, 10])
#        fig_pn.canvas.manager.window.wm_geometry("+%d+%d" % fig_position )
#        fig_pn.tight_layout()
        
        ax_conc = plt.subplot(rs, cs, 1)
        ax_orn = plt.subplot(rs, cs, 1+cs)
        ax_pn = plt.subplot(rs, cs, 1+cs*2)
        ax_ln = plt.subplot(rs, cs, 1+cs*3)
        
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
        ax_conc.spines['right'].set_color('none')
        ax_conc.spines['top'].set_color('none')
        ax_orn.spines['right'].set_color('none')
        ax_orn.spines['top'].set_color('none')
        ax_pn.spines['right'].set_color('none')
        ax_pn.spines['top'].set_color('none')
        ax_ln.spines['right'].set_color('none')
        ax_ln.spines['top'].set_color('none')
        
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
    
    
        #    # *****************************************************************
        #    # %%   CORRELATION BETWEEN PNs trains of spikes
        #    cor_pn_m = np.zeros((np.size(conds), num_glo+1))
        #    
        #    for stim_cond in ['spon', 'stim']:
        #        cor_pn = np.nan*np.ones((num_pns_tot, num_pns_tot))
        #        for pp in range(num_pns_tot-1):
        #            if stim_cond == 'spon':
        #                pn0 = num_spike_pn[spon_time, pp] 
        #            elif stim_cond == 'stim':
        #                pn0 = num_spike_pn[stim_time, pp] 
        #            pn0 = running_sum(pn0, wind_len)/wind_len
        #            pn0 = np.array(pn0[::ss_len])
        #            if sum(pn0) != 0:
        #                for pp1 in range(pp+1, num_pns_tot):
        #                    if stim_cond == 'spon':
        #                        pn1 = num_spike_pn[spon_time, pp1] 
        #                    elif stim_cond == 'stim':
        #                        pn1 = num_spike_pn[stim_time, pp1] 
        #                    pn1 = running_sum(pn1, wind_len)/wind_len
        #                    pn1 = np.array(pn1[::ss_len])                
        #                    if sum(pn1) != 0:
        #                        cor_pn[pp,pp1] = np.corrcoef(pn1, pn0)[0, 1]
        #                        cor_pn[pp1,pp] = cor_pn[pp,pp1]
        #                            
        #        cor_pn_g1 = cor_pn[:num_pns_glo,:num_pns_glo]
        #        cor_pn_g2 = cor_pn[num_pns_glo:,num_pns_glo:]
        #        cor_pn_g12 = cor_pn[:num_pns_glo,num_pns_glo:]
        #        
        #        # mean are calculated from the Fisher transform:
        #        if stim_cond == 'spon':
        #            cor_pn_m[0, 0] = np.tanh(np.nanmean(np.arctanh(cor_pn_g1)))
        #            cor_pn_m[0, 1] = np.tanh(np.nanmean(np.arctanh(cor_pn_g2)))
        #            cor_pn_m[0, 2] = np.tanh(np.nanmean(np.arctanh(cor_pn_g12)))
        #            if verbose:
        #                print('cor PN g1 spontan:' + '%.2f'%cor_pn_m[0, 0])
        #                print('cor PN g2 spontan:' + '%.2f'%cor_pn_m[0, 1])
        #                print('cor PN g1-g2 spontan:' + '%.2f'%cor_pn_m[0, 2])
        #        elif stim_cond == 'stim':
        #            cor_pn_m[1, 0] = np.tanh(np.nanmean(np.arctanh(cor_pn_g1)))
        #            cor_pn_m[1, 1] = np.tanh(np.nanmean(np.arctanh(cor_pn_g2)))
        #            cor_pn_m[1, 2] = np.tanh(np.nanmean(np.arctanh(cor_pn_g12)))
        #            if verbose:
        #                print('cor PN g1 stimul:' + '%.2f'%cor_pn_m[1, 0])
        #                print('cor PN g2 un-stim:' + '%.2f'%cor_pn_m[1, 1])
        #                print('cor PN g1-g2 stimul:' + '%.2f'%cor_pn_m[1, 2])               
        #        # %%******************************************
        #        # FIGURE CORRELATION of PNs
        #        fig2 = plt.figure(figsize=(15,8), )
        #        rs = 3 # number of rows
        #        cs = 3 # number of cols
        #    
        #        ax1 = plt.subplot(rs,cs,1)
        #        sh1 = ax1.imshow(cor_pn_g1)
        #        plt.colorbar(sh1, ax=ax1, fraction=0.046, pad=0.04)
        #        cor_tmp = cor_pn_g1[~np.isnan(cor_pn_g1)].flatten()
        #        if ~np.any(np.isinf(cor_tmp)):
        #            ax5 = plt.subplot(rs,cs,2)
        #            ax5.hist(cor_tmp)
        #            ax6 = plt.subplot(rs,cs,3)
        #            ax6.plot(cor_tmp)
        #        else:
        #            print('some inf value')
        #        
        #        ax4 = plt.subplot(rs,cs,4)
        #        sh2 = ax4.imshow(cor_pn_g2)
        #        plt.colorbar(sh2, ax=ax4, fraction=0.046, pad=0.04)
        #        cor_tmp = cor_pn_g2[~np.isnan(cor_pn_g2)].flatten()
        #        if ~np.any(np.isinf(cor_tmp)):
        #            ax5 = plt.subplot(rs,cs,5)
        #            ax5.hist(cor_tmp)
        #            ax6 = plt.subplot(rs,cs,6)
        #            ax6.plot(cor_tmp)
        #        else:
        #            print('some inf value')
        #
        #        ax7 = plt.subplot(rs,cs, 7)
        #        sh3 = ax7.imshow(cor_pn_g12)
        #        plt.colorbar(sh3, ax=ax7, fraction=0.046, pad=0.04)
        #        cor_tmp = cor_pn_g12[~np.isnan(cor_pn_g12)].flatten()
        #        if ~np.any(np.isinf(cor_tmp)):
        #            ax8 = plt.subplot(rs,cs,8)
        #            ax8.hist(cor_tmp)
        #            ax9 = plt.subplot(rs,cs,9)
        #            ax9.plot(cor_tmp)
        #        else:
        #            print('some inf value')
        #        if stim_cond == 'spon':
        #            ax1.set_title('Spontaneous g1')
        #            ax4.set_title('Spontaneous g2')
        #            ax7.set_title('Spontaneous g1 and g2')
        #            if fig_save:
        #                fig2.savefig(fld_analysis + '/Noise_Corr_PN_spon.png')
        #        elif stim_cond == 'stim':
        #            ax1.set_title('Stimulated g1')
        #            ax4.set_title('Non-Stimulated g2')
        #            ax7.set_title('Stim g1 and non-Stimulated g2')
        #            if fig_save:
        #                fig2.savefig(fld_analysis + '/Noise_Corr_PN_stim.png')
        #        
        #
    
    orn_stim = [orn_avg1, orn_avg2, orn_peak1, orn_peak2, ]
    if al_dyn:
        pn_stim = [pn_avg1, pn_avg2, pn_peak1, pn_peak2,]
    else:
        pn_stim = np.zeros(4)
    
    return  [orn_stim, pn_stim, ]
# old return command:
#    return [nu_pn_spon, nu_pn_stim]

if __name__ == '__main__':
    print('run directly')

    for stim_seed in [0]:
#        #***********************************************
#        # Olsen-Wilson 2010 figure
#        fld_analysis = '../Olsen2010/data'
#        ln_spike_h  = 0.4
#        nsi_str     = 0.3
#        inh_conds   = ['noin'] #['nsi', 'ln', 'noin'] #
#        stim_type   = 'ss' # 'ss'  # 'ts'
#        stim_dur    = 500
#        delays2an   = 0
#        peak_ratio  = 1
#        peaks       = np.linspace(0,7,11)
#        orn_fig     = 0
#        al_fig      = 0
#        fig_ui      = 0      
#        fig_save    = 0
#        #***********************************************
        
        # FIG. ORN_response
        fld_analysis = '../NSI_analysis/ORN_dynamics' #/sdf_test
        inh_conds   = ['noin'] #
        stim_type   = 'ss' # 'ts' # 'ss' # 'rp'# '
        ln_spike_h  = 0.4
        nsi_str     = 0.3
        stim_dur    = 500
        delays2an   = 0
        peaks       = [0.8]
        peak_ratio  = 1
        orn_fig     = 1
        al_fig      = 0
        fig_ui      = 1
        fig_save    = 1
        #***********************************************        
        
#        # FIG. DelayResponse
#        fld_analysis = '../NSI_analysis/triangle_stim/triangles_delay' #
#        inh_conds = ['nsi', 'ln', 'noin'] 
#        stim_type   = 'ts' 
#        ln_spike_h  = 0.4
#        nsi_str     = 0.3
#        stim_dur    = 50  # 10 20 50 100 200 
#        delays2an   = 100 
#        peaks       = [1.8]
#        peak_ratio  = 1
#        orn_fig     = 0
#        al_fig      = 1
#        fig_ui      = 1
#        fig_save    = 1
#        #***********************************************
        
#        #***********************************************
#        # Fig.ImpulseResponse
#        fld_analysis = '../NSI_analysis/triangle_stim/ImpulseResponse'
#        inh_conds   = ['nsi', 'ln', 'noin'] #
#        stim_type   = 'ts'  # 'ts'
#        stim_dur    = 50
#        ln_spike_h  = 0.6
#        nsi_str     = 0.3
#        delays2an   = 0
#        peak_ratio  = 1
#        peaks       = [1.4,] 
#        orn_fig     = 0
#        al_fig      = 1
#        fig_ui      = 1        
        
#        #***********************************************
#        # Trials and errors
#        fld_analysis = '../NSI_analysis/trialserrors'
#        inh_conds   = ['nsi', ] #'ln', 'noin'
#        stim_type   = 'pl'  # 'ts' # 'ss'
#        stim_dur    = 10000
#        ln_spike_h  = 0.4
#        nsi_str     = 0.3
#        delays2an   = 0
#        peak_ratio  = 1
#        peaks       = [1.,] 
#        orn_fig     = 0
#        al_fig      = 0
#        fig_ui      = 0        
#        fig_save    = 0

#        #***********************************************
#        # Real plumes, example figure
#        fld_analysis = '../NSI_analysis/analysis_real_plumes/example'
#        inh_conds   = ['nsi', ] #'ln', 'noin'
#        stim_type   = 'pl'  # 'ts' # 'ss'
#        stim_dur    = 10000
#        ln_spike_h  = 0.4
#        nsi_str     = 0.3
#        delays2an   = 0
#        peak_ratio  = 1
#        peaks       = [1.5,] 
#        orn_fig     = 0
#        al_fig      = 1
#        fig_ui      = 1        
#        fig_save    = 1

        
        
        fig_opts = [orn_fig, al_fig, fig_ui]
        if path.isdir(fld_analysis):
            print('OLD analysis fld: ' + fld_analysis)    
        else:
            print('NEW analysis fld: ' + fld_analysis)    
            mkdir(fld_analysis)
        copyfile('flynose.py', fld_analysis+'/flynose.py') 
        
        n_loops     = 1
        pn_avg_dif  = np.zeros(n_loops)
        pn_avg      = np.zeros(n_loops)
        pn_peak     = np.zeros(n_loops)

        for peak in peaks:
            print('conc: %.1f, stim_dur:%dms, spike LN: %.1f, NSI strength: %.1f'
              %(peak, stim_dur,ln_spike_h,nsi_str))
            for b_max in [3]: # 3, 50, 150
                for w_max in [3]: # 3, 50, 150
                    for rho in [0]: #[0, 1, 3, 5]: 
                        params2an = [0, .0, stim_dur, delays2an, peak, 
                                     peak_ratio, rho, stim_type,w_max,b_max]
                        tic = timeit.default_timer()
                        for inh_cond in inh_conds:
                            if inh_cond == 'nsi':
                                params2an[0:2] = [nsi_str, .0, ]
                            elif inh_cond == 'noin':
                                params2an[0:2] = [0, 0, ]
                            elif inh_cond == 'ln':
                                params2an[0:2] = [.0, ln_spike_h,]
                            
                            #    params2an = [nsi_value, ln_spike_height, dur2an, delays2an, peak, peak_ratio]
                            plt.ion()      # ion() # to avoid showing the plot every time     
                            for id_loop in range(n_loops):
                                [orn_stim, pn_stim,] = main(params2an, fig_opts, 
                                    verbose = False, fld_analysis = fld_analysis, 
                                    stim_seed=stim_seed)
                                pn_avg_dif[id_loop] = (pn_stim[0]-pn_stim[1])
                                pn_avg[id_loop] = (pn_stim[0]+pn_stim[1])/2
                                pn_peak[id_loop] = (pn_stim[2]+pn_stim[3])/2        
                            
                            print(inh_cond+' inh, peak:%.1f, avg:%.1f, avg dif:%.1f'%(np.mean(pn_peak), np.mean(pn_avg), np.mean(pn_avg_dif)))
                            toc = timeit.default_timer()
                        print('time to run %d sims: %.1fs'%(np.size(inh_conds),toc-tic))
                        print('')
                        
else:
    print('run from import')
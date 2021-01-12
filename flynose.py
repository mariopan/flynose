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

import sys
sys.path.insert(0, '/flynose_branch/')

import corr_plumes
import sdf_krofczik
import stats_for_plumes as stats


# *****************************************************************
# FUNCTIONS
def depalo_eq2(z,t,u,orn_params,num_recep):
    a_y = orn_params[0]
    c_x = orn_params[1]
    b_y = orn_params[2]
    
    b_x = orn_params[3]
    d_x = orn_params[4]
    
    b_r = orn_params[5]
    d_r = orn_params[6]
    a_r = orn_params[7]
    n = orn_params[8]
    omega_nsi = orn_params[9]
    
    if num_recep == 0 or num_recep > 4:
        print('Error: number of glomeruli has to be non-zero and not greater than 4')
    
    elif num_recep == 1:
            r = z[0]
            x = z[1]
            y = z[2]
            
            drdt = b_r*u[0]**n*(1-r) - d_r*r
            
            dydt = a_r*r - c_x*x*(1+d_x*y) - b_x*y - omega_nsi*y 
            dxdt = a_y*y - b_y*x
            dzdt = [drdt,dxdt,dydt]
            
    elif num_recep == 2:
            r = z[0]
            x = z[1]
            y = z[2]
            
            s = z[3]  # r2 
            q = z[4]  # x2
            w = z[5]  # y2
            
            drdt = b_r*u[0]**n*(1-r) - d_r*r
            dsdt = b_r*u[1]**n*(1-s) - d_r*s
            
            dydt = a_r*r - c_x*x*(1+d_x*y) - b_x*y - omega_nsi*w*y 
            dxdt = a_y*y - b_y*x
            
            dwdt = a_r*s - c_x*q*(1+d_x*w) - b_x*w - omega_nsi*y*w
            dqdt = a_y*w - b_y*q
            
            dzdt = [drdt,dxdt,dydt,dsdt,dqdt,dwdt]
            
    elif num_recep == 3:
            r = z[0]
            x = z[1]
            y = z[2]
            
            s = z[3]  # r2
            q = z[4]  # x2
            w = z[5]  # y2
            
            f = z[6]  # r3
            p = z[7]  # x3
            m = z[8]  # y2
            
            drdt = b_r*u[0]**n*(1-r) - d_r*r
            dsdt = b_r*u[1]**n*(1-s) - d_r*s
            dfdt = b_r*u[2]**n*(1-f) - d_r*f
            
            dydt = a_r*r - c_x*x*(1+d_x*y) - b_x*y - omega_nsi*w*y - omega_nsi*m*y
            dxdt = a_y*y - b_y*x
            
            dwdt = a_r*s - c_x*q*(1+d_x*w) - b_x*w - omega_nsi*y*w - omega_nsi*m*w
            dqdt = a_y*w - b_y*q
            
            dmdt = a_r*f - c_x*p*(1+d_x*m) - b_x*m - omega_nsi*w*m - omega_nsi*y*m
            dpdt = a_y*m - b_y*p   
            
            dzdt = [drdt,dxdt,dydt,dsdt,dqdt,dwdt,dfdt,dpdt,dmdt] 
            
    elif num_recep == 4:
            r = z[0]
            x = z[1]
            y = z[2]
            
            s = z[3]  # r2 
            q = z[4]  # x2
            w = z[5]  # y2
            
            f = z[6]  # r3
            p = z[7]  # x3
            m = z[8]  # y3
            
            h = z[9]  # r4
            g = z[10] # x4
            c = z[11] # y4

            drdt = b_r*u[0]**n*(1-r) - d_r*r
            dsdt = b_r*u[1]**n*(1-s) - d_r*s
            dfdt = b_r*u[2]**n*(1-f) - d_r*f
            dhdt = b_r*u[3]**n*(1-h) - d_r*h
            
            dydt = a_r*r - c_x*x*(1+d_x*y) - b_x*y - omega_nsi*w*y - omega_nsi*m*y - omega_nsi*c*y
            dxdt = a_y*y - b_y*x
            
            dwdt = a_r*s - c_x*q*(1+d_x*w) - b_x*w - omega_nsi*y*w - omega_nsi*m*w - omega_nsi*c*w
            dqdt = a_y*w - b_y*q
            
            dmdt = a_r*f - c_x*p*(1+d_x*m) - b_x*m - omega_nsi*w*m - omega_nsi*y*m - omega_nsi*c*m
            dpdt = a_y*m - b_y*p   
            
            dcdt = a_r*h - c_x*g*(1+d_x*c) - b_x*c - omega_nsi*w*c - omega_nsi*m*c - omega_nsi*y*c
            dgdt = a_y*c - b_y*g  
            
            dzdt = [drdt,dxdt,dydt,dsdt,dqdt,dwdt,dfdt,dpdt,dmdt,dhdt,dgdt,dcdt]
            
    return dzdt

def rect_func(b, x):
    nu_max = b[0]
    a_rect = b[1]
    c_rect = b[2]
    ot = nu_max/(1 + np.exp(-a_rect*(x-c_rect)))
    return ot


def pn2ln_v_ex(x0,t, s, ln_params, ):
#    ln_params = np.array([tau_s, tau_v, a_s_ln, vrev_ln, vrest_ln])
    tau_v = ln_params[1]
    
    vrev = ln_params[3]
    vrest = ln_params[4]
    
    # PN -> LN equations:
    # ORN -> PN equations:
    dt = t[1]-t[0]
    b = -(1 + s)/tau_v
    a = (vrest + s*vrev)/tau_v
    y = (x0 + a/b)*np.exp(b*dt)-a/b
    #dvdt = ((vrest-v) + s*(vrev-v) )/tau_v
    return y

def pn2ln_s_ex(x0,t, u_pn, ln_params, ):
    #    ln_params = np.array([tau_s, tau_v, a_s_ln, vrev_ln, vrest_ln])
    tau_s = ln_params[0]
    a_s = ln_params[2]
    
    # PN -> LN equation of s:
    b = (-1-a_s*u_pn)/tau_s
    a = a_s*u_pn/tau_s
    dt = t[1]-t[0]
    y = (x0 + a/b)*np.exp(b*dt)-a/b
#    dsdt = (a_s*u_pn*(1-s) - s)/tau_s       
    return y


def y_ln_fun_ex(y0, t, u_ln, tau_y, alpha_ln,):
    b = (-alpha_ln*u_ln-1)/tau_y
    a = alpha_ln*u_ln/tau_y
    dt = t[1]-t[0]
    y = (y0 + a/b)*np.exp(b*dt)-a/b
    return y

def orn2pn_s_ex(x0,t, u_orn, x_pn,y_ln,pn_params,):
    #    pn_params  = np.array([tau_s, tau_v, a_s_pn, vrev_pn, vrest_pn])
    tau_s = pn_params[0]
    a_s = pn_params[2]
    
    # ORN -> PN equations:
    b = (-1-a_s*u_orn*(1-x_pn)*(1-y_ln))/tau_s
    a = a_s*u_orn*(1-x_pn)*(1-y_ln)/tau_s
    dt = t[1]-t[0]
    y = (x0 + a/b)*np.exp(b*dt)-a/b
    return y

def orn2pn_v_ex(x0,t, s, pn_params,):
#    pn_params  = np.array([tau_s, tau_v, a_s_pn, vrev_pn, vrest_pn])
    tau_v = pn_params[1]
    
    vrev = pn_params[3]
    vrest = pn_params[4]
    
    # ORN -> PN equations:
    dt = t[1]-t[0]
    b = -(1 + s)/tau_v
    a = (vrest + s*vrev)/tau_v
    y = (x0 + a/b)*np.exp(b*dt)-a/b
#    dvdt = (vrest + s*vrev)/tau_v  - v*(1 + g*s)/tau_v
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
ticks_fs = label_fs - 3
panel_fs = 30 # font size of panel's letter
black   = 'xkcd:black'
blue    = 'xkcd:blue'
red     = 'xkcd:red'
green   = 'xkcd:green'
purple  = 'xkcd:purple'
orange  = 'xkcd:orange'
cmap    = plt.get_cmap('rainbow')

id_colors2u     = [1, 5]
col_glo         = np.zeros((2,4))
col_glo[0,:]    = cmap(int(255/(id_colors2u[0]+1.3)))
col_glo[1,:]    = cmap(int(255/(id_colors2u[1]+1.3)))
    
# *****************************************************************


def main(params2an, fig_opts):
    
    [orn_fig, al_fig, fig_ui, fig_save, data_save, al_dyn, 
                verbose, fld_analysis] = fig_opts
    
    omega_nsi   = params2an[0] # 0.3
    alpha_ln    = params2an[1] # 10 ORN input coeff for adaptation variable y_ln

    
    # *****************************************************************
    # STIMULUS GENERATION    
    stim_params = params2an[2]
    stim_type = stim_params[0]
    if len(stim_type)==2:
        ext_stimulus = False # flag indicating an external stimulus is provided        
        [stim_type, pts_ms, t_tot, [t_on, t_on2], [t_off, t_off2], 
         [peak1, peak2], plume_params] = stim_params
    
    else:
        ext_stimulus = False # flag indicating an external stimulus is provided        
        [stim_type, pts_ms, t_tot, [t_on, t_on2], [t_off, t_off2], 
         [peak1, peak2], plume_params, fld_stim] = stim_params
        stim_data_name = fld_stim+stim_type+'.dat'
    
    tau_sdf = 20
    dt_sdf = 5
    sdf_size = int(t_tot/dt_sdf)
    
    n2sim           = pts_ms*t_tot + 1      # number of time points
    t               = np.linspace(0, t_tot, n2sim) # time points
    
    # Stimulus params
    stim_on         = t_on*pts_ms   # [num. of samples]
    stim_off        = t_off*pts_ms    
    stim_on2        = t_on2*pts_ms
    stim_off2       = t_off2*pts_ms
    
    stim_dur          = t_off-t_on
    delay           = t_on2-t_on

    # initialize output vectors
    num_glo_list    = [2]#[4,3,2,1]     # number of glomeruli per sensilla
    n_sens_type        = len(num_glo_list)  # number of sensilla
    num_glo_tot     = sum(num_glo_list) # number of glomeruli in total
    # TEMPORARY FIX U_OD SCALING
    u_od            = np.zeros((n2sim, 2))
#    cor_stim        = np.nan
#    overlap_stim    = np.nan
#    cor_whiff       = np.nan
    
    if stim_type == 'ss':
        # Single Step Stimuli
        
        tau_on          = 50
        t_tmp           = np.linspace(0, t_off-t_on, stim_off-stim_on)    
        t_tmp2          = np.linspace(0, t_off2-t_on2, stim_off2-stim_on2)               
            
        u_od[stim_on:stim_off, 0] = peak1*(1-np.exp(-t_tmp/tau_on))
        u_od[stim_on2:stim_off2, 1] = peak2*(1-np.exp(-t_tmp2/tau_on))
        
        t_tmp           = np.linspace(0, t_tot-t_off, 1+t_tot*pts_ms-stim_off)    
        t_tmp2          = np.linspace(0, t_tot-t_off2, 1+t_tot*pts_ms-stim_off2)               
        
        u_od[stim_off:, 0] = u_od[stim_off-1, 0]*np.exp(-t_tmp/tau_on)
        u_od[stim_off2:, 1] = u_od[stim_off2-1, 1]*np.exp(-t_tmp2/tau_on)
        
        
    elif stim_type == 'ts':
        # Single Step Stimuli
        
        t_peak          = t_on + stim_dur/2     # [ms]
        stim_peak       = int(t_peak*pts_ms)
        
        t_peak2         = t_on2 + stim_dur/2     # [ms]
        stim_peak2      = int(t_peak2*pts_ms)
        
        t_tmp           = np.linspace(0, t_off-t_on, stim_off-stim_on)    
        t_tmp2          = np.linspace(0, t_off2-t_on2, stim_off2-stim_on2)               
            
        u_od[stim_on:stim_peak, 0] = np.linspace(0, peak1, stim_peak-stim_on)
        u_od[stim_peak:stim_off, 0] = np.linspace(peak1, 0, stim_off-stim_peak)
                
        u_od[stim_on2:stim_peak2, 1] = np.linspace(0, peak2, stim_peak2-stim_on2)
        u_od[stim_peak2:stim_off2, 1] = np.linspace(peak2, 0, stim_off2-stim_peak2)
        

    elif stim_type == 'pl':
        
        plume_params    = stim_params[6]
        # *******************************************************************
        # PARAMS FOR GENERATION OF PLUMES
        quenched        = True          # if True Tbl and Twh are chosen to compensate the distance between stimuli
        t2sim_s         = (t_tot-t_on)/1000  # [s]
        sample_rate     = 1000*pts_ms   # [Hz] num of samples per each sec
        n_sample2       = 5             # [ms] num of samples with constant concentration
        
        # *******************************************************************
        #  PARAMS FOR WHIFF AND BLANK DISTRIOBUTIONS
        g               = -1/2# 1    # -1/2 for a power law of -3/2, 1 for uniform distribution
        whiff_min       = 3e-3      # [s]
        whiff_max       = plume_params[1]        # [s] 3, 50,150
        
        blank_min       = 3e-3      # [s]
        blank_max       = plume_params[2]       # [s]  25, 35
        
        # *******************************************************************
        # PARAMS FOR CONCENTRATION DISTRIBUTIONS
        # fit of average concentration at 75 m, Mylne and Mason 1991, Fig.10 
        b_conc          = -(np.log10(1-.997) + np.log10(1-.5))/10.7
        a_conc          = -0.3*b_conc - np.log10(1-.5)
        
        rho_c           = 1      # correlation between normal distribution to generate whiffs and blanks
        rho_t_exp       = plume_params[0]     # correlation between normal distribution to generate concentration        
        rho_t           = 1-10**-rho_t_exp
        
        stim_seed       = plume_params[3]
        
        # ******************************************************************* 
        # arguments for the generation of stimuli function
        #np.random.seed()
        corr_plumes_in = [t2sim_s, sample_rate, n_sample2, g, whiff_min, whiff_max, 
               blank_min, blank_max, a_conc, b_conc,rho_c, rho_t, quenched, stim_seed]
        
        # *******************************************************************
        # PLUME GENERATION
        out_y, out_w, _, _= corr_plumes.main(*corr_plumes_in)
        u_od[stim_on:, 0] = out_y*peak1 # params2an[4]
        u_od[stim_on:, 1] = out_w*peak2 # params2an[4]*params2an[5]
               
#        cor_stim        = -2
#        overlap_stim    = -2
#        cor_whiff       = -2
#        
#        interm_est_1 = np.sum(out_y>0)/(t2sim_s*sample_rate)
#        interm_est_2 = np.sum(out_w>0)/(t2sim_s*sample_rate)
#
#        if (np.sum(out_y)!=0) & (np.sum(out_w)!=0):
#            cor_stim        = np.corrcoef(out_y, out_w)[1,0]
#            overlap_stim    = stats.overlap(out_y, out_w)
#            nonzero_concs1  = out_y[(out_y>0) & (out_w>0)]
#            nonzero_concs2  = out_w[(out_y>0) & (out_w>0)]
#            cor_whiff       = np.corrcoef(nonzero_concs1, nonzero_concs2)[0, 1] # np.corrcoef(concs1, concs2)[0, 1]
    else:
        ex_stim = np.loadtxt(stim_data_name)
     
        # Sims params
        t_tot      = ex_stim[-1,0]*1e3 # [ms] t_tot depends on data
        pts_ms          = 5             # simulated points per ms
        n2sim           = np.size(ex_stim, axis=0)#pts_ms*t_tot + 1      # number of time points
        t               = np.linspace(0, t_tot, n2sim) # time points
        
        # Stimulus params
        t_on            = 0      # [ms]
        stim_on         = t_on*pts_ms 
        t_off           = t_on + stim_dur     # [ms]
        stim_off        = t_off*pts_ms    
        t_on2           = t_on+delay     # [ms]
        stim_on2        = t_on2*pts_ms
        t_off2          = t_on2 + stim_dur    # [ms]
        stim_off2       = t_off2*pts_ms
        
        u_od            = np.zeros((n2sim, 2))
        u_od[:, 0]      = .01*ex_stim[:,1]
        u_od[:, 1]      = .01*(ex_stim[0,1]+ex_stim[-1,1])/2
    
    # *****************************************************************
    # CONNECTIVITY PARAMETERS
    
    # *****************************************************************
    # NETWORK PARAMETERS 
    n_orns_pn         = 18    # number of ORNs per each PN in each glomerulus
    n_orns_recep        = 40    # number of ORNs per each glomerulus
    n_orns_tot        = n_orns_recep*num_glo_tot  # total number of ORNs 
    n_pns_recep         = 5     # number of PNs per each glomerulus
    n_lns_recep         = 3     # number of LNs per each glomerulus
    n_pns_tot         = n_pns_recep*num_glo_tot # number of total PNs
    n_lns_tot         = n_lns_recep*num_glo_tot # number of total LNs 
    
    # *****************************************************************
    # ORN PARAMETERS 
    
    cov_hom         = 0.4 # Covariance value homotypic ORNs
    nu_pn_noise     = 200 # Hz  - PNs Noise into each PNs
    nu_ln_noise     = 0 # Hz    - LNs Noise into each PNs

    # rectification params
    c_rect              = 1
    a_rect              = 3.3 
    nu_max_rect         = 250
    B0                  = [nu_max_rect, a_rect, c_rect]
    
    # Spiking machine params
    a_y                  = 0.25  
    b_y                  = 0.002  
    
    c_x                  = 0.0028     # 0.004
    b_x                  = 0.2       # 0.12
    d_x                  = 1           
    a_r                  = 1
    
    # Transduction params
    n                   = 1                 # 1
    b_r                 = 0.01              #*100# 1.75
    d_r                 = 0.009             #*100# 1.1
    orn_params          = np.array([a_y, c_x, b_y, b_x,d_x,b_r,d_r,a_r,n,omega_nsi,])

    #**************************************
    # ORN, PN and LN PARAMETERS
    spike_length        = int(4*pts_ms)     # [ms]
    t_ref               = 2*pts_ms          # ms; refractory period 
    theta               = 1                 # [mV] firing threshold
    
    orn_spike_height    = .3
    pn_spike_height     = .3
    ln_spike_height     = .3
    
    # *****************************************************************
    # GENERATION OF THE CONNECTIVITY MATRIX
    
    # Each ORN belongs to ONLY one of the glomeruli
    ids_orn_glo     = np.zeros((n_orns_tot, num_glo_tot), dtype=int)
    for pp in range (int(num_glo_tot)):
        ids_orn_glo[pp*n_orns_recep:(pp+1)*n_orns_recep,pp] = 1
    
    # Correlation is high only on the same glomerulus ORNs
    mv_mean     = np.zeros(n_orns_tot)
    mv_cov      = np.zeros((n_orns_tot,n_orns_tot))
    mv_cov_tmp  = ((1-cov_hom)*np.identity(n_orns_recep) +
                 cov_hom*np.ones((n_orns_recep, n_orns_recep))) # diagonal covariance
    for pp in range(int(num_glo_tot)):
        mv_cov[pp*n_orns_recep:(pp+1)*n_orns_recep,
               pp*n_orns_recep:(pp+1)*n_orns_recep] = mv_cov_tmp
        
    # Each PN belongs to ONLY one of the glomeruli
    ids_pn_glo     = np.zeros((n_pns_tot), dtype=int)
    for pp in range (int(num_glo_tot)):
        ids_pn_glo[pp*n_pns_recep:(pp+1)*n_pns_recep] = pp
        
    # Each LN belongs to ONLY one of the glomeruli
    ids_ln_glo     = np.zeros((n_lns_tot, num_glo_tot), dtype=int)
    for pp in range (int(num_glo_tot)):
        ids_ln_glo[pp*n_lns_recep:(pp+1)*n_lns_recep,pp] = 1
    
    # Each PN is connected randomly with a sub-sample of ORNs
    ids_orn_pn          = np.zeros((n_pns_tot, n_orns_pn), dtype=int)
    
    # Connectivity matrices between ORNs and PNs 
    orn_pn_mat          = np.zeros((n_orns_tot, n_pns_tot))
   
    for pp in range(n_pns_tot):
        rnd_ids         = np.random.permutation(n_orns_recep) 
        tmp_ids          = rnd_ids[:n_orns_pn] + n_orns_recep*ids_pn_glo[pp]
        ids_orn_pn[pp,:] = tmp_ids
        orn_pn_mat[tmp_ids, pp] = orn_spike_height
    
    # Connectivity matrices between PNs and LNs
    pn_ln_mat           = np.zeros((n_pns_tot, n_lns_tot))
    for pp in range(num_glo_tot):
        pn_ln_mat[pp*n_pns_recep:(pp+1)*n_pns_recep,
                  pp*n_lns_recep:(pp+1)*n_lns_recep] = pn_spike_height
    
    recep_id = 0        
    ln_pn_mat           = np.zeros((n_lns_tot,n_pns_tot))
    for pp in range(n_sens_type):
        num_recep = num_glo_list[pp]
        # Inhibitory LN connectivity within glomeruli cluster
        ln_pn_mat[(recep_id*n_lns_recep):((recep_id+num_recep)*n_lns_recep),
                  (recep_id*n_pns_recep):((recep_id+num_recep)*n_pns_recep)] = ln_spike_height
        for qq in range(num_recep):
            # PN innervating LN are not inhibited
            ln_pn_mat[((recep_id+qq)*n_lns_recep):((recep_id+qq+1)*n_lns_recep),
                      ((recep_id+qq)*n_pns_recep):((recep_id+qq+1)*n_pns_recep)] = 0
        recep_id = recep_id + num_recep
    
    # *****************************************************************
    # GENERATE ORN RESPONSE TO ODOR INPUT 
    # num_spike_orn       = np.zeros((n2sim, num_recep))
    # r_orn               = np.zeros((n2sim, num_recep))
    # x_orn               = np.zeros((n2sim, num_recep))
    # y_orn               = np.zeros((n2sim, num_recep))
    # nu_orn              = np.zeros((n2sim, num_recep))   
    tnu_orn             = np.zeros((n2sim, num_glo_tot))
    tr_orn              = np.zeros((n2sim, num_glo_tot))
    tx_orn              = np.zeros((n2sim, num_glo_tot))
    ty_orn              = np.zeros((n2sim, num_glo_tot))
    
    # initial conditions
    # z_orn0          = np.ones((num_recep, 3))*[0, 0, 0]
    # r_orn[0,:]        = z_orn0[:, 0]
    # x_orn[0,:]        = z_orn0[:, 1]
    # y_orn[0,:]        = z_orn0[:, 2]
    # tot_od            = np.zeros([n2sim, num_recep])
    
    # ODOUR PREFERENCE
    od_pref = np.array([[1,0],
                        [0,1],])
                        # [0,0],
                        # [1,0],
                        # [1,0],
                        # [0,0],
                        # [0,1],
                        # [1,0],
                        # [0,0],
                        # [0,1]])
    recep_id = 0
    sen_id = 0
    for pp in range(n_sens_type):     
        num_recep = num_glo_list[pp]
        # GENERATE ORN RESPONSE TO ODOR INPUT
        num_spike_orn       = np.zeros((n2sim, num_recep))
        r_orn               = np.zeros((n2sim, num_recep))
        x_orn               = np.zeros((n2sim, num_recep))
        y_orn               = np.zeros((n2sim, num_recep))
        nu_orn              = np.zeros((n2sim, num_recep))  
        # initial conditions
        z_orn0            = np.ones((num_recep, 3))*[0, 0, 0]
        r_orn[0,:]        = z_orn0[:, 0]
        x_orn[0,:]        = z_orn0[:, 1]
        y_orn[0,:]        = z_orn0[:, 2]
        tot_od            = np.zeros([n2sim, num_recep])
        for qq in range(num_recep):
            temp_od_pref = u_od*od_pref[recep_id,:]
            tot_od[:,qq] = np.sum(temp_od_pref, axis=1)
            recep_id = recep_id+1

        for tt in range(1, n2sim-t_ref-1):
            # span for next time step
            tspan = [t[tt-1],t[tt]]
            
            z0_unid = np.zeros(num_recep*3)
            for zz in range(num_recep):
                z0_unid[zz*3:(zz+1)*3] = z_orn0[zz,:]

            z_orn = odeint(depalo_eq2, z0_unid, tspan,
                           args=(tot_od[tt,:], orn_params, num_recep))
            for gg in range(num_recep):
                z_orn0[gg,0] = z_orn[1][0+gg*3]
                z_orn0[gg,1] = z_orn[1][1+gg*3]
                z_orn0[gg,2] = z_orn[1][2+gg*3]
            
                r_orn[tt,gg] = z_orn[1][0+gg*3]
                x_orn[tt,gg] = z_orn[1][1+gg*3]
                y_orn[tt,gg] = z_orn[1][2+gg*3]
                nu_orn[tt,gg] = rect_func(B0, y_orn[tt,gg])
                
        tnu_orn[:,sen_id:(sen_id+num_recep)] =  nu_orn
        tr_orn[:,sen_id:(sen_id+num_recep)]  =  r_orn
        tx_orn[:,sen_id:(sen_id+num_recep)]  =  x_orn
        ty_orn[:,sen_id:(sen_id+num_recep)]  =  y_orn
        sen_id = sen_id + num_recep
    
    # *****************************************************************
    # Transform the average nu_orn into a spiking 
    # matrix n2sim by n_orns_tot of correlated spikes:
    num_spike_orn   = np.zeros((n2sim, n_orns_tot))
    u_orn           = np.zeros((n2sim, n_pns_tot))
    
    # generate a matrix n2sim by n_orns_tot of correlated spikes:
    rnd     = np.random.multivariate_normal(mv_mean, mv_cov, n2sim)
    rnd     = spst.norm.cdf(rnd)
    
    nu_tmp = np.zeros((n2sim,n_orns_tot))
    
    orns_id = 0
    recep_id  = 0
    for pp in range(n_sens_type):
        num_recep = num_glo_list[pp]
        for qq in range(num_recep):
            nu_tmp[:,orns_id:(orns_id+n_orns_recep)] = (np.tile(tnu_orn[
                :,recep_id], (n_orns_recep,1))).transpose()
            recep_id = recep_id +1
            orns_id = orns_id + n_orns_recep
    
    t_zeros = np.zeros((1, n_pns_tot))
    num_spike_orn = (rnd < nu_tmp/(pts_ms*1e3))*1.0
    orn_spike_all = num_spike_orn.dot(orn_pn_mat) 
    u_orn = u_orn + orn_spike_all
    for tt in range(spike_length-1):
        orn_spike_all = np.concatenate((t_zeros, orn_spike_all[:-1,:]))
        u_orn = u_orn + orn_spike_all
        
    # *****************************************************************
    # Per each PNs, add a noise signal coming from other PNs, LNs, ...
    rnd_pn  = np.random.random_sample((n2sim,n_pns_tot))
    rnd_ln  = np.random.random_sample((n2sim,n_pns_tot))    
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

   
    
    if orn_fig | al_fig:
        orn_sdf, orn_sdf_time = sdf_krofczik.main(orn_spike_matrix, sdf_size,
                                                   tau_sdf, dt_sdf)  # (Hz, ms)
        orn_sdf = orn_sdf*1e3

    
    # *****************************************************************
    # FIGURE ORN dynamics

    if orn_fig:  
        t2plot = -200, 1000 #t_tot #-t_on, t_tot-t_on
        rs = 4 # number of rows
        cs = 1 # number of cols

        if stim_type == 'pl':
            t2plot = 0, 1000#2000, 4000
            rs = 2 # number of rows
            cs = 2 # number of cols
            
        if ext_stimulus:
            t2plot = 0, t_tot
        
        orn_id = 0
        for pp in range(n_sens_type):
            panels_id = ['a', 'b', 'c', 'd']
            
            fig_orn = plt.figure(figsize=[8.5, 8])
    #       fig_orn.canvas.manager.window.wm_geometry("+%d+%d" % fig_position )
            # fig_orn.tight_layout()
            
            ax_orn1 = plt.subplot(rs, cs, 1)
            ax_orn2 = ax_orn1.twinx()
            ax_orn3 = plt.subplot(rs, cs, 2)
            ax_orn4 = ax_orn3.twinx()
            
            ax_orn_sc = plt.subplot(rs, cs, 3)
            ax_orn_fr = plt.subplot(rs, cs, 4)
            
            ax_orn1.plot(t-t_on, u_od[:,0], linewidth=lw+1, color=black, 
                         label=r'Glom %d'%(1))
            ax_orn2.plot(t-t_on, r_orn[:,0], linestyle='--',color=black,  linewidth=lw+1, 
                         label=r'r, glom: %d'%(1))
            ax_orn3.plot(t-t_on, x_orn[:,0], linewidth=lw+1, color=black, 
                         label=r'Od, glom : %d'%(0))
            ax_orn4.plot(t-t_on, y_orn[:,0], linestyle='--', color=black, linewidth=lw+1, 
                         label=r'Od, glom : %d'%(0))
            trsp = .3
            
            recep_clrs = ['purple','green','cyan','red']
            num_recep = num_glo_list[pp]
            for qq in range(num_recep):
                x1 = orn_sdf[:,orn_id:(orn_id+n_orns_recep)]
                mu1 = x1.mean(axis=1)
                sigma1 = x1.std(axis=1)
                ax_orn_fr.plot(orn_sdf_time-t_on, mu1, linewidth=lw+1, color=recep_clrs[qq])
                ax_orn_fr.fill_between(orn_sdf_time-t_on, 
                    mu1+sigma1, mu1-sigma1, facecolor=recep_clrs[qq], alpha=trsp,label='sdf glo '+str(qq))
    
                spikes_orn = np.argwhere(num_spike_orn[:,orn_id:(orn_id+n_orns_recep)])
                
                ax_orn_sc.scatter(spikes_orn[:,0]/pts_ms-t_on, 
                                (n_orns_recep*qq)+spikes_orn[:,1], color=recep_clrs[qq], s=10)
                orn_id = orn_id + n_orns_recep
    
            # FIGURE SETTINGS
            ax_orn1.tick_params(axis='both', which='major', labelsize=ticks_fs)
            ax_orn2.tick_params(axis='both', which='major', labelsize=ticks_fs)
            ax_orn3.tick_params(axis='both', which='major', labelsize=ticks_fs)
            ax_orn4.tick_params(axis='both', which='major', labelsize=ticks_fs)
            ax_orn_fr.tick_params(axis='both', which='major', labelsize=ticks_fs)
            ax_orn_sc.tick_params(axis='both', which='major', labelsize=ticks_fs)
            
            ax_orn1.set_xticklabels('')
            ax_orn2.set_xticklabels('')
            ax_orn3.set_xticklabels('')
            ax_orn4.set_xticklabels('')
    #        ax_orn_fr.set_xticklabels('')
            ax_orn_sc.set_xticklabels('')
            
            
            # ax_orn1.yaxis.label.set_color(green)
            ax_orn1.set_ylabel('Input (a.u.)', fontsize=label_fs)
            # ax_orn2.yaxis.label.set_color(col_glo[0,:]/2)
            ax_orn2.set_ylabel(r'r (a.u.) ', fontsize=label_fs)
            # ax_orn3.yaxis.label.set_color(green)
            ax_orn3.set_ylabel(r'y (a.u.)', fontsize=label_fs)
            # ax_orn4.yaxis.label.set_color(col_glo[1,:]/2)
            ax_orn4.set_ylabel(r'x (a.u.)', fontsize=label_fs)
            ax_orn_fr.set_ylabel('firing rates (Hz)', fontsize=label_fs)
            ax_orn_fr.set_xlabel('Time  (ms)', fontsize=label_fs) 
            ax_orn_sc.set_ylabel('Neuron id', fontsize=label_fs)
    
            ax_orn1.text(-.15, 1.25, panels_id[0], transform=ax_orn1.transAxes, 
                              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            ax_orn3.text(-.15, 1.25, panels_id[1], transform=ax_orn3.transAxes, 
                              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            ax_orn_sc.text(-.15, 1.25, panels_id[2], transform=ax_orn_sc.transAxes,
                              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            ax_orn_fr.text(-.15, 1.25, panels_id[3], transform=ax_orn_fr.transAxes, 
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
            ll_new = ll+.025
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
                orn_fig_name = '/ORN_' + params2an[7] + '.png'
            else:
                #%%
                if stim_type == 'pl':
                    orn_fig_name = '/ORNdyn' + \
                                '_stim_' + stim_type +\
                                '_nsi_%.1f'%(params2an[0]) +\
                                '_lnspH_%.2f'%(params2an[1]) +\
                                '_dur2an_%d'%(t_off-t_on) +\
                                '_delay2an_%d'%(t_on2-t_on) +\
                                '_peak_%.1f'%(peak1) +\
                                '_peakratio_%.1f'%(peak1/peak2) +\
                                '_rho_%d'%(plume_params[0]) +\
                                '_wmax_%.1g'%(plume_params[1]) +\
                                '_bmax_%.1g'%(plume_params[2]) +\
                                '.png'
                else:
                    orn_fig_name = '/ORNdyn' + \
                            '_stim_' + stim_type +\
                            '_nsi_%.1f'%(params2an[0]) +\
                            '_lnspH_%.2f'%(params2an[1]) +\
                            '_dur2an_%d'%(t_off-t_on) +\
                            '_delay2an_%d'%(t_on2-t_on) +\
                            '_peak_%.1f'%(peak1) +\
                            '_peakratio_%.1f'%(peak1/peak2) +\
                            '.png'
            fig_orn.savefig(fld_analysis + orn_fig_name)
    # ******************************************************************
    
    # *****************************************************************
    # AL SIMULATION 
    # *****************************************************************
      
                                
    # *****************************************************************
    # PN and LN PARAMETERS and OUTPUT VECTORS

    tau_v               = .5        # [ms]
    tau_s               = 10        # [ms]
    
    #**************************************
    # PN PARAMETERS
    a_s_pn              = 2.5       #     
    vrest_pn            = -6.5      # [mV] resting potential
    vrev_pn             = 15.0      # [mV] reversal potential
    
    alpha_x             = 2.         # ORN input coeff for adaptation variable x_pn
    tau_x               = 600    # [ms] time scale for dynamics of adaptation variable x_pn
    x_pn0               = 0.48*np.ones(n_pns_tot)     # 0.27
    
    pn_params  = np.array([tau_s, tau_v, a_s_pn, vrev_pn, vrest_pn])
    
    #**************************************
    # LN PARAMETERS
    a_s_ln              = 2.5       #     
    vrest_ln            = -3.0      # -1.5 [mV] resting potential
    vrev_ln             = 15.0      # [mV] reversal potential
    

    tau_y               = 600    # [ms] time scale for dynamics of adaptation variable y_ln
    y_ln0               = 0.025*np.ones(n_pns_tot) # 0.2
    ln_params = np.array([tau_s, tau_v, a_s_ln, vrev_ln, vrest_ln])
    #**************************************
    
    # INITIALIZE LN to PN output vectors
    x_pn            = np.zeros((n2sim, n_pns_tot))
    u_pn            = np.zeros((n2sim, n_lns_tot))
    u_ln            = np.zeros((n2sim, n_pns_tot))
    y_ln            = np.zeros((n2sim, n_pns_tot))
    
    # INITIALIZE PN output vectors
    num_spike_pn    = np.zeros((n2sim, n_pns_tot))
    
    # INITIALIZE LN output vectors
    s_ln            = np.zeros((n2sim, n_lns_tot))
    v_ln            = np.zeros((n2sim, n_lns_tot))
    num_spike_ln    = np.zeros((n2sim, n_lns_tot))  
    
    # PN and LN params initial conditions
    x_pn[0, :]      = x_pn0
    s_pn            = np.zeros((n2sim, n_pns_tot))
    v_pn            = np.ones((n2sim, n_pns_tot))*vrest_pn
    pn_ref_cnt      = np.zeros(n_pns_tot) # Refractory period counter starts from 0
    
    y_ln[0, :]      = y_ln0
    s_ln            = np.zeros((n2sim, n_lns_tot))
    v_ln            = np.ones((n2sim, n_lns_tot))*vrest_ln
    ln_ref_cnt      = np.zeros(n_lns_tot) # initially the ref period cnter is equal to 0
            
    
    
    if al_dyn:
        # *****************************************************************
        # solve ODE for PN and LN
        for tt in range(1, n2sim-t_ref-1):
            # span for next time step
            tspan = [t[tt-1],t[tt]]
            
            pp_rnd = np.arange(n_pns_tot) # np.random.permutation(n_pns_tot)
            
            # ******************************************************************
            # Vectorized and fast UPDATE PNS 
            # ******************************************************************
            # adaptation variable of PN neuron
            x_pn[tt, pp_rnd] = x_adapt_ex(x_pn[tt-1,pp_rnd],tspan, 
                    u_orn[tt, pp_rnd], tau_x, alpha_x, )        
        
            # Inhibitory input to PNs
            y_ln[tt, pp_rnd] = y_ln_fun_ex(y_ln[tt-1, pp_rnd],tspan, 
                    u_ln[tt, pp_rnd], tau_y, alpha_ln, )
        
            # *********************************
            # ORN -> PN synapses
            
            # *********************************
            # PNs whose ref_cnt is equal to zero:
            pn_ref_0 = pn_ref_cnt==0
            s_pn[tt, pn_ref_0] = orn2pn_s_ex(s_pn[tt-1, pn_ref_0],tspan, 
                u_orn[tt, pn_ref_0], x_pn[tt-1, pn_ref_0], y_ln[tt-1, pn_ref_0], pn_params, )
            v_pn[tt, pn_ref_0] = orn2pn_v_ex(v_pn[tt-1, pn_ref_0],tspan, 
                    s_pn[tt-1, pn_ref_0], pn_params, )
            
            # *********************************
            # PNs whose ref_cnt is different from zero:
            pn_ref_no0 = pn_ref_cnt!=0
            # Refractory period count down
            pn_ref_cnt[pn_ref_no0] = pn_ref_cnt[pn_ref_no0] - 1  
            
            # PNs whose Voltage is above threshold AND whose ref_cnt is equal to zero:
            pn_above_thr = (v_pn[tt, :] >= theta) & (pn_ref_cnt==0)
            num_spike_pn[tt, pn_above_thr] = num_spike_pn[tt, pn_above_thr] + 1
            u_pn[tt:tt+spike_length, :] = (u_pn[tt:tt+spike_length, :] + 
                    np.sum(pn_ln_mat[pn_above_thr,:], axis=0))
            pn_ref_cnt[pn_above_thr] = t_ref
            
            # *********************************
            # PN -> LN synapses        
            
            # *********************************
            # LNs whose ref_cnt is equal to zero:
            ln_ref_0 = ln_ref_cnt==0
            s_ln[tt, ln_ref_0] = pn2ln_s_ex(s_ln[tt-1, ln_ref_0], tspan, 
                        u_pn[tt, ln_ref_0], ln_params, )
            v_ln[tt, ln_ref_0] = pn2ln_v_ex(v_ln[tt-1, ln_ref_0], tspan, 
                        s_ln[tt-1, ln_ref_0], ln_params, )
            
            # *********************************
            # LNs whose ref_cnt is different from zero:
            ln_ref_no0 = ln_ref_cnt!=0
            # Refractory period count down
            ln_ref_cnt[ln_ref_no0] = ln_ref_cnt[ln_ref_no0] - 1  
            
            # LNs whose Voltage is above threshold AND whose ref_cnt is equal to zero:
            ln_above_thr = (v_ln[tt, :] >= theta) & (ln_ref_cnt==0)
            num_spike_ln[tt, ln_above_thr] = num_spike_ln[tt, ln_above_thr] + 1
            u_ln[tt:tt+spike_length, :] = (u_ln[tt:tt+spike_length, :] + 
                        np.sum(ln_pn_mat[ln_above_thr,:], axis=0))
            ln_ref_cnt[ln_above_thr] = t_ref
            # ******************************************************************
            
        # *****************************************************************
        # Calculate the spike matrix of PNs and LNs
        pn_spike_matrix = np.asarray(np.where(num_spike_pn))
        pn_spike_matrix[0,:] = pn_spike_matrix[0,:]/pts_ms
        pn_spike_matrix = np.transpose(pn_spike_matrix)
        
        ln_spike_matrix = np.asarray(np.where(num_spike_ln))
        ln_spike_matrix[0,:] = ln_spike_matrix[0,:]/pts_ms
        ln_spike_matrix = np.transpose(ln_spike_matrix)
        
        

    else:
        [pn_spike_matrix, ln_spike_matrix, ] = [np.nan, np.nan]
    

    # %******************************************
    # FIGURE ORN, PN, LN
    if al_dyn & al_fig:
        # *****************************************************************
        # Calculate the SDF for PNs and LNs
        pn_sdf, pn_sdf_time = sdf_krofczik.main(pn_spike_matrix, sdf_size,
                                                     tau_sdf, dt_sdf)  # (Hz, ms)
        pn_sdf= pn_sdf*1e3
    
        ln_sdf, ln_sdf_time = sdf_krofczik.main(ln_spike_matrix, sdf_size,
                                                     tau_sdf, dt_sdf)  # (Hz, ms)
        ln_sdf= ln_sdf*1e3
        
        t2plot = -300, t_tot-300
        rs = 4 # number of rows
        cs = 1 # number of cols
        fig_size = [7, 8] 
        recep_clrs = ['purple','green','cyan','red']
        trsp = 0.3 # level of transparency in hose plot
        
        if stim_type == 'pl':
            t2plot = 0, 4000
            rs = 2 # number of rows
            cs = 2 # number of cols
            fig_size = [10, 5]

        recep_id = 0
        for qq in range(n_sens_type):
            num_recep = num_glo_list[qq]
            
            fig_pn = plt.figure(figsize=fig_size)
            
            ax_conc = plt.subplot(rs, cs, 1)
            ax_orn = plt.subplot(rs, cs, 2)
            ax_pn = plt.subplot(rs, cs, 3)
            ax_ln = plt.subplot(rs, cs, 4)
            
            ax_conc.plot(t-t_on, 100*u_od[:,0], color=purple, linewidth=lw+2, 
                              label='glom : '+'%d'%(1))
            ax_conc.plot(t-t_on, 100*u_od[:,1], '--',color=green, linewidth=lw+1, 
                              label='glom : '+'%d'%(2))
            
            for ll in range(num_recep):
                X1 = orn_sdf[:, recep_id*n_orns_recep:((recep_id+1)*n_orns_recep)] # np.mean(orn_sdf_norm[:,:,n_orns_recep:], axis=2)
                mu1 = X1.mean(axis=1)
                sigma1 = X1.std(axis=1)
                ax_orn.plot(orn_sdf_time-t_on, mu1, 
                            color=recep_clrs[ll], linewidth=lw-1, )
                ax_orn.fill_between(orn_sdf_time-t_on, mu1+sigma1, mu1-sigma1, 
                                facecolor=recep_clrs[ll], alpha=trsp)
                
                ax_orn.plot(orn_sdf_time-t_on, np.mean(orn_sdf[:,recep_id*n_orns_recep:((recep_id+1)*n_orns_recep)], axis=1),
                                                      color=recep_clrs[ll], linewidth=lw+1,label='sdf glo')
                
                ax_pn.plot(pn_sdf_time-t_on, pn_sdf[:,recep_id*n_pns_recep:((recep_id+1)*n_pns_recep)], '--',color=recep_clrs[ll], 
                                      linewidth=lw, label='PN')
                
                ax_ln.plot(ln_sdf_time-t_on, ln_sdf[:,recep_id*n_lns_recep:((recep_id+1)*n_lns_recep)], '--',color=recep_clrs[ll], 
                                      linewidth=lw, label='LN')
                ax_ln.plot(pn_sdf_time-t_on, 
                        pn_sdf[:, recep_id*n_pns_recep:((recep_id+1)*n_pns_recep)], '--', #pn_sdf
                        color=recep_clrs[ll], linewidth=lw,)
                recep_id = recep_id+1
                
            ax_conc.set_xlim(t2plot)
            ax_orn.set_xlim(t2plot)
            ax_pn.set_xlim(t2plot)
            ax_ln.set_xlim(t2plot)
            
            ax_orn.set_ylim((0, 150))
            ax_pn.set_ylim((0, 180))
            ax_ln.set_ylim((0, 230))
    
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
            if stim_type == 'pl':
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
    
            # tmp
            if not(stim_type == 'pl'):
                title_fs = 30
                if (params2an[1] ==0) & (params2an[1] ==0):
                    ax_conc.set_title('a. Independent', fontsize=title_fs)
                elif (params2an[1] >0):
                    ax_conc.set_title('b. LN inhib.', fontsize=title_fs)
                else:
                     ax_conc.set_title('c. NSI', fontsize=title_fs)   
                 
            ax_conc.spines['right'].set_color('none')
            ax_conc.spines['top'].set_color('none')
            ax_orn.spines['right'].set_color('none')
            ax_orn.spines['top'].set_color('none')
            ax_pn.spines['right'].set_color('none')
            ax_pn.spines['top'].set_color('none')
            ax_ln.spines['right'].set_color('none')
            ax_ln.spines['top'].set_color('none')
            
            if (stim_type == 'pl'):
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
            if stim_type == 'ts':
                fig_pn.savefig(fld_analysis+  '/ORNPNLN' +
                            '_stim_' + stim_params[0] +
                            '_nsi_%.1f'%(params2an[0]) +
                            '_ln_' + '%.2f'%(params2an[1]) +
                            '_dur2an_%d'%(stim_dur) +
                            '_delay2an_%d'%(delay) +
                            '_peak_%.1f'%(peak1) +
                            '_peakratio_%.1f'%(peak2/peak1) +
                            '.png')
            elif stim_type == 'pl':
                fig_pn.savefig(fld_analysis+  '/ORNPNLN' +
                            '_stim_' + stim_params[0] +
                            '_nsi_%.1f'%(params2an[0]) +
                            '_ln_' + '%.2f'%(params2an[1]) +
                            '_dur2an_%d'%(stim_dur) +
                            '_delay2an_%d'%(delay) +
                            '_peak_%.1f'%(peak1) +
                            '_rho_' + '%d'%(plume_params[0]) +
                            '_wmax_' + '%.1g'%(plume_params[1]) +
                            '_bmax_' + '%.1g'%(plume_params[2]) +
                            '.png')
        if fig_opts[2]==False:
            plt.close()
        plt.show()    
    # *************************************************************************

    flynose_out = [t, u_od, orn_spike_matrix, pn_spike_matrix, ln_spike_matrix, ]
    
    return  flynose_out 


if __name__ == '__main__':
    print('run directly')
    stim_data_fld = ''
    
    #***********************************************
    # analysis params
    tau_sdf         = 41
    dt_sdf          = 5

    # ORN NSI params
    alpha_ln        = 0#16.6  # 13.3 #10.0 # 0.0 # ln spike h=0.4
    nsi_str         = 0.0   # 0.3 # 0.0
    
    # Trials and errors 

    # output params 
    fld_analysis    = 'NSI_analysis/trials'
   
    # #***********************************************
    # # stimulus params
    stim_dur        = 500
    delay           = 0    
    stim_type       = 'ss'          # 'ts'  # 'ss' # 'pl'
    pts_ms          = 1
    t_tot           = 2000        # ms 
    t_on            = [300, 300+delay]    # ms
    t_off           = np.array(t_on)+stim_dur # ms
    concs           = [.7, .7]
    sdf_size        = int(t_tot/dt_sdf)
    # real plumes params
    b_max           = np.nan # 3, 50, 150
    w_max           = np.nan #3, 50, 150
    rho             = np.nan #[0, 1, 3, 5]: 
    stim_seed       = 0   # if =np.nan() no traceable random
    
    #***********************************************
    # Real plumes, example figure
    # stim_type   = 'pl'  # 'ts' # 'ss'
    # dur         = 4000
    # delay       = 0
    
    # pts_ms      = 5
    # t_tot       = 4300        # ms 
    # t_on        = [300, 300+delay]    # ms
    # t_off       = np.array(t_on)+dur # ms
    # concs       = [1.5, 1.5]
    # sdf_size    = int(t_tot/dt_sdf)
    
    # # real plumes params
    # b_max       = 25#, 50, 150
    # w_max       = 3#np.nan #3, 50, 150
    # rho         = 1#np.nan #[0, 1, 3, 5]: 
    # stim_seed   = 0   # if =np.nan() no traceable random
    #***********************************************

    plume_params = [rho, w_max, b_max, stim_seed]
    
    stim_params = [stim_type, pts_ms, t_tot, t_on, t_off, concs, plume_params]
    
    params2an = [nsi_str, alpha_ln, stim_params,]
    
    orn_fig     = 0
    al_fig      = 1
    fig_ui      = 1        
    fig_save    = 1    
    data_save   = 0
    al_dyn      = 1
    verbose     = 0    

    fig_opts = [orn_fig, al_fig, fig_ui, fig_save, data_save, al_dyn, 
                verbose, fld_analysis]
    

    if path.isdir(fld_analysis):
        print('OLD analysis fld: ' + fld_analysis)    
    else:
        print('NEW analysis fld: ' + fld_analysis)    
        mkdir(fld_analysis)
    
    
    n_loops         = 1
    pn_avg_dif      = np.zeros(n_loops)
    pn_avg_ratio    = np.zeros(n_loops)
    pn_peak_ratio   = np.zeros(n_loops)
    pn_peak_s   = np.zeros(n_loops)
    pn_peak_w   = np.zeros(n_loops)
    
    if len(stim_type)>2:
        params2an.append(stim_data_fld)
    tic = timeit.default_timer()
        
    plt.ion()      # ioff() # to avoid showing the plot every time     
    
    for id_loop in range(n_loops):
        flynose_out = main(params2an, fig_opts)
        [t, u_od, orn_spike_matrix, pn_spike_matrix, 
         ln_spike_matrix, ] = flynose_out
        
        # *************************************************************************
        # COLLECT AND SAVE DATA
        
        # Calculate the SDF for PNs and LNs
        if al_dyn:
            pn_sdf, pn_sdf_time = sdf_krofczik.main(pn_spike_matrix, sdf_size,
                                                         tau_sdf, dt_sdf)  # (Hz, ms)
            pn_sdf = pn_sdf*1e3
        
            ln_sdf, ln_sdf_time = sdf_krofczik.main(ln_spike_matrix, sdf_size,
                                                         tau_sdf, dt_sdf)  # (Hz, ms)
            ln_sdf = ln_sdf*1e3
             
            n_pns_recep         = 5     # number of PNs per each glomerulus
            id_stim_w = np.flatnonzero((pn_sdf_time>t_on[0]) & (pn_sdf_time<t_on[0]+100))
            id_stim_s = np.flatnonzero((pn_sdf_time>t_on[1]) & (pn_sdf_time<t_on[1]+100))
            pn_peak_w[id_loop]  = np.max(np.mean(pn_sdf[id_stim_w, :n_pns_recep], axis=1)) # using average PN
            pn_peak_s[id_loop]  = np.max(np.mean(pn_sdf[id_stim_s, n_pns_recep:], axis=1)) # using average PN
            pn_avg_w  = np.mean(pn_sdf[id_stim_w, :n_pns_recep])
            pn_avg_s  = np.mean(pn_sdf[id_stim_s, n_pns_recep:])
            
            # Calculate the ratio for PN responses
            pn_avg_dif[id_loop] = pn_avg_w-pn_avg_s
            pn_avg_ratio[id_loop] = pn_avg_s/pn_avg_w
            pn_peak_ratio[id_loop]  = pn_peak_s[id_loop]/pn_peak_w[id_loop]
            
            if stim_type == 'pl':
                # Calculate the mean and the peak for PN responses
                pn_sdf_dt = pn_sdf_time[1]-pn_sdf_time[0]
                pn_tmp = np.zeros((np.size(id_stim_w),2))
                
                pn_tmp[:,0] = np.mean(pn_sdf[id_stim_w, :n_pns_recep], axis=1)
                pn_tmp[:,1] = np.mean(pn_sdf[id_stim_w, n_pns_recep:], axis=1)
                perf_time = np.zeros((2, 3))
                perf_avg = np.zeros((2, 3))
                id_glo = None
                for id_glo in range(2):
                    for thr_id, thr in enumerate([50, 100, 150]):
                        perf_time[id_glo, thr_id, ] = np.sum(pn_tmp[:, id_glo]>thr)*pn_sdf_dt
                        if perf_time[id_glo, thr_id, ]>0:
                            perf_avg[id_glo, thr_id, ] = np.average(pn_tmp[:, id_glo], 
                                weights=(pn_tmp[:, id_glo]>thr))
                 
                print('mean time weak')
                print(perf_time[0,:])
                
                print('mean time strong')
                print(perf_time[1,:])
            
            print('peak strong:%.1f Hz, peak weak:%.1f Hz'
                  %(np.mean(pn_peak_s), np.mean(pn_peak_w)))
            print('peak ratio:%.1f, avg ratio:%.1f, avg dif:%.1f Hz'
                  %(np.mean(np.ma.masked_invalid(pn_peak_ratio)), 
                    np.mean(np.ma.masked_invalid(pn_avg_ratio)), np.mean(pn_avg_dif)))
    
    toc = timeit.default_timer()
    print('time to do %d sims: %.1f s'%(n_loops, toc-tic))
    print('')
                        
else:
    print('run from import')

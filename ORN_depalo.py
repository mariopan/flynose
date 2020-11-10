#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 10:06:34 2020

ORN model with dynamics based on a modified version of linearnonlinear (LN) 
model described by De Palo 2010 coupled with a 
nonlinear function rectifying the frequency activity.

The modified LN model is a system of 2 coupled ODE.

This script is compound of the first two part of flynose: 
    stimulus generation and ORN simulation.

@author: mario
"""


import numpy as np
import matplotlib.pyplot as plt
import timeit
from scipy.integrate import odeint


# *****************************************************************
# FUNCTIONS


# stimulus
def stim_fcn(stim_params):
    
    tmp_ks = ['stim_type', 'stim_dur', 'pts_ms', 't_tot', 't_on', 'concs', 'conc0',]    
    [stim_type, stim_dur, pts_ms, t_tot, t_on, concs, conc0] = [
        stim_params[x] for x in tmp_ks]  
    
    # Stimulus params    
    t_off           = t_on+stim_dur
    stim_on         = t_on*pts_ms   # [num. of samples]
    stim_off        = t_off*pts_ms    
    
    n2sim           = pts_ms*t_tot + 1      # number of time points
    
    u_od            = np.ones((n2sim, 1)) * conc0
    
    if (stim_type == 'ss'):
        # Single Step Stimuli
        
        tau_on          = 50
        t_tmp           = np.linspace(0, t_off-t_on, stim_off-stim_on)    
            
        u_od[stim_on:stim_off, 0] = conc0 + concs*(1-np.exp(-t_tmp/tau_on))
        
        t_tmp           = np.linspace(0, t_tot-t_off, 1+t_tot*pts_ms-stim_off)    
        
        u_od[stim_off:, 0] = conc0 + u_od[stim_off-1, 0]*np.exp(-t_tmp/tau_on)
        
    return u_od

# transduction function
def transd(r0,t,u,orn_params,):

    alpha_r = orn_params['alpha_r']
    beta_r = orn_params['beta_r']
    n = orn_params['n']
    
    dt = t[1]-t[0]
    b = -alpha_r * u**n - beta_r
    a = alpha_r * u**n 
    r = (r0 + a/b)*np.exp(b*dt)-a/b
    # drdt = alpha_r*u**n*(1-r) - beta_r*r
    return r    

def depalo_eq(z,t,r,orn_params,): 
    
    tmp_ks = ['a_y', 'b_y',  'b_x', 'a_r', 'c_x',  'd_x',]    
    [a_y, b_y, b_x, a_r, c_x, d_x] = [
        orn_params[x] for x in tmp_ks] 
    
    x = z[0]
    y = z[1]
    
    # dxdt = ax*y - bx*x
    # dydt = ar*r - cx*x*(1+dy*y) - by*y 
    dydt = a_r*r - c_x*x*(1+d_x*y) - b_x*y# - omega_nsi*w*y 
    dxdt = a_y*y - b_y*x
    
    dzdt = [dxdt,dydt]
    return dzdt


def rect_func(x, orn_params):
    nu_max = orn_params['nu_max_rect']
    a_rect = orn_params['a_rect']
    c_rect = orn_params['c_rect']
    ot = nu_max/(1 + np.exp(-a_rect*(x-c_rect)))
    return ot


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



def main(orn_params, stim_params):
    
    # *****************************************************************
    # STIMULUS GENERATION    
    tmp_ks = ['pts_ms', 't_tot',]    
    [pts_ms, t_tot, ] = [
        stim_params[x] for x in tmp_ks] 
    
    n2sim       = pts_ms*t_tot + 1      # number of time points
    t           = np.linspace(0, t_tot, n2sim) # time points
    
    u_od        = stim_fcn(stim_params)
        
    # *****************************************************************
    # ORN PARAMETERS 
    tmp_ks = ['t_ref', ]    
    [t_ref, ] = [orn_params[x] for x in tmp_ks] 
    
    # *****************************************************************
    # initialize output vectors
    r_orn               = np.zeros((n2sim, ))
    x_orn               = np.zeros((n2sim,))
    y_orn               = np.zeros((n2sim, ))
    nu_orn              = np.zeros((n2sim, ))   
    z0_unid             = np.zeros(2)      

    # ********************************************************
    # Trasnduction output
    for tt in range(1, n2sim-1):
        # span for next time step
        tspan = [t[tt-1],t[tt]]
        r_orn[tt, ] = transd(r_orn[tt-1, ], tspan, u_od[tt], orn_params)   

    # *****************************************************************
    # GENERATE ORN RESPONSE TO ODOR INPUT 
        
    for tt in range(1, n2sim-t_ref-1):
        # span for next time step
        tspan = [t[tt-1,],t[tt,]]
        
        z_orn = odeint(depalo_eq, z0_unid, tspan,
                       args=(r_orn[tt, ], orn_params,))
        z0_unid[0:2] = z_orn[1,:]
       
        x_orn[tt, ] = z_orn[1,0]
        y_orn[tt, ] = z_orn[1,1]
       
        nu_orn[tt, ] = rect_func(y_orn[tt, ], orn_params)
    
    
    orn_depalo_out = [t, u_od, r_orn, x_orn, y_orn, 
                      nu_orn]
    
    return  orn_depalo_out 


if __name__ == '__main__':
    print('run directly')
    stim_data_fld = ''
    
    #***********************************************
    # output params 
    fld_analysis    = 'NSI_analysis/orn_depalo/trials'
   
    #***********************************************
    # stimulus params
    stim_params         = dict([
                    ('stim_dur' , 500),
                    ('stim_type' , 'ss'),   # 'ts'  # 'ss' # 'pl'
                    ('pts_ms' , 1),         # simulated pts per ms 
                    ('t_tot', 1000),        # ms 
                    ('t_on', 160),          # ms
                    ('concs', .01),
                    ('conc0', .00001),
                    ])
    
    # ORN PARAMETERS 
    orn_params = dict([
        # Transduction params
                        ('n', .5),
                        ('alpha_r', .9), 
                        ('beta_r', .090),
        # rectification params        
                        ('c_rect', 1),
                        ('a_rect', 3.3), 
                        ('nu_max_rect', 250),
        # Spiking machine params
                        ('t_ref', 2*stim_params['pts_ms']), # ms; refractory period 
                        ('a_y', 0.25),  
                        ('b_y', 0.002),  
                        ('b_x', 0.2),       # 0.12
                        ('c_x', 0.0028),     # 0.004
                        ('d_x', 1),           
                        ('a_r', 1),
                        ])                    
    
    # *****************************************************************
    # LIF ORN simulation
    n_loops         = 1    
    
    tic = timeit.default_timer()
    for id_loop in range(n_loops):
        orn_depalo_out = main(orn_params, stim_params)
        [t, u_od, r_orn, x_orn, y_orn, 
         nu_orn, ]  = orn_depalo_out
    toc = timeit.default_timer()
    print('time to do %d sims: %.1f s'%(n_loops, toc-tic))
    print('')

    # %% *****************************************************************
    # FIGURE ORN dynamics
    t_on = stim_params['t_on']
    t_tot = stim_params['t_tot']
    pts_ms = stim_params['pts_ms']
    
    t2plot = -t_on-10, t_tot-t_on+10
    rs = 3 # number of rows
    cs = 1 # number of cols
                    
    panels_id = ['a', 'b', 'c', 'd']
    fig_orn = plt.figure(figsize=[8.5, 8])
    fig_orn.tight_layout()
    
    ax_orn1 = plt.subplot(rs, cs, 1)
    ax_orn2 = ax_orn1.twinx()
    ax_orn3 = plt.subplot(rs, cs, 2)
    ax_orn4 = ax_orn3.twinx()
    # ax_orn_sc = plt.subplot(rs, cs, 3)
    ax_orn_fr = plt.subplot(rs, cs, 3)
    
    
    ax_orn1.plot(t-t_on, u_od[:, ], linewidth=lw+1, color=black, 
                 label=r'Glom %d'%(1))
    ax_orn2.plot(t-t_on, r_orn[:, ], linewidth=lw+1, color=blue,  
                 label=r'r, glom: %d'%(1))
    ax_orn3.plot(t-t_on, x_orn[:, ], linewidth=lw+1, color=black, 
                 label=r'Od, glom : %d'%(0))
    ax_orn4.plot(t-t_on, y_orn[:, ], linewidth=lw+1, color=blue,  
                 label=r'Od, glom : %d'%(0))

    ax_orn_fr.plot(t-t_on, nu_orn, linewidth=lw+1, color=green)
    
    # spikes_orn_0 = np.argwhere(num_spike_orn)        
    # ax_orn_sc.scatter(spikes_orn_0[:,0]/pts_ms-t_on, 
                    # spikes_orn_0[:,1], color=purple, s=10)

    # FIGURE SETTINGS
    ax_orn1.tick_params(axis='both', which='major', labelsize=ticks_fs)
    ax_orn2.tick_params(axis='both', which='major', labelsize=ticks_fs)
    ax_orn3.tick_params(axis='both', which='major', labelsize=ticks_fs)
    ax_orn4.tick_params(axis='both', which='major', labelsize=ticks_fs)
    ax_orn_fr.tick_params(axis='both', which='major', labelsize=ticks_fs)
    # ax_orn_sc.tick_params(axis='both', which='major', labelsize=ticks_fs)
    
    ax_orn1.set_xticklabels('')
    ax_orn2.set_xticklabels('')
    ax_orn3.set_xticklabels('')
    ax_orn4.set_xticklabels('')
    # ax_orn_sc.set_xticklabels('')
    
    
    ax_orn1.set_ylabel('Input (a.u.)', fontsize=label_fs)
    ax_orn2.set_ylabel(r'r (a.u.) ', fontsize=label_fs, color=blue,)
    ax_orn3.set_ylabel(r'Potential (x)', fontsize=label_fs)
    ax_orn4.set_ylabel(r'adapt (y)', fontsize=label_fs, color=blue,)
    ax_orn_fr.set_ylabel('firing rates (Hz)', fontsize=label_fs)
    ax_orn_fr.set_xlabel('Time  (ms)', fontsize=label_fs) 
    # ax_orn_sc.set_ylabel('Neuron id', fontsize=label_fs)

    ax_orn1.text(-.15, 1.25, panels_id[0], transform=ax_orn1.transAxes, 
                      fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    ax_orn3.text(-.15, 1.25, panels_id[1], transform=ax_orn3.transAxes, 
                      fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    # ax_orn_sc.text(-.15, 1.25, panels_id[2], transform=ax_orn_sc.transAxes,
    #                   fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    ax_orn_fr.text(-.15, 1.25, panels_id[3], transform=ax_orn_fr.transAxes, 
                      fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    
    ax_orn1.spines['top'].set_color('none')
    ax_orn2.spines['top'].set_color('none')
    ax_orn3.spines['top'].set_color('none')
    ax_orn4.spines['top'].set_color('none')
    # ax_orn_sc.spines['right'].set_color('none')
    # ax_orn_sc.spines['top'].set_color('none')
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
    # ll, bb, ww, hh = ax_orn_sc.get_position().bounds
    # ax_orn_sc.set_position([ll_new, bb+bb_plus, ww_new, hh])
    ll, bb, ww, hh = ax_orn_fr.get_position().bounds
    ax_orn_fr.set_position([ll_new, bb-bb_plus, ww_new, hh])
    
    ax_orn1.set_xlim((t2plot))
    ax_orn2.set_xlim((t2plot))
    ax_orn3.set_xlim((t2plot))
    ax_orn4.set_xlim((t2plot))
    # ax_orn_sc.set_xlim((t2plot))
    ax_orn_fr.set_xlim((t2plot))
    plt.show()
                        
else:
    print('run from import')



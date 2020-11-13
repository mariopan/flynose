#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:04:01 2020

This script runs the simulations for the ORNs with a LIF dynamics:
We describe ORN activity in terms of an odorant transduction process combined 
with a biophysical spike generator \cite{lazar2020molecular}. During 
transduction, odorants bind and unbind at olfactory receptors according to 
simple rate equations. As we are not interested in the competition of 
different odorants for the same receptors, we simplify the customary 
two-stages binding and activation model \cite{rospars2008competitive, 
                                              nowotny2013data,chan2018odorant} 
to a single binding rate equation for the fraction r of receptors bound to an 
odorant, 

\begin{eqnarray}
\label{eq:transduction}
\dot{r} &=& b_r C^n (1-r) - d_r r 
\end{eqnarray}

We described the spike generator by a leaky integrate-and-fire neuron with 
adaptation,

\begin{eqnarray}
  \tau_{\text{V}} \dot{V} &=& (1+y)(V_{\text{rest}}^{\text{ORN}} - V) 
  + r\, (V_{\text{rev}}^{\text{ORN}} - V) \\
  \dot{y} &=& - b_y y + a_y  \sum_{t_S \in S} \delta(t-t_s) 
\end{eqnarray}

where, $V_{\text{rest}}$ is the resting potential, $V_{\text{rev}}$ the 
reversal potential of the ion channels opened due to receptor activation,  
$\zeta_x$ is a normally distributed white noise process representing receptor 
noise, and $y$ is a spike rate adaptation variable with decay time constant 
$b_y$. 


See also: 
    Treves, A (1993). Mean-field analysis of neuronal spike dynamics Network: 
        Computation in Neural Systems 4: 259-284.
    http://www.scholarpedia.org/article/Spike_frequency_adaptation

@author: mario
"""
import numpy as np
import matplotlib.pyplot as plt
import timeit

import sdf_krofczik

# %% STANDARD FIGURE PARAMS
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


#%% DEFINE FUNCTIONS

# tic toc
def tictoc():
    return timeit.default_timer()

# stimulus
def stim_fcn(stim_params):
    
    tmp_ks = \
        ['stim_type', 'stim_dur', 'pts_ms', 't_tot', 't_on', 'concs', 'conc0',]    
    [stim_type, stim_dur, pts_ms, t_tot, t_on, concs, conc0] = [
        stim_params[x] for x in tmp_ks]  
    
    # Stimulus params    
    n_od            = len(concs)
    t_off           = t_on+stim_dur
    
    n2sim           = pts_ms*t_tot + 1      # number of time points
    
    u_od            = np.ones((n2sim, n_od)) * conc0
    
    if (stim_type == 'ss'):
        # Single Step Stimuli
        
        tau_on          = 50
        for nn in range(n_od):
            stim_on         = t_on[nn]*pts_ms   # [num. of samples]
            stim_off        = t_off[nn]*pts_ms    
            # stimulus onset
            t_tmp           = \
                np.linspace(0, t_off[nn]-t_on[nn], stim_off-stim_on)
            
            u_od[stim_on:stim_off, nn] = \
                conc0[nn] + concs[nn]*(1 - np.exp(-t_tmp/tau_on))
            
            # stimulus offset
            t_tmp           = \
                np.linspace(0, t_tot-t_off[nn], 1 + t_tot*pts_ms-stim_off)    
            
            u_od[stim_off:, nn] = conc0[nn] + \
                u_od[stim_off-1, nn]*np.exp(-t_tmp/tau_on)
 
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

# LIF 1
def v_orn_ode(v0, t, r, y, vrev, orn_params, ):
    tau_v = orn_params['tau_v']
    # vrev = orn_params['vrev']
    vrest = orn_params['vrest']
    r = r*orn_params['g_r']
    y = y*orn_params['g_y']
    
    dt = t[1]-t[0]
    b = -(1 + y + r)/tau_v
    a = (vrest*(1+y) + r*vrev)/tau_v
    v = (v0 + a/b)*np.exp(b*dt)-a/b
    #dvdt = ((1+g_y*y)*(vrest-v) + g_r*r*(vrev-v) )/tau_v
    return v

# adaptation
def y_adapt(y0, t, orn_params):
#   orn_params = np.array([b_r, d_r, n, tau_v, vrev, vrest, g, t_ref, theta, ay, tau_y])
    beta_y = orn_params['beta_y']
    
    dt = t[1] - t[0]
    y = y0 * np.exp(-dt*beta_y)
    #dydt = -y/tau_y + ay * sum(delta(t-t_spike))
    return y

    
# ************************************************************************
# main function of the LIF ORN 
def main(orn_params, stim_params, sdf_params):
    
    
    # ********************************************************
    [tau_sdf, dt_sdf] = sdf_params
    
    # ********************************************************
    # Stimulus generation
    tmp_ks = ['pts_ms', 't_tot',]    
    [pts_ms, t_tot, ] = [
        stim_params[x] for x in tmp_ks]    
    
    n2sim       = pts_ms*t_tot + 1      # number of time points
    t           = np.linspace(0, t_tot, n2sim) # time points
    n_neu       = 2
    
    u_od        = np.zeros((n2sim, ))
    u_od        = stim_fcn(stim_params)
    
    
    # *****************************************************************
    # ORN PARAMETERS 
    t_ref           = orn_params['t_ref']
    theta           = orn_params['theta']
    alpha_y         = orn_params['alpha_y']
    vrest           = orn_params['vrest']
    vrev            = orn_params['vrev']
    nsi_mat         = [1, 0] # neuron 0 NSIs neuron 1 and viceversa
    w_nsi           = orn_params['w_nsi']
    
    # *****************************************************************
    # initialize output vectors
    r_orn           = np.zeros((n2sim, n_neu)) 
    v_orn           = vrest*np.ones((n2sim, n_neu)) 
    y_orn           = np.zeros((n2sim, n_neu))
    vrev_t          = np.ones(n_neu)*vrev
    num_spikes      = np.zeros((n2sim, n_neu))
    orn_ref         = np.zeros(n_neu)
    
    #%% ********************************************************
    # Trasnduction output
    tic  = tictoc()
    for tt in range(1, n2sim-1):
        # span for next time step
        tspan = [t[tt-1],t[tt]]
        r_orn[tt, :] = transd(r_orn[tt-1, :], tspan, u_od[tt, :], orn_params)   
    toc  = tictoc()-tic
    print(toc)
    
    #%% ********************************************************
    # LIF ORN generation
    tic  = tictoc()
    for tt in range(1, n2sim-t_ref-1):
        # span for next time step
        tspan = [t[tt-1],t[tt]]
        
        y_orn[tt, :] = y_adapt(y_orn[tt-1, :], tspan, orn_params)
        vrev_t = vrev*(1 - w_nsi*(v_orn[tt-1, nsi_mat]-vrest))

        # *********************************
        # ORNs whose ref_cnt is equal to zero:
        orn_ref0 = (orn_ref==0)
        v_orn[tt, orn_ref0] = v_orn_ode(v_orn[tt-1, orn_ref0], tspan, 
                                        r_orn[tt, orn_ref0], y_orn[tt, orn_ref0], 
                                        vrev_t[orn_ref0], orn_params)
        
        # ORNs whose Voltage is above threshold AND whose ref_cnt is equal to zero:
        orn_above_thr = (v_orn[tt, :] >= theta) & (orn_ref==0)
        num_spikes[tt, orn_above_thr] = num_spikes[tt, orn_above_thr] + 1
        orn_ref[orn_above_thr] = t_ref
        y_orn[tt:(tt+t_ref), orn_above_thr] = y_orn[tt, orn_above_thr]+alpha_y
           
            
        # ORNs whose ref_cnt is different from zero:
        orn_ref_no0 = (orn_ref!=0)
        # Refractory period count down
        orn_ref[orn_ref_no0] = orn_ref[orn_ref_no0] - 1 
    
    toc  = tictoc()-tic
    print(toc)
    
    #%% *****************************************************************
    # Calculate the spike matrix 
    spike_matrix = np.asarray(np.where(num_spikes))
    spike_matrix[0,:] = spike_matrix[0,:]/pts_ms
    spike_matrix = np.transpose(spike_matrix)
    
    # *****************************************************************
    # SDF extraction from the spike matrix
    sdf_size    = int(stim_params['t_tot']/dt_sdf)
    t_sdf = np.linspace(0, dt_sdf*sdf_size, sdf_size)
    orn_sdf = np.zeros_like(t_sdf)
    
    if ~(np.sum(spike_matrix) == 0):
        orn_sdf, t_sdf = sdf_krofczik.main(spike_matrix, sdf_size,
                                                tau_sdf, dt_sdf)  # (Hz, ms)
        orn_sdf = orn_sdf[:,0]*1e3    
    
    orn_lif_out = [t, u_od, r_orn, v_orn, y_orn, 
                   num_spikes, spike_matrix, orn_sdf, t_sdf]
    
    return  orn_lif_out 


# ************************************************************
# Launching script and Figure
if __name__ == '__main__':
    print('run directly')
    stim_data_fld = ''
    # output params 
    fld_analysis    = 'NSI_analysis/trials'
   
    #***********************************************
    # analysis params
    tau_sdf = 50
    dt_sdf  = 5
    sdf_params = [tau_sdf, dt_sdf]
    
    # stimulus params
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
                        
    stim_params     = dict([
                        ('stim_dur' , 500),
                        ('stim_type' , 'ss'),   # 'ts'  # 'ss' # 'pl'
                        ('pts_ms' , 5),         # simulated pts per ms 
                        ('t_tot', 1500),        # ms 
                        ('od_pref', od_pref)])
    n_od = 1
    if n_od == 1:
        concs_params    = dict([
                        ('t_on', np.array([300])),          # ms
                        ('concs', [0.01]),
                        ('conc0', [1e-3]),
                        ])
    elif n_od == 2:
        concs_params    = dict([
                        ('t_on', [300, 300]),          # ms
                        ('concs', [.01, .1e-3]),
                        ('conc0', [1e-3, 1e-3]),
                        ])
    
    stim_params.update(concs_params)
    
    # ORN PARAMETERS 
    orn_params = dict([
        # Transduction params
                        ('n', .6),
                        ('alpha_r', 1), 
                        ('beta_r', .090),
        # LIF params
                        ('t_ref', 2*stim_params['pts_ms']), # ms; refractory period 
                        ('theta', 1),                 # [mV] firing threshold
                        ('tau_v', 2),        # [ms]
                        ('vrest', -.5),      # [mV] resting potential
                        ('vrev', 12),  # [mV] reversal potential
                        # ('v_k', vrest),
                        ('g_y', .5),       
                        ('g_r', 1),       
        # Adaptation params
                        ('alpha_y', .25), 
                        ('beta_y', .002), 
        # NSI params
                        ('w_nsi', .2), ])
    
    tic = timeit.default_timer()
    orn_lif_out         = main(orn_params, stim_params, sdf_params)
    toc = timeit.default_timer()
    
    print('sim run time: %.2f s' %(toc-tic))
    
    [t, u_od, r_orn, v_orn, y_orn, num_spikes, spike_matrix, orn_sdf,
     t_sdf]  = orn_lif_out
    
    
    
    #%% *****************************************************************
    # FIGURE ORN dynamics
    t_on = stim_params['t_on']
    pts_ms = stim_params['pts_ms']
    vrest = orn_params['vrest']
    vrev = orn_params['vrev']
    
    t2plot = -500, 1000 #t_tot #-t_on, t_tot-t_on
    rs = 3 # number of rows
    cs = 1 # number of cols
                    
    panels_id = ['a', 'b', 'c', 'd']
    fig_orn = plt.figure(figsize=[8.5, 8])
    fig_orn.tight_layout()
    
    ax_orn1 = plt.subplot(rs, cs, 1)
    ax_orn2 = ax_orn1.twinx()
    ax_orn3 = plt.subplot(rs, cs, 2)
    ax_orn4 = ax_orn3.twinx()
    ax_orn_fr = plt.subplot(rs, cs, 3)
    
    
    ax_orn1.plot(t-t_on, u_od[:,], linewidth=lw+1, color=black,)
    ax_orn2.plot(t-t_on, r_orn[:,], linewidth=lw+1, color=blue,)
    ax_orn3.plot(t-t_on, v_orn[:,], linewidth=lw+1, color=black,)
    ax_orn3.plot([t[0]-t_on, t[-1]-t_on], [vrest, vrest], '--', linewidth=lw, color=red,)
    ax_orn3.plot([t[0]-t_on, t[-1]-t_on], [vrev, vrev], '-.', linewidth=lw, color=red,)
    ax_orn4.plot(t-t_on, y_orn[:,], linewidth=lw+1, color=blue,)

    ax_orn_fr.plot(t_sdf-t_on, orn_sdf, color=green, linewidth=lw+1, 
                      label='\nu')
            
    spikes_orn_0 = np.argwhere(num_spikes)        
    # ax_orn_sc.scatter(spikes_orn_0[:,0]/pts_ms-t_on, 
    #                 spikes_orn_0[:,1], color=purple, s=10)

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
    ax_orn3.set_ylabel(r'V (a.u.)', fontsize=label_fs)
    ax_orn4.set_ylabel(r'y adapt (a.u.)', fontsize=label_fs, color=blue,)
    ax_orn_fr.set_ylabel('firing rates (Hz)', fontsize=label_fs)
    ax_orn_fr.set_xlabel('Time  (ms)', fontsize=label_fs) 
    # ax_orn_sc.set_ylabel('Neuron id', fontsize=label_fs)

    ax_orn1.text(-.15, 1.25, panels_id[0], transform=ax_orn1.transAxes, 
                      fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    ax_orn3.text(-.15, 1.25, panels_id[1], transform=ax_orn3.transAxes, 
                      fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    # ax_orn_sc.text(-.15, 1.25, panels_id[2], transform=ax_orn_sc.transAxes,
                      # fontsize=panel_fs, fontweight='bold', va='top', ha='right')
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
       

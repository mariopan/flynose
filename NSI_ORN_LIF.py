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
                conc0 + concs[nn]*(1 - np.exp(-t_tmp/tau_on))
            
            # stimulus offset
            t_tmp           = \
                np.linspace(0, t_tot-t_off[nn], 1 + t_tot*pts_ms-stim_off)    
            
            u_od[stim_off:, nn] = conc0 + \
                (u_od[stim_off-1, nn]-conc0)*np.exp(-t_tmp/tau_on)
 
    return u_od


# transduction function
def transd(r0,t,u,orn_params,):

    alpha_r = orn_params['alpha_r']
    alpha_r[alpha_r==0] = 1e-16
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

def rev_fcn(vrev, w_nsi, v_orn, nsi_mat, vrest, t, n_neu, ):
    # 1 Co-housed ORN
    def solo_ORN():
        vrev_t = vrev
        return vrev_t
    
    # 2 Co-housed ORNs
    def duo_ORN():
        vect_a = nsi_vect[:, 1]
        vrev_t = vrev*(1 - w_nsi*(v_orn[t, vect_a]-vrest))
        return vrev_t
    
    # 3 Co-housed ORNs
    def tri_ORN():
        vect_a = [nsi_vect[x, 1] for x in range(0, len(nsi_vect[:, 0]), 2)]
        vect_b = [nsi_vect[x, 1] for x in range(1, len(nsi_vect[:, 0]), 2)]
         
        vrev_t = vrev*(1 - (w_nsi*(v_orn[t, vect_a] - vrest)+
                            w_nsi*(v_orn[t, vect_b] - vrest)))
        return vrev_t
    
    # 4 Co-housed ORNs
    def quad_ORN():
        vect_a = [nsi_vect[x, 1] for x in range(0, len(nsi_vect[:, 0]), 3)]
        vect_b = [nsi_vect[x, 1] for x in range(1, len(nsi_vect[:, 0]), 3)]
        vect_c = [nsi_vect[x, 1] for x in range(2, len(nsi_vect[:, 0]), 3)]
        
        vrev_t = vrev*(1 - (w_nsi*(v_orn[t, vect_a] - vrest)+
                            w_nsi*(v_orn[t, vect_b] - vrest)+
                            w_nsi*(v_orn[t, vect_c] - vrest)))
        return vrev_t
    
    # Convert matrix to vector
    nsi_vect = np.transpose(np.asarray(np.where(nsi_mat == 1)))
    
    # Run correct ORN number
    rev_dict = {
        1 : solo_ORN,
        2 :  duo_ORN, 
        3 :  tri_ORN,
        4 : quad_ORN,
        }
    
    vrev_t = rev_dict[n_neu]()
    return vrev_t
   
        
    
# ************************************************************************
# main function of the LIF ORN 
def main(orn_params, stim_params, sdf_params, sens_params):
    
    # SDF PARAMETERS 
    [tau_sdf, dt_sdf] = sdf_params
    
    # STIMULI PARAMETERS 
    tmp_ks = ['pts_ms', 't_tot', 'n_od', ]    
    [pts_ms, t_tot, n_od, ] = [
        stim_params[x] for x in tmp_ks]    
    # SENSILLUM PARAMETERS
    n_neu           = sens_params['n_neu']
    # od_pref         = sens_params['od_pref']
    w_nsi           = sens_params['w_nsi']
    n_sens          = sens_params['n_sens']
    
    # Connectivity matrix for ORNs
    nsi_mat = np.zeros((n_neu*n_sens, n_neu*n_sens))
    
    for pp in range(n_sens*n_neu):
        nn = np.arange(np.mod(pp,n_sens), n_neu*n_sens, 
                       n_sens,dtype='int')
        nsi_mat[pp, nn] = 1
    np.fill_diagonal(nsi_mat, 0)
    
    # n_linkth = n_sens*n_neu*(n_neu-1)
    # print('Link Th: %d, Link eff: %d' %(n_linkth,np.sum(nsi_mat)))
    # fig= plt.figure()
    # plt.imshow(nsi_mat, )
    # plt.show()
    
    # # OLD Connectivity matrix for ORNs
    # nsi_mat = np.zeros((n_neu*n_sens, n_neu*n_sens))
    # for pp in range(n_sens):
    #     nsi_mat[pp*n_neu:(pp+1)*n_neu, pp*n_neu:(pp+1)*n_neu] = 1
    #     np.fill_diagonal(nsi_mat[pp*n_neu:(pp+1)*n_neu, \
    #                               pp*n_neu:(pp+1)*n_neu], 0)
    
    # ORN PARAMETERS 
    t_ref           = orn_params['t_ref']
    theta           = orn_params['theta']
    alpha_y         = orn_params['alpha_y']
    vrest           = orn_params['vrest']
    vrev            = orn_params['vrev']
    
    # INITIALIZE OUTPUT VECTORS
    n2sim           = pts_ms*t_tot + 1      # number of time points
    t               = np.linspace(0, t_tot, n2sim) # time points
    n_neu_tot       = n_neu*n_sens

    u_od            = np.zeros((n2sim, n_od))
    
    r_orn           = np.zeros((n2sim, n_neu, n_od)) 
    v_orn           = vrest*np.ones((n2sim, n_neu_tot)) 
    y_orn           = np.zeros((n2sim, n_neu_tot))
    vrev_t          = np.ones(n_neu_tot)*vrev
    num_spikes      = np.zeros((n2sim, n_neu_tot))
    orn_ref         = np.zeros(n_neu_tot)
    
    # ODOUR STIMULUS/I
    u_od            = stim_fcn(stim_params)
    
    # Transduction output
    for tt in range(1, n2sim-1):
        # span for next time step
        tspan = [t[tt-1],t[tt]] 
        r_orn[tt, :, :] = transd(r_orn[tt-1, :, :], tspan, 
                                  u_od[tt, :], orn_params)   

    r_tmp = np.sum(r_orn, axis=2)
    r_tot = np.zeros((n2sim, n_neu*n_sens))
    for ss in range(n_neu):
        nn_s = np.arange(ss, n_neu*n_sens, n_neu, dtype='int')
        for nn in nn_s:
            r_tot[:, nn] = r_tmp[:, ss] # + noise
    
    # ********************************************************
    # LIF ORN DYNAMICS
    for tt in range(1, n2sim-t_ref-1):
        # span for next time step
        tspan = [t[tt-1],t[tt]]
        
        # adaptation variable
        y_orn[tt, :] = y_adapt(y_orn[tt-1, :], tspan, orn_params)
        
        # NSI effect on reversal potential 
        vrev_t = rev_fcn(vrev, w_nsi, v_orn, nsi_mat, vrest, tt-1, n_neu)

        
        # ORNs whose ref_cnt is equal to zero:
        orn_ref0 = (orn_ref==0)
        if n_neu == 1:
            v_orn[tt, orn_ref0] = v_orn_ode(v_orn[tt-1, orn_ref0], tspan, 
                                        r_tot[tt, orn_ref0], y_orn[tt, orn_ref0], 
                                        vrev_t, orn_params)
        else:
            v_orn[tt, orn_ref0] = v_orn_ode(v_orn[tt-1, orn_ref0], tspan, 
                                        r_tot[tt, orn_ref0], y_orn[tt, orn_ref0], 
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
    
    # Calculate the spike matrix 
    spike_matrix = np.asarray(np.where(num_spikes))
    spike_matrix[0,:] = spike_matrix[0,:]/pts_ms
    spike_matrix = np.transpose(spike_matrix)
    
    # SDF extraction from the spike matrix
    sdf_size    = int(stim_params['t_tot']/dt_sdf)
    t_sdf = np.linspace(0, dt_sdf*sdf_size, sdf_size)
    orn_sdf = np.zeros_like(t_sdf)
    
    if ~(np.sum(spike_matrix) == 0):
        orn_sdf, t_sdf = sdf_krofczik.main(spike_matrix, sdf_size,
                                                tau_sdf, dt_sdf)  # (Hz, ms)
        # orn_sdf = orn_sdf[:,0]*1e3    
        orn_sdf = orn_sdf*1e3 
    orn_lif_out = [t, u_od, r_orn, v_orn, y_orn, 
                   num_spikes, spike_matrix, orn_sdf, t_sdf,]
    
    return  orn_lif_out 


# ************************************************************
# Launching script and Figure
if __name__ == '__main__':
    print('run directly')
    
    # output params 
    stim_data_fld = ''
    fld_analysis    = 'NSI_analysis/trials'
   
    # stimulus params
    stim_params     = dict([
                        ('stim_type' , 'ss'),   # 'ts'  # 'ss' # 'pl'
                        ('pts_ms' , 5),         # simulated pts per ms 
                        ('n_od', 2), 
                        ('t_tot', 2000),        # ms 
                        ('conc0', [2.853669391e-04]),
                        ])
    
    n_od = stim_params['n_od']
    if n_od == 1:
        concs_params    = dict([
                        ('stim_dur' , np.array([500])),
                        ('t_on', np.array([300])),          # ms
                        ('concs', np.array([0.01])),
                        ])
    elif n_od == 2:
        concs_params    = dict([
                        ('stim_dur' , np.array([500, 500])),
                        ('t_on', np.array([800, 800])),          # ms
                        ('concs', np.array([.002, .002])),
                        ])
    
    stim_params.update(concs_params)
    
    #%% Sensilla/network parameters
    n_sens = 1
    n_neu = 3
    
    sens_params     = dict([
                        ('n_neu', n_neu),
                        ('n_sens', n_sens),
                        # NSI params
                        ('w_nsi', .05), 
                        ('od_pref' , \
                         np.array([[1,0], [0,1], [0,1], [1,0], 
                                   [0,0], [1,0], [0,1], [1,0]]))
                               #  [0,0],
                               #  [1,0],
                               #  [1,0],
                               #  [0,0],
                               #  [0,1],
                               #  [1,0],
                               #  [0,0],
                               #  [0,1]
                        ])
    
    # Create Transduction Matrix
    transd_mat = np.zeros((n_neu, n_od))
    for pp in range(n_neu):
        transd_mat[pp,:] = sens_params['od_pref'][pp,:]
    
    # ORN Parameters 
    orn_params = dict([
        # Transduction params
                        ('n', .822066870*transd_mat), 
                        ('alpha_r', 12.6228808*transd_mat), 
                        ('beta_r', 7.6758436748e-02*transd_mat),
        # LIF params
                        ('t_ref', 2*stim_params['pts_ms']), # ms; refractory period 
                        ('theta', 1),                 # [mV] firing threshold
                        ('tau_v', 2.26183540),        # [ms]
                        ('vrest', -0.969461053),      # [mV] resting potential
                        ('vrev', 21.1784081),  # [mV] reversal potential
                        # ('v_k', vrest),
                        ('g_y', .5853575783),       
                        ('g_r', .864162073),       
        # Adaptation params
                        ('alpha_y', .45310619), 
                        ('beta_y', 3.467184e-03), 
                        ])
    
    # analysis params
    tau_sdf         = 41
    dt_sdf          = 5
    sdf_params      = [tau_sdf, dt_sdf]
    
    #*********************************************************************
    # ORN LIF SIMULATION
    tic = timeit.default_timer()
    orn_lif_out         = main(orn_params, stim_params, sdf_params, sens_params)
    toc = timeit.default_timer()
    
    print('sim run time: %.2f s' %(toc-tic))
    
    [t, u_od, r_orn, v_orn, y_orn, num_spikes, spike_matrix, orn_sdf,
     t_sdf,]  = orn_lif_out
    
      
    #%% *****************************************************************
    # FIGURE ORN dynamics
    t_on = np.min(stim_params['t_on'])
    pts_ms = stim_params['pts_ms']
    vrest = orn_params['vrest']
    vrev = orn_params['vrev']
    n_neu = sens_params['n_neu']
    
    t2plot = -t_on, 1000 #t_tot #-t_on, t_tot-t_on
    rs = 3 # number of rows
    cs = n_neu # number of cols
                    
    panels_id = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
    fig_orn, ax_orn = plt.subplots(rs, cs, figsize=[8.5, 6.5])
    fig_orn.tight_layout()
    
    if n_neu == 1:
        ax_orn2a = ax_orn[0].twinx()
        ax_orn4a = ax_orn[1].twinx()
        weight_od = u_od*transd_mat[0,:]
        ax_orn[0].plot(t-t_on, weight_od, linewidth=lw+1, color=black,)
        ax_orn2a.plot(t-t_on, r_orn[:, 0], linewidth=lw+1, color=blue,)
        ax_orn4a.plot(t-t_on, y_orn[:, 0], linewidth=lw+1, color=blue,)
        ax_orn[1].plot(t-t_on, v_orn[:, 0], linewidth=lw+1, color=black,)
        ax_orn[1].plot([t[0]-t_on, t[-1]-t_on], [vrest, vrest], '--', linewidth=lw, color=red,)
        ax_orn[1].plot([t[0]-t_on, t[-1]-t_on], [vrev, vrev], '-.', linewidth=lw, color=red,)
        ax_orn[2].plot(t_sdf-t_on, orn_sdf[:, 0], color=green, linewidth=lw+1, 
                              label='\nu')
        ax_orn[0].tick_params(axis='both', which='major', labelsize=ticks_fs)
        ax_orn2a.tick_params(axis='both', which='major', labelsize=ticks_fs)
        ax_orn4a.tick_params(axis='both', which='major', labelsize=ticks_fs)
        ax_orn[1].tick_params(axis='both', which='major', labelsize=ticks_fs)
        ax_orn[2].tick_params(axis='both', which='major', labelsize=ticks_fs)
        ax_orn[0].set_xticklabels('')
        ax_orn[1].set_xticklabels('')
        ax_orn2a.set_xticklabels('')
        ax_orn4a.set_xticklabels('')
        ax_orn2a.set_yticklabels('')        
        ax_orn4a.set_yticklabels('')   
        ax_orn[0].set_ylabel('Input (a.u.)', fontsize=label_fs)
        ax_orn[1].set_ylabel(r'V (a.u.)', fontsize=label_fs)
        ax_orn[2].set_ylabel('firing rates (Hz)', fontsize=label_fs)   
        ax_orn2a.set_ylabel(r'r (a.u.) ', fontsize=label_fs, color=blue,)
        ax_orn4a.set_ylabel(r'y adapt (a.u.)', fontsize=label_fs, color=blue,)
        ax_orn[2].set_xlabel('Time  (ms)', fontsize=label_fs) 
            # ax_orn_sc.set_ylabel('Neuron id', fontsize=label_fs)
        
        ax_orn[0].text(-.15, 1.25, panels_id[0], transform=ax_orn[0].transAxes, 
                             fontsize=panel_fs, fontweight='bold', va='top', ha='right')
        ax_orn[1].text(-.15, 1.25, panels_id[2], transform=ax_orn[1].transAxes, 
                             fontsize=panel_fs, fontweight='bold', va='top', ha='right')
        ax_orn[2].text(-.15, 1.25, panels_id[4], transform=ax_orn[2].transAxes, 
                             fontsize=panel_fs, fontweight='bold', va='top', ha='right')
        
        ax_orn[0].spines['top'].set_color('none')
        ax_orn2a.spines['top'].set_color('none')
        ax_orn4a.spines['top'].set_color('none')
        
        ax_orn[1].spines['top'].set_color('none')
        # ax_orn_sc.spines['right'].set_color('none')
        # ax_orn_sc.spines['top'].set_color('none')
        ax_orn[2].spines['right'].set_color('none')
        ax_orn[2].spines['top'].set_color('none')
        
        ll, bb, ww, hh = ax_orn[0].get_position().bounds
        ww_new = ww - 0.08
        bb_plus = 0.015
        ll_new = ll + 0.075
        hh_new = hh - 0.05
        ax_orn[0].set_position([ll_new, bb+2*bb_plus, ww_new, hh_new])
        ll, bb, ww, hh = ax_orn[1].get_position().bounds
        ax_orn[1].set_position([ll_new, bb+1.5*bb_plus, ww_new, hh])
        # ll, bb, ww, hh = ax_orn_sc.get_position().bounds
        # ax_orn_sc.set_position([ll_new, bb+bb_plus, ww_new, hh])
        ll, bb, ww, hh = ax_orn[2].get_position().bounds
        ax_orn[2].set_position([ll_new, bb-bb_plus, ww_new, hh])
        
        ax_orn[0].set_xlim((t2plot))
        ax_orn2a.set_xlim((t2plot))
        ax_orn4a.set_xlim((t2plot))
        ax_orn[1].set_xlim((t2plot))
        # ax_orn_sc.set_xlim((t2plot))
        ax_orn[2].set_xlim((t2plot))
        
        plt.show()
        
    else:
        ax_orn2a = ax_orn[0,0].twinx()
        ax_orn4a = ax_orn[1,0].twinx()
        if n_neu > 1:
            ax_orn2b = ax_orn[0,1].twinx()
            ax_orn4b = ax_orn[1,1].twinx()
            if n_neu > 2:
                ax_orn2c = ax_orn[0,2].twinx()
                ax_orn4c = ax_orn[1,2].twinx()
                if n_neu > 3:
                    ax_orn2d = ax_orn[0,3].twinx()
                    ax_orn4d = ax_orn[1,3].twinx()
        
        #weight_od = u_od*transd_mat
        for id_neu in range(n_neu):
            # PLOT
            weight_od = u_od*transd_mat[id_neu,:]
            ax_orn[0, id_neu].plot(t-t_on, weight_od, linewidth=lw+1, color=black,) 
            if id_neu == 0:
                ax_orn2a.plot(t-t_on, r_orn[:, id_neu], linewidth=lw+1, color=blue,)
                ax_orn4a.plot(t-t_on, y_orn[:, id_neu], linewidth=lw+1, color=blue,)
            elif id_neu == 1:        
                ax_orn2b.plot(t-t_on, r_orn[:, id_neu], linewidth=lw+1, color=blue,)
                ax_orn4b.plot(t-t_on, y_orn[:, id_neu], linewidth=lw+1, color=blue,)
            elif id_neu == 2:
                ax_orn2c.plot(t-t_on, r_orn[:, id_neu], linewidth=lw+1, color=blue,)
                ax_orn4c.plot(t-t_on, y_orn[:, id_neu], linewidth=lw+1, color=blue,)
            elif id_neu == 3:
                ax_orn2d.plot(t-t_on, r_orn[:, id_neu], linewidth=lw+1, color=blue,)
                ax_orn4d.plot(t-t_on, y_orn[:, id_neu], linewidth=lw+1, color=blue,)
                
            ax_orn[1, id_neu].plot(t-t_on, v_orn[:, id_neu], linewidth=lw+1, color=black,)
            ax_orn[1, id_neu].plot([t[0]-t_on, t[-1]-t_on], [vrest, vrest], '--', linewidth=lw, color=red,)
            ax_orn[1, id_neu].plot([t[0]-t_on, t[-1]-t_on], [vrev, vrev], '-.', linewidth=lw, color=red,)
            ax_orn[2, id_neu].plot(t_sdf-t_on, orn_sdf[:, id_neu], color=green, linewidth=lw+1, 
                              label='\nu')
        
            spikes_orn_0 = np.argwhere(num_spikes)        
            # ax_orn_sc.scatter(spikes_orn_0[:,0]/pts_ms-t_on, 
            #                 spikes_orn_0[:,1], color=purple, s=10)
        
            # FIGURE SETTINGS
            ax_orn[0, id_neu].tick_params(axis='both', which='major', labelsize=ticks_fs)
            if id_neu == 0:
                ax_orn2a.tick_params(axis='both', which='major', labelsize=ticks_fs)
                ax_orn4a.tick_params(axis='both', which='major', labelsize=ticks_fs)
            elif id_neu == 1:
                ax_orn2b.tick_params(axis='both', which='major', labelsize=ticks_fs)
                ax_orn4b.tick_params(axis='both', which='major', labelsize=ticks_fs)   
            elif id_neu == 2:
                ax_orn2c.tick_params(axis='both', which='major', labelsize=ticks_fs)
                ax_orn4c.tick_params(axis='both', which='major', labelsize=ticks_fs)  
            elif id_neu == 3:
                ax_orn2d.tick_params(axis='both', which='major', labelsize=ticks_fs)
                ax_orn4d.tick_params(axis='both', which='major', labelsize=ticks_fs)  
                
            ax_orn[1, id_neu].tick_params(axis='both', which='major', labelsize=ticks_fs)
            ax_orn[2, id_neu].tick_params(axis='both', which='major', labelsize=ticks_fs)
            # ax_orn_sc.tick_params(axis='both', which='major', labelsize=ticks_fs)
            
            ax_orn[0, id_neu].set_xticklabels('')
            ax_orn[1, id_neu].set_xticklabels('')
           
            if id_neu == 0:
                ax_orn2a.set_xticklabels('')
                ax_orn4a.set_xticklabels('')
                ax_orn2a.set_yticklabels('')        
                ax_orn4a.set_yticklabels('')        
            elif id_neu == 1 :
                ax_orn2b.set_xticklabels('')
                ax_orn4b.set_xticklabels('')        
                ax_orn2b.set_yticklabels('')
                ax_orn4b.set_yticklabels('')
                # ax_orn[2, id_neu].set_yticklabels('')  
            elif id_neu == 2:
                ax_orn2c.set_xticklabels('')
                ax_orn4c.set_xticklabels('')        
                ax_orn2c.set_yticklabels('')
                ax_orn4c.set_yticklabels('')
            elif id_neu == 3:
                ax_orn2d.set_xticklabels('')
                ax_orn4d.set_xticklabels('')        
                ax_orn2d.set_yticklabels('')
                ax_orn4d.set_yticklabels('')
            # ax_orn_sc.set_xticklabels('')        
            # if n_neu = 1 if n_neu = 2 if n_neu = 3 if n_neu = 4
            if id_neu == 0:
                ax_orn[0, id_neu].set_ylabel('Input (a.u.)', fontsize=label_fs)
                ax_orn[1, id_neu].set_ylabel(r'V (a.u.)', fontsize=label_fs)
                ax_orn[2, id_neu].set_ylabel('firing rates (Hz)', fontsize=label_fs)        
            elif id_neu == n_neu-1:
                if id_neu == 1:
                    ax_orn2b.set_ylabel(r'r (a.u.) ', fontsize=label_fs, color=blue,)
                    ax_orn4b.set_ylabel(r'y adapt (a.u.)', fontsize=label_fs, color=blue,)
                elif id_neu == 2:
                    ax_orn2c.set_ylabel(r'r (a.u.) ', fontsize=label_fs, color=blue,)
                    ax_orn4c.set_ylabel(r'y adapt (a.u.)', fontsize=label_fs, color=blue,)
                elif id_neu == 3:
                    ax_orn2d.set_ylabel(r'r (a.u.) ', fontsize=label_fs, color=blue,)
                    ax_orn4d.set_ylabel(r'y adapt (a.u.)', fontsize=label_fs, color=blue,)
    
                     
            ax_orn[2, id_neu].set_xlabel('Time  (ms)', fontsize=label_fs) 
            # ax_orn_sc.set_ylabel('Neuron id', fontsize=label_fs)
        
            ax_orn[0, id_neu].text(-.15, 1.25, panels_id[0+id_neu], transform=ax_orn[0, id_neu].transAxes, 
                              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            ax_orn[1, id_neu].text(-.15, 1.25, panels_id[n_neu+id_neu], transform=ax_orn[1, id_neu].transAxes, 
                              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            ax_orn[2, id_neu].text(-.15, 1.25, panels_id[(n_neu*2)+id_neu], transform=ax_orn[2, id_neu].transAxes, 
                              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            
            ax_orn[0, id_neu].spines['top'].set_color('none')
            if id_neu == 0:
                ax_orn2a.spines['top'].set_color('none')
                ax_orn4a.spines['top'].set_color('none')
            elif id_neu == 1:
                ax_orn2b.spines['top'].set_color('none')
                ax_orn4b.spines['top'].set_color('none')
            elif id_neu == 2:
                ax_orn2c.spines['top'].set_color('none')
                ax_orn4c.spines['top'].set_color('none')
            elif id_neu == 3:
                ax_orn2d.spines['top'].set_color('none')
                ax_orn4d.spines['top'].set_color('none')
            
            ax_orn[1, id_neu].spines['top'].set_color('none')
            # ax_orn_sc.spines['right'].set_color('none')
            # ax_orn_sc.spines['top'].set_color('none')
            ax_orn[2, id_neu].spines['right'].set_color('none')
            ax_orn[2, id_neu].spines['top'].set_color('none')
            
            if id_neu == 0:
                ll, bb, ww, hh = ax_orn[0, id_neu].get_position().bounds
                ww_new = ww - 0.08
                bb_plus = 0.015
                ll_new = ll + 0.075
                hh_new = hh - 0.05
                ax_orn[0, id_neu].set_position([ll_new, bb+2*bb_plus, ww_new, hh_new])
                ll, bb, ww, hh = ax_orn[1, id_neu].get_position().bounds
                ax_orn[1, id_neu].set_position([ll_new, bb+1.5*bb_plus, ww_new, hh])
                # ll, bb, ww, hh = ax_orn_sc.get_position().bounds
                # ax_orn_sc.set_position([ll_new, bb+bb_plus, ww_new, hh])
                ll, bb, ww, hh = ax_orn[2, id_neu].get_position().bounds
                ax_orn[2, id_neu].set_position([ll_new, bb-bb_plus, ww_new, hh])
            else:
                ll, bb, ww, hh = ax_orn[0, id_neu].get_position().bounds
                ww_new = ww - 0.08
                bb_plus = 0.015
                ll_new = ll + (0.075-(0.08*id_neu))
                hh_new = hh - 0.05
                ax_orn[0, id_neu].set_position([ll_new, bb+2*bb_plus, ww_new, hh_new])
                ll, bb, ww, hh = ax_orn[1, id_neu].get_position().bounds
                ax_orn[1, id_neu].set_position([ll_new, bb+1.5*bb_plus, ww_new, hh])
                # ll, bb, ww, hh = ax_orn_sc.get_position().bounds
                # ax_orn_sc.set_position([ll_new, bb+bb_plus, ww_new, hh])
                ll, bb, ww, hh = ax_orn[2, id_neu].get_position().bounds
                ax_orn[2, id_neu].set_position([ll_new, bb-bb_plus, ww_new, hh])
                
            ax_orn[0, id_neu].set_xlim((t2plot))
            if id_neu == 0:
                ax_orn2a.set_xlim((t2plot))
                ax_orn4a.set_xlim((t2plot))
            elif id_neu == 1:
                ax_orn2b.set_xlim((t2plot))
                ax_orn4b.set_xlim((t2plot))
            elif id_neu == 2:
                ax_orn2c.set_xlim((t2plot))
                ax_orn4c.set_xlim((t2plot))
            elif id_neu == 3:
                ax_orn2d.set_xlim((t2plot))
                ax_orn4d.set_xlim((t2plot))
            ax_orn[1, id_neu].set_xlim((t2plot))
            # ax_orn_sc.set_xlim((t2plot))
            ax_orn[2, id_neu].set_xlim((t2plot))
    
        plt.show()
       

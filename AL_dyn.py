#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 16:25:36 2021

This script simulate the activity of the AL. 
It receives as input the activity of the ORN layer and its structure.

AL.py
@author: mario
"""

import numpy as np
import timeit

import sdf_krofczik
import matplotlib.pyplot as plt

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
recep_clrs = ['green','purple','cyan','red']


def tictoc():
    return timeit.default_timer()

def pn2ln_v_ex(v0,t, s, pn_ln_params, ):
#    ln_params = np.array([tau_s, tau_v, alpha_pn, vrev_ln, vrest_ln, vln_noise])
    tau_v = pn_ln_params['tau_v']
    
    vrev = pn_ln_params['vrev_ln']
    vrest = pn_ln_params['vrest_ln']
    vln_noise = pn_ln_params['vln_noise']*1*(-.5+np.random.uniform(0, 1, size=np.shape(v0)))
    
    # PN -> LN equations:
    # ORN -> PN equations:
    dt = t[1]-t[0]
    b = -(1 + s)/tau_v
    a = (vrest + s*vrev + vln_noise)/tau_v
    v = (v0 + a/b)*np.exp(b*dt)-a/b
    #dvdt = ((vrest-v) + s*(vrev-v) + v_bckgnd)/tau_v
    return v

def pn2ln_s_ex(s0,t, u_pn, pn_ln_params, ):
    tau_s = pn_ln_params['tau_s']
    alpha_pn = pn_ln_params['alpha_pn']
    
    # PN -> LN equation of s:
    b = (-1-alpha_pn*u_pn)/tau_s
    a = alpha_pn*u_pn/tau_s
    dt = t[1]-t[0]
    s = (s0 + a/b)*np.exp(b*dt)-a/b
#    dsdt = (a_s*u_pn*(1-s) - s)/tau_s       
    return s

def y_ln_fun_ex(y0, t, u_ln, pn_ln_params,):
    alpha_ln = pn_ln_params['alpha_ln']
    tau_y = pn_ln_params['tau_y']
    
    b = (-alpha_ln*u_ln-1)/tau_y
    a = alpha_ln*u_ln/tau_y
    dt = t[1]-t[0]
    y = (y0 + a/b)*np.exp(b*dt)-a/b
    return y

def orn2pn_s_ex(s0,t, u_orn, x_pn,y_ln,pn_ln_params,):
    #    pn_params  = np.array([tau_s, tau_v, alpha_orn, vrev_pn, vrest_pn])
    tau_s = pn_ln_params['tau_s']
    alpha_orn = pn_ln_params['alpha_orn']
    
    # ORN -> PN equations:
    b = (-1-alpha_orn*u_orn*(1-x_pn)*(1-y_ln))/tau_s
    a = alpha_orn*u_orn*(1-x_pn)*(1-y_ln)/tau_s
    dt = t[1]-t[0]
    s = (s0 + a/b)*np.exp(b*dt)-a/b
    return s

def orn2pn_v_ex(v0,t, s, pn_ln_params,):
#    pn_params  = np.array([tau_s, tau_v, alpha_orn, vrev_pn, vrest_pn, vpn_noise])
    tau_v = pn_ln_params['tau_v']
    
    vrev    = pn_ln_params['vrev_pn']
    vrest   = pn_ln_params['vrest_pn']
    vpn_noise = pn_ln_params['vpn_noise']*(-.5+np.random.uniform(0, 1, size=np.shape(v0)))

    g_l     = pn_ln_params['g_l']
    g_s     = pn_ln_params['g_s']
    
    # ORN -> PN equations:
    dt = t[1]-t[0]
    b = -(g_l + g_s* s)/tau_v
    a = (g_l*vrest + g_s*s*vrev + vpn_noise)/tau_v
    vtmp  = np.exp(b*dt)
    
    v = (v0 + a/b)*vtmp-a/b
#    dvdt = (vrest + s*vrev + v_bckgnd)/tau_v  - v*(1 + g*s)/tau_v
    return v

def x_adapt_ex(x0,t,u_orn, pn_ln_params,):
    tau_x = pn_ln_params['tau_x']
    alpha_x = pn_ln_params['alpha_x']
    
    b = (-alpha_x*u_orn-1)/tau_x
    a = alpha_x*u_orn/tau_x
    dt = t[1]-t[0]
    x = (x0 + a/b)*np.exp(b*dt)-a/b
    return x

# ************************************************************************
# main function of the LIF ORN 
def main(params_al_orn, orn_spikes_t, verbose=False, corr_an=False):
    
    tic = tictoc()
    
    stim_params = params_al_orn['stim_params']
    # sens_params = params_al_orn['sens_params']
    # orn_params = params_al_orn['orn_params']
    sdf_params = params_al_orn['sdf_params']
    al_params = params_al_orn['al_params']
    pn_ln_params = params_al_orn['pn_ln_params']
    
    # ORN, PN and LN PARAMETERS
    t_ref               = al_params['t_ref']          # ms; refractory period 
    theta               = al_params['theta']                 # [mV] firing threshold
    n_recep_list        = al_params['n_recep_list']
    n_orns_recep        = al_params['n_orns_recep']
    n_sens_type         = al_params['n_sens_type']
    
    n_recep_tot       = sum(n_recep_list) # number of receptors in total
    # AL + ORN layer network parameters
    n_orns_pn         = n_orns_recep    # number of ORNs per each PN in each glomerulus
    n_orns_tot        = n_orns_recep*n_recep_tot  # total number of ORNs 
    
    stim_type           = stim_params['stim_type']
    pts_ms              = stim_params['pts_ms']
    t_tot               = stim_params['t_tot']
    n_od                = stim_params['n_od']
    dt_sdf              = sdf_params['dt_sdf']
    tau_sdf             = sdf_params['tau_sdf']
    
    n2sim               = int(pts_ms*t_tot) + 1    # number of time points
    sdf_size            = int(t_tot/dt_sdf)
    
    t                   = np.linspace(0, t_tot, n2sim) # time points
    
    n_pns_recep         = al_params['n_pns_recep']     # number of PNs per each glomerulus
    n_lns_recep         = al_params['n_lns_recep']     # number of LNs per each glomerulus
    n_pns_tot           = n_pns_recep*n_recep_tot # number of total PNs
    n_lns_tot           = n_lns_recep*n_recep_tot # number of total LNs    
    
    if verbose:
        # flynose verbose description 
        print('flynose Simulation ')    
        print('')
        print('In the ORNs layer there are %d type/s of sensilla' %(n_sens_type, ))
        print('and %d identical sensilla of each type' %(n_orns_recep, ))
        
        for st in range(n_sens_type):
            print('   Sensillum %d has %d ORNs of different type' %(st, n_recep_list[st]))
        print('In total, there are %d ORNs of %d different types' %(n_orns_tot, n_recep_tot))
        print('')
        
        print('In the AL there are %d glomeruli. One per each receptor type.' %n_recep_tot)
        print('Each glomerulus has %d PNs and %d LNs' %(n_pns_recep, n_lns_recep))
        print('In total, AL is compound of %d PNs and %d LNs' %(n_pns_tot, n_lns_tot))
        print('Each PNs receives input from %d random ORNs of the same type' %(n_orns_pn))
        print('')
        
        print('flynose is presented with an odour mixtures containing %d odorants' %n_od)
        print('The stimulus is a '+stim_type)
    
    
    # Each PN belongs to ONLY one of the glomeruli
    ids_recep     = np.arange(n_recep_tot)
    ids_pn_recep  = np.repeat(ids_recep, n_pns_recep)
    
    # Connectivity matrices between ORNs and PNs 
    orn_pn_mat          = np.zeros((n_orns_tot, n_pns_tot))  
    for pp in range(n_pns_tot):
        rnd_ids             = np.random.permutation(n_orns_recep) 
        tmp_ids             = rnd_ids[:n_orns_pn] + \
            n_orns_recep*ids_pn_recep[pp]
        orn_pn_mat[tmp_ids, pp] = 1
    
    # Connectivity matrices between PNs and LNs
    pn_ln_mat           = np.zeros((n_pns_tot, n_lns_tot))
    for pp in range(n_recep_tot):
        pn_ln_mat[pp*n_pns_recep:(pp+1)*n_pns_recep,
                  pp*n_lns_recep:(pp+1)*n_lns_recep] = 1 # pn_spike_height
    
    recep_id = 0        
    ln_pn_mat           = np.zeros((n_lns_tot,n_pns_tot))
    for pp in range(n_sens_type):
        num_recep = n_recep_list[pp]
        # Inhibitory LN connectivity within receptors cluster
        ln_pn_mat[(recep_id*n_lns_recep):((recep_id+num_recep)*n_lns_recep),
                  (recep_id*n_pns_recep):((recep_id+num_recep)*n_pns_recep)] = 1#ln_spike_height
        for qq in range(num_recep):
            # PN innervating LN are not inhibited
            ln_pn_mat[((recep_id+qq)*n_lns_recep):((recep_id+qq+1)*n_lns_recep),
                      ((recep_id+qq)*n_pns_recep):((recep_id+qq+1)*n_pns_recep)] = 0
        recep_id = recep_id + num_recep
    
    # Generate input to PNs
    u_orn = orn_spikes_t.dot(orn_pn_mat) 
      
    # PN and LN PARAMETERS and OUTPUT VECTORS
    x_pn0               = 0#.0048 # .48
    s_pn0               = 0#.02   #.2
    v_pn0               = 0#.05   # .5
    
    y_ln0               = 0#.0025
    s_ln0               = 0#.02
    v_ln0               = 0#.05
    
    # Initialize LN to PN output vectors
    x_pn            = np.zeros((n2sim, n_pns_tot))
    u_pn            = np.zeros((n2sim, n_lns_tot))
    s_pn            = np.zeros((n2sim, n_pns_tot))
    v_pn            = np.ones((n2sim, n_pns_tot))*pn_ln_params['vrest_pn']
    
    u_ln            = np.zeros((n2sim, n_pns_tot))
    y_ln            = np.zeros((n2sim, n_pns_tot))
    
    # Initialize PN output vectors
    num_spike_pn    = np.zeros((n2sim, n_pns_tot))
    
    pn_ref_cnt      = np.zeros(n_pns_tot) # Refractory period counter starts from 0
    
    # Initialize LN output vectors
    s_ln            = np.zeros((n2sim, n_lns_tot))
    v_ln            = np.ones((n2sim, n_lns_tot))*pn_ln_params['vrest_ln']
    num_spike_ln    = np.zeros((n2sim, n_lns_tot))  
    
    # PN and LN params initial conditions
    s_pn[0, :]      = s_pn0*(1 + np.random.standard_normal((1, n_pns_tot)))
    x_pn[0, :]      = x_pn0*(1 + np.random.standard_normal((1, n_pns_tot)))
    v_pn[0,:]       = v_pn0*np.ones((1, n_pns_tot)) \
        + np.random.standard_normal((1, n_pns_tot)) 
    
    s_ln[0, :]      = s_ln0*(1 + np.random.standard_normal((1, n_lns_tot)))
    y_ln[0, :]      = y_ln0*(1 + np.random.standard_normal((1, n_pns_tot)))
    v_ln[0,:]       = v_ln0*np.ones((1, n_lns_tot)) \
        + np.random.standard_normal((1, n_lns_tot)) 
    
    ln_ref_cnt      = np.zeros(n_lns_tot) # initially the ref period cnter is equal to 0
    
    
    # solve ODE for PN and LN
    for tt in range(1, n2sim-t_ref-1):
        # span for next time step
        tspan = [t[tt-1],t[tt]]
        
        pp_rnd = np.arange(n_pns_tot) # np.random.permutation(n_pns_tot)
        
        # Adaptation variable of PN neuron
        x_pn[tt, pp_rnd] = x_adapt_ex(x_pn[tt-1,pp_rnd],tspan, 
                u_orn[tt, pp_rnd], pn_ln_params, )        
    
        # Inhibitory input to PNs
        y_ln[tt, pp_rnd] = y_ln_fun_ex(y_ln[tt-1, pp_rnd],tspan, 
                u_ln[tt-1, pp_rnd], pn_ln_params, )
        
        # ORN -> PN synapses
        
        # PNs whose ref_cnt is equal to zero:
        pn_ref_0 = pn_ref_cnt==0
        s_pn[tt, pn_ref_0] = orn2pn_s_ex(s_pn[tt-1, pn_ref_0],tspan, 
            u_orn[tt, pn_ref_0], x_pn[tt-1, pn_ref_0], y_ln[tt-1, pn_ref_0], pn_ln_params, )
        v_pn[tt, pn_ref_0] = orn2pn_v_ex(v_pn[tt-1, pn_ref_0],tspan, 
                s_pn[tt-1, pn_ref_0], pn_ln_params, )
        
        # PNs whose ref_cnt is different from zero:
        pn_ref_no0 = pn_ref_cnt!=0
        # Refractory period count down
        pn_ref_cnt[pn_ref_no0] = pn_ref_cnt[pn_ref_no0] - 1  
        
        # PNs whose Voltage is above threshold AND whose ref_cnt is equal to zero:
        pn_above_thr = (v_pn[tt, :] >= theta) & (pn_ref_cnt==0)
        num_spike_pn[tt, pn_above_thr] = num_spike_pn[tt, pn_above_thr] + 1
        u_pn[tt, :] += np.sum(pn_ln_mat[pn_above_thr,:], axis=0)
        pn_ref_cnt[pn_above_thr] = t_ref
        
        # PN -> LN synapses        
            
        # LNs whose ref_cnt is equal to zero:
        ln_ref_0 = ln_ref_cnt==0
        s_ln[tt, ln_ref_0] = pn2ln_s_ex(s_ln[tt-1, ln_ref_0], tspan, 
                    u_pn[tt, ln_ref_0], pn_ln_params, )
        v_ln[tt, ln_ref_0] = pn2ln_v_ex(v_ln[tt-1, ln_ref_0], tspan, 
                    s_ln[tt-1, ln_ref_0], pn_ln_params, )
        
        # LNs whose ref_cnt is different from zero:
        ln_ref_no0 = ln_ref_cnt!=0
        # Refractory period count down
        ln_ref_cnt[ln_ref_no0] = ln_ref_cnt[ln_ref_no0] - 1  
        
        # LNs whose Voltage is above threshold AND whose ref_cnt is equal to zero:
        ln_above_thr = (v_ln[tt, :] >= theta) & (ln_ref_cnt==0)
        num_spike_ln[tt, ln_above_thr] = num_spike_ln[tt, ln_above_thr] + 1
        u_ln[tt, :] += np.sum(ln_pn_mat[ln_above_thr,:], axis=0)
        ln_ref_cnt[ln_above_thr] = t_ref
        
    # Calculate the spike matrix of PNs and LNs
    pn_spike_matrix = np.asarray(np.where(num_spike_pn))
    pn_spike_matrix[0,:] = pn_spike_matrix[0,:]/pts_ms
    pn_spike_matrix = np.transpose(pn_spike_matrix)
    
    ln_spike_matrix = np.asarray(np.where(num_spike_ln))
    ln_spike_matrix[0,:] = ln_spike_matrix[0,:]/pts_ms
    ln_spike_matrix = np.transpose(ln_spike_matrix)
    toc = tictoc()

        
    # Calculate the SDF for PNs and LNs
    pn_sdf_time = np.linspace(0, dt_sdf*sdf_size, sdf_size)
    pn_sdf = np.zeros((sdf_size, n_pns_tot))
    
    if ~(np.sum(pn_spike_matrix) == 0):
        pn_sdf_tmp, pn_sdf_time = sdf_krofczik.main(pn_spike_matrix, sdf_size,
                                                 tau_sdf, dt_sdf)  # (Hz, ms)
        for nn in range(np.size(pn_sdf_tmp,1)):
            pn_sdf[:, nn] = pn_sdf_tmp[:, nn]*1e3 

    ln_sdf_time = np.linspace(0, dt_sdf*sdf_size, sdf_size)
    ln_sdf = np.zeros((sdf_size, n_lns_tot))
    
    if ~(np.sum(ln_spike_matrix) == 0):
        ln_sdf_tmp, ln_sdf_time = sdf_krofczik.main(ln_spike_matrix, sdf_size,
                                                 tau_sdf, dt_sdf)  # (Hz, ms)
        for nn in range(np.size(ln_sdf_tmp,1)):
            ln_sdf[:, nn] = ln_sdf_tmp[:, nn]*1e3 
    
    if verbose:
        print('AL sim time: %.2f s' %(toc-tic,))
    
    al_out = [t, pn_spike_matrix, pn_sdf, pn_sdf_time,
              ln_spike_matrix, ln_sdf, ln_sdf_time,]
    
    
    #%%  AL correlation analysis
    if corr_an:   
        tic = tictoc()
        corr_pn = np.zeros((n_pns_tot, n_pns_tot))
        corr_vpn = np.zeros((n_pns_tot, n_pns_tot))
        for nn1 in range(n_pns_tot):
            for nn2 in range(n_pns_tot):
                if nn2>nn1:
                    pip1 = np.zeros(t_tot)
                    pip2 = np.zeros(t_tot)
                    pip1[pn_spike_matrix[pn_spike_matrix[:,1] == nn1, 0]] = 1
                    pip2[pn_spike_matrix[pn_spike_matrix[:,1] == nn2, 0]] = 1
        
                    corr_pn[nn1, nn2] = np.corrcoef((pip1,pip2))[0,1]
                    corr_pn[nn2, nn1] = corr_pn[nn1, nn2]
                    
                    pip1 = v_pn[::pts_ms, nn1]
                    pip2 = v_pn[::pts_ms, nn2]
                    corr_vpn[nn1, nn2] = np.corrcoef((pip1, pip2))[0,1]
                    corr_vpn[nn2, nn1] = corr_vpn[nn1, nn2]
                    
        tmp_corr = corr_vpn[:n_pns_recep, :n_pns_recep]
        tmp_corr[tmp_corr!=0]
        corr_pn_hom = np.mean(tmp_corr[tmp_corr!=0])
        corr_pn_het = np.mean(corr_vpn[:n_pns_recep, n_pns_recep:]) # corr_pn[0,-1]
        print('PNs, Hom and Het Potent corr: %.3f and %.3f' 
              %(corr_pn_hom, corr_pn_het))
        
        tmp_corr = corr_pn[:n_pns_recep, :n_pns_recep]
        tmp_corr[tmp_corr!=0]
        corr_pn_hom = np.mean(tmp_corr[tmp_corr!=0])
        corr_pn_het = np.mean(corr_pn[:n_pns_recep, n_pns_recep:]) # corr_pn[0,-1]
        print('PNs, Hom and Het spk cnt corr: %.3f and %.3f' 
              %(corr_pn_hom, corr_pn_het))
        
        print('')
        # LNs correlation analysis
        corr_ln = np.zeros((n_lns_tot, n_lns_tot))
        corr_vln = np.zeros((n_lns_tot, n_lns_tot))
        for nn1 in range(n_lns_tot):
            for nn2 in range(n_lns_tot):
                if nn2>nn1:
                    pip1 = np.zeros(t_tot)
                    pip2 = np.zeros(t_tot)
                    pip1[pn_spike_matrix[pn_spike_matrix[:,1] == nn1, 0]] = 1
                    pip2[pn_spike_matrix[pn_spike_matrix[:,1] == nn2, 0]] = 1
        
                    corr_ln[nn1, nn2] = np.corrcoef((pip1,pip2))[0,1]
                    corr_ln[nn2, nn1] = corr_ln[nn1, nn2]
                    
                    pip1 = v_ln[::pts_ms, nn1]
                    pip2 = v_ln[::pts_ms, nn2]
                    corr_vln[nn1, nn2] = np.corrcoef((pip1, pip2))[0,1]
                    corr_vln[nn2, nn1] = corr_vln[nn1, nn2]
                    
        tmp_corr = corr_vln[:n_lns_recep, :n_lns_recep]
        tmp_corr[tmp_corr!=0]
        corr_ln_hom = np.mean(tmp_corr[tmp_corr!=0])
        corr_ln_het = np.mean(corr_vln[:n_lns_recep, n_lns_recep:])
        print('LNs, Hom and Het Potent corr: %.3f and %.3f' 
              %(corr_ln_hom, corr_ln_het))
        
        tmp_corr = corr_ln[:n_lns_recep, :n_lns_recep]
        tmp_corr[tmp_corr!=0]
        corr_ln_hom = np.mean(tmp_corr[tmp_corr!=0])
        corr_ln_het = np.mean(corr_ln[:n_lns_recep, n_lns_recep:]) 
        print('LNs, Hom and Het spk cnt corr: %.3f and %.3f' 
              %(corr_ln_hom, corr_ln_het))
        print('')
        toc = tictoc()
        print('time to corr analysis: %.2f s' %(toc-tic))
    return  al_out 
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 16:25:36 2021

This script simulate the activity of the AL. 
It receives as input the activity of the ORN layer and its structure.

AL_dyn.py
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



def ln2pn_y(y0, t, spike_ln, pn_ln_params, ):
    # LN to ORN-PN synapse synaptic activation y_j (pre-synapse), j=0...nLN-1
    # spike_ln is an array of length nLN containing 1 for LN spiking and 0 otherwise
    alpha_ln = pn_ln_params['alpha_ln']
    tau_ln = pn_ln_params['tau_ln']
    
    dt = t[1]-t[0]
    y = y0*np.exp(-dt/tau_ln) + alpha_ln*spike_ln*(1-y0)
       
    return y



''' TN version'''

def orn2pn_s(s0, t, spike_orn, pn_ln_params, ):
    # ORN to PN synaptic activation s_j (pre-synapse) 
    # spike_orn is an array of length nORN containing 1 for ORN spiking and 0 otherwise
    tau_orn = pn_ln_params['tau_orn']
    alpha_orn = pn_ln_params['alpha_orn']

    # per PN inhibition 
    dt = t[1]-t[0]
    s = s0*np.exp(-dt/tau_orn) + alpha_orn*spike_orn*(1-s0)
            
    return s



def x_adapt(x0, t, spike_pn, pn_ln_params,):
    # PN spike adaptation variable, driven by PN spiking
    # spike_pn is an array of length nPN containing 1 for ORN spiking and 0 otherwise
    tau_ad = pn_ln_params['tau_ad']
    alpha_ad = pn_ln_params['alpha_ad']
    
    dt = t[1]-t[0]
    x = x0*np.exp(-dt/tau_ad) + alpha_ad*spike_pn*(1-x0)
    return x




def pn_v(v0, t, s_ornpn, y_lnpn, x_pn, pn_ln_params,):
    # PN potential equations:
    c_pn_ln = pn_ln_params['c_pn_ln']
    
    vrev_ex     = pn_ln_params['vrev_ex']
    vrev_inh    = pn_ln_params['vrev_inh']
    vrest       = pn_ln_params['vrest_pn']
    vpn_noise   = pn_ln_params['vpn_noise']*np.random.standard_normal(size=np.shape(v0))

    g_l         = pn_ln_params['g_l_pn']
    g_orn       = pn_ln_params['g_orn']
    g_ln        = pn_ln_params['g_ln']
    g_adapt     = pn_ln_params['g_ad']
    
    dt = t[1]-t[0]
    # b = -(g_l + g_orn*s_ornpn + g_ln*y_lnpn )/c_pn_ln
    b = -(g_l + g_orn*s_ornpn + g_ln*y_lnpn + g_adapt*x_pn)/c_pn_ln
    a = (g_l*vrest + g_orn*s_ornpn*vrev_ex + g_ln*y_lnpn*vrev_inh + g_adapt*x_pn*vrev_inh)/c_pn_ln
    v = (v0 + a/b)*np.exp(b*dt)-a/b + vpn_noise*np.sqrt(dt)
    return v


def pn2ln_s(s0, t, spike_pn, pn_ln_params, ):
    # PN to LN synaptic activation s_j (pre-synapse), j=0...nPN-1
    # spike_pn is an array of length nPN containing 1 for PN spiking and 0 otherwise
    tau_pn = pn_ln_params['tau_pn']
    alpha_pn = pn_ln_params['alpha_pn']
    
    # equation of presynatpic activation s:
    dt = t[1]-t[0]
    s = s0*np.exp(-dt/tau_pn) + alpha_pn*spike_pn*(1-s0)
    
    return s

   

def pn2ln_v_ex(v0, t, s_pnln, pn_ln_params, ):
    # LN potential equations:
    c_pn_ln = pn_ln_params['c_pn_ln']
    
    vrev_ex     = pn_ln_params['vrev_ex']
    vrest       = pn_ln_params['vrest_ln']
    vln_noise   = pn_ln_params['vln_noise']*(np.random.standard_normal(size=np.shape(v0)))

    g_l         = pn_ln_params['g_l_ln']
    g_pn        = pn_ln_params['g_pn']
    
    dt = t[1]-t[0]
    b = -(g_l + g_pn*s_pnln)/c_pn_ln
    a = (g_l*vrest + g_pn*s_pnln*vrev_ex )/c_pn_ln 
    v = (v0 + a/b)*np.exp(b*dt)-a/b + vln_noise*np.sqrt(dt)
    return v



# ************************************************************************
# main function of the LIF ORN 
def main(params_al_orn, spike_orn, verbose=False, corr_an=False):
    
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
    
      
    # PN and LN PARAMETERS and OUTPUT VECTORS
    x_pn0               = 0
    s_pn0               = 0
    v_pn0               = 0
    
    y_ln0               = 0
    v_ln0               = 0

    s_orn           = np.zeros((n2sim, n_orns_tot))
    s_ornpn        = np.zeros((n2sim, n_pns_tot))
    
    # Initialize LN to PN output vectors
    x_pn            = np.zeros((n2sim, n_pns_tot))
    s_pn            = np.zeros((n2sim, n_pns_tot))
    s_pnln          = np.zeros((n2sim, n_lns_tot))
    v_pn            = np.ones((n2sim, n_pns_tot))*pn_ln_params['vrest_pn']
    
    y_ln            = np.zeros((n2sim, n_lns_tot))
    y_lnpn          = np.zeros((n2sim, n_pns_tot))

    # Initialize PN output vectors
    spike_pn    = np.zeros((n2sim, n_pns_tot))
    pn_ref_cnt      = np.zeros(n_pns_tot) # Refractory period counter starts from 0
    
    # Initialize LN output vectors
    v_ln            = np.ones((n2sim, n_lns_tot))*pn_ln_params['vrest_ln']
    spike_ln    = np.zeros((n2sim, n_lns_tot))  
    
    # PN and LN params initial conditions
    s_pn[0, :]      = s_pn0*(1 + np.random.standard_normal((1, n_pns_tot)))
    x_pn[0, :]      = x_pn0*(1 + np.random.standard_normal((1, n_pns_tot)))
    v_pn[0,:]       = v_pn0*np.ones((1, n_pns_tot)) \
        + np.random.standard_normal((1, n_pns_tot)) 
    
    y_ln[0, :]      = y_ln0*(1 + np.random.standard_normal((1, n_lns_tot)))
    v_ln[0,:]       = v_ln0*np.ones((1, n_lns_tot)) \
        + np.random.standard_normal((1, n_lns_tot)) 
    
    ln_ref_cnt      = np.zeros(n_lns_tot) # initially the ref period cnter is equal to 0
    
    
    # solve ODE for PN and LN
    for tt in range(1, n2sim-t_ref-1):
        # span for next time step
        tspan = [t[tt-1],t[tt]]
        
        #pp_rnd = np.arange(n_pns_tot) # np.random.permutation(n_pns_tot)
        
        # ORN input to PNs
        s_orn[tt, :] = orn2pn_s(s_orn[tt-1, :],tspan, spike_orn[tt, :], pn_ln_params, )
        # summing inputs:
        s_ornpn[tt, :] = s_orn[tt, :].dot(orn_pn_mat) 
        # PN dynamics:
        # PNs whose ref_cnt is equal to zero:
        pn_ref_0 = pn_ref_cnt==0
        v_pn[tt, pn_ref_0] = pn_v(v_pn[tt-1, pn_ref_0],tspan, 
                                  s_ornpn[tt-1, pn_ref_0], y_lnpn[tt-1, pn_ref_0], x_pn[tt-1, pn_ref_0], pn_ln_params, )
        # handle spiking:
        # PNs whose ref_cnt is different from zero:
        pn_ref_no0 = pn_ref_cnt!=0
        # Refractory period count down
        pn_ref_cnt[pn_ref_no0] = pn_ref_cnt[pn_ref_no0] - 1  
        
        # PNs whose Voltage is above threshold AND whose ref_cnt is equal to zero:
        pn_above_thr = (v_pn[tt, :] >= theta) & (pn_ref_cnt==0)
        spike_pn[tt, pn_above_thr] = 1
        pn_ref_cnt[pn_above_thr] = t_ref

        # Adaptation variable of PN neuron
        x_pn[tt, :] = x_adapt(x_pn[tt-1, :],tspan, spike_pn[tt, :], pn_ln_params, )        

        # PN to LN synapses activation
        s_pn[tt, :] = pn2ln_s(s_pn[tt-1, :], tspan, spike_pn[tt, :], pn_ln_params, )        
        # summing inputs:
        s_pnln[tt, :] = s_pn[tt, :].dot(pn_ln_mat)
        
        # LN dynamics:
        # LNs whose ref_cnt is equal to zero:
        ln_ref_0 = ln_ref_cnt==0
        v_ln[tt, ln_ref_0] = pn2ln_v_ex(v_ln[tt-1, ln_ref_0], tspan, 
                    s_pnln[tt-1, ln_ref_0], pn_ln_params, )
        
        # handle spiking:
        # LNs whose ref_cnt is different from zero:
        ln_ref_no0 = ln_ref_cnt!=0
        # Refractory period count down
        ln_ref_cnt[ln_ref_no0] = ln_ref_cnt[ln_ref_no0] - 1  
        
        # LNs whose Voltage is above threshold AND whose ref_cnt is equal to zero:
        ln_above_thr = (v_ln[tt, :] >= theta) & (ln_ref_cnt==0)
        spike_ln[tt, ln_above_thr] = 1
        ln_ref_cnt[ln_above_thr] = t_ref

        # Inhibitory LN input to PNs
        y_ln[tt, :] = ln2pn_y(y_ln[tt-1, :],tspan, spike_ln[tt, :], pn_ln_params, )
        # summing inputs:
        y_lnpn[tt, :] = y_ln[tt, :].dot(ln_pn_mat)
           

    # Calculate the spike matrix of PNs and LNs
    pn_spike_matrix = np.asarray(np.where(spike_pn))
    pn_spike_matrix[0,:] = pn_spike_matrix[0,:]/pts_ms
    pn_spike_matrix = np.transpose(pn_spike_matrix)
    
    ln_spike_matrix = np.asarray(np.where(spike_ln))
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
    

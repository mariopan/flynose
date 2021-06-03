#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 10:29:39 2021

plot_hist_isi.py

@author: mario
"""

import numpy as np
import matplotlib.pyplot as plt


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
recep_clrs = ['green','purple','cyan','red']



#%% FIGURE, time course and histogram of ISI and POTENTIAL of ORNs
def main (params_1sens, orn_lif_out):
    
    [t, u_od, r_orn, v_orn, y_orn, num_spikes, spike_matrix, orn_sdf, t_sdf,]  = orn_lif_out
    
    stim_params     = params_1sens['stim_params']
    orn_params      = params_1sens['orn_params']
    sens_params     = params_1sens['sens_params']
    
    # sdf_params          = params_al_orn['sdf_params']
    # al_params           = params_al_orn['al_params']
    # pn_ln_params        = params_al_orn['pn_ln_params']
    
    
    t_on    = np.min(stim_params['t_on'])
    # stim_dur = stim_params['stim_dur'][0]
    # t_tot   = stim_params['t_tot']
    pts_ms  = stim_params['pts_ms']
    vrest   = orn_params['vrest']
    # vrev    = orn_params['vrev']
    n_neu   = sens_params['n_neu']
    
    # SENSILLUM PARAMETERS
    n_neu           = sens_params['n_neu']
    n_orns_recep    = sens_params['n_orns_recep']
    
    n_neu_tot       = n_neu*n_orns_recep
    n_isi = np.zeros((n_neu_tot,))
    rs = 2
    cs = 2
    
    fig, axs = plt.subplots(rs, cs, figsize=(7,7))    
    
    for nn1 in range(n_neu):
        isi = []
        for nn2 in range(n_orns_recep):
            nn = nn2+n_orns_recep*nn1     
            min_isi = 10
            spks_tmp = spike_matrix[spike_matrix[:,1]==nn][:,0]
            spks_tmp = spks_tmp[spks_tmp>10]
            if stim_params['stim_type'] != 'rs':
                spks_tmp = spks_tmp[spks_tmp<t_on]
            n_isi[nn] = len(spks_tmp)-1
            isi = np.append(isi, np.diff(spks_tmp))
            if np.shape(isi)[0]>0:
                min_isi = np.min((np.min(isi), min_isi))
                
            axs[0,0].plot(np.diff(spks_tmp), '.-', color=recep_clrs[nn1], alpha=.25)
        
        if len(isi)>3:
            axs[0, 1].hist(isi, bins=int(len(isi)/3), color=recep_clrs[nn1], alpha=.25, 
                    orientation='horizontal')
    
    fr_mean_rs = 1000/np.mean(isi)
    print('ORNs, FR avg no stimulus: %.2f Hz' %fr_mean_rs)
    
    fr_peak = np.max(np.mean(orn_sdf[:, :n_orns_recep], axis=1)) 
    print('ORNs, FR peak: %.2f Hz' %fr_peak)
    
    # Comparison with Poissonian hypothesis
    # t_tmp = np.linspace(0, np.max(isi),100)
    # isi_pois = fr_mean_rs*np.exp(-fr_mean_rs*t_tmp*1e-3) # poisson    
    # axs[1].plot(isi_pois, t_tmp, 'k.-')
    # SETTINGS
    axs[0, 0].set_xlabel('id spikes', fontsize=label_fs)
    axs[0, 0].set_ylabel('ISI spikes (ms)', fontsize=label_fs)
    
    dbb = 1.5
    ll, bb, ww, hh = axs[0,0].get_position().bounds
    axs[0,0].set_position([ll, bb, ww*dbb , hh])
    
    ll, bb, ww, hh = axs[0,1].get_position().bounds
    axs[0, 1].set_position(
        [ll+(dbb - 1)*ww, bb, ww*(2-dbb), hh])
    
    # V ORNs    
    X0 = t-t_on
    trsp = .3
    if n_neu == 1:
        X1 = v_orn
        axs[1, 0].plot([t[0]-t_on, t[-1]-t_on], [vrest, vrest], 
              '--', linewidth=lw, color=black,)
        mu1 = X1.mean(axis=1)
        # sigma1 = X1.std(axis=1)
        
        axs[1, 0].plot(X0, mu1, linewidth=lw+1, 
                color=recep_clrs[0], )
        for nn in range(n_orns_recep):
            axs[1, 0].plot(X0, X1[:, nn], '.', linewidth= lw-1, 
                color=recep_clrs[0], alpha=trsp)
            
        axs[1, 1].hist(X1[(t_on*pts_ms):(t_on+250)*pts_ms, nn], 
            bins=50, color=recep_clrs[0], alpha=.25, 
                    orientation='horizontal')
    
    
    else:
        for id_neu in range(n_neu):
            X1 = v_orn[:, id_neu*n_orns_recep:((id_neu+1)*n_orns_recep)]
            axs[1, 0].plot([t[0]-t_on, t[-1]-t_on], [vrest, vrest], 
                          '--', linewidth=lw, color=red,)
            mu1 = X1.mean(axis=1)
            # sigma1 = X1.std(axis=1)
            
            # axs[1, 0].fill_between(X0, mu1+sigma1, mu1-sigma1, 
                        # facecolor=recep_clrs[id_neu], alpha=trsp)
            
            axs[1, 0].plot(X0, mu1,  
                linewidth=lw+1, color=recep_clrs[id_neu],)
            
            for nn in range(n_orns_recep):
                axs[1, 0].plot(X0, X1[:, nn], '.', linewidth= lw-1, 
                    color=recep_clrs[id_neu], alpha=trsp)
            
            axs[1, 1].hist(X1[(t_on*pts_ms):(t_on+250)*pts_ms, nn], bins=50, 
                    alpha=.25, color=recep_clrs[id_neu], 
                    orientation='horizontal')
    
    axs[1, 0].set_xlabel('time (ms)', fontsize=label_fs)
    axs[1, 0].set_ylabel('V (mV)', fontsize=label_fs)
    axs[1, 1].set_ylabel('pdf', fontsize=label_fs)
    
    dbb = 1.5
    ll, bb, ww, hh = axs[1,0].get_position().bounds
    axs[1, 0].set_position([ll, bb, ww*dbb , hh])
    
    ll, bb, ww, hh = axs[1,1].get_position().bounds
    axs[1, 1].set_position(
        [ll+(dbb - 1)*ww, bb, ww*(2-dbb), hh])
    
                        
    
    return fig, axs

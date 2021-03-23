#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:00:32 2021

The dynamics of all the sensilla in the ORNs layer 

Simulation of all the sensilla in the ORNs layer. Essentially a loop over 
NSI_ORN_LIF.py

ORNs_layer_dyn.py

@author: mario
"""
import numpy as np
import timeit

import NSI_ORN_LIF

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

# %% ORNS SIMULATION
def main(params_all_sens, verbose=False, corr_an=False):
    stim_params = params_all_sens['stim_params']
    orn_layer_params = params_all_sens['orn_layer_params']
    orn_params = params_all_sens['orn_params']
    sdf_params = params_all_sens['sdf_params']

    n_sens_type       = orn_layer_params.__len__()  # number of type of sensilla
    n_recep_list      = np.zeros(n_sens_type, dtype=int)
    for st in range(n_sens_type):
        n_recep_list[st]      = orn_layer_params[st]['n_neu'] #[n_neu, n_neu]    # number of ORNs per sensilla

    tic = tictoc()

    pts_ms              = stim_params['pts_ms']
    t_tot               = stim_params['t_tot']
    dt_sdf              = sdf_params['dt_sdf']
    
    n2sim               = int(pts_ms*t_tot) + 1    # number of time points
    sdf_size            = int(t_tot/dt_sdf)
    
    n_orns_recep        = orn_layer_params[0]['n_orns_recep']
    n_recep_tot         = sum(n_recep_list) # number of receptors in total
    n_orns_tot          = n_orns_recep*n_recep_tot  # total number of ORNs 
    
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
    
    spike_orn = np.zeros((n2sim, n_orns_tot))
    orn_sdf = np.zeros((sdf_size, n_orns_tot))
    
    v_orn_t = np.zeros((n2sim, n_orns_tot))
    
    id_orn0 = 0 
    
    for id_sens, n_neu in enumerate(n_recep_list):
        params_1sens   = dict([
                    ('stim_params', stim_params),
                    ('sens_params', orn_layer_params[id_sens]),
                    ('orn_params', orn_params),
                    ('sdf_params', sdf_params),
                    ])
        #####################################################################
        orn_lif_out   = NSI_ORN_LIF.main(params_1sens, verbose=verbose)
        [t, u_od, r_orn, v_orn, y_orn, 
         n_spikes_orn_tmp, spike_matrix, orn_sdf_tmp, orn_sdf_time] = orn_lif_out 
        #####################################################################
        
        
        ids_orn = np.arange(n_neu*n_orns_recep) + id_orn0 
        
        spike_orn[:, ids_orn] = n_spikes_orn_tmp
        orn_sdf[:, ids_orn] = orn_sdf_tmp
        v_orn_t[:, ids_orn] = v_orn
        id_orn0 = ids_orn[-1]+1
        
    if corr_an:
        # ORN correlation analysis
        corr_orn = np.zeros((n_orns_tot, n_orns_tot))
        corr_vorn = np.zeros((n_orns_tot, n_orns_tot))
        for nn1 in range(n_orns_tot):
            for nn2 in range(n_orns_tot):
                if nn2>nn1:
                    pip1 = v_orn_t[::5, nn1]
                    pip2 = v_orn_t[::5, nn2]
                    corr_vorn[nn1, nn2] = np.corrcoef((pip1,pip2))[0,1]
                    corr_vorn[nn2, nn1] = corr_vorn[nn1, nn2]
                    
                    pip1 = np.zeros(t_tot)
                    pip2 = np.zeros(t_tot)
                    pip1[spike_matrix[spike_matrix[:,1] == nn1, 0]] = 1
                    pip2[spike_matrix[spike_matrix[:,1] == nn2, 0]] = 1
                    corr_orn[nn1, nn2] = np.corrcoef((pip1,pip2))[0,1]
                    corr_orn[nn2, nn1] = corr_orn[nn1, nn2]
                    
        tmp_corr = corr_vorn[:n_orns_recep, :n_orns_recep]
        tmp_corr[tmp_corr!=0]
        corr_orn_hom = np.mean(tmp_corr[tmp_corr!=0])
        corr_orn_het = np.mean(corr_vorn[:n_orns_recep, n_orns_recep:]) # corr_pn[0,-1]
        print('ORNs, Hom and Het Potent corr: %.3f and %.3f' 
              %(corr_orn_hom, corr_orn_het))
        
        tmp_corr = corr_orn[:n_orns_recep, :n_orns_recep]
        tmp_corr[tmp_corr!=0]
        corr_orn_hom = np.mean(tmp_corr[tmp_corr!=0])
        corr_orn_het = np.mean(corr_orn[:n_orns_recep, n_orns_recep:]) # corr_pn[0,-1]
        print('ORNs, Hom and Het spk cnt corr: %.3f and %.3f' 
              %(corr_orn_hom, corr_orn_het))
    
    
    # orn_avg = np.mean(orn_sdf)
    # print('ORNs, FR avg: %.2f Hz' %orn_avg)
    # print('')
    
    
    toc = tictoc()-tic
    if verbose:
        print('ORNs layer sim time: %.2f s' %(toc,))
        
    #collect output variables in 'out_orns_layer'
    out_orns_layer = [t, u_od,  spike_orn, orn_sdf,orn_sdf_time]
    return out_orns_layer

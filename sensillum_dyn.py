#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:04:01 2020

It simulates single sensensillum dynamics with LIF interacting neurons.

sensillum_dyn.py

Input:
    params_1sens:   parameters of the simulation of a Single sensillum
    verbose:        if True print the most important features of the simulation
    u_od:           Output of the plume generation

Output:
        

@author: mario
"""
import numpy as np

import sdf_krofczik
import sensillum_tools
     
        
def main(params_1sens, u_od, verbose=False):
    """ Dynamics of the single type sensillum """
    
    t_part      = 20000      # [ms] repetition time window 
    
    
    ######################################################################
    #       RECEPTORS DYNAMICS    
    r_orn_od = sensillum_tools.transd_1sens(params_1sens, u_od, t_part)
    
    # add noise separately per each sensillum
    r_orn = sensillum_tools.noise_in_transd(params_1sens, r_orn_od, t_part)
    
    r_orn += .12        # if we want to start from lower values than 1e-4
    ######################################################################
    #       LIF ORN DYNAMICS
    stim_params = params_1sens['stim_params']    
    
    # Stimuli params
    tmp_ks = ['pts_ms', 't_tot', ]    
    [pts_ms, t_tot,] = [stim_params[x] for x in tmp_ks]    
    
    if t_tot >= t_part:
        n_rep       = int(np.ceil(t_tot / t_part))
        extra_time  = int(t_tot% t_part)
    else:
        n_rep       = 1
        extra_time  = t_tot
        t_part      = t_tot
    
    
    # SENSILLUM PARAMETERS
    sens_params     = params_1sens['sens_params']
    n_neu           = sens_params['n_neu']
    n_orns_recep    = sens_params['n_orns_recep']
    
    n_neu_tot       = n_neu*n_orns_recep
    
    # Output params 
    v_orn_last      = []
    y_orn_last      = []        
    spike_matrix    = np.zeros((2,0), dtype=int)
    
    n2sim           = int(pts_ms*t_part)         # number of time points
    t_rep           = np.linspace(0, t_part, n2sim) # time points
                
    # Instantiate ORNs of the sensillum
    sensillum_orns = sensillum_tools.SensillumORNS(params_1sens, n2sim)
    
    tt_rep          = 0
    for id_rep in range(n_rep):
        
        if (extra_time>0) &  (id_rep == (n_rep-1)):
            n2sim = int(pts_ms*extra_time)     # number of time points
        
        r_orn_rep = r_orn[tt_rep:(tt_rep+n2sim), :] #-t_ref
        sim_duration = range(1, n2sim)        #-t_ref
        
        if id_rep > 0:
            # starting values are the last of the previous iteration
            sensillum_orns.update_t0values(y_orn_last, v_orn_last)
    
        
        for tt in sim_duration: 
            sensillum_orns.run_1step(r_orn_rep, tt, t_rep)
            
        
        [v_orn_last,y_orn_last] = sensillum_orns.last_values(tt)
                
        # Calculate the spike matrix from orns_sens.spikes_orn
        tmp_sp_mat        = np.asarray(np.where(sensillum_orns.spikes_orn[:tt-1, :]))
        tmp_sp_mat[0, :] += tt_rep
        spike_matrix = np.concatenate((spike_matrix, tmp_sp_mat),axis=1)
        tt_rep  += tt
        

    # save variables for the whole simulation duration:
    v_orn = sensillum_orns.v_orn
    y_orn = sensillum_orns.y_orn
    
    
    n2sim_tot       = int(pts_ms*t_tot)   # number of time points
    t               = np.linspace(0, t_tot, n2sim_tot) # time points
    spikes_orn      = np.zeros((n2sim_tot, n_neu_tot)) 
    spikes_orn[spike_matrix[0,:], spike_matrix[1,:]] = True
    
    spike_matrix[0,:]   = spike_matrix[0,:]/pts_ms #    convert the time column into ms
    spike_matrix        = np.transpose(spike_matrix)  
    
    # SDF extraction from the spike matrix
    sdf_params = params_1sens['sdf_params']
    tau_sdf = sdf_params['tau_sdf']
    dt_sdf  = sdf_params['dt_sdf']
    
    sdf_size        = int(stim_params['t_tot']/dt_sdf)
    
    orn_sdf_time    = np.linspace(0, dt_sdf*sdf_size, sdf_size)
    orn_sdf         = np.zeros((sdf_size, n_neu_tot))
    
    if ~(np.sum(spike_matrix) == 0):
        orn_sdf_tmp, orn_sdf_time = sdf_krofczik.main(spike_matrix, sdf_size,
                                                tau_sdf, dt_sdf)  # (Hz, ms)
        for nn in range(np.size(orn_sdf_tmp,1)):
            orn_sdf[:, nn] = orn_sdf_tmp[:, nn]*1e3  
    # else:
    #     print('doh')
                
        
    orn_lif_out = dict([
        ('t', t), ('u_od', u_od), 
        ('r_orn', r_orn),
        ('v_orn', v_orn),
        ('y_orn', y_orn),
        ('spikes_orn', spikes_orn), 
                   ('spike_matrix', spike_matrix), 
                   ('orn_sdf', orn_sdf), ('orn_sdf_time', orn_sdf_time),])
   
    return  orn_lif_out 

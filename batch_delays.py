#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:08:11 2019

script name: batch_delays.py
@author: mp525
"""
import timeit

import sys
import pickle        
#from os import path
#from os import mkdir
from shutil import copyfile
import matplotlib.pyplot as plt

import flynose
import sdf_krofczik

delays2an   = [0, 10, 20, 50, 100, 200, 500,]
delay_id    = 0#int(sys.argv[1])-1 # jobs run only starting from 1 ...
delay       = delays2an[delay_id]

#%%
import numpy as np
import datetime
now = datetime.datetime.now()

n_loops         =  50
n_ratios        =  1
n_concs         =  4
nsi_ln_par      = [[0,0],[.3,0],[0,13.3],[.4,0],[0,10],[.2,0],[0,16.6]]
dur2an          =  [10, 20, 50, 100, 200]
peak2an         =  np.linspace(.2, 1.4, n_concs)
pr2an           =  np.linspace(1, 20, n_ratios)

n_durs           = np.size(dur2an)

sims_to_run = len(nsi_ln_par)*n_loops*n_concs*n_ratios*n_durs
print('Number of Simulations to run: %d '%sims_to_run)

# approximately 3 secs per run:
Tot_sim_time = sims_to_run*2/60  # mins
run_sim_time = 0
print('Estimated Simulation time: %.2f hours (%.2f mins):'%(Tot_sim_time/60, Tot_sim_time))
endsim = now+datetime.timedelta(minutes=Tot_sim_time)
print(endsim)


#%%***********************************************
# Standard params
fld_analysis= 'delays_trials'

batch_params = [n_loops,pr2an,peak2an,nsi_ln_par,dur2an,delays2an,]

with open(fld_analysis+'/batch_params.pickle', 'wb') as f:
            pickle.dump(batch_params, f)

# Stimulus params 
stim_type   = 'ts'          # 'ts'  # 'ss' # 'pl'
pts_ms      = 5
onset_stim  = 300
# real plumes params
b_max       = np.nan # 3, 50, 150
w_max       = np.nan #3, 50, 150
rho         = np.nan #[0, 1, 3, 5]: 
stim_seed   = np.nan

tau_sdf     = 20
dt_sdf      = 5      

plume_params = [rho, w_max, b_max, stim_seed]

# figure and output options
orn_fig     = 0
al_dyn      = 1
al_fig      = 0
fig_ui      = 0        
verbose     = 0    
fig_save    = 1
data_save   = 0    

fig_opts    = [orn_fig, al_fig, fig_ui, fig_save, data_save, al_dyn, 
            verbose, fld_analysis] 


num_pns_glo         = 5     # number of PNs per each glomerulus
num_orns_glo        = 40    # number of ORNs per each glomerulus

peak_pnw    = np.zeros((n_ratios, n_concs, n_durs, n_loops))
peak_pns    = np.zeros((n_ratios, n_concs, n_durs, n_loops))
avg_pnw    = np.zeros((n_ratios, n_concs,n_durs, n_loops))
avg_pns    = np.zeros((n_ratios, n_concs, n_durs, n_loops))

peak_ornw    = np.zeros((n_ratios, n_concs, n_durs, n_loops))
peak_orns    = np.zeros((n_ratios, n_concs, n_durs, n_loops))
avg_ornw    = np.zeros((n_ratios, n_concs,n_durs, n_loops))
avg_orns    = np.zeros((n_ratios, n_concs, n_durs, n_loops))

   
date_str = now.strftime("%Y%m%d")
copyfile('flynose.py', 'flynose.' + date_str + '.py') 
copyfile('batch_delays.py', 'batch_delays.' + date_str + '.py') 

for nsi_str, alpha_ln in nsi_ln_par:
    data_name = 'ratio_nsi_%.2f_ln_%.1f_delay_%d'%(nsi_str, alpha_ln, delay)
    print(data_name)
    for peak_id, peak in enumerate(peak2an):
        
        for pr_id, peak_ratio in enumerate(pr2an):
            for dur_id, stim_dur in enumerate(dur2an):
                t_tot       = onset_stim + stim_dur +100+max(delay, 200)# ms 
                t_on        = [onset_stim, onset_stim+delay]    # ms
                t_off       = np.array(t_on)+stim_dur # ms
                concs       = [peak, peak*peak_ratio]
                stim_params = [stim_type, pts_ms, t_tot, t_on, t_off, 
                                       concs, plume_params]
                sdf_size    = int(t_tot/dt_sdf)
                sdf_params  = [sdf_size, tau_sdf, dt_sdf]
                params2an = [nsi_str, alpha_ln, stim_params]
                                    
                for ii in range(n_loops):
                                        
                    plt.ioff() # plt.ion() # to avoid showing the plot every time     
                    
                    tic = timeit.default_timer()
                    flynose_out = flynose.main(params2an, fig_opts)
                    [t, u_od, orn_spike_matrix, pn_spike_matrix, 
                     ln_spike_matrix, ] = flynose_out
                    toc = timeit.default_timer()
                    eff_dur = (toc-tic)
                    
                    if orn_spike_matrix.size >0:
                        # Calculate the SDF for PNs and LNs
                        orn_sdf, orn_sdf_time = sdf_krofczik.main(orn_spike_matrix, sdf_size,
                                                                     tau_sdf, dt_sdf)  # (Hz, ms)
                        orn_sdf = orn_sdf*1e3                        
                         
                        id_stim_w = np.flatnonzero((orn_sdf_time>t_on[0]) 
                                                & (orn_sdf_time<t_on[0]+100))
                        id_stim_s = np.flatnonzero((orn_sdf_time>t_on[1]) 
                                                & (orn_sdf_time<t_on[1]+100))
                        orn_peak_w  = np.max(np.mean(orn_sdf[id_stim_w, :num_orns_glo], axis=1)) # using average PN
                        orn_peak_s  = np.max(np.mean(orn_sdf[id_stim_s, num_orns_glo:], axis=1)) # using average PN
                        orn_avg_w  = np.mean(orn_sdf[id_stim_w, :num_orns_glo])
                        orn_avg_s  = np.mean(orn_sdf[id_stim_s, num_orns_glo:])

                        avg_ornw[pr_id, peak_id, dur_id,ii] = orn_avg_w
                        avg_orns[pr_id, peak_id, dur_id,ii] = orn_avg_s
                        
                        peak_ornw[pr_id, peak_id, dur_id,ii] = orn_peak_w
                        peak_orns[pr_id, peak_id, dur_id,ii] = orn_peak_s
                    
                    if pn_spike_matrix.size >0:
                        # Calculate the SDF for PNs and LNs
                        pn_sdf, pn_sdf_time = sdf_krofczik.main(pn_spike_matrix, sdf_size,
                                                                     tau_sdf, dt_sdf)  # (Hz, ms)
                        pn_sdf = pn_sdf*1e3                        
                         
                        id_stim_w = np.flatnonzero((pn_sdf_time>t_on[0]) 
                                        & (pn_sdf_time<t_on[0]+100))
                        id_stim_s = np.flatnonzero((pn_sdf_time>t_on[1]) 
                                        & (pn_sdf_time<t_on[1]+100))
                        
                        pn_peak_w  = np.max(np.mean(pn_sdf[id_stim_w, :num_pns_glo], axis=1)) # using average PN
                        pn_peak_s  = np.max(np.mean(pn_sdf[id_stim_s, num_pns_glo:], axis=1)) # using average PN
                        pn_avg_w  = np.mean(pn_sdf[id_stim_w, :num_pns_glo])
                        pn_avg_s  = np.mean(pn_sdf[id_stim_s, num_pns_glo:])

                        avg_pnw[pr_id, peak_id, dur_id,ii] = pn_avg_w
                        avg_pns[pr_id, peak_id, dur_id,ii] = pn_avg_s
                        
                        peak_pnw[pr_id, peak_id, dur_id,ii] = pn_peak_w
                        peak_pns[pr_id, peak_id, dur_id,ii] = pn_peak_s
                    
                    
                    sims_to_run = sims_to_run - 1
                    print('run time for sim of %d ms: %.2fs'%(t_tot,eff_dur))
                    print('Remaining Simulations to run: %d '%(sims_to_run))
                    print('Approx Remaining time: %.0f mins'%(sims_to_run*eff_dur/60))
    with open(fld_analysis+'/' + data_name+'.pickle', 'wb') as f:
        saved_pars = ['params2an', 'sdf_params', 'peak2an', 'pr2an', 'dur2an',
                     'avg_ornw', 'avg_orns', 'avg_pnw', 'avg_pns', 
                     'peak_ornw', 'peak_orns', 'peak_pnw', 'peak_pns']
        pickle.dump([params2an, sdf_params, peak2an, pr2an, dur2an,
                     avg_ornw, avg_orns, avg_pnw, avg_pns, 
                     peak_ornw, peak_orns, peak_pnw, peak_pns, saved_pars], f)
        

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:08:11 2019

script name: batch_delays2.py

@author: mp525
"""

import timeit
import pickle        
import matplotlib.pyplot as plt
import numpy as np
import datetime

import ORNs_layer_dyn
import AL_dyn

# delay batch params
now = datetime.datetime.now()

delays2an   = [0, 10, 20, 50, 100, 200, 500,]
delay_id    = 0#int(sys.argv[1])-1 # jobs run only starting from 1 ...
delay       = delays2an[delay_id]

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

#%%  LOAD Standard PARAMS FROM A FILE
fld_params      = 'NSI_analysis/trials/' #Olsen2010
name_params     = 'params_al_orn.ini'
params_al_orn   = pickle.load(open(fld_params + name_params,  "rb" ))

stim_params         = params_al_orn['stim_params']
orn_layer_params    = params_al_orn['orn_layer_params']
# orn_params          = params_al_orn['orn_params']
sdf_params          = params_al_orn['sdf_params']
al_params           = params_al_orn['al_params']
pn_ln_params        = params_al_orn['pn_ln_params']

# Stimulus params 
stim_params['stim_type']    = 'ts' # 'ss'  # 'ts'
onset_stim                  = 700
stim_params['conc0']        = 1.85e-4    # 2.85e-4
   
n_pns_recep         = al_params['n_pns_recep'] # number of PNs per each glomerulus
n_orns_recep        = orn_layer_params[0]['n_orns_recep']   # number of ORNs per each glomerulus

verbose             = False
data_save           = 1
fld_analysis        = 'NSI_analysis/analysis_delays/trials/'
date_str            = now.strftime("%Y%m%d")

# initialize output variables
peak_pnw    = np.zeros((n_ratios, n_concs, n_durs, n_loops))
peak_pns    = np.zeros((n_ratios, n_concs, n_durs, n_loops))
avg_pnw    = np.zeros((n_ratios, n_concs,n_durs, n_loops))
avg_pns    = np.zeros((n_ratios, n_concs, n_durs, n_loops))

peak_ornw    = np.zeros((n_ratios, n_concs, n_durs, n_loops))
peak_orns    = np.zeros((n_ratios, n_concs, n_durs, n_loops))
avg_ornw    = np.zeros((n_ratios, n_concs,n_durs, n_loops))
avg_orns    = np.zeros((n_ratios, n_concs, n_durs, n_loops))

for nsi_str, alpha_ln in nsi_ln_par:
    orn_layer_params[0]['w_nsi'] = nsi_str
    pn_ln_params['alpha_ln']        = alpha_ln
    data_name  = 'delay_' +\
                'stim_' + stim_params['stim_type'] +\
                '_nsi_%.1f'%(nsi_str) +\
                '_ln_%d'%(alpha_ln) +\
                '_delay2an_%d'%(delay) +\
                '.pickle'        
    print(data_name)
    for peak_id, peak in enumerate(peak2an):
        
        for pr_id, peak_ratio in enumerate(pr2an):
            for dur_id, stim_dur in enumerate(dur2an):
                # set stim_params
                stim_params['t_tot']       = onset_stim + stim_dur + 100 + max(delay, 200)        # ms 
                stim_params['t_on']        = np.array([onset_stim, onset_stim+delay])    # ms
                stim_params['stim_dur']    = np.array([stim_dur, stim_dur]) # ms
                stim_params['concs']       = np.array([peak, peak*peak_ratio])
                
                for ii in range(n_loops):
                                        
                    plt.ioff() # plt.ion() # to avoid showing the plot every time     
                    
                    tic = timeit.default_timer()
                    # Run flynose 
                    # ORNs layer dynamics
                    [t, u_od,  orn_spikes_t, orn_sdf, orn_sdf_time] = \
                        ORNs_layer_dyn.main(params_al_orn, verbose=verbose, )                    
                    
                    # AL dynamics
                    [t, pn_spike_matrix, pn_sdf, pn_sdf_time,
                        ln_spike_matrix, ln_sdf, ln_sdf_time,] = \
                        AL_dyn.main(params_al_orn, orn_spikes_t, verbose=verbose, )
                                        
                    toc = timeit.default_timer()
                    eff_dur = (toc-tic)
                    
                    # Calculate avg and peak SDF for ORNs
                    if orn_spikes_t.size >0:
                        id_stim_w = np.flatnonzero((orn_sdf_time>onset_stim) 
                                                & (orn_sdf_time<onset_stim+100))
                        id_stim_s = np.flatnonzero((orn_sdf_time>onset_stim+delay) 
                                                & (orn_sdf_time<onset_stim+delay+100))
                        orn_peak_w  = np.max(np.mean(orn_sdf[id_stim_w, :n_orns_recep], axis=1)) # using average PN
                        orn_peak_s  = np.max(np.mean(orn_sdf[id_stim_s, n_orns_recep:], axis=1)) # using average PN
                        orn_avg_w  = np.mean(orn_sdf[id_stim_w, :n_orns_recep])
                        orn_avg_s  = np.mean(orn_sdf[id_stim_s, n_orns_recep:])

                        avg_ornw[pr_id, peak_id, dur_id,ii] = orn_avg_w
                        avg_orns[pr_id, peak_id, dur_id,ii] = orn_avg_s
                        
                        peak_ornw[pr_id, peak_id, dur_id,ii] = orn_peak_w
                        peak_orns[pr_id, peak_id, dur_id,ii] = orn_peak_s
                    
                    # Calculate avg and peak SDF for PNs 
                    if pn_spike_matrix.size >0:
                        id_stim_w = np.flatnonzero((pn_sdf_time>onset_stim) 
                                        & (pn_sdf_time<onset_stim+100))
                        id_stim_s = np.flatnonzero((pn_sdf_time>onset_stim+delay) 
                                        & (pn_sdf_time<onset_stim+delay+100))
                        
                        pn_peak_w  = np.max(np.mean(pn_sdf[id_stim_w, :n_pns_recep], axis=1)) # using average PN
                        pn_peak_s  = np.max(np.mean(pn_sdf[id_stim_s, n_pns_recep:], axis=1)) # using average PN
                        pn_avg_w  = np.mean(pn_sdf[id_stim_w, :n_pns_recep])
                        pn_avg_s  = np.mean(pn_sdf[id_stim_s, n_pns_recep:])

                        avg_pnw[pr_id, peak_id, dur_id,ii] = pn_avg_w
                        avg_pns[pr_id, peak_id, dur_id,ii] = pn_avg_s
                        
                        peak_pnw[pr_id, peak_id, dur_id,ii] = pn_peak_w
                        peak_pns[pr_id, peak_id, dur_id,ii] = pn_peak_s
                    
                    
                    sims_to_run = sims_to_run - 1
                    # print('run time for sim of %d ms: %.2fs'%(stim_params['t_tot'],eff_dur))
                    print('Remaining Simulations to run: %d '%(sims_to_run))
                    print('Approx Remaining time: %.0f mins'%(sims_to_run*eff_dur/60))
    
    # SAVE SDF OF ORN  and PNs FIRING RATE and the params
    if data_save:
        params_batch = dict([
                    ('n_loops', n_loops),
                    ('peak2an', peak2an),
                    ('pr2an', pr2an),
                    ('dur2an', dur2an),
                     ])
        
        output2an = dict([
                    ('peak_pnw', peak_pnw),
                    ('peak_pns',peak_pns),
                    ('avg_pnw',avg_pnw),
                    ('avg_pns',avg_pns),
                    ('peak_ornw',peak_ornw),
                    ('peak_orns',peak_orns),
                    ('avg_ornw',avg_ornw),
                    ('avg_orns',avg_orns),
                    ])                            
        
        with open(fld_analysis+data_name , 'wb') as f:
            pickle.dump([params_al_orn, output2an, params_batch], f)
     
    
        

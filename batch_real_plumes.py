#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:08:11 2019

script name: batch_flynose_real_plumes.py
@author: mp525
"""
import flynose
import sdf_krofczik
import stats_for_plumes as stats

import pickle
import timeit
from shutil import copyfile
import sys


stim_seed_start =0# int(sys.argv[1])
#%%

import numpy as np
import datetime


now = datetime.datetime.now()

## ORN NSI params
nsi_ln_par = [[0,0],[.3,0],[0,13.3],[.2,0],[0,10],[.4,0],[0,16.6]]

stim_dur  = 201000 # 201000[ms]
n_seeds   = 1 # 50
    
w_maxs  = [.01,.03,.3, 3, 25, 50, ] # max value in the whiff distribution
b_maxs  = [25]                      # max value in the blank distribution
rhos    = [0, 1, 3, 5]
peak    = 1.5
peak_ratio = 1

sims_to_run = n_seeds*len(nsi_ln_par)*np.size(w_maxs)*np.size(b_maxs)*np.size(rhos)
print('Number of Simulations to run: %d '%sims_to_run)

Tot_sim_time = 3.8*sims_to_run*stim_dur/1000/60/60  # hours
run_sim_time = 0
print('Estimated Simulation time: %.3g hours (%.2g mins)'%(Tot_sim_time, Tot_sim_time*60))
endsim = now+datetime.timedelta(hours=Tot_sim_time)
print(endsim)


#%%***********************************************
# Standard params
fld_analysis= 'NSI_analysis/real_plumes/trials'

n_loops = 50
batch_params = [n_loops,nsi_ln_par,stim_dur,w_maxs,b_maxs, rhos,peak,peak_ratio  ]

with open(fld_analysis+'/batch_params.pickle', 'wb') as f:
            pickle.dump(batch_params, f)
            
date_str = now.strftime("%Y%m%d")
copyfile('flynose.py', fld_analysis+'/flynose.' + date_str + '.py') 
copyfile('batch_real_plumes.py', fld_analysis+'/batch_flynose_real_plumes.' + date_str + '.py') 


# Stimulus params 
stim_type   = 'pl'          # 'ts'  # 'ss' # 'pl'
pts_ms      = 5
onset_stim  = 300
delay       = 0

t_tot       = onset_stim+stim_dur        # ms 
t_on        = [onset_stim, onset_stim+delay]    # ms
t_off       = np.array(t_on)+stim_dur # ms
concs       = [peak, peak*peak_ratio]

#  PARAMS FOR WHIFF AND BLANK DISTRIOBUTIONS
g           = -1/2# 1    # -1/2 for a power law of -3/2, 1 for uniform distribution
whiff_min   = 3e-3      # [s]
blank_min   = 3e-3      # [s]
        
tau_sdf     = 20
dt_sdf      = 5      
sdf_size    = int(t_tot/dt_sdf)

sdf_params  = [sdf_size, tau_sdf, dt_sdf]
                
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
    
for stim_seed in np.arange(stim_seed_start, stim_seed_start+n_seeds):
    for nsi_str, alpha_ln in nsi_ln_par:
        

        for b_max in b_maxs:
            for w_max in w_maxs:
                # CALCULATE THE THEORETICAL MEAN WHIFF, MEAN BLANK DURATIONS AND INTERMITTENCY
                pdf_wh, logbins, wh_mean = stats.whiffs_blanks_pdf(whiff_min, w_max, g)
                pdf_bl, logbins, bl_mean = stats.whiffs_blanks_pdf(blank_min, b_max, g)
                
                interm_th = wh_mean/(wh_mean+bl_mean)
                for rho in rhos:
                    # real plumes params
                    plume_params = [rho, w_max, b_max, stim_seed]
                    stim_params = [stim_type, pts_ms, t_tot, t_on, t_off, 
                                       concs, plume_params]
                    params2an = [nsi_str, alpha_ln, stim_params]
                            
                    tic = timeit.default_timer()
                    
                    # RUN SIMULATIONS
                    [t, u_od, orn_spike_matrix, pn_spike_matrix,_] = \
                    flynose.main(params2an, fig_opts)
                    
                    #**************************************************
                    # Calculate AND SAVE DATA
                    # stimulus analysis
                    t_id_stim = np.flatnonzero((t>t_on[0]) & (t<t_off[0]))
                    
                    od_avg_1 = np.mean(u_od[t_id_stim, 0])
                    od_avg_2 = np.mean(u_od[t_id_stim, 1])
                    cor_stim        = -2
                    overlap_stim    = -2
                    cor_whiff       = -2
                    out_1 = u_od[t_id_stim, 0]
                    out_2 = u_od[t_id_stim, 1]
                                        
                    interm_est_1 = np.sum(out_1>0)/(t_tot*pts_ms)
                    interm_est_2 = np.sum(out_2>0)/(t_tot*pts_ms)
                    
                    if (np.sum(out_2)!=0) & (np.sum(out_1)!=0):
                        cor_stim        = np.corrcoef(out_2, out_1)[1,0]
                        overlap_stim    = stats.overlap(out_2, out_1)
                        nonzero_concs1  = out_2[(out_2>0) & (out_1>0)]
                        nonzero_concs2  = out_1[(out_2>0) & (out_1>0)]
                        cor_whiff       = np.corrcoef(nonzero_concs1, nonzero_concs2)[0, 1] # np.corrcoef(concs1, concs2)[0, 1]
                    
                    # Calculate the SDF for PNs and LNs
                    orn_sdf, orn_sdf_time = sdf_krofczik.main(orn_spike_matrix, sdf_size,
                                                                 tau_sdf, dt_sdf)  # (Hz, ms)
                    orn_sdf = orn_sdf*1e3
                    
                    pn_sdf, pn_sdf_time = sdf_krofczik.main(pn_spike_matrix, sdf_size,
                                                                 tau_sdf, dt_sdf)  # (Hz, ms)
                    pn_sdf = pn_sdf*1e3
                
                    
                    id_stim_1 = np.flatnonzero((pn_sdf_time>t_on[0]) & (pn_sdf_time<t_off[0]))
                    id_stim_2 = np.flatnonzero((pn_sdf_time>t_on[1]) & (pn_sdf_time<t_off[1]))
                    
                    pn_peak_1  = np.max(np.mean(pn_sdf[id_stim_1, :num_pns_glo], axis=1)) # using average PN
                    pn_peak_2  = np.max(np.mean(pn_sdf[id_stim_2, num_pns_glo:], axis=1)) # using average PN
                    pn_avg_1  = np.mean(pn_sdf[id_stim_1, :num_pns_glo])
                    pn_avg_2  = np.mean(pn_sdf[id_stim_2, num_pns_glo:])
                    
                    orn_peak_1  = np.max(np.mean(orn_sdf[id_stim_1, :num_orns_glo], axis=1)) # using average PN
                    orn_peak_2  = np.max(np.mean(orn_sdf[id_stim_2, num_orns_glo:], axis=1)) # using average PN
                    orn_avg_1  = np.mean(orn_sdf[id_stim_1, :num_orns_glo])
                    orn_avg_2  = np.mean(orn_sdf[id_stim_2, num_orns_glo:])
                    
                    #%% Calculate the mean and the peak for PN responses
                    pn_sdf_dt = pn_sdf_time[1]-pn_sdf_time[0]
                    pn_tmp = np.zeros((np.size(id_stim_1),2))
                    
                    pn_tmp[:,0] = np.mean(pn_sdf[id_stim_1, :num_pns_glo], axis=1)
                    pn_tmp[:,1] = np.mean(pn_sdf[id_stim_1, num_pns_glo:], axis=1)
                    perf_time = np.zeros((2, 3))
                    perf_avg = np.zeros((2, 3))
                    
                    for id_glo in range(2):
                        for thr_id, thr in enumerate([50, 100, 150]):
                            perf_time[id_glo, thr_id, ] = np.sum(pn_tmp[:, id_glo]>thr)*pn_sdf_dt
                            if perf_time[id_glo, thr_id, ]>0:
                                perf_avg[id_glo, thr_id, ] = np.average(pn_tmp[:, id_glo], 
                                    weights=(pn_tmp[:, id_glo]>thr))

                    data_name = '/real_plumes' + \
                        '_nsi_%.2f'%(params2an[0]) +\
                        '_ln_%.1f'%(params2an[1]) +\
                        '_rho_%d'%(rho) +\
                        '_wmax_%.2f'%(w_max) +\
                        '_seed_%d'%stim_seed +\
                        '.pickle'
                    
                    output_names = ['cor_stim', 'overlap_stim', 'cor_whiff', 
                                     'interm_th', 'interm_est_1', 'interm_est_2', 'od_avg1', 
                                     'od_avg2', 'orn_avg1', 'orn_avg2', 'pn_avg1', 'pn_avg2', 
                                     'perf_avg', 'perf_time', ]                
                
                    params2an_names = ['omega_nsi', 'alpha_ln', 'dur2an', 'delays2an', 
                                       'peak', 'peak_ratio', 'rho', 'stim_type', 'w_max', 'b_max']
                
                    with open(fld_analysis+data_name, 'wb') as f:
                        pickle.dump([params2an, cor_stim, overlap_stim, cor_whiff, 
                                     interm_th, interm_est_1, interm_est_2, od_avg_1, od_avg_2, 
                                     orn_avg_1, orn_avg_2, pn_avg_1, pn_avg_2, perf_avg, perf_time, 
                                     params2an_names, output_names], f)

                    toc = timeit.default_timer()
                    sims_to_run = sims_to_run - 1
                    print('Remaining Simulations to run: %d '%(sims_to_run))
                    print('Approx Remaining time: %.0f mins'%(sims_to_run*(toc-tic)/60))
    
    print('Tot sim time:%.0f mins'%(run_sim_time))
	    

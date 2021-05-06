#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:08:11 2019

script name: batch_real_plumes2.py
@author: mp525
"""


import timeit
import pickle        
# import matplotlib.pyplot as plt
import numpy as np
import datetime

import ORNs_layer_dyn
import AL_dyn
import set_orn_al_params
import stats_for_plumes as stats
import stim_fcn
import sys



def tictoc():
    return timeit.default_timer()


now = datetime.datetime.now()

analysis_name   = 'real_plumes'

nsi_ln_par      = [[0,0], [.6, 0], [0, .6], ]

if sys.argv == ['']:
    stim_seed_start = 0 # int(sys.argv[1])
    
else:
    stim_seed_start = int(sys.argv[1])


print('real plumes simulations')
    

## ORN NSI params
stim_dur  = 10000        # 201000[ms]
n_seeds   = 1           # 50
    
w_maxs      = [.03,.3, 3, 25, 50, ] # max value in the whiff distribution
rhos        = [0, 1, 3, 5] 
rhocs       = [0, 0.33, .66, 1] 

# w_maxs  = [.01,.03,.3, 3, 25, 50, ]#[.01,.03,.3, 3, 25, 50, ] # max value in the whiff distribution
b_maxs  = [25]                      # max value in the blank distribution
peak    = 10e-4
peak_ratio = 1

sims_to_run = n_seeds*len(nsi_ln_par)*len(w_maxs)*len(b_maxs)*len(rhos)
print('Number of Simulations to run: %d '%sims_to_run)

# approx 3s per each second of simulation
Tot_sim_time = sims_to_run*3.4*stim_dur/1000/60/60  # hours
run_sim_time = 0
print('Estimated Simulation time: %.3g hours (%.1f mins)'%(Tot_sim_time, Tot_sim_time*60))
endsim = now+datetime.timedelta(hours=Tot_sim_time)
print(endsim)

#%%
#  LOAD Standard NET PARAMS FROM A FILE
params_al_orn = set_orn_al_params.main(2)

stim_params         = params_al_orn['stim_params']
orn_layer_params    = params_al_orn['orn_layer_params']
# orn_params          = params_al_orn['orn_params']
sdf_params          = params_al_orn['sdf_params']
al_params           = params_al_orn['al_params']
pn_ln_params        = params_al_orn['pn_ln_params']
plume_params        = stim_params['plume_params'] 

# Stimulus params 
stim_params['stim_type']    = 'pl' # 'ss'  # 'ts'
t0                          = 1000
stim_params['pts_ms']       = 10
stim_params['conc0']        = 1.85e-4    # 2.85e-4


stim_params['t_tot']       = t0 + stim_dur                  # ms 
stim_params['t_on']        = np.array([t0, t0])    # ms
stim_params['stim_dur']    = np.array([stim_dur, stim_dur]) # ms
stim_params['concs']       = np.array([peak, peak*peak_ratio])


n_pns_recep                 = al_params['n_pns_recep'] # number of PNs per each glomerulus
n_orns_recep                = orn_layer_params[0]['n_orns_recep']   # number of ORNs per each glomerulus

verbose                     = False
data_save                   = 1
fld_analysis                = 'NSI_analysis/analysis_'+analysis_name+'_1/'

sdf_params['tau_sdf']       = 20
pn_ln_params['tau_ln']      = 25

n_sens_type       = orn_layer_params.__len__()  # number of type of sensilla


#  PARAMS FOR WHIFF AND BLANK DISTRIBUTIONS
g           = plume_params['g']  # -1/2 for a power law of -3/2, 1 for uniform distribution
whiff_min   = plume_params['whiff_min']      # [s]
blank_min   = plume_params['blank_min']      # [s]

t_tot       = stim_params['t_tot']
pts_ms      = stim_params['pts_ms']

batch_params = dict([
    ('nsi_ln_par', nsi_ln_par),
    ('stim_dur', stim_dur),
    ('w_maxs', w_maxs),
    ('b_maxs', b_maxs),
    ('rhos', rhos),
    ('rhocs', rhocs),
    ('peak', peak),
    ('peak_ratio', peak_ratio),
    ])


# save batch and net params 
if data_save:
    with open(fld_analysis+analysis_name+'_batch_params.pickle', 'wb') as f:
                pickle.dump(batch_params, f)
    with open(fld_analysis+analysis_name+'_params_al_orn.ini', 'wb') as f:
                pickle.dump(params_al_orn, f)  

#%% RUN BATCHES OF SIMULATIONS
tic_whole = tictoc()

for id_seed in np.arange(stim_seed_start, stim_seed_start+n_seeds):
    
# for id_seed in range(n_seeds):
    for b_max in b_maxs:
        plume_params['blank_max'] = b_max
        for w_max in w_maxs:
            plume_params['whiff_max'] = w_max
                
            # CALCULATE THE THEORETICAL MEAN WHIFF, MEAN BLANK DURATIONS AND INTERMITTENCY
            pdf_wh, logbins, wh_mean = stats.whiffs_blanks_pdf(whiff_min, w_max, g)
            pdf_bl, logbins, bl_mean = stats.whiffs_blanks_pdf(blank_min, b_max, g)
            
            interm_th = wh_mean/(wh_mean+bl_mean)
            
            ########################################################
            # choose a good seed for all inh and rho conditions
            n_attempts = 500
            diff_ab = np.ones((n_attempts))
            
            rand_list  = np.random.randint(2000, size=n_attempts)
            diff_min = .05#0.01

            
            for ii in range(n_attempts):
                stim_params['stim_seed']   = rand_list[ii]
                plume_params['rho_t_exp']   = 0         # [0, 5]
                plume_params['rho_c']       = 0         # [0, 1]
                u_od = stim_fcn.main(stim_params)
                out_w = u_od[t0:, 0]
                out_s = u_od[t0:, 1]
                diff_ab[ii] = abs(np.mean(out_s) -np.mean(out_w))/np.mean(out_s+out_w)
                
                plume_params['rho_t_exp']   = 5         # [0, 5]
                plume_params['rho_c']       = 1         # [0, 1]
                u_od = stim_fcn.main(stim_params)
                out_w_hi = u_od[t0:, 0]
                
                diff_ab[ii] += abs(np.mean(out_w_hi) -np.mean(out_w))/np.mean(out_w_hi+out_w)
            
                min_diff_ab = np.min(diff_ab)
                if min_diff_ab<diff_min:
                    break
             
            stim_seed   = rand_list[np.argmin(diff_ab,)]    
            stim_params['stim_seed'] = stim_seed
            print('seed: %d'%stim_seed)
            ############################################
            
            for id_rho, rho_t_exp in enumerate(rhos):
                
                # set stim_params
                plume_params['rho_t_exp'] = rho_t_exp
                rho_c = rhocs[id_rho]
                plume_params['rho_c'] = rho_c
        
                            
                for nsi_str, alpha_ln in nsi_ln_par:
                    pn_ln_params['alpha_ln']        = alpha_ln
                    for sst in range(n_sens_type):
                        orn_layer_params[sst]['w_nsi']    = nsi_str
                        
                    tic = timeit.default_timer()
                    
                    # RUN flynose SIMULATIONS                    
                    # ORNs layer dynamics
                    output_orn = ORNs_layer_dyn.main(params_al_orn, verbose=verbose, )                    
                    [t, u_od,  orn_spikes_t, orn_sdf, orn_sdf_time] = output_orn
                    
                    # AL dynamics
                    output_al = AL_dyn.main(params_al_orn, orn_spikes_t, verbose=verbose, )
                    [t, pn_spike_matrix, pn_sdf, pn_sdf_time,
                        ln_spike_matrix, ln_sdf, ln_sdf_time,] = output_al
                    
                    toc = timeit.default_timer()
                    eff_dur = (toc-tic)
                    
                    # CALCULATE AND SAVE DATA
                    t_id_stim = np.flatnonzero((t>t0) & (t<t0 + stim_dur))
                    
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
                    
                    
                    
                    id_stim_1 = np.flatnonzero((pn_sdf_time>t0) & 
                                               (pn_sdf_time<t0 + stim_dur))
                    id_stim_2 = np.flatnonzero((pn_sdf_time>t0) & 
                                               (pn_sdf_time<t0 + stim_dur))
                    
                    pn_peak_1  = np.max(np.mean(pn_sdf[id_stim_1, :n_pns_recep], axis=1)) # using average PN
                    pn_peak_2  = np.max(np.mean(pn_sdf[id_stim_2, n_pns_recep:], axis=1)) # using average PN
                    pn_avg_1  = np.mean(pn_sdf[id_stim_1, :n_pns_recep])
                    pn_avg_2  = np.mean(pn_sdf[id_stim_2, n_pns_recep:])
                    
                    orn_peak_1  = np.max(np.mean(orn_sdf[id_stim_1, :n_orns_recep], axis=1)) # using average PN
                    orn_peak_2  = np.max(np.mean(orn_sdf[id_stim_2, n_orns_recep:], axis=1)) # using average PN
                    orn_avg_1  = np.mean(orn_sdf[id_stim_1, :n_orns_recep])
                    orn_avg_2  = np.mean(orn_sdf[id_stim_2, n_orns_recep:])
                    
                    # Calculate the mean and the peak for PN responses
                    pn_sdf_dt = pn_sdf_time[1]-pn_sdf_time[0]
                    pn_tmp = np.zeros((np.size(id_stim_1),2))
                    
                    pn_tmp[:,0] = np.mean(pn_sdf[id_stim_1, :n_pns_recep], axis=1)
                    pn_tmp[:,1] = np.mean(pn_sdf[id_stim_1, n_pns_recep:], axis=1)
                    perf_time = np.zeros((2, 3))
                    perf_avg = np.zeros((2, 3))
                    
                    for id_glo in range(2):
                        for thr_id, thr in enumerate([50, 100, 150]):
                            perf_time[id_glo, thr_id, ] = np.sum(pn_tmp[:, id_glo]>thr)*pn_sdf_dt
                            if perf_time[id_glo, thr_id, ]>0:
                                perf_avg[id_glo, thr_id, ] = np.average(pn_tmp[:, id_glo], 
                                    weights=(pn_tmp[:, id_glo]>thr))

                    
                    # SAVE SDF OF ORN  and PNs FIRING RATE and the params
                    data_name = analysis_name + \
                        '_nsi_%.1f'%(nsi_str) +\
                        '_ln_%.1f'%(alpha_ln) +\
                        '_rho_%d'%(rho_t_exp) +\
                        '_wmax_%.2f'%(w_max) +\
                        '_seed_%d'%id_seed +\
                        '.pickle'

                    output2an = dict([
                        ('cor_stim', cor_stim),
                        ('overlap_stim', overlap_stim),
                        ('cor_whiff', cor_whiff),
                        ('interm_th', interm_th),
                        ('interm_est_1', interm_est_1),
                        ('interm_est_2',  interm_est_2),
                        ('od_avg_1', od_avg_1),
                        ('od_avg_2', od_avg_2),
                        ('orn_avg_1', orn_avg_1),
                        ('orn_avg_2', orn_avg_2),
                        ('pn_avg_1', pn_avg_1),
                        ('pn_avg_2', pn_avg_2),
                        ('perf_avg', perf_avg),
                        ('perf_time', perf_time),
                         ])                
                
                    if data_save:
                        print(data_name)
                        with open(fld_analysis + data_name , 'wb') as f:
                            pickle.dump([params_al_orn, output2an,], f)

                    toc = timeit.default_timer()
                    sims_to_run = sims_to_run - 1
                    print('Remaining Simulations to run: %d '%(sims_to_run))
                    print('Approx Remaining time: %.0f mins'%(sims_to_run*(toc-tic)/60))
    
	    
toc_whole = (tictoc()-tic_whole)/3600 # [hours]
print('Tot time: %.2f hours'%toc_whole)

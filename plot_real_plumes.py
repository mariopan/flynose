#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:57:28 2021

This function plot the response to a single real plumes stimulus with a single 
inhibitory condition.


plot_real_plumes.py

@author: mario
"""




# Setting parameters and define functions

import numpy as np
import timeit
import pickle        
import matplotlib.pyplot as plt
import matplotlib as mpl

import AL_dyn
import ORNs_layer_dyn
import plot_al_orn
import set_orn_al_params
import stats_for_plumes as stats
import stim_fcn 


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
legend_fs = 12
black   = 'xkcd:black'
blue    = 'xkcd:blue'
red     = 'xkcd:red'
green   = 'xkcd:green'
purple  = 'xkcd:purple'
orange  = 'xkcd:orange'


cmap    = plt.get_cmap('rainbow')
recep_clrs = ['green','purple','cyan','red']
np.set_printoptions(precision=2)


def tictoc():
    return timeit.default_timer()



# LOAD PARAMS FROM A FILE
params_al_orn = set_orn_al_params.main(2)

stim_params         = params_al_orn['stim_params']
orn_layer_params    = params_al_orn['orn_layer_params']
orn_params          = params_al_orn['orn_params']
sdf_params          = params_al_orn['sdf_params']
al_params           = params_al_orn['al_params']
pn_ln_params        = params_al_orn['pn_ln_params']

plume_params        = stim_params['plume_params']        

n_sens_type         = orn_layer_params.__len__()  # number of type of sensilla



# Stimulus parameters
t0                          = 1000
delay                       = 0
stim_dur                    = 5000      # [ms]
peak_ratio                  = 1
peak                        = 20e-4 # 5e-4, 1.5e-3, 2e-2,]
stim_params['pts_ms']       = 10
sdf_params['tau_sdf']       = 20
sdf_params['dt_sdf']        = 5
pn_ln_params['tau_ln']      = 25

# nsi params
nsi_str                     = .6
alpha_ln                    = .6
inh_conds                   = ['nsi', 'ln', 'noin']
corr_conds                  = ['lo', 'hi']#


stim_params['stim_type']    = 'pl'      # 'ss'  # 'ts' # 'rs' # 'pl'
stim_params['t_on']         = np.array([t0, t0+delay])
stim_params['conc0']        = 1.85e-4    # fitted value: 2.85e-4
stim_params['stim_dur']     = np.array([stim_dur, stim_dur+delay])
stim_params['t_tot']        = t0+delay+stim_dur
stim_params['concs']        = np.array([peak, peak*peak_ratio])


# Output parameters
figs_save                   = 0
fld_analysis                = 'NSI_analysis/real_plumes_example/'

time2analyse        = stim_dur
n_pns_recep         = al_params['n_pns_recep'] # number of PNs per each glomerulus
n_orns_recep        = orn_layer_params[0]['n_orns_recep']   # number of ORNs per each glomerulus
t_tot               = stim_params['t_tot']
pts_ms              = stim_params['pts_ms']                    

# Initialize output variables
pn_peak_w   = np.zeros((1,1))
pn_avg_w    = np.zeros((1,1))
pn_peak_s   = np.zeros((1,1))
pn_avg_s    = np.zeros((1,1))

orn_peak_w  = np.zeros((1,1))
orn_avg_w   = np.zeros((1,1))
orn_peak_s  = np.zeros((1,1))
orn_avg_s   = np.zeros((1,1))    
od_avg_w   = np.zeros((1,1))
od_avg_s   = np.zeros((1,1))
cor_stim   = np.zeros((1,1))

overlap_stim    = np.ones((1,1))*-2
cor_whiff       = np.ones((1,1))*-2


#%% look for a stimulus with small difference between two odorants
tic = tictoc()
n_attempts = 100
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
 
stim_params['stim_seed']   = rand_list[np.argmin(diff_ab,)]    
        
print('Stim seed: %d' %stim_params['stim_seed'])


# PLOT odors' concentrations

for id_cor, corr_cond in enumerate(corr_conds):
    
    if corr_cond == 'lo':
        plume_params['rho_t_exp']   = 0         # [0, 5]
        plume_params['rho_c']       = 0         # [0, 1]
    elif corr_cond == 'hi':
        plume_params['rho_t_exp']   = 5         # [0, 5]
        plume_params['rho_c']       = 1         # [0, 1]
    
    tmp_od = stim_fcn.main(stim_params)
    out_w = tmp_od[t0:, 0]
    out_s = tmp_od[t0:, 1]
    
    print(corr_cond+', Odour s, w, corr: %.2g, %.2g, %.2g' 
          %(np.mean(out_s), np.mean(out_w), np.corrcoef(out_s, out_w)[1,0]))
toc = tictoc()
print('time to select the seed: %.2f s' %(toc-tic))

# %% 
# stim_params['stim_seed'] = 710 #1293        
tic = tictoc()
    
print('conc: %2g, conc ratio: %d'%(peak, peak_ratio))

print('')
for inh_cond in inh_conds:
    print('Inh. condition: '+ inh_cond)
    
    if inh_cond == 'nsi':
        w_nsi = nsi_str    
        for sst in range(n_sens_type):
            orn_layer_params[sst]['w_nsi']  = nsi_str    
        pn_ln_params['alpha_ln']        = 0
    elif inh_cond == 'noin':
        w_nsi = 0    
        for sst in range(n_sens_type):
            orn_layer_params[sst]['w_nsi']  = 0
        pn_ln_params['alpha_ln']        = 0
    elif inh_cond == 'ln':
        w_nsi = 0    
        for sst in range(n_sens_type):
            orn_layer_params[sst]['w_nsi']  = 0    
        pn_ln_params['alpha_ln']        = alpha_ln
        
    
   
    for corr_cond in corr_conds:
        if corr_cond == 'lo':
            plume_params['rho_t_exp']   = 0         # [0, 5]
            plume_params['rho_c']       = 0         # [0, 1]
        elif corr_cond == 'hi':
            plume_params['rho_t_exp']   = 5         # [0, 5]
            plume_params['rho_c']       = 1         # [0, 1]
            
        print('Corr. condition: '+ corr_cond)
        
        
        output_orn = ORNs_layer_dyn.main(params_al_orn)
        [t, u_od,  orn_spikes_t, orn_sdf,orn_sdf_time] = output_orn 
        
        # AL dynamics
        output_al = AL_dyn.main(params_al_orn, orn_spikes_t)
        [t, pn_spike_matrix, pn_sdf, pn_sdf_time,
                      ln_spike_matrix, ln_sdf, ln_sdf_time,] = output_al
        
        fig_al_orn = plot_al_orn.main(params_al_orn, output_orn, output_al)
        
        # CALCULATE AND SAVE DATA
 
        cor_stim[0]     = -2
        overlap_stim    = -2
        cor_whiff       = -2
        out_w = u_od[t0:, 0]
        out_s = u_od[t0:, 1]
        od_avg_w[0] = np.mean(out_w)
        od_avg_s[0] = np.mean(out_s)
                            
        interm_est_1 = np.sum(out_w>0)/(t_tot*pts_ms)
        interm_est_2 = np.sum(out_s>0)/(t_tot*pts_ms)
        
        if (np.sum(out_s)!=0) & (np.sum(out_w)!=0):
            cor_stim[0]     = np.corrcoef(out_s, out_w)[1,0]
            overlap_stim    = stats.overlap(out_s, out_w)
            nonzero_concs1  = out_s[(out_s>0) & (out_w>0)]
            nonzero_concs2  = out_s[(out_s>0) & (out_w>0)]
            cor_whiff       = np.corrcoef(nonzero_concs1, nonzero_concs2)[0, 1] # np.corrcoef(concs1, concs2)[0, 1]
        
        
        if orn_spikes_t.size >0:
            id_stim_w = np.flatnonzero((orn_sdf_time>t0) 
                                    & (orn_sdf_time<t0+time2analyse))
            
            
            id_stim_s = np.flatnonzero((orn_sdf_time>t0+delay) 
                                    & (orn_sdf_time<t0+delay+time2analyse))
            
            orn_peak_w[0]  = np.max(np.mean(orn_sdf[id_stim_w, :n_orns_recep], axis=1)) # using average PN
            orn_peak_s[0]  = np.max(np.mean(orn_sdf[id_stim_s, n_orns_recep:], axis=1)) # using average PN
            orn_avg_w[0]  = np.mean(orn_sdf[id_stim_w, :n_orns_recep])
            orn_avg_s[0]  = np.mean(orn_sdf[id_stim_s, n_orns_recep:])
        
        # Calculate avg and peak SDF for PNs 
        if pn_spike_matrix.size >0:
            id_stim_w = np.flatnonzero((pn_sdf_time>t0) 
                            & (pn_sdf_time<t0+time2analyse))
            id_stim_s = np.flatnonzero((pn_sdf_time>t0+delay) 
                            & (pn_sdf_time<t0+delay+time2analyse))
            
            pn_peak_w[0]  = np.max(np.mean(pn_sdf[id_stim_w, :n_pns_recep], axis=1)) # using average PN
            pn_peak_s[0]  = np.max(np.mean(pn_sdf[id_stim_s, n_pns_recep:], axis=1)) # using average PN
            pn_avg_w[0]  = np.mean(pn_sdf[id_stim_w, :n_pns_recep])
            pn_avg_s[0]  = np.mean(pn_sdf[id_stim_s, n_pns_recep:])
        
            
        pn_sdf_dt = pn_sdf_time[1]-pn_sdf_time[0]
        pn_tmp = np.zeros((np.size(id_stim_s),2))
                    
        pn_tmp[:,0] = np.mean(pn_sdf[id_stim_s, :n_pns_recep], axis=1)
        pn_tmp[:,1] = np.mean(pn_sdf[id_stim_s, n_pns_recep:], axis=1)
        perf_time = np.zeros((2, 3))
        perf_avg = np.zeros((2, 3))
        
        for id_glo in range(2):
            for thr_id, thr in enumerate([50, 100, 150]):
                perf_time[id_glo, thr_id, ] = np.sum(pn_tmp[:, id_glo]>thr)*pn_sdf_dt
                if perf_time[id_glo, thr_id, ]>0:
                    perf_avg[id_glo, thr_id, ] = np.average(pn_tmp[:, id_glo], 
                        weights=(pn_tmp[:, id_glo]>thr))
        perf_at = perf_avg*perf_time/1e3
        
        print('Odour s, w, corr: %.2g, %.2g, %.2g' %(od_avg_s, od_avg_w, cor_stim))
        print('nu ORN s, w: %.2f, %.2f' %(orn_avg_s, orn_avg_w))
        print('nu PN s, w: %.2f, %.2f' %(pn_avg_s, pn_avg_w))
        print('performance 50, 100, 150 Hz: ')
        print(perf_at[0,:])
        print('')

toc = tictoc()
print('Sim + plot time: %.2f m' %((toc-tic)/60))


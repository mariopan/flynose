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



# %% LOAD PARAMS FROM A FILE
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
stim_dur                    = 200000#00      # [ms]
peak_ratio                  = 1e-8
peak                        = .5e-4 # 5e-4, 1.5e-3, 2e-2,]
stim_params['pts_ms']       = 10
sdf_params['tau_sdf']       = 20

# nsi params
nsi_str                     = .6
alpha_ln                    = .6
inh_cond                    = ['noin']


stim_params['stim_type']    = 'pl'      # 'ss'  # 'ts' # 'rs' # 'pl'
stim_params['t_on']         = np.array([t0, t0+delay])
stim_params['conc0']        = 1.85e-4    # fitted value: 2.85e-4
stim_params['stim_dur']     = np.array([stim_dur, stim_dur+delay])
stim_params['t_tot']        = t0+delay+stim_dur+300
stim_params['concs']        = np.array([peak, peak*peak_ratio])
plume_params['stim_seed']   = np.nan


# Output parameters
figs_save                   = 0
fld_analysis                = 'NSI_analysis/real_plumes_example/'

time2analyse        = stim_dur+300
n_pns_recep         = al_params['n_pns_recep'] # number of PNs per each glomerulus
n_orns_recep        = orn_layer_params[0]['n_orns_recep']   # number of ORNs per each glomerulus

# Initialize output variables
pn_peak_w   = np.zeros((1,1))
pn_avg_w    = np.zeros((1,1))
pn_peak_s   = np.zeros((1,1))
pn_avg_s    = np.zeros((1,1))

orn_peak_w  = np.zeros((1,1))
orn_avg_w   = np.zeros((1,1))
orn_peak_s  = np.zeros((1,1))
orn_avg_s   = np.zeros((1,1))



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
    

tic = tictoc()

output_orn = ORNs_layer_dyn.main(params_al_orn)
[t, u_od,  orn_spikes_t, orn_sdf,orn_sdf_time] = output_orn 

# AL dynamics
output_al = AL_dyn.main(params_al_orn, orn_spikes_t)
[t, pn_spike_matrix, pn_sdf, pn_sdf_time,
              ln_spike_matrix, ln_sdf, ln_sdf_time,] = output_al

fig_al_orn = plot_al_orn.main(params_al_orn, output_orn, output_al)

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

    
print('conc ratio: %d'%peak_ratio)
print('nu PN strong avg:')        
print(pn_avg_s)
print('nu PN weak avg:')        
print(pn_avg_w)
print('nu PN ratio avg:')        
print(pn_avg_s/pn_avg_w)
print('')
  
print('nu ORN strong avg:')        
print(orn_avg_s)
print('nu ORN weak avg:')        
print(orn_avg_w)
print('nu ORN ratio avg:')        
print(orn_avg_s/orn_avg_w)
print('')

toc = tictoc()
print('Sim + plot time: %.2f' %(toc-tic))


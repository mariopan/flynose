#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:26:53 2021

flynose2.py
flynose 2.0: Runs a singl trial of AL and ORNs layer dynamics and 
make the standard plot

@author: mario
"""

# Setting parameters and define functions

import numpy as np
import timeit

import AL_dyn
import ORNs_layer_dyn
import figure_al_orn

import pickle        

def tictoc():
    return timeit.default_timer()

# LOAD standard PARAMS FROM A FILE
name_data = 'AL_ORN_rate.pickle'            
fld_analysis = 'NSI_analysis/trials/' #Olsen2010
file_params = 'params_al_orn.ini'
params_al_orn = pickle.load(open(fld_analysis+ file_params,  "rb" ))

stim_params = params_al_orn['stim_params']
orn_layer_params= params_al_orn['orn_layer_params']
orn_params= params_al_orn['orn_params']
sdf_params= params_al_orn['sdf_params']
al_params= params_al_orn['al_params']
pn_ln_params= params_al_orn['pn_ln_params']

n_sens_type       = orn_layer_params.__len__()  # number of type of sensilla

# update params of interest
stim_params['stim_type']    = 'ss' # 'ss'  # 'ts' # 'rs' # 'pl'
stim_params['stim_dur']     = np.array([500, 500])      # ms
stim_params['t_tot']        = 2000
stim_params['t_on']         = np.array([700, 700])      # ms
stim_params['conc0']        = 1.85e-4    # 2.85e-4
stim_params['concs']        = np.array([1e-2, 1.85e-4])

for sst in range(n_sens_type):
    orn_layer_params[sst]['w_nsi']  = .8 # typical values (0, 0.8)
pn_ln_params['alpha_ln']        = 0      # typical values (0, 500)


# ORNs layer dynamics
tic = tictoc()
output_orn = ORNs_layer_dyn.main(params_al_orn)
[t, u_od,  orn_spikes_t, orn_sdf,orn_sdf_time] = output_orn 

# AL dynamics
output_al = AL_dyn.main(params_al_orn, orn_spikes_t)
[t, pn_spike_matrix, pn_sdf, pn_sdf_time,
              ln_spike_matrix, ln_sdf, ln_sdf_time,] = output_al
toc = tictoc()
print('time to run a flynose sims: %.1fs'%(toc-tic))

figure_al_orn.main(params_al_orn, output_orn, output_al)

# Calculate the mean and max for PNs and LNs                
t2avg0              = 50
orn_avg = np.max(orn_sdf[t2avg0:,:])
pn_avg = np.max(pn_sdf[t2avg0:,:])
ln_avg = np.max(ln_sdf[t2avg0:,:])
print('FR max ORNs: %.2f Hz' %orn_avg)
print('FR max PNs: %.2f Hz' %pn_avg)
print('FR max LNs: %.2f Hz' %ln_avg)
orn_avg = np.mean(orn_sdf[t2avg0:,:])
pn_avg = np.mean(pn_sdf[t2avg0:,:])
ln_avg = np.mean(ln_sdf[t2avg0:,:])
print('FR mean ORNs: %.2f Hz' %orn_avg)
print('FR mean PNs: %.2f Hz' %pn_avg)
print('FR mean LNs: %.2f Hz' %ln_avg)

output2an = dict([
            ('t', t),
            ('u_od',u_od),
            ('orn_sdf', orn_sdf),
            ('orn_sdf_time',orn_sdf_time), 
            ('pn_sdf', pn_sdf),
            ('pn_sdf_time', pn_sdf_time), 
            ('ln_sdf', ln_sdf),
            ('ln_sdf_time', ln_sdf_time), 
            ])

with open(fld_analysis+name_data, 'wb') as f:
    pickle.dump([params_al_orn, output2an], f)
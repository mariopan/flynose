#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 15:39:06 2021

@author: mario
"""

import numpy as np
# import matplotlib.pyplot as plt
import timeit

import ORNs_layer_dyn
import set_orn_al_params

import NSI_ORN_LIF 
import plot_orn


# tic toc
def tictoc():
    return timeit.default_timer()         

params_al_orn = set_orn_al_params.main(2)

stim_params         = params_al_orn['stim_params']
orn_layer_params    = params_al_orn['orn_layer_params']
orn_params          = params_al_orn['orn_params']
sdf_params          = params_al_orn['sdf_params']
al_params           = params_al_orn['al_params']
pn_ln_params        = params_al_orn['pn_ln_params']
plume_params        = stim_params['plume_params']
sens_params         = orn_layer_params[0]

n_sens_type         = orn_layer_params.__len__()  # number of type of sensilla

# ORN NSI params

# fig_id options:  # 'ts_s' #  'ts_a' # 'pl'
fig_id                  = 'ss_s' 

fld_analysis            = 'NSI_analysis/triangle_stim/'
nsi_str                 = 0.6
alpha_ln                = 0.6
pn_ln_params['tau_ln']  = 250

    
# figure and output options
fig_save    = 0
data_save   = 0    
verbose     = 0
olsen_fig   = 0

# stim params
delay                       = 0    
stim_params['stim_type']    = 'ss' 
stim_params['stim_dur']     = np.array([500, 500])
stim_params['t_tot']        = 2000
t_on                        = 1000
stim_params['conc0']        = 1.85e-4    # 2.85e-4

peak                        = 4e-2 #[1.85e-4, 5e-4, 1.5e-3, 2e-2, 2e-1]#[0.001, 0.005] #
peak_ratio                  = .1
   

sdf_params['dt_sdf'] = 5
sdf_params['tau_sdf'] = 40

stim_params['concs'] = np.array([peak, peak*peak_ratio])

inh_cond = 'nsi'
nsi_str = 0.6
alpha_ln = 0.6

if inh_cond == 'nsi':
    sens_params['w_nsi']  = nsi_str    
    pn_ln_params['alpha_ln']        = 0
elif inh_cond == 'noin':
    sens_params['w_nsi']  = 0
    pn_ln_params['alpha_ln']        = 0
elif inh_cond == 'ln':
    sens_params['w_nsi']  = 0    
    pn_ln_params['alpha_ln']        = alpha_ln
    
    

# ORNs layer dynamics
params_1sens   = dict([
                ('stim_params', stim_params),
                ('sens_params', sens_params),
                ('orn_params', orn_params),
                ('sdf_params', sdf_params),
                ])

# ORN LIF SIMULATION
tic = timeit.default_timer()
output_orn = NSI_ORN_LIF.main(params_1sens)

[t, u_od, r_orn, v_orn, y_orn, 
                   num_spikes, spike_matrix, orn_sdf, t_sdf,]   = output_orn    
    
    
fig = plot_orn.main(params_1sens, output_orn, )
fld_analysis = 'NSI_analysis/offset_peak/'
fig_name = 'fast_odorant.png'
fig.savefig(fld_analysis + fig_name)
    
toc = timeit.default_timer()

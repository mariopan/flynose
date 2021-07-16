#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 10:17:26 2021

test_stim_fcn.py

@author: mario
"""

import numpy as np
import matplotlib.pyplot as plt

import stim_fcn_oop
import stim_fcn
import set_orn_al_params


params_al_orn   = set_orn_al_params.main(1)

stim_params     = params_al_orn['stim_params']

stim_params['conc0'] = 1.85e-4
stim_params['t_tot']  = 2000
stim_params['t_on']  = np.array([1000, 1000])

stim_params['stim_type'] = 'rs'
stim_params['stim_seed'] = 10


verbose = True

# GENERATE ODOUR STIMULUS/I and UPDATE STIM PARAMS
u_od            = stim_fcn.main(stim_params, verbose=True)

print(u_od[:10, :])



verbose_dict = {
        'ss' : 'u_od is single step',
        'ts' :  'u_od is triangular pulse', 
        'pl' :  'u_od is extracted from real plumes',
        'ext' : 'ext stimuli, from Kim et al. 2011',
        'rs' : 'u_od is constant',
        }
    
plume_dict = {
    'ss' : stim_fcn_oop.PulseStep,
    'ts' : stim_fcn_oop.PulseTriangular, 
    'pl' : stim_fcn_oop.RealPlume,
    'ext' : stim_fcn_oop.ExtPlume,
    'rs' : stim_fcn_oop.SimPlume,
    } 

plume_type = stim_params['stim_type']
plume = plume_dict[plume_type](stim_params)

if verbose: 
    print(verbose_dict[plume_type])
    
    
# stim_obj = stim_fcn_oop.main(stim_params, verbose=True)

rs = 1
cs = 2
fig_od, ax_od = plt.subplots(rs, cs, figsize=[8.5, 9])
ax_od[0].plot(plume.u_od)
ax_od[1].plot(u_od)
plt.show()

print(plume.u_od[:10, :])

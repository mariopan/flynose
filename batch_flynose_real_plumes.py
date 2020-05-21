#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:08:11 2019

script name: batch_flynose_real_plumes.py
@author: mp525
"""
import flynose

import timeit
from os import path
from os import mkdir
from shutil import copyfile
import sys


stim_seed_start = int(sys.argv[1])
#%%

import numpy as np
import datetime
now = datetime.datetime.now()

inh_conds = ['nsi', 'ln', 'noin']
stim_type = 'pl' 
stim_dur  = 201000#0
n_seeds   = 1
    
w_maxs  = [.01,.03,.3, 3, 25, 50, ] # [3,50, 150] # max value in the whiff distribution
b_maxs  = [25] # [25] max value in the blank distribution
rhos    = [0, 1, 3, 5]
peak    = 1.5
peak_ratio = 1
delay   = 0 
nsi_value = 0.3
alpha_ln  = 13.3
                    
orn_fig     = 0
al_fig      = 0
fig_ui      = 0        
fig_save    = 0
data_save   = 1
fig_opts = [orn_fig, al_fig, fig_ui, fig_save, data_save]

sims_to_run = n_seeds*np.size(inh_conds)*np.size(w_maxs)*np.size(b_maxs)*np.size(rhos)
print('Number of Simulations to run: %d '%sims_to_run)

Tot_sim_time = 3.8*sims_to_run*stim_dur/1000/60/60  # hours
run_sim_time = 0
print('Estimated Simulation time: %.3g hours (%.2g mins)'%(Tot_sim_time, Tot_sim_time*60))
endsim = now+datetime.timedelta(hours=Tot_sim_time)
print(endsim)
#%%

#fld_analysis = '/'#'NSI_analysis/analysis_real_plumes/stim_60s_long_w/'
#if path.isdir(fld_analysis):
#    print('OLD analysis fld: ' + fld_analysis)    
#else:
#    print('NEW analysis fld: ' + fld_analysis)    
#    mkdir(fld_analysis)
date_str = now.strftime("%Y%m%d")
copyfile('flynose.py', 'flynose.' + date_str + '.py') 
copyfile('batch_flynose_real_plumes.py', 'batch_flynose_real_plumes.' + date_str + '.py') 
    
for stim_seed in np.arange(stim_seed_start, stim_seed_start+n_seeds):
    fld_analysis_tmp = 'real_plumes_%d'%stim_seed
    if path.isdir(fld_analysis_tmp):
        print('OLD analysis fld: ' + fld_analysis_tmp)    
    else:
        print('NEW analysis fld: ' + fld_analysis_tmp)    
        mkdir(fld_analysis_tmp)

    for b_max in b_maxs:
        for w_max in w_maxs:
            for rho in rhos:
                params2an = [.0, .0, stim_dur, delay, peak, peak_ratio, rho, stim_type,w_max,b_max]
                for inh_cond in inh_conds:
                    if inh_cond == 'nsi':
                        params2an[0:2] = [nsi_value, 0]
                    elif inh_cond == 'noin':
                        params2an[0:2] = [0, 0]
                    elif inh_cond == 'ln':
                        params2an[0:2] = [0, alpha_ln]
                        
#                    plt.ioff()      # ion() # to avoid showing the plot every time     
                    tic = timeit.default_timer()
                    [_, _, _, orn_stim, pn_stim,] = flynose.main(params2an, fig_opts, 
                                verbose = False, fld_analysis = fld_analysis_tmp, 
                                stim_seed=stim_seed)

#                    plt.close()
                    toc = timeit.default_timer()
                    sims_to_run = sims_to_run - 1
                    print('Remaining Simulations to run: %d '%(sims_to_run))
                    print('Approx Remaining time: %.0f mins'%(sims_to_run*(toc-tic)/60))
    
    print('Tot sim time:%.0f mins'%(run_sim_time))
	    

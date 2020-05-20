#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:08:11 2019

script name: batch_flynose_ratio.py
@author: mp525
"""
import timeit

import sys
import pickle        
from os import path
from os import mkdir
from shutil import copyfile
import matplotlib.pyplot as plt

import flynose
delays2an   = [0,0, 10, 20, 50, 100, 200, 500,]
delay_id    = int(sys.argv[1])
delay       = delays2an[delay_id]
#%%

import numpy as np
import datetime
now = datetime.datetime.now()

n_loops         =  50
n_peak_ratio    =  10
n_peak          =  4
inh_conds       = ['nsi', 'ln', 'noin'] #
dur2an          = [10, 20, 50, 100, 200]
n_delays        = np.size(dur2an)

sims_to_run = np.size(inh_conds)*n_loops*n_peak*n_peak_ratio*n_delays
print('Number of Simulations to run: %d '%sims_to_run)

# approximately 3 secs per run:
Tot_sim_time = sims_to_run*3/60  # mins
run_sim_time = 0
print('Estimated Simulation time: %.2f mins:'%Tot_sim_time)
endsim = now+datetime.timedelta(minutes=Tot_sim_time)
print(endsim)
#%%

ratio_peak  = np.zeros((n_peak_ratio, n_peak, n_loops))
peak_1      = np.zeros((n_peak_ratio, n_peak, n_loops))

peak_orn1   = np.zeros((n_peak_ratio, n_peak, n_loops))
peak_orn2   = np.zeros((n_peak_ratio, n_peak, n_loops))
peak_pn1    = np.zeros((n_peak_ratio, n_peak, n_loops))
peak_pn2    = np.zeros((n_peak_ratio, n_peak, n_loops))
avg_orn1   = np.zeros((n_peak_ratio, n_peak, n_loops))
avg_orn2   = np.zeros((n_peak_ratio, n_peak, n_loops))
avg_pn1    = np.zeros((n_peak_ratio, n_peak, n_loops))
avg_pn2    = np.zeros((n_peak_ratio, n_peak, n_loops))

   
date_str = now.strftime("%Y%m%d")
copyfile('flynose.py', 'flynose.' + date_str + '.py') 
copyfile('batch_flynose_ratio.py', 'batch_flynose_ratio.' + date_str + '.py') 

orn_fig     = 0
al_fig      = 0
fig_ui      = 0        
fig_save    = 0
data_save   = 0
stim_type   = 'ts'  # 'ts'
b_max       = np.nan # 3
w_max       = np.nan # 3, 50, 150
rho         = np.nan #[0, 1, 3, 5]: 

fig_opts = [orn_fig, al_fig, fig_ui, fig_save, data_save]

for stim_dur in dur2an:
    fld_analysis_tmp = 'ratio_stim_dur_%d_delay_%d'%(stim_dur,delay)
    if path.isdir(fld_analysis_tmp):
        print('OLD analysis fld: ' + fld_analysis_tmp)    
    else:
        print('NEW analysis fld: ' + fld_analysis_tmp)    
        mkdir(fld_analysis_tmp)
    for inh_c in inh_conds:
                
        if inh_c =='nsi':
            nsi_value = .3
            alpha_ln = 0
            data_name = 'NSI_ratio_an'
        elif inh_c =='ln':
            nsi_value = 0
            alpha_ln  = 6#13.3
            data_name = 'LN_ratio_an'
        else:
            nsi_value = 0
            alpha_ln = 0                    
            data_name = 'NoIn_ratio_an'
                            
        peak_id = 0
        for peak in np.linspace(.2, 1.4, n_peak):            
            peak_ratio_id = 0
            for peak_ratio in np.linspace(1, 20, n_peak_ratio):
                for ii in range(n_loops):
                    params2an = [nsi_value, alpha_ln, stim_dur, delay, peak, 
                             peak_ratio, rho, stim_type,w_max,b_max]
                    
                    plt.ioff() # plt.ion() # to avoid showing the plot every time     
                    
                    tic = timeit.default_timer()
                    [orn_stim, pn_stim,] = flynose.main(params2an, fig_opts, 
                            verbose = False, fld_analysis = fld_analysis_tmp, 
                            stim_seed=(ii+57))
                    toc = timeit.default_timer()
                    eff_dur = (toc-tic)

                    ratio_peak[peak_ratio_id, peak_id, ii] = peak_ratio
                    peak_1[peak_ratio_id, peak_id, ii] = peak
#                            orn_stim = [orn_avg1, orn_avg2, orn_peak1, orn_peak2, ]
#                            pn_stim = [pn_avg1, pn_avg2, pn_peak1, pn_peak2,]
                    avg_orn1[peak_ratio_id, peak_id, ii] = orn_stim[0]
                    avg_orn2[peak_ratio_id, peak_id, ii] = orn_stim[1]
                    avg_pn1[peak_ratio_id, peak_id, ii] = pn_stim[0]
                    avg_pn2[peak_ratio_id, peak_id, ii] = pn_stim[1]
                    
                    peak_orn1[peak_ratio_id, peak_id, ii] = orn_stim[2]
                    peak_orn2[peak_ratio_id, peak_id, ii] = orn_stim[3]
                    peak_pn1[peak_ratio_id, peak_id, ii] = pn_stim[2]
                    peak_pn2[peak_ratio_id, peak_id, ii] = pn_stim[3]
                    
                    sims_to_run = sims_to_run - 1
                    print('Remaining Simulations to run: %d '%(sims_to_run))
                    print('Approx Remaining time: %.0f mins'%(sims_to_run*eff_dur/60))
                peak_ratio_id = peak_ratio_id + 1                
            peak_id = peak_id + 1
        print(data_name)
        print('parameters: ')
        print(params2an)
        
        with open(fld_analysis_tmp+'/' + data_name + '.pickle', 'wb') as f:
            pickle.dump([params2an, peak_1, ratio_peak, 
                         avg_orn1, avg_orn2, avg_pn1, avg_pn2,
                         peak_orn1, peak_orn2, peak_pn1, peak_pn2], f)
        

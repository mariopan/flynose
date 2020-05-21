#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:48:07 2020

flynose_examples.py

@author: mario
"""



import numpy as np
import matplotlib.pyplot as plt
import timeit

# import pickle        
from os import path
from os import mkdir
from shutil import copyfile

import flynose



stim_data_fld = ''
for stim_seed in [0]:
    
#    #***********************************************
#    # Trials and errors
#    fld_analysis    = 'NSI_analysis/trialserrors'
#    inh_conds       = ['nsi', ] #'ln', 'noin'
#    stim_type       = 'ss' # 'ts'  # 'ts' # 'ss'
#    stim_dur        = 100
#    alpha_ln        = 13.3  #    ln_spike_h      = 0.4
#    nsi_str         = 0.3
#    delays2an       = 0
#    peak_ratio      = 1
#    peaks           = [1,] 
#    orn_fig         = 1
#    al_fig          = 1
#    fig_ui          = 1        
#    fig_save        = 0
#    data_save       = 1    

#    #***********************************************
#    # FIG. ORN_response
#    fld_analysis = 'NSI_analysis/ORN_dynamics' #/sdf_test
#    inh_conds   = ['noin'] #
#    stim_type   = 'ss' # 'ts' # 'ss' # 'rp'# '
#    alpha_ln        = 13.3  ##    ln_spike_h  = 0.4
#    nsi_str     = 0.3
#    stim_dur    = 500
#    delays2an   = 0
#    peaks       = [0.8]
#    peak_ratio  = 1
#    orn_fig     = 1
#    al_fig      = 0
#    fig_ui      = 1
#    fig_save    = 0
#    data_save       = 1    
        
#    #***********************************************
#    # Lazar and Kim data reproduction
#    fld_analysis    = 'NSI_analysis/lazar_sim/'
#    inh_conds       = ['nsi', ] #'ln', 'noin'
#    ext_stimulus    = True
#    stim_type       = 'ramp_3' # 'step_3' 'parabola_3' 'ramp_3'
#    stim_data_fld   = 'lazar_data_hr/'
#    stim_dur        = 1000
#    alpha_ln        = 13.3  ##    ln_spike_h      = 0.4
#    nsi_str         = 0.3
#    delays2an       = 0
#    peak_ratio      = 1
#    peaks           = [1,] 
#    orn_fig         = 1
#    al_fig          = 0
#    fig_ui          = 1        
#    fig_save        = 0
#    data_save       = 1    

#    #***********************************************
#    # Olsen-Wilson 2010 figure
#    fld_analysis = 'Olsen2010_Martelli2013/data'
#    alpha_ln        = 13.3  #ln_spike_h  = 0.4
#    nsi_str     = 0.3
#    inh_conds   = ['noin'] #['nsi', 'ln', 'noin'] #
#    stim_type   = 'ss' # 'ss'  # 'ts'
#    stim_dur    = 500
#    delays2an   = 0
#    peak_ratio  = 1
#    peaks       = np.linspace(0,7,11)
#    orn_fig     = 0
#    al_fig      = 0
#    fig_ui      = 0      
#    fig_save    = 0
#    data_save   = 0    
    
    #***********************************************
    # Fig.ImpulseResponse
    fld_analysis = 'NSI_analysis/triangle_stim/ImpulseResponse'
    inh_conds   = ['nsi', 'ln', 'noin'] #
    stim_type   = 'ts'  # 'ts'
    stim_dur    = 50
    alpha_ln    = 13.3  #ln_spike_h  = 0.6
    nsi_str     = 0.3
    delays2an   = 0
    peak_ratio  = 1
    peaks       = [1.4,] 
    orn_fig     = 0
    al_fig      = 1
    fig_ui      = 1
    fig_save    = 0
    data_save   = 0    
    
#    #***********************************************        
#    # FIG. DelayResponse
#    fld_analysis = 'NSI_analysis/triangle_stim/triangles_delay' #
#    inh_conds = ['nsi', 'ln', 'noin'] 
#    stim_type   = 'ts' 
#    alpha_ln    = 13.3  ##    ln_spike_h  = 0.4
#    nsi_str     = 0.3
#    stim_dur    = 50  # 10 20 50 100 200 
#    delays2an   = 100 
#    peaks       = [1.8]
#    peak_ratio  = 1
#    orn_fig     = 0
#    al_fig      = 1
#    fig_ui      = 1
#    fig_save    = 0
#    data_save   = 0    
        
#    #***********************************************
#    # Real plumes, example figure
#    fld_analysis = 'NSI_analysis/analysis_real_plumes/example'
#    inh_conds   = ['nsi', ] #'ln', 'noin'
#    stim_type   = 'pl'  # 'ts' # 'ss'
#    stim_dur    = 4000
#    alpha_ln    = 13.3  #    ln_spike_h  = 0.4
#    nsi_str     = 0.3
#    delays2an   = 0
#    peak_ratio  = 1
#    peaks       = [1.5,] 
#    orn_fig     = 0
#    al_fig      = 1
#    fig_ui      = 1        
#    fig_save    = 0    
#    data_save   = 1    
    
    
    
    fig_opts = [orn_fig, al_fig, fig_ui, fig_save, data_save]
    if path.isdir(fld_analysis):
        print('OLD analysis fld: ' + fld_analysis)    
    else:
        print('NEW analysis fld: ' + fld_analysis)    
        mkdir(fld_analysis)
    copyfile('flynose.py', fld_analysis+'/flynose.py') 
    
    n_loops     = 1
    pn_avg_dif  = np.zeros(n_loops)
    pn_avg      = np.zeros(n_loops)
    pn_peak     = np.zeros(n_loops)
    
    for peak in peaks:
        print('conc: %.1f, stim_dur:%dms, spike LN: %.1f, NSI strength: %.1f'
          %(peak, stim_dur,alpha_ln,nsi_str))
        for b_max in [3]: # 3, 50, 150
            for w_max in [3]: # 3, 50, 150
                for rho in [0]: #[0, 1, 3, 5]: 
                    params2an = [0, .0, stim_dur, delays2an, peak, 
                                 peak_ratio, rho, stim_type,w_max,b_max]
                    if len(stim_type)>2:
                        params2an.append(stim_data_fld)
                    tic = timeit.default_timer()
                    for inh_cond in inh_conds:
                        if inh_cond == 'nsi':
                            params2an[0:2] = [nsi_str, .0, ]
                        elif inh_cond == 'noin':
                            params2an[0:2] = [0, 0, ]
                        elif inh_cond == 'ln':
                            params2an[0:2] = [.0, alpha_ln,]
                        
                        #    params2an = [nsi_value, ln_spike_height, dur2an, delays2an, peak, peak_ratio]
                        plt.ion()      # ion() # to avoid showing the plot every time     
                        for id_loop in range(n_loops):
                            [_, _, _, orn_stim, pn_stim,] = flynose.main(params2an, fig_opts, 
                                verbose = False, fld_analysis = fld_analysis, 
                                stim_seed=stim_seed)
                            pn_avg_dif[id_loop] = (pn_stim[0]-pn_stim[1])
                            pn_avg[id_loop] = (pn_stim[0]+pn_stim[1])/2
                            pn_peak[id_loop] = (pn_stim[2]+pn_stim[3])/2        
                        
                        print(inh_cond+' inh, peak:%.1f, avg:%.1f, avg dif:%.1f'%(np.mean(pn_peak), np.mean(pn_avg), np.mean(pn_avg_dif)))
                        toc = timeit.default_timer()
                    print('time to run %d sims: %.1fs'%(np.size(inh_conds),toc-tic))
                    print('')

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

import pickle        
from os import path
from os import mkdir
from shutil import copyfile

#import flynose_old
import flynose
import sdf_krofczik



stim_data_fld = ''

#***********************************************
# Standard params

# ORN NSI params
inh_conds = ['nsi', ] #'ln', 'noin'
alpha_ln        = 13.3# ln spike h=0.4
nsi_str         = 0.3
# output params 
pts_ms      = 5
t_tot       = 1200        # ms 
# real plumes params
b_max       = np.nan # 3, 50, 150
w_max       = np.nan #3, 50, 150
rho         = np.nan #[0, 1, 3, 5]: 
stim_seed   = np.nan   # if =np.nan() no traceable random
# figure and output options
verbose     = 0    
tau_sdf     = 20
dt_sdf      = 5      


# #***********************************************
# # FIG. ORN_response
# fld_analysis = 'NSI_analysis/ORN_dynamics' #/sdf_test
# inh_conds   = ['noin'] 
# stim_type   = 'ss' # 'ts' # 'ss' # 'rp'# '
# stim_dur    = 500
# delay       = 0
# peaks       = [0.8]
# peak_ratio  = 1
# t_tot       = 1200        # ms 
# orn_fig     = 1
# al_fig      = 0
# fig_ui      = 1
# fig_save    = 1
# data_save   = 0  
# al_dyn      = 0
        
#***********************************************
# Lazar and Kim data reproduction
fld_analysis    = 'NSI_analysis/lazar_sim/'
inh_conds       = ['nsi', ] #'ln', 'noin'
ext_stimulus    = True
stim_type       = 'ramp_1' # 'step_3' 'parabola_3' 'ramp_3'
stim_data_fld   = 'lazar_data_hr/'

stim_dur        = np.nan
delay           = np.nan
peak_ratio      = np.nan
peaks           = [1,] 
al_dyn          = 0
orn_fig         = 0
al_fig          = 0
fig_ui          = 1        
fig_save        = 0
data_save       = 1    
t_tot       = 3500 # ms 
tau_sdf     = 60
dt_sdf      = 5      

##***********************************************
## Olsen-Wilson 2010 figure
#fld_analysis = 'Olsen2010_Martelli2013/data'
#inh_conds   = ['noin'] #['nsi', 'ln', 'noin'] #
#stim_type   = 'ss' # 'ss'  # 'ts'
#stim_dur    = 500
#delay       = 0
#peak_ratio  = 1
#peaks       = np.linspace(0,7,11)
#al_dyn      = 1
#orn_fig     = 0
#al_fig      = 0
#fig_ui      = 0      
#fig_save    = 0
#data_save   = 1    
    
##***********************************************
## Fig.ImpulseResponse
#fld_analysis = 'NSI_analysis/triangle_stim/ImpulseResponse'
#inh_conds   = ['nsi',]# 'ln', 'noin'] #
#stim_type   = 'ts'  # 'ts'
#stim_dur    = 50
#delay       = 0
#peak_ratio  = 1
#peaks       = [1.4,] 
#al_dyn      = 1
#orn_fig     = 0
#al_fig      = 1
#fig_ui      = 1
#fig_save    = 0
#data_save   = 0    
    
##***********************************************        
## FIG. DelayResponse
#fld_analysis = 'NSI_analysis/triangle_stim/triangles_delay' #
#inh_conds = ['nsi', 'ln', 'noin'] 
#stim_type   = 'ts' 
#stim_dur    = 50  # 10 20 50 100 200 
#delay       = 100 
#peaks       = [1.8]
#peak_ratio  = 1
#orn_fig     = 0
#al_dyn      = 1
#al_fig      = 1
#fig_ui      = 1
#fig_save    = 0
#data_save   = 0    

##***********************************************
## Real plumes, example figure
#fld_analysis = 'NSI_analysis/analysis_real_plumes/example'
#inh_conds   = ['nsi', ] #'ln', 'noin'
#stim_type   = 'pl'  # 'ts' # 'ss'
#t_tot       = 5000
#stim_dur    = 4000
#delay       = 0
#peak_ratio  = 1
#peaks       = [1.5,] 
## real plumes params
#b_max       = 25 #, 50, 150
#w_max       = 3  #, 50, 150
#rho         = 1 #np.nan # 0, 1, 3, 5: 
#stim_seed   = 0   # if =np.nan() no traceable random
##fig opts
#orn_fig     = 0
#al_dyn      = 1
#al_fig      = 1
#fig_ui      = 1        
#fig_save    = 0    
#data_save   = 0    

   
# Standard params
t_on        = [300, 300+delay]    # ms
t_off       = np.array(t_on)+stim_dur # ms
concs       = [0, 0]
sdf_size    = int(t_tot/dt_sdf)       

for stim_seed in range(1):
    plume_params = [rho, w_max, b_max, stim_seed]
    fig_opts = [orn_fig, al_fig, fig_ui, fig_save, data_save, al_dyn, 
                verbose, fld_analysis] 

    if path.isdir(fld_analysis):
        print('OLD analysis fld: ' + fld_analysis)    
    else:
        print('NEW analysis fld: ' + fld_analysis)    
        mkdir(fld_analysis)
        
    n_loops     = 1
    pn_avg_dif  = np.zeros(n_loops)
    
    pn_avg_ratio    = np.zeros(n_loops)
    pn_peak_ratio   = np.zeros(n_loops)
    
    for peak in peaks:
        concs = [peak, peak*peak_ratio]
        stim_params = [stim_type, pts_ms, t_tot, t_on, t_off, 
                       concs, plume_params]
    
        print('conc: %.1f, stim_dur:%.1f ms, spike LN: %.1f, NSI strength: %.1f'
          %(peak, stim_dur,alpha_ln,nsi_str))

        if len(stim_type)>2:
            ext_stimulus = True
            stim_params.append(stim_data_fld)
        else:
            ext_stimulus = False
            
        params2an = [nsi_str, alpha_ln, stim_params,]
        tic = timeit.default_timer()
        for inh_cond in inh_conds:
            if inh_cond == 'nsi':
                params2an[0:2] = [nsi_str, .0, ]
            elif inh_cond == 'noin':
                params2an[0:2] = [0, 0, ]
            elif inh_cond == 'ln':
                params2an[0:2] = [.0, alpha_ln,]
            
            # SAVE params of the simulations
            with open(fld_analysis+'/params_'+inh_cond+'_peak_%.2f'%(peak)+
                      '.pickle', 'wb') as f:
                pickle.dump([params2an, fig_opts,], f)
            
            plt.ion()      # ion() # to avoid showing the plot every time     
            for id_loop in range(n_loops):
                flynose_out = flynose.main(params2an, fig_opts)
                [t, u_od, orn_spike_matrix, pn_spike_matrix, 
                 ln_spike_matrix, ] = flynose_out
                
                # Calculate the SDF for PNs and LNs
                orn_sdf_norm, orn_sdf_time = sdf_krofczik.main(orn_spike_matrix, 
                                                sdf_size, tau_sdf, dt_sdf)  # (Hz, ms)
                orn_sdf_norm = orn_sdf_norm*1e3
                
                # SAVE SDF OF ORN FIRING RATE
                if data_save & (al_dyn==0):
                    if ext_stimulus:
                        name_data = '/ORNrate' +\
                                '_stim_' + stim_type +\
                                '_nsi_%.1f'%(params2an[0]) +\
                                '_ln_%.2f'%(params2an[1]) +\
                                '.pickle'
                    else:
                        name_data = '/ORNrate' +\
                                '_stim_' + stim_params[0]+\
                                '_nsi_%.1f'%(params2an[0]) +\
                                '_ln_%.2f'%(params2an[1]) +\
                                '_dur2an_%d'%(stim_params[4][0]-stim_params[3][0]) +\
                                '_delay2an_%d'%(stim_params[3][1]-stim_params[3][0]) +\
                                '_peak_%.1f'%(peak) +\
                                '_peakratio_%.1f'%(peak_ratio) +\
                                '.pickle'

                        
                    output_names = ['t', 'u_od', 'orn_sdf_norm', 'orn_sdf_time',
                                    ]        
                    
                    params2an_names = ['omega_nsi', 'alpha_ln', 'dur2an', 'delays2an', 
                                       'peak', 'peak_ratio', 'rho', 'stim_type', ]
            
                    with open(fld_analysis+name_data, 'wb') as f:
                        pickle.dump([params2an, t, u_od, orn_sdf_norm, orn_sdf_time, 
                             params2an_names, output_names], f)
                if al_dyn:
                    
                    # *************************************************************************
                    # COLLECT AND SAVE DATA
                    
                    # Calculate the SDF for PNs and LNs
                    pn_sdf_norm, pn_sdf_time = sdf_krofczik.main(pn_spike_matrix, sdf_size,
                                                     tau_sdf, dt_sdf)  # (Hz, ms)
                    pn_sdf_norm = pn_sdf_norm*1e3
                
                    ln_sdf_norm, ln_sdf_time = sdf_krofczik.main(ln_spike_matrix, sdf_size,
                                                     tau_sdf, dt_sdf)  # (Hz, ms)
                    ln_sdf_norm = ln_sdf_norm*1e3
                  
                    # Calculate the mean and the peak for PN responses
                    num_pns_glo         = 5     # number of PNs per each glomerulus
                    id_stim_w = np.flatnonzero((pn_sdf_time>t_on[0]) & (pn_sdf_time<t_off[0]))
                    id_stim_s = np.flatnonzero((pn_sdf_time>t_on[1]) & (pn_sdf_time<t_off[1]))
                    pn_sdf_dt = pn_sdf_time[1]-pn_sdf_time[0]
    
                     
                    pn_peak_w  = np.max(np.mean(pn_sdf_norm[id_stim_w, :num_pns_glo], axis=1)) # using average PN
                    pn_peak_s  = np.max(np.mean(pn_sdf_norm[id_stim_s, num_pns_glo:], axis=1)) # using average PN
                    pn_avg_w  = np.mean(pn_sdf_norm[id_stim_w, :num_pns_glo])
                    pn_avg_s  = np.mean(pn_sdf_norm[id_stim_s, num_pns_glo:])
                    # Calculate the ratio for PN responses
                    pn_avg_dif[id_loop] = pn_avg_w-pn_avg_s
                    pn_avg_ratio[id_loop] = pn_avg_s/pn_avg_w
                    pn_peak_ratio[id_loop]  = pn_peak_s/pn_peak_w
                    
                    print('peak ratio:%.1f, conc weak:%.1f Hz, conc strong:'\
                        '%.1f Hz'%(pn_peak_ratio[id_loop], pn_peak_w, pn_peak_s))
                    
                    # ************************************************
                    # SAVE SDF OF conc, ORN, PN and LN FIRING RATE
                    if data_save:
                        if ext_stimulus:
                            name_data = ['/ORNALrate' +
                                    '_stim_' + stim_type +\
                                    '_nsi_%.1f'%(params2an[0]) +\
                                    '_ln_%.2f'%(params2an[1]) +\
                                    '.pickle']
                        else:
                            name_data = ['/ORNALrate' +
                                    '_stim_' + stim_type +\
                                    '_nsi_%.1f'%(params2an[0]) +\
                                    '_ln_%.2f'%(params2an[1]) +\
                                    '_dur2an_%d'%(stim_dur) +\
                                    '_delay2an_%d'%(delay) +\
                                    '_peak_%.1f'%(peak) +\
                                    '_peakratio_%.1f'%(peak_ratio) +\
                                    '.pickle'] 
                                            
                        output_names = ['t', 'u_od', 'orn_sdf_norm', 'orn_sdf_time', 
                                        'pn_sdf_norm', 'pn_sdf_time', 
                                        'ln_sdf_norm', 'ln_sdf_time', ]
                        
                        params2an_names = ['omega_nsi', 'alpha_ln', 'stim_dur', 'delay', 
                                           'peak', 'peak_ratio', 'rho', 'stim_type', ]
                
                        if not(stim_type == 'pl'):
                            with open(fld_analysis+name_data[0], 'wb') as f:
                                pickle.dump([params2an, t, u_od, 
                                             orn_sdf_norm, orn_sdf_time, 
                                             pn_sdf_norm, pn_sdf_time, 
                                             ln_sdf_norm, ln_sdf_time, 
                                             params2an_names, output_names], f)
                    
            print('peak ratio:%.1f, avg ratio:%.1f, avg dif:%.1f Hz'
                  %(np.mean(np.ma.masked_invalid(pn_peak_ratio)), 
                    np.mean(np.ma.masked_invalid(pn_avg_ratio)), np.mean(pn_avg_dif)))
    
            
            toc = timeit.default_timer()
        print('time to run %d sims: %.1fs'%(np.size(inh_conds),toc-tic))
        print('')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:21:20 2021

ORN_dyn_plot.py

This script run NSI_ORN_LIF.py one or multiple times and saves the data 
the following figures of the NSI paper:
    fig.3 ORN dynamics of all its components (ORN_response)
    fig.3 ORN firing rates for several values of the stimulations (martelli)
    fig.3s ORN dynamics for stimuli a la Lazar (lazar)   

@author: mario
"""



import numpy as np
import matplotlib.pyplot as plt
import timeit
# from scipy import signal

import pickle        
# from os import path
# from os import mkdir
# from shutil import copyfile

import matplotlib as mpl

# import NSI_ORN_LIF
# import sdf_krofczik

# tic toc
def tictoc():
    return timeit.default_timer()         

def martelli_plot(data2an, params2an, id_c):
    
    n_orns_recep = params2an['sens_params']['n_orns_recep']
    
    t           = data2an['t']
    t2simulate  = t[-1]
    u_od        = data2an['u_od']
    orn_sdf     = data2an['orn_sdf']
    orn_sdf_time = data2an['orn_sdf_time']
    
    orn2plot = np.mean(orn_sdf[:,:n_orns_recep], axis=1)
    stim_type= params2an['stim_params']['stim_type']
    
    # SETTINGS
    if len(stim_type) == 2:
        print('normalized to the peak') 
        orn2plot = orn2plot/np.max(orn2plot)
        panel_letters = ['e', 'f']
    else:
        
        stim_type= stim_type[:-1]
        print(stim_type)
        if stim_type== 'parabola_':
            panel_letters = ['e', 'f']
        elif stim_type== 'step_':
            panel_letters = ['a', 'b']
        elif stim_type== 'ramp_':
            panel_letters = ['c', 'd']
    
    
    ax_conc_m.set_xlim(t2plot)
    ax_orn_m.set_xlim(t2plot)

    if id_c==0:
        ax_conc_m.tick_params(axis='both', labelsize=ticks_fs)
        ax_orn_m.tick_params(axis='both', labelsize=ticks_fs)
        
        ax_conc_m.set_xticklabels('')
        
        if (len(stim_type) == 2):
            ax_conc_m.set_ylabel('Input (a.u.)', fontsize=label_fs)
            ax_orn_m.set_ylabel(' Norm. firing rates \n (unitless)', fontsize=label_fs)
            
        if (stim_type== 'step_'):
            ax_conc_m.set_ylabel('Input (a.u.)', fontsize=label_fs)
            ax_orn_m.set_ylabel(' Firing rates \n (unitless)', fontsize=label_fs)
            
        if len(stim_type) > 2:
            ax_conc_m.set_title(stim_type[:-1], fontsize = title_fs+10)
            
            ax_conc_m.set_xticks(np.linspace(0, t2simulate, 8))
            ax_conc_m.set_xticklabels('')
            ax_orn_m.set_xticks(np.linspace(0, t2simulate, 8))
            ax_orn_m.set_xticklabels(['0', '', '', '1500', '', '', '3000'])
            
            ax_conc_m.set_yticks(np.linspace(0,400, 5))
            ax_orn_m.set_yticks(np.linspace(0,250, 6))
            
            ax_conc_m.set_ylim((-5, 400))
            ax_orn_m.set_ylim((0, 250))            
            
            ax_conc_m.grid(True, which='both',lw=1, ls=':')
            ax_orn_m.grid(True, which='both',lw=1, ls=':')
        else:
            ax_conc_m.set_xticks(np.linspace(0, t2simulate, 5))
            ax_orn_m.set_xticks(np.linspace(0, t2simulate, 5))
            
            ax_conc_m.spines['right'].set_color('none')
            ax_conc_m.spines['top'].set_color('none')
            ax_orn_m.spines['right'].set_color('none')
            ax_orn_m.spines['top'].set_color('none')
        ax_orn_m.set_xlabel('Time  (ms)', fontsize=label_fs)

        ll, bb, ww, hh = ax_conc_m.get_position().bounds
        ax_conc_m.set_position([ll+.075, bb, ww, hh])
        ll, bb, ww, hh = ax_orn_m.get_position().bounds
        ax_orn_m.set_position([ll+.075, bb, ww, hh])
    
        ax_conc_m.text(-.15, 1.1, panel_letters[0], transform=ax_conc_m.transAxes,
              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
        ax_orn_m.text(-.15, 1.1, panel_letters[1], transform=ax_orn_m.transAxes, 
              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            
    # PLOT
    id_col = id_c + 3
    ax_conc_m.plot(t-t_on, 100*u_od[:,0], color=greenmap.to_rgba(id_col), linewidth=lw, 
              label='glom : '+'%d'%(1))
    
    ax_orn_m.plot(orn_sdf_time-t_on, orn2plot, 
             color=greenmap.to_rgba(id_col), linewidth=lw-1,label='sdf glo 1')
    
    
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
black   = 'xkcd:black'
blue    = 'xkcd:blue'
red     = 'xkcd:red'
green   = 'xkcd:green'
purple  = 'xkcd:purple'
orange  = 'xkcd:orange'
cmap    = plt.get_cmap('rainbow')



stim_data_fld = ''

fig2plot = 'martelli2013' # 'orn_response' # 'martelli' # 'lazar'
                    

# Martelli 2013 figure

if fig2plot == 'martelli2013':

    fld_analysis = 'NSI_analysis/Martelli2013'

    # stim params
    stim_dur = 500
    stim_type = 'ss'
    
    stim_params = dict([
                    ('stim_type', stim_type),
                    ('stim_dur', [stim_dur, stim_dur]),
                     ])
    sens_params = dict([
                    ('w_nsi', 0), 
                    ])
    delay       = 0
    peak_ratio  = 1
    peaks       = np.linspace(0.05, .5, 10)

    # nsi params
    inh_conds   = ['noin'] #['nsi', 'ln', 'noin'] #
    
    ext_stimulus = 0            # it is equal to 1 only for lazar stimuli
    fig_save    = 0
    data_save   = 0  
    n_loops     = 1
    fig_martelli_name   = '/ORN-Martelli2013_dur_%d'%stim_dur

# FIG. ORN_response
# elif fig2plot == 'orn_response':
    
#     # output params 
#     fld_analysis = 'NSI_analysis/ORN_LIF_dynamics' #/sdf_test
#     orn_fig_name = '/ORN_lif_dyn.png'
#     # stim params
#     stim_params = dict([
#                     ('stim_type', 'ss'),
#                     ('stim_dur', [500, 500]),
#                      ])
#     sens_params = dict([
#                     ('w_nsi', 0), 
#                     ])
                    
#     delay       = 0
#     peaks       = [0.8]         # concentration value for ORN1
#     peak_ratio  = 1             # concentration value for ORN1/ORN2    
    
#     # nsi params
#     inh_conds   = ['noin'] 
    
#     ext_stimulus = 0            # it is equal to 1 only for lazar stimuli
#     fig_save    = 0
#     data_save   = 1  
#     n_loops     = 1
        

# Lazar and Kim data reproduction
elif fig2plot == 'lazar':
    ext_stimulus = 1            # it is equal to 1 only for lazar stimuli
# fld_analysis    = 'NSI_analysis/lazar_sim/'
# inh_conds       = ['nsi', ] #'ln', 'noin'
# ext_stimulus    = True
# stim_type       = 'ramp_1' # 'step_3' 'parabola_3' 'ramp_3'
# stim_data_fld   = 'lazar_data_hr/'

# stim_dur        = np.nan
# delay           = np.nan
# peak_ratio      = np.nan
# peaks           = [1,] 
# al_dyn          = 0
# orn_fig         = 0
# al_fig          = 0
# fig_ui          = 1        
# fig_save        = 0
# data_save       = 1    
# t_tot       = 3500 # ms 
# tau_sdf     = 60
# dt_sdf      = 5      
n_lines     = np.size(peaks)

c = np.arange(1, n_lines + 4)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
greenmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Greens)
purplemap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Purples)
   
# %% PLOT


for stim_seed in range(1):
    
    for inh_cond in inh_conds:
        
        # FIGURE Martelli 2013
        if fig2plot == 'martelli2013':
            t2plot = -200,stim_dur+300
            if len(stim_type) > 2:
                t2plot = 0, 3500
            rs = 2 # number of rows
            cs = 1 # number of cols
            
            fig_pn_m = plt.figure(figsize=[5.71, 8])
            ax_conc_m = plt.subplot(rs, cs, 1)
            ax_orn_m = plt.subplot(rs, cs, 1+cs)
        
        for id_c, peak in enumerate(peaks):
    
            
            if ext_stimulus:
                name_data = '/ORNrate' +\
                    '_stim_' + stim_params['stim_type'] +\
                    '_nsi_%.1f'%(sens_params['w_nsi']) +\
                    '.pickle'
            else:
                name_data = '/ORNrate' +\
                    '_stim_' + stim_params['stim_type'] +\
                    '_nsi_%.1f'%(sens_params['w_nsi']) +\
                    '_dur2an_%d'%(stim_params['stim_dur'][0]) +\
                    '_delay2an_%d'%(delay) +\
                    '_peak_%.1f'%(peak) +\
                    '_peakratio_%.1f'%(peak_ratio) +\
                        '.pickle'
                        
            data_params = pickle.load(open(fld_analysis+ name_data,  "rb" ))
            params2an = data_params[0]
            data2an = data_params[1]
            t_on = params2an['stim_params']['t_on'][0]
            if fig2plot == 'martelli2013':
                martelli_plot(data2an, params2an, id_c)
                fig_pn_m.savefig(fld_analysis+  fig_martelli_name+'_'+inh_cond+'.png')
            plt.show()
            
                        
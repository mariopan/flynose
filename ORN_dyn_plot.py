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
    
    orn2plot  = np.mean(orn_sdf[:,:n_orns_recep], axis=1)
    stim_type = params2an['stim_params']['stim_type']
    
    # SETTINGS
    if stim_params['stim_type'] == 'ext':    
        #stim_type= stim_type[:-1]
        print(fig2plot)
        if fig2plot == 'parabola':
            panel_letters = ['e', 'f']
        elif fig2plot == 'step':
            panel_letters = ['a', 'b']
        elif fig2plot == 'ramp':
            panel_letters = ['c', 'd']
    else:
        print('normalized to the peak') 
        orn2plot = orn2plot/np.max(orn2plot)
        panel_letters = ['e', 'f']
    
    
    ax_conc_m.set_xlim(t2plot)
    ax_orn_m.set_xlim(t2plot)

    if id_c==0:
        ax_conc_m.tick_params(axis='both', labelsize=ticks_fs)
        ax_orn_m.tick_params(axis='both', labelsize=ticks_fs)
        
        ax_conc_m.set_xticklabels('')
        
        if fig2plot  == 'martelli2013':
            ax_conc_m.set_ylabel('Input (a.u.)', fontsize=label_fs)
            ax_orn_m.set_ylabel(' Norm. firing rates \n (unitless)', fontsize=label_fs)
            
        if fig2plot == 'step':
            ax_conc_m.set_ylabel('Input (a.u.)', fontsize=label_fs)
            ax_orn_m.set_ylabel(' Firing rates \n (unitless)', fontsize=label_fs)
            
        if stim_params['stim_type'] == 'ext':
            ax_conc_m.set_title(stim_type[:-1], fontsize = title_fs+10)
            
            ax_conc_m.set_xticks(np.linspace(0, t2simulate, 7))
            ax_conc_m.set_xticklabels('')
            ax_orn_m.set_xticks(np.linspace(0, t2simulate, 7))
            ax_orn_m.set_xticklabels(['0', '', '', '1500', '', '', '3000'])
            
            ax_conc_m.set_yticks(np.linspace(0,400, 5))
            ax_orn_m.set_yticks(np.linspace(0,250, 6))
            
            # ax_conc_m.set_ylim((-5, 400))
            # ax_orn_m.set_ylim((0, 250))            
            
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


fig2plot = 'step' # 'martelli' # 'ramp' # 'parabola' # 'step'

if fig2plot == 'martelli2013':
    # Martelli 2013 figure    
    fld_analysis = 'NSI_analysis/Martelli2013'

    # stim params
    stim_dur = 500
    stim_type = 'ss'
    
    stim_params = dict([
                    ('stim_type', stim_type),
                    ('stim_dur', [stim_dur, stim_dur]),
                     ])
    sens_params = dict([('w_nsi', 0), ])
    
    delay       = 0
    peak_ratio  = 1
    peaks       = np.linspace(0.05, .5, 10)

    # nsi params
    inh_conds   = ['noin'] #['nsi', 'ln', 'noin'] #
    
    ext_stimulus = 0            # it is equal to 1 only for lazar stimuli
    fig_save    = 0
    data_save   = 0  
    n_loops     = 1
    fig_name   = '/ORN-Martelli2013_dur_%d'%stim_dur


elif (fig2plot == 'ramp') | (fig2plot == 'parabola') | (fig2plot == 'step'):
    # Lazar and Kim data reproduction    
    fld_analysis    = 'NSI_analysis/lazar_sim2/'
    
    # stim params 
    stim_params = dict([
                    ('stim_type', 'ext'), 
                    ('stim_data_name', 'lazar_data_hr/'+fig2plot+'_1'), #.dat
                    ])
    sens_params = dict([('w_nsi', 0), ])
    
    stim_name   = fig2plot
    peaks       = np.array([1, 2, 3])
    
    # nsi params 
    inh_conds       = ['noin', ] #'ln', 'noin'

    fig_save    = 1
    n_loops     = 1

    # tau_sdf     = 60
    # dt_sdf      = 5      
    # sdf_params      = [tau_sdf, dt_sdf]
    fig_name   = '/lazar_'+fig2plot
    
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
        elif stim_params['stim_type'] == 'ext':
            t2plot = 0, 3500
        rs = 2 # number of rows
        cs = 1 # number of cols
        
        fig_pn_m = plt.figure(figsize=[5.71, 8])
        ax_conc_m = plt.subplot(rs, cs, 1)
        ax_orn_m = plt.subplot(rs, cs, 1+cs)
        
        for id_c, peak in enumerate(peaks):
            
            if stim_params['stim_type'] == 'ext':
                name_data = '/ORNrate' +\
                            '_stim_' + stim_name + str(peak) +\
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
            
            martelli_plot(data2an, params2an, id_c)
            if fig_save:
                fig_pn_m.savefig(fld_analysis + fig_name + '.png')
            plt.show()
            
                        
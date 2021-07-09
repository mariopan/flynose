#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:21:20 2021

ORN_dyn_plot.py

This script run NSI_ORN_LIF.py one or multiple times and saves the data 
the following figures of the NSI paper:
    fig.3 ORN dynamics of all its components (ORN_response)
    fig.3 ORN firing rates for several values of the stimulations (martelli)
    fig.s1 ORN dynamics for stimuli a la Lazar (lazar)   

@author: mario
"""

import numpy as np
import matplotlib.pyplot as plt
import timeit
import pickle        
from os import path
from os import mkdir
import matplotlib as mpl

import NSI_ORN_LIF
import plot_orn  
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
black   = 'xkcd:black'
blue    = 'xkcd:blue'
red     = 'xkcd:red'
green   = 'xkcd:green'
purple  = 'xkcd:purple'
orange  = 'xkcd:orange'
cmap    = plt.get_cmap('rainbow')
recep_clrs = ['green','purple','cyan','red']



# tic toc
def tictoc():
    return timeit.default_timer()         

def figure_multipeaks(data2an, params2an, id_c):
    
    n_orns_recep = params2an['sens_params']['n_orns_recep']
    
    t           = data2an['t']
    t2simulate  = t[-1]
    u_od        = data2an['u_od']
    orn_sdf     = data2an['orn_sdf']
    orn_sdf_time = data2an['orn_sdf_time']
    
    t_on        = params2an['stim_params']['t_on'][0]            
    
    
    # PLOT
    id_col = id_c + 3
    ax_conc_m.plot(t-t_on, 100*u_od[:,0], color=greenmap.to_rgba(id_col), linewidth=lw, 
              label='glom : '+'%d'%(1))
    
    orn2plot    = np.mean(orn_sdf[:, :n_orns_recep], axis=1)
    if fig2plot == 'martelli2013':
        print('normalized to the peak') 
        orn2plot = orn2plot/np.max(orn2plot)
    ax_orn_m.plot(orn_sdf_time-t_on, orn2plot, 
             color=greenmap.to_rgba(id_col), linewidth=lw-1,)
    
    # second ORNs
    # ax_conc_m.plot(t-t_on, 100*u_od[:,1], color=purplemap.to_rgba(id_col), linewidth=lw, 
    #           label='glom : '+'%d'%(1))
    
    # orn2plot    = np.mean(orn_sdf[:, n_orns_recep:], axis=1)
    # ax_orn_m.plot(orn_sdf_time-t_on, orn2plot, 
    #          color=purplemap.to_rgba(id_col), linewidth=lw-1,)
    
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
        panel_letters = ['f', 'g']
    
    
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
            ax_conc_m.set_title(fig2plot, fontsize = title_fs+10)
            
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
            
    
    




# LOAD PARAMS FROM A FILE
params_al_orn = set_orn_al_params.main(1)

# fld_analysis = 'NSI_analysis/trials/' #Olsen2010
# name_data = 'params_al_orn.ini'
# params_al_orn = pickle.load(open(fld_analysis+ name_data,  "rb" ))
# stimulus params
stim_params         = params_al_orn['stim_params']
sens_params         = params_al_orn['orn_layer_params'][0]
orn_params          = params_al_orn['orn_params']
sdf_params          = params_al_orn['sdf_params']
# al_params           = params_al_orn['al_params']
# pn_ln_params        = params_al_orn['pn_ln_params']

# fig2plot options:  
# 'trials' 'martelli2013' 'ramp' 'parabola' 'step' 'orn_response' 
fig2plot = 'orn_response' 

# Figures options
fig_save            = 0
fig_orn_dyn         = 1
fig_multipeaks      = 0


max_stim_seed       = 1

# nsi params
inh_conds           = ['noin'] #['nsi', 'ln', 'noin'] #
data_save           = 1


    
if fig2plot == 'trials':
    fld_analysis = 'NSI_analysis/trials/'
    
    
    # stim paramsnp.array([1.85e-4, 0.002]) # 
    delay                       = 0
    peak_ratio                  = 1
    peaks                       = np.logspace(-3.3, -2, 5) # np.linspace(0.05, .5, 10)
    stim_dur                    = 500
    stim_type                   = 'ss'
    
    stim_params['stim_type']    = stim_type
    stim_params['stim_dur']     = np.array([stim_dur, stim_dur+delay])
    stim_params['t_tot']        = 2000        # ms 
    stim_params['t_on']         =  np.array([1000, 1000])
    stim_params['conc0']        = 1.85e-4
        
    sens_params['w_nsi']        = 0.0
    
    t2plot                      = -200,stim_dur+300
    fig_name                    = 'orn_dur_%d'%stim_dur
    
elif fig2plot == 'martelli2013':
    # Martelli 2013 figure    
    fld_analysis = 'NSI_analysis/Martelli2013/'
    
    # stim paramsnp.array([1.85e-4, 0.002]) # 
    delay           = 0
    peak_ratio      = 1
    peaks           = np.logspace(-3.3, -2, 5) # np.linspace(0.05, .5, 10)
    stim_dur        = 500
    stim_type       = 'ss'
    
    stim_params['stim_type']    = stim_type
    stim_params['stim_dur']     = np.array([stim_dur, stim_dur+delay])
    stim_params['t_tot']        = 2000        # ms 
    stim_params['t_on']         =  np.array([1000, 1000])
    stim_params['conc0']        = 1.85e-4
        
    sens_params['w_nsi']  = 0.0
    
    t2plot          = -200,stim_dur+300
    fig_name   = 'ORN-Martelli2013_dur_%d'%stim_dur

elif (fig2plot == 'ramp') | (fig2plot == 'parabola') | (fig2plot == 'step'):
    # Lazar and Kim data reproduction    
    fld_analysis    = 'NSI_analysis/lazar_sim2/'
    
    # stim params 
    stim_params['stim_type']    = 'ext'
    stim_params['stim_data_name'] = 'lazar_data_hr/'+fig2plot+'_1' #.dat
    stim_params['t_on']         =  np.array([0, 0])
    stim_params['pts_ms' ]= 5
    peaks       = np.array([1, 2, 3])
    
    # tau_sdf     = 60
    # dt_sdf      = 5      
    t2plot = 0, 3500

    fig_name   = 'lazar_'+fig2plot

elif fig2plot == 'orn_response':
    
    fld_analysis = 'NSI_analysis/ORN_LIF_dynamics/' #/sdf_test
    orn_fig_name = 'ORN_lif_dyn_500ms.png'
    
    # stim params
    delay                       = 0
    stim_dur                    = 500
    stim_params['stim_type']    = 'ss' # 'ts' # 'ss' # 'rp'# '
    stim_params['t_tot']        = 2000        # ms 
    stim_params['t_on']         =  np.array([1000, 1000])
    stim_params['stim_dur']     = np.array([stim_dur, stim_dur])
    stim_params['conc0']        = 1.85e-4
    peaks                       = np.array([1e-3])         # concentration value for ORN1
    peak_ratio                  = .01         # concentration ratio: ORN2/ORN1    
    sdf_params['tau_sdf']       = 41
    
    t2plot          = -200,stim_dur+300


n_lines     = np.size(peaks)

c = np.arange(1, n_lines + 4)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
greenmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Greens)
purplemap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Purples)

print(fig2plot)
params_1sens   = dict([
                ('stim_params', stim_params),
                ('sens_params', sens_params),
                ('orn_params', orn_params),
                ('sdf_params', sdf_params),
                ])

# %% RUN SIMS AND PLOT
for stim_seed in range(max_stim_seed):
    
    if path.isdir(fld_analysis):
        print('OLD analysis fld: ' + fld_analysis)    
    else:
        print('NEW analysis fld: ' + fld_analysis)    
        mkdir(fld_analysis)
        
    for inh_cond in inh_conds:
        
        # FIGURE multiple concentrations

        if fig_multipeaks:

            rs = 2 # number of rows
            cs = 1 # number of cols
            
            fig_pn_m = plt.figure(figsize=[5.71, 8])
            ax_conc_m = plt.subplot(rs, cs, 1)
            ax_orn_m = plt.subplot(rs, cs, 1+cs)


        
        for id_c, peak in enumerate(peaks):
            
            if stim_params['stim_type'] == 'ext':
                stim_params['stim_data_name'] = stim_params['stim_data_name'][:-1]+str(peak)
                print(stim_params['stim_data_name'])
                name_data = 'ORNrate' +\
                            '_stim_' + fig2plot + str(peak) +\
                            '_nsi_%.1f'%(sens_params['w_nsi']) +\
                            '.pickle'
            else:
                stim_params['concs'] = np.array([peak, peak*peak_ratio])
                name_data = 'ORNrate' +\
                    '_stim_' + stim_params['stim_type'] +\
                    '_nsi_%.1f'%(sens_params['w_nsi']) +\
                    '_dur2an_%d'%(stim_params['stim_dur'][0]) +\
                    '_delay2an_%d'%(delay) +\
                    '_peak_%.2f'%(np.log10(peak)) +\
                    '_peakratio_%.1f'%(peak_ratio) +\
                        '.pickle'
            
            # RUN SIM
            orn_lif_out = NSI_ORN_LIF.main(params_1sens, )
            [t, u_od, r_orn, v_orn, y_orn, num_spikes, orn_spike_matrix, 
                 orn_sdf, orn_sdf_time,] = orn_lif_out
            output2an = dict([
                        ('t', t),
                        ('u_od',u_od),
                        ('orn_sdf', orn_sdf),
                        ('orn_sdf_time',orn_sdf_time), ])   
            if data_save:
                with open(fld_analysis+name_data, 'wb') as f:
                    pickle.dump([params_1sens, output2an], f)
        
            
            # FIGURE ORN DYNAMICS OR MULTIPLTE PEAKS 
            if fig_orn_dyn:
                fig_orn = plot_orn.main(params_1sens, orn_lif_out)
                    
                if fig_save:
                    fig_orn.savefig(fld_analysis + orn_fig_name)
                    with open(fld_analysis+'param_1sens.pickle', 'wb') as f:
                        pickle.dump([params_1sens, ], f)
                                
            elif fig_multipeaks:
                figure_multipeaks(output2an, params_1sens, id_c)
                if fig_save:
                    fig_pn_m.savefig(fld_analysis + fig_name + '.png')
                    if data_save:
                        with open(fld_analysis+'param_1sens.pickle', 'wb') as f:
                            pickle.dump([params_1sens, ], f)
            plt.show()
                        
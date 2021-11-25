#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:30:03 2020

plot_examples.py

Plot the dynamics of AL and ORN for the three conditions (ctrl, NSI, LN inh) 
in a fancy diagonal way or overlapped in a single row.


@author: mario
"""

import numpy as np
import matplotlib.pyplot as plt
import timeit

import ORNs_layer_dyn
import AL_dyn
import set_orn_al_params

import stats_for_plumes as stats

# STANDARD FIGURE PARAMS
lw = 2
fs = 20
plt.rc('text', usetex=True)  # laTex in the polot
#plt.rc('font', family='serif')
fig_position = 1300,10
title_fs = 18 # font size of title
label_fs = 20 # font size of labels
panel_fs = 30 # font size of panel' letters
legend_fs = 12
scale_fs = label_fs-5

black   = 'xkcd:black'
blue    = 'xkcd:blue'
red     = 'xkcd:red'
green   = 'xkcd:green'
purple  = 'xkcd:purple'
orange  = 'xkcd:orange'
pink    = 'xkcd:pink'
cyan    = 'xkcd:cyan'
gray    = 'xkcd:gray'

#cmap    = plt.get_cmap('rainbow')
 


def orn_al_settings():
    """ FIGURE orn_al SETTING"""
    for id_col in range(3):
        axs[id_col].set_xlim(t2plot)
        axs[id_col].tick_params(axis='both', labelsize=label_fs)
        axs[id_col].set_xticklabels('')
        axs[id_col].axis('off')
    
    axs[1].set_ylim((0, 300))
    axs[2].set_ylim((0, 300))
    
    axs[0].set_title('Input ', fontsize=title_fs)
    axs[1].set_title(r' ORN  ', fontsize=title_fs)
    axs[2].set_title(r' PN  ', fontsize=title_fs)

    if fig_id == 'ts_a':
        axs[0].text(-.2, 1.0, 'a', transform=axs[0].transAxes, 
            fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    
    # vertical/horizontal lines for time/Hz scale
    if fig_id=='ts_a':
        # horizontal line
        axs[1].plot([300, 350], [150, 150], color=black) 
        axs[1].text(305, 120, '50 ms', color=black, fontsize=scale_fs)
        # vertical line
        axs[1].plot([300, 300], [150, 250], color=black) 
        axs[1].text(260, 160, '100 Hz', color=black,rotation=90, fontsize=scale_fs)
    
        axs[1].set_ylim((0, 370))
        axs[2].set_ylim((0, 280))
        
        ax2.set_ylim([200, 285])
        ax2.set_xlim([30, 150])
        ax2.axis('on')
        for idh in range(4):
            inh_c = inh_conds[idh]
            axs[2].text(250, 20+30*idh, model_names[inh_c], color=model_colours[inh_c], fontsize=scale_fs+2)

    elif fig_id=='ts_s':
        # horizontal line
        axs[1].plot([200, 250], [150, 150], color=black) 
        axs[1].text(205, 120, '50 ms', color=black, fontsize=scale_fs)
        # vertical line
        axs[1].plot([200, 200], [150, 250], color=black) 
        axs[1].text(160, 160, '100 Hz', color=black,rotation=90, fontsize=scale_fs)
        axs[1].set_ylim((0, 250))
        axs[2].set_ylim((0, 250))
        
        for idh in range(4):
            inh_c = inh_conds[idh]
            axs[2].text(220, 110+30*idh, model_names[inh_c], color=model_colours[inh_c], fontsize=scale_fs+2)
        # ax2.set_ylim([100, 200])
        # ax2.set_xlim([70, 120])
        # ax2.axis('off')
        

    elif fig_id=='pl':
        # horizzontal line
        axs[1].plot([2000, 3000], [150+30, 150+30], color=black) 
        axs[1].text(2000, 120+30, '1000 ms', color=black, fontsize=scale_fs)
        # vertical line
        axs[1].plot([2000, 2000], [150+30, 250+30], color=black) 
        axs[1].text(1600, 160+30, '100 Hz', color=black,rotation=90, fontsize=scale_fs)
    
    
        
        
def orn_al_plot(data_tmp, params_tmp, inh_cond):
    """ Figure orn_al PLOT """
    
    t           = data_tmp['t']
    u_od        = data_tmp['u_od']
    orn_sdf     = data_tmp['orn_sdf']
    orn_sdf_time = data_tmp['orn_sdf_time']
    pn_sdf = data_tmp['pn_sdf']
    pn_sdf_time = data_tmp['pn_sdf_time']
    
    t_zero  =   params_tmp['stim_params']['t_on'][0]
    
    n_orns_recep = params_tmp['al_params']['n_orns_recep']
    n_pns_recep = params_tmp['al_params']['n_pns_recep']
    
    # PLOTTING DATA
    trsp = .3
    
    if inh_cond == 'noin':
        axs[0].plot(t-t_zero, 100*u_od[:,0], color=black, 
               linewidth=lw, linestyle='-', )
        # 2nd ORNs
        if (fig_id == 'ts_a') or (fig_id == 'pl'):
            axs[0].plot(t-t_zero, 100*u_od[:,1], color=gray, 
                linewidth=lw, linestyle='--', )
    
    if (inh_cond == 'noin') | (inh_cond == 'nsi'):
        X1 = np.mean(orn_sdf[:,:,:n_orns_recep], axis=2)
        mu1 = X1.mean(axis=0)
        sigma1 = X1.std(axis=0)
        axs[1].plot(orn_sdf_time-t_zero, mu1, linestyle='-', 
                    linewidth=lw, color=model_colours[inh_cond])
        axs[1].fill_between(orn_sdf_time-t_zero, 
           mu1+sigma1, mu1-sigma1, facecolor=green, alpha=trsp)
        
        # 2nd ORNs
        if (fig_id == 'ts_a') or (fig_id == 'pl'):
            X1 = np.mean(orn_sdf[:,:,n_orns_recep:], axis=2)
            mu1 = X1.mean(axis=0)
            sigma1 = X1.std(axis=0)
            axs[1].plot(orn_sdf_time-t_zero, mu1, linestyle='--', linewidth=lw, color=model_colours[inh_cond])
            axs[1].fill_between(orn_sdf_time-t_zero, 
                mu1+sigma1, mu1-sigma1, facecolor=model_colours[inh_cond], alpha=trsp)
        
    X1 = np.mean(pn_sdf[:,:,:n_pns_recep], axis=2)
    mu1 = X1.mean(axis=0)
    sigma1 = X1.std(axis=0)
    axs[2].plot(pn_sdf_time-t_zero, mu1, linestyle='-', linewidth=lw, 
                color=model_colours[inh_cond], label=model_names[inh_cond])
    axs[2].fill_between(orn_sdf_time-t_zero, 
           mu1+sigma1, mu1-sigma1, facecolor=model_colours[inh_cond], alpha=trsp)
    
    
    
    
    if (fig_id == 'ts_a') or (fig_id == 'pl'):
        ax2.plot(pn_sdf_time-t_zero, mu1, linestyle='-', linewidth=lw, color=model_colours[inh_cond])
    
    
        # 2nd ORNs
        X1 = np.mean(pn_sdf[:,:,n_pns_recep:], axis=2)
        mu1 = X1.mean(axis=0)
        sigma1 = X1.std(axis=0)
        axs[2].plot(pn_sdf_time-t_zero, mu1, linestyle='--', linewidth=lw, color=model_colours[inh_cond])
        axs[2].fill_between(pn_sdf_time-t_zero, 
                mu1+sigma1, mu1-sigma1, facecolor=model_colours[inh_cond], alpha=trsp)
        
        ax2.plot(pn_sdf_time-t_zero, mu1, linestyle='--', linewidth=lw, color=model_colours[inh_cond])
        
def orn_al_diag_settings():
    # FIGURE SETTING
    
    dl = .29
    ll_new = .04+np.array([0, .005+ dl, .01+ 2*dl, ])
    
    dh = .22
    bb_new = .02+np.array([.015+ 3*dh, .01+ 2*dh, .005+ dh, 0, ])
    
    for id_row in range(4):
        for id_col in range(3):
            ll, bb, ww, hh = axs[id_row, id_col].get_position().bounds
            axs[id_row, id_col].set_position([ll_new[id_col], 
                                             bb_new[id_row], dl, dh])
    
    for id_row in range(4):
        for id_col in range(3):
            axs[id_row, id_col].set_xlim(t2plot)
            axs[id_row, id_col].tick_params(axis='both', labelsize=label_fs)
            axs[id_row, id_col].set_xticklabels('')
            axs[id_row, id_col].axis('off')
    
    for id_row in range(4):
        axs[id_row, 1].set_ylim((0, 250))
        axs[id_row, 2].set_ylim((0, 250))
    
    axs[0, 0].set_title('Input ', fontsize=title_fs)
    axs[0, 1].set_title(r' ORN  ', fontsize=title_fs)
    axs[0, 2].set_title(r' PN  ', fontsize=title_fs)
    

    # vertical/horizontal lines for time/Hz scale
    
    if fig_id=='ts_s':
        # horizontal line
        axs[2, 1].plot([100, 150], [50, 50], color=black) 
        axs[2, 1].text(105, 10, '50 ms', color=black, fontsize=scale_fs)
        # vertical line
        axs[2, 1].plot([100, 100], [50, 150], color=black) 
        axs[2, 1].text(80, 40, '100 Hz', color=black,rotation=90, fontsize=scale_fs)
        
    elif fig_id=='ts_a':
        # horizzontal line
        axs[2, 1].plot([100, 150], [50, 50], color=black) 
        axs[2, 1].text(105, 10, '50 ms', color=black, fontsize=scale_fs)
        # vertical line
        axs[2, 1].plot([100, 100], [50, 150], color=black) 
        axs[2, 1].text(75, 40, '100 Hz', color=black,rotation=90, fontsize=scale_fs)

    elif fig_id=='pl':
        # horizzontal line
        axs[1, 1].plot([2000, 2500], [50, 50], color=black) 
        axs[1, 1].text(2000, 2, '500 ms', color=black, fontsize=scale_fs)
        # vertical line
        axs[1, 1].plot([2000, 2000], [50, 150], color=black) 
        axs[1, 1].text(1600, 40, '100 Hz', color=black,rotation=90, fontsize=scale_fs)
    
    if (fig_id=='ts_a') | (fig_id=='pl'):
        axs[0, 0].text(.05, 1.4, 'a', transform=axs[0,0].transAxes, 
                 fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    
    
    if fig_id=='ts_a':
        x4text = 280
    elif fig_id=='ts_s':
        x4text = 190
    else:
        x4text = t2plot[1]+200
    
    axs[0, 2].text(x4text, 75, 'ctrl\n model', fontsize=scale_fs, fontweight='bold',rotation=90)
    axs[1, 2].text(x4text, 75, 'LN\n model', fontsize=scale_fs, fontweight='bold', rotation=90)
    axs[2, 2].text(x4text, 75, 'NSI\n model', fontsize=scale_fs, fontweight='bold',rotation=90)
    axs[3, 2].text(x4text, 75, 'Mix\n model', fontsize=scale_fs, fontweight='bold', rotation=90)
        
def orn_al_diag_plot(data_tmp, params_tmp, inh_cond):
    
    t           = data_tmp['t']
    u_od        = data_tmp['u_od']
    orn_sdf     = data_tmp['orn_sdf']
    orn_sdf_time = data_tmp['orn_sdf_time']
    pn_sdf = data_tmp['pn_sdf']
    pn_sdf_time = data_tmp['pn_sdf_time']
    
    t_zero  =   params_tmp['stim_params']['t_on'][0]
    
    n_orns_recep = params_tmp['al_params']['n_orns_recep']
    n_pns_recep = params_tmp['al_params']['n_pns_recep']
    
    # ****************************************************
    # PLOTTING DATA
    
    id_row = id_inh
    
    trsp = .3

    if inh_cond == 'noin':
        axs[0, 0].plot(t-t_zero, 100*u_od[:,0], color=green, 
               linewidth=lw, )
        if (fig_id == 'ts_a') or (fig_id == 'pl'):
            axs[0, 0].plot(t-t_zero, 100*u_od[:,1], color=purple, 
               linewidth=lw, )
    
    if (inh_cond == 'noin') | (inh_cond == 'nsi') : #| (inh_cond == 'mix')
        X1 = np.mean(orn_sdf[:,:,:n_orns_recep], axis=2)
        mu1 = X1.mean(axis=0)
        sigma1 = X1.std(axis=0)
        axs[id_row, 1].plot(orn_sdf_time-t_zero, mu1, linewidth=lw, color=green)
        axs[id_row, 1].fill_between(orn_sdf_time-t_zero, 
           mu1+sigma1, mu1-sigma1, facecolor=green, alpha=trsp)
    
        X1 = np.mean(orn_sdf[:,:,n_orns_recep:], axis=2)
        mu1 = X1.mean(axis=0)
        sigma1 = X1.std(axis=0)
        if (fig_id == 'ts_a') or (fig_id == 'pl'):
        
            axs[id_row, 1].plot(orn_sdf_time-t_zero, mu1, linewidth=lw, color=purple)
            axs[id_row, 1].fill_between(orn_sdf_time-t_zero, 
                mu1+sigma1, mu1-sigma1, facecolor=purple, alpha=trsp)
        
    X1 = np.mean(pn_sdf[:,:,:n_pns_recep], axis=2)
    mu1 = X1.mean(axis=0)
    sigma1 = X1.std(axis=0)
    axs[id_row, 2].plot(pn_sdf_time-t_zero, mu1, linewidth=lw, color=green)
    axs[id_row, 2].fill_between(orn_sdf_time-t_zero, 
           mu1+sigma1, mu1-sigma1, facecolor=green, alpha=trsp)
    
    X1 = np.mean(pn_sdf[:,:,n_pns_recep:], axis=2)
    mu1 = X1.mean(axis=0)
    sigma1 = X1.std(axis=0)
    if (fig_id == 'ts_a') or (fig_id == 'pl'):
        axs[id_row, 2].plot(pn_sdf_time-t_zero, mu1, linewidth=lw, color=purple)
        axs[id_row, 2].fill_between(pn_sdf_time-t_zero, 
                mu1+sigma1, mu1-sigma1, facecolor=purple, alpha=trsp)
    
    
    
    
    
    
#***********************************************
# Standard params
# fld_analysis= 'NSI_analysis/triangle_stim/diag_figs'
inh_conds   = ['noin', 'ln', 'nsi', 'mix'] 

model_colours = dict([('noin', pink),
                 ('ln', orange),
                 ('nsi', cyan),
                 ('mix', green), ])

model_names = dict([('noin', 'control'),
                 ('ln', 'LN inhib'),
                 ('nsi', 'NSI'),
                 ('mix', 'mix'), ])

model_ls = dict([('noin','-'),
                 ('ln', ':'),
                 ('nsi', '--'),
                 ('mix', '-.'), ])

model_lw = dict([('noin', lw),
                 ('ln', lw-1),
                 ('nsi', lw),
                 ('mix', lw-1), ])

# LOAD PARAMS FROM A FILE
params_al_orn = set_orn_al_params.main(2)

stim_params         = params_al_orn['stim_params']
orn_layer_params    = params_al_orn['orn_layer_params']
orn_params          = params_al_orn['orn_params']
sdf_params          = params_al_orn['sdf_params']
al_params           = params_al_orn['al_params']
pn_ln_params        = params_al_orn['pn_ln_params']
plume_params        = stim_params['plume_params']

n_sens_type         = orn_layer_params.__len__()  # number of type of sensilla

# ORN NSI params

# fig_id options:  # 'ts_s' #  'ts_a' # 'pl'
fig_id                  = 'pl' 

fld_analysis            = 'NSI_analysis/triangle_stim/'
nsi_str                 = 0.6
alpha_ln                = 0.6
pn_ln_params['tau_ln']  = 250

    
# figure and output options
fig_save    = 0
fig_dpi     = 350
data_save   = 0    
verbose     = 0
olsen_fig   = 0


n_lines     = 10

if fig_id == 'ts_s':
    # stim params
    delay                       = 0    
    stim_params['stim_type']    = 'ts' # 'id_l'  # 'ts'
    stim_params['stim_dur']     = np.array([50, 50])
    stim_params['t_tot']        = 1300
    t_on                        = 1000
    stim_params['conc0']        = 1.85e-6    # 2.85e-4
    
    peaks                       = [.5e-3] # 0.05 [1.85e-4, 5e-4, 1.5e-3, 2e-2, 2e-1]#[0.001, 0.005] #

elif fig_id == 'ts_a':
    # Stimulus params 
    delay                       = 50
    stim_params['stim_type']    = 'ts' # 'id_l'  # 'ts'
    stim_params['stim_dur']     = np.array([50, 50])
    stim_params['t_tot']        = 1300+delay
    t_on                        = 1000
    stim_params['conc0']        = 1.85e-6    # 2.85e-4
    sdf_params['tau_sdf']       = 20

    peaks                       = [5e-2] #[1.85e-4, 5e-4, 1.5e-3, 2e-2, 2e-1]#[0.001, 0.005] #

elif fig_id == 'pl':
    fld_analysis                = 'NSI_analysis/real_plumes_example/'
    fig_orn_dyn                 = 1
    max_stim_seed               = 3
    orn_fig_name                = '/ORN_lif_dyn_realplume.png'
       
    # stim params
    delay                       = 0    
    stim_params['stim_type']    = 'pl'      
    stim_params['t_tot']        = 5000     # ms 
    t_on                        = 1000
    stim_params['conc0']        = 1.85e-4
    peaks                       = [10e-4]
    stim_params['stim_seed']    = 22
    
    # real plumes params
    plume_params['whiff_max']   = 3
    plume_params['rho_t_exp']   = 0   #[0, 1, 3, 5]
    
    sdf_params['tau_sdf']       = 30
    sdf_params['dt_sdf']        = 5
    
    stim_params['plume_params'] = plume_params
    
    #('g_y',  .5853575783),      # 0.3), # 
    orn_params['g_y']           = 0.58 #0.3#
    

conc0           = stim_params['conc0']
t_tot           = stim_params['t_tot']
stim_dur        = stim_params['stim_dur'][0]
peak_ratio      = 1
stim_params['t_on'] = np.array([t_on, t_on+delay])      # ms 

pts_ms          =   stim_params['pts_ms']

dt_sdf          = sdf_params['dt_sdf']
sdf_size        = int(stim_params['t_tot']/dt_sdf)


n_neu           = orn_layer_params[0]['n_neu']
n_orns_recep    = al_params['n_orns_recep']# number of ORNs per each receptor
n_pns_recep     = al_params['n_pns_recep'] # number of PNs per each receptor

orn_sdf_all     = np.zeros((n_lines, sdf_size, n_neu*n_orns_recep))
pn_sdf_all      = np.zeros((n_lines, sdf_size, n_neu*n_pns_recep))


tic = timeit.default_timer()

# peaks                       = [5e-4] #[1.85e-4, 5e-4, 1.5e-3, 2e-2, 2e-1]#[0.001, 0.005] #
n_peaks = len(peaks)
time2analyse = 200

conc_s    = np.zeros((n_peaks, len(inh_conds)))
conc_th = np.zeros((n_peaks, len(inh_conds)))

avg_ornw = np.zeros((n_peaks, n_lines, len(inh_conds)))
avg_orns = np.zeros((n_peaks, n_lines, len(inh_conds))) 
avg_pnw = np.zeros((n_peaks, n_lines, len(inh_conds)))
avg_pns = np.zeros((n_peaks, n_lines, len(inh_conds)))

peak_ornw = np.zeros((n_peaks, n_lines, len(inh_conds)))
peak_orns  = np.zeros((n_peaks, n_lines, len(inh_conds)))
peak_pnw = np.zeros((n_peaks, n_lines, len(inh_conds)))
peak_pns = np.zeros((n_peaks, n_lines, len(inh_conds)))



for id_p, peak in enumerate(peaks):
    stim_params['concs'] = np.array([peak, peak*peak_ratio])

    # FIGURE         
    fig_name    = 'diag_stim_'+ stim_params['stim_type']+'_delay_%d'%delay + \
                    '_nsi%.2f'%nsi_str+'_ln%.1f'%alpha_ln + '_peak%.5f'%peak
    if fig_id == 'ts_a':
        fig_name = fig_name + '_tauln_%d'%pn_ln_params['tau_ln']
    
    fig_name = fig_name + '_MIX'
    
    
    if fig_id == 'pl':
        t2plot = -20, stim_params['t_tot']-t_on-500
        rs = 4 # number of rows
        cs = 3 # number of cols
        fig_pn, axs = plt.subplots(rs, cs, figsize=[9, 4])
        orn_al_diag_settings()
        
    else:
        t2plot = -50, stim_params['t_tot']-t_on
        rs = 1 # number of rows
        cs = 3 # number of cols
        fig_pn, axs = plt.subplots(rs, cs, figsize=[9, 2.8])
        if fig_id == 'ts_a':
            left, bottom, width, height = [0.85, 0.5, 0.14, 0.45]
            ax2 = fig_pn.add_axes([left, bottom, width, height])
            ax2.set_xticklabels('')
            ax2.set_yticklabels('')
        orn_al_settings()
        
        
    # simulations and append to figure         
    for id_inh, inh_cond in enumerate(inh_conds):
                                
        # setting NSI params
        for sst in range(n_sens_type):
            if inh_cond == 'nsi':
                orn_layer_params[sst]['w_nsi']  = nsi_str    
                pn_ln_params['alpha_ln']        = 0
            elif inh_cond == 'noin':
                orn_layer_params[sst]['w_nsi']  = 0
                pn_ln_params['alpha_ln']        = 0
            elif inh_cond == 'ln':
                orn_layer_params[sst]['w_nsi']  = 0    
                pn_ln_params['alpha_ln']        = alpha_ln
            elif inh_cond == 'mix':
                orn_layer_params[sst]['w_nsi']  = nsi_str
                pn_ln_params['alpha_ln']        = alpha_ln
    
        for id_l in range(n_lines):  
                     
            # Run flynose 
            
            # ORNs layer dynamics
            [t, u_od,  orn_spikes_t, orn_sdf, orn_sdf_time] = \
                ORNs_layer_dyn.main(params_al_orn, verbose=verbose, )
            
            orn_sdf_all[id_l,:,:] = orn_sdf
            
            # AL dynamics
            [t, pn_spike_matrix, pn_sdf, pn_sdf_time,
                ln_spike_matrix, ln_sdf, ln_sdf_time,] = \
                AL_dyn.main(params_al_orn, orn_spikes_t, verbose=verbose, )
            pn_sdf_all[id_l,:,:] = pn_sdf
            
            
            
            # Calculate avg and peak SDF for ORNs
            if orn_spikes_t.size >0:
                id_stim_w = np.flatnonzero((orn_sdf_time>t_on) 
                                        & (orn_sdf_time<t_on+time2analyse))
                
                
                id_stim_s = np.flatnonzero((orn_sdf_time>t_on+delay) 
                                        & (orn_sdf_time<t_on+delay+time2analyse))
                
                orn_peak_w  = np.max(np.mean(orn_sdf[id_stim_w, :n_orns_recep], axis=1)) # using average PN
                orn_peak_s  = np.max(np.mean(orn_sdf[id_stim_s, n_orns_recep:], axis=1)) # using average PN
                orn_avg_w  = np.mean(orn_sdf[id_stim_w, :n_orns_recep])
                orn_avg_s  = np.mean(orn_sdf[id_stim_s, n_orns_recep:])

                avg_ornw[id_p, id_l, id_inh] = orn_avg_w
                avg_orns[id_p, id_l, id_inh] = orn_avg_s
                
                peak_ornw[id_p, id_l, id_inh] = orn_peak_w
                peak_orns[id_p, id_l, id_inh] = orn_peak_s
            
            # Calculate avg and peak SDF for PNs 
            if pn_spike_matrix.size >0:
                id_stim_w = np.flatnonzero((pn_sdf_time>t_on) 
                                & (pn_sdf_time<t_on+time2analyse))
                id_stim_s = np.flatnonzero((pn_sdf_time>t_on+delay) 
                                & (pn_sdf_time<t_on+delay+time2analyse))
                
                pn_peak_w  = np.max(np.mean(pn_sdf[id_stim_w, :n_pns_recep], axis=1)) # using average PN
                pn_peak_s  = np.max(np.mean(pn_sdf[id_stim_s, n_pns_recep:], axis=1)) # using average PN
                pn_avg_w  = np.mean(pn_sdf[id_stim_w, :n_pns_recep])
                pn_avg_s  = np.mean(pn_sdf[id_stim_s, n_pns_recep:])

                avg_pnw[id_p, id_l, id_inh] = pn_avg_w
                avg_pns[id_p, id_l, id_inh] = pn_avg_s
                
                peak_pnw[id_p, id_l, id_inh] = pn_peak_w
                peak_pns[id_p, id_l, id_inh] = pn_peak_s
            if fig_id == 'pl':
                # CALCULATE AND SAVE DATA
                t_id_stim = np.flatnonzero((t>t_on) & (t<t_tot))
                
                od_avg_1 = np.mean(u_od[t_id_stim, 0])
                od_avg_2 = np.mean(u_od[t_id_stim, 1])
                cor_stim        = -2
                overlap_stim    = -2
                cor_whiff       = -2
                out_1 = u_od[t_id_stim, 0]
                out_2 = u_od[t_id_stim, 1]
                                    
                interm_est_1 = np.sum(out_1>0)/(t_tot*pts_ms)
                interm_est_2 = np.sum(out_2>0)/(t_tot*pts_ms)
                
                if (np.sum(out_2)!=0) & (np.sum(out_1)!=0):
                    cor_stim        = np.corrcoef(out_2, out_1)[1,0]
                    overlap_stim    = stats.overlap(out_2, out_1, conc0, conc0)
                    nonzero_concs1  = out_2[(out_2>0) & (out_1>0)]
                    nonzero_concs2  = out_1[(out_2>0) & (out_1>0)]
                    cor_whiff       = np.corrcoef(nonzero_concs1, nonzero_concs2)[0, 1] # np.corrcoef(concs1, concs2)[0, 1]
            
        data2plot = dict([
                        ('t', t),
                        ('u_od', u_od),
                        ('orn_sdf', orn_sdf_all),
                        ('orn_sdf_time',orn_sdf_time), 
                        ('pn_sdf', pn_sdf_all),
                        ('pn_sdf_time',pn_sdf_time), 
                        ])          
    
    
        if fig_id == 'pl':
            orn_al_diag_plot(data2plot, params_al_orn, inh_cond)
        else:
            orn_al_plot(data2plot, params_al_orn, inh_cond)
        
 
        
    
    plt.show()    

    if fig_save:
        print('saving figure in '+fld_analysis)
        fig_pn.savefig(fld_analysis + fig_name + '_'+str(fig_dpi)+'dpi.png',
                       dpi=fig_dpi)

       
print('PN avg S:')
print(np.mean(avg_pns, axis=1))
print('PN avg w:')
print(np.mean(avg_pnw, axis=1))
pn_avg_ratio = np.ma.masked_invalid(avg_pns/avg_pnw)
print('PN avg ratio: ')
print(np.mean(pn_avg_ratio, axis=1))

print('ORN S:')
print(np.mean(avg_orns, axis=1))
print('ORN w:')
print(np.mean(avg_ornw, axis=1))
orn_avg_ratio = np.ma.masked_invalid(avg_orns/avg_ornw)
print('ORN peak ratio: ')
print(np.mean(orn_avg_ratio, axis=1))

toc = timeit.default_timer()-tic

print('Diag shade plot elapsed time: %.1f'%(toc))
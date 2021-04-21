#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:30:03 2020

plot_diag_shades2.py

@author: mario
"""

import numpy as np
import matplotlib.pyplot as plt
import timeit
import pickle

import ORNs_layer_dyn
import AL_dyn
import set_orn_al_params



# STANDARD FIGURE PARAMS
lw = 2
fs = 13
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
#cmap    = plt.get_cmap('rainbow')

def olsen2010_data(data_tmp, params_tmp):
    """ return the average activity of PNs and ORNs during stimulation"""
    
    pts_ms  =   params_tmp['stim_params']['pts_ms']

    t_on    =   params_tmp['stim_params']['t_on'][0]
    stim_dur   =   params_tmp['stim_params']['stim_dur'][0]
    t_off   = t_on + stim_dur
    
    n_orns_recep = params_tmp['al_params']['n_orns_recep']
    n_pns_recep = params_tmp['al_params']['n_pns_recep']
    
    stim_on  = t_on*pts_ms 
    stim_off = t_off*pts_ms 
    
    u_od        = data_tmp['u_od']
    orn_sdf     = data_tmp['orn_sdf']
    orn_sdf_time = data_tmp['orn_sdf_time']
    pn_sdf      = data_tmp['pn_sdf']
    pn_sdf_time = data_tmp['pn_sdf_time']
    
    if stim_dur == 500:
        orn_id_stim = np.flatnonzero((orn_sdf_time>t_on) & (orn_sdf_time<t_off))
        pn_id_stim = np.flatnonzero((pn_sdf_time>t_on) & (pn_sdf_time<t_off))
    else:
        orn_id_stim = np.flatnonzero((orn_sdf_time>t_on) & (orn_sdf_time<t_on+200))
        pn_id_stim = np.flatnonzero((pn_sdf_time>t_on) & (pn_sdf_time<t_on+200))
        
        
    nu_orn_w = np.mean(orn_sdf[orn_id_stim, :n_orns_recep])
    nu_pn_w = np.mean(pn_sdf[pn_id_stim, :n_pns_recep])
    
    nu_orn_s = np.mean(orn_sdf[orn_id_stim, n_orns_recep:])
    nu_pn_s = np.mean(pn_sdf[pn_id_stim, n_pns_recep:])
    conc_s = np.mean(u_od[stim_on:stim_off, 0], axis=0)
    conc_w = np.mean(u_od[stim_on:stim_off, 1], axis=0)
    
    nu_orn_s_err = np.std(orn_sdf[orn_id_stim, :n_orns_recep])/np.sqrt(n_orns_recep)
    nu_orn_w_err = np.std(orn_sdf[orn_id_stim, n_orns_recep:])/np.sqrt(n_orns_recep)
    nu_pn_s_err = np.std(pn_sdf[pn_id_stim, :n_pns_recep])/np.sqrt(n_pns_recep)
    nu_pn_w_err = np.std(pn_sdf[pn_id_stim, n_pns_recep:])/np.sqrt(n_pns_recep)
    
    out_olsen = dict([
        ('conc_s', conc_s),
        ('conc_w', conc_w),
        ('nu_orn_s', nu_orn_s),
        ('nu_orn_w', nu_orn_w),
        ('nu_pn_s', nu_pn_s),
        ('nu_pn_w', nu_pn_w),
        ('nu_orn_s_err', nu_orn_s_err),
        ('nu_orn_w_err', nu_orn_w_err),
        ('nu_pn_s_err', nu_pn_s_err),
        ('nu_pn_w_err', nu_pn_w_err),
        ])
    return out_olsen
    
   
    


def orn_al_settings(axs):
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
        
    
    elif fig_id=='ts_s':
        # horizontal line
        axs[1].plot([200, 250], [150, 150], color=black) 
        axs[1].text(205, 120, '50 ms', color=black, fontsize=scale_fs)
        # vertical line
        axs[1].plot([200, 200], [150, 250], color=black) 
        axs[1].text(160, 160, '100 Hz', color=black,rotation=90, fontsize=scale_fs)


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
    if inh_cond == 'noin':
        linest = '-'
    elif inh_cond == 'nsi':
        linest = '--'
    elif inh_cond == 'ln':
        linest = '-.'
    trsp = .3
    
    if inh_cond == 'noin':
        axs[0].plot(t-t_zero, 100*u_od[:,0], color=green, 
               linewidth=lw, linestyle=linest, label='glom : '+'%d'%(1))
        # 2nd ORNs
        if (fig_id == 'ts_a') or (fig_id == 'pl'):
            axs[0].plot(t-t_zero, 100*u_od[:,1], color=purple, 
                linewidth=lw, label='glom : '+'%d'%(2))
    
    if (inh_cond == 'noin') | (inh_cond == 'nsi'):
        X1 = np.mean(orn_sdf[:,:,:n_orns_recep], axis=2)
        mu1 = X1.mean(axis=0)
        sigma1 = X1.std(axis=0)
        axs[1].plot(orn_sdf_time-t_zero, mu1, linestyle=linest, linewidth=lw, color=green)
        axs[1].fill_between(orn_sdf_time-t_zero, 
           mu1+sigma1, mu1-sigma1, facecolor=green, alpha=trsp)
    
        # 2nd ORNs
        if (fig_id == 'ts_a') or (fig_id == 'pl'):
            X1 = np.mean(orn_sdf[:,:,n_orns_recep:], axis=2)
            mu1 = X1.mean(axis=0)
            sigma1 = X1.std(axis=0)
            axs[1].plot(orn_sdf_time-t_zero, mu1, linestyle=linest, linewidth=lw, color=purple)
            axs[1].fill_between(orn_sdf_time-t_zero, 
                mu1+sigma1, mu1-sigma1, facecolor=purple, alpha=trsp)
        
    X1 = np.mean(pn_sdf[:,:,:n_pns_recep], axis=2)
    mu1 = X1.mean(axis=0)
    sigma1 = X1.std(axis=0)
    axs[2].plot(pn_sdf_time-t_zero, mu1, linestyle=linest, linewidth=lw, color=green)
    axs[2].fill_between(orn_sdf_time-t_zero, 
           mu1+sigma1, mu1-sigma1, facecolor=green, alpha=trsp)
    
    # 2nd ORNs
    if (fig_id == 'ts_a') or (fig_id == 'pl'):
        X1 = np.mean(pn_sdf[:,:,n_pns_recep:], axis=2)
        mu1 = X1.mean(axis=0)
        sigma1 = X1.std(axis=0)
        axs[2].plot(pn_sdf_time-t_zero, mu1, linestyle=linest, linewidth=lw, color=purple)
        axs[2].fill_between(pn_sdf_time-t_zero, 
                mu1+sigma1, mu1-sigma1, facecolor=purple, alpha=trsp)
    
def orn_al_diag_settings(axs):
    # FIGURE SETTING
    
    dl = .29
    ll_new = .04+np.array([0,.005+ dl,.01+ 2*dl])
    dh = .29
    bb_new = .02+np.array([.01+ 2*dh,.005+ dh,0])
    
    for id_row in range(3):
        for id_col in range(3):
            ll, bb, ww, hh = axs[id_row, id_col].get_position().bounds
            axs[id_row, id_col].set_position([ll_new[id_col], 
                                             bb_new[id_row], dl, dh])
    
    for id_row in range(3):
        for id_col in range(3):
            axs[id_row, id_col].set_xlim(t2plot)
            axs[id_row, id_col].tick_params(axis='both', labelsize=label_fs)
            axs[id_row, id_col].set_xticklabels('')
            # axs[id_row, id_col].axis('off')
    
    for id_row in range(3):
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
        axs[2, 1].plot([2000, 2500], [50, 50], color=black) 
        axs[2, 1].text(2000, 2, '500 ms', color=black, fontsize=scale_fs)
        # vertical line
        axs[2, 1].plot([2000, 2000], [50, 150], color=black) 
        axs[2, 1].text(1600, 40, '100 Hz', color=black,rotation=90, fontsize=scale_fs)
    
    if (fig_id=='ts_a') | (fig_id=='pl'):
        axs[0, 0].text(.05, 1.4, 'a', transform=axs[0,0].transAxes, 
                 fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    if fig_id=='ts_a':
        x4text = 280
    elif fig_id=='ts_s':
        x4text = 190
    else:
        x4text = 3400
    
    axs[0, 2].text(x4text, 75, 'ctrl\n model', fontsize=scale_fs, fontweight='bold',rotation=90)
    axs[1, 2].text(x4text, 75, 'NSI\n model', fontsize=scale_fs, fontweight='bold',rotation=90)
    axs[2, 2].text(x4text, 75, 'LN\n model', fontsize=scale_fs, fontweight='bold', rotation=90)
        
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
    if inh_cond == 'noin':
        id_row = 0
    elif inh_cond == 'nsi':
        id_row = 1
    elif inh_cond == 'ln':
        id_row = 2
    trsp = .3

    if inh_cond == 'noin':
        axs[0, 0].plot(t-t_zero, 100*u_od[:,0], color=green, 
               linewidth=lw, label='glom : '+'%d'%(1))
        if (fig_id == 'ts_a') or (fig_id == 'pl'):
            axs[0, 0].plot(t-t_zero, 100*u_od[:,1], '--', color=purple, 
               linewidth=lw, label='glom : '+'%d'%(2))
    
    if (inh_cond == 'noin') | (inh_cond == 'nsi'):
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
inh_conds   = ['noin', 'ln', 'nsi',] 

# LOAD PARAMS FROM A FILE
# fld_params = 'NSI_analysis/trials/' #Olsen2010
# name_data = 'params_al_orn.ini'
# params_al_orn = pickle.load(open(fld_params + name_data,  "rb" ))
params_al_orn = set_orn_al_params.main(2)

stim_params         = params_al_orn['stim_params']
orn_layer_params    = params_al_orn['orn_layer_params']
# orn_params          = params_al_orn['orn_params']
sdf_params          = params_al_orn['sdf_params']
al_params           = params_al_orn['al_params']
pn_ln_params        = params_al_orn['pn_ln_params']

n_sens_type         = orn_layer_params.__len__()  # number of type of sensilla

# ORN NSI params

# fig_id options:  # 'ts_s' #  'ts_a' # 'pl'
fig_id                  = 'pl' 

fld_analysis            = 'NSI_analysis/triangle_stim/'
nsi_str                 = 0.6
alpha_ln                = 0.6
pn_ln_params['tau_ln']  = 250


# figure and output options
fig_save    = 1
data_save   = 0    
verbose     = 0
olsen_fig   = 0


n_lines     = 10

if fig_id == 'ts_s':
    # stim params
    delay                       = 0    
    stim_params['stim_type']    = 'ts' # 'id_l'  # 'ts'
    stim_params['stim_dur']     = np.array([50, 50])
    stim_params['t_tot']        = 1000
    t_on                        = 700
    stim_params['conc0']        = 1.85e-4    # 2.85e-4
    peak                        = 2e-2
    
elif fig_id == 'ts_a':
    # Stimulus params 
    delay                       = 100
    stim_params['stim_type']    = 'ts' # 'id_l'  # 'ts'
    stim_params['stim_dur']     = np.array([50, 50])
    stim_params['t_tot']        = 1000+delay
    t_on                        = 700
    stim_params['conc0']        = 1.85e-4    # 2.85e-4
    peak                        = 1e-3  
    
elif fig_id == 'pl':
    fld_analysis        = 'NSI_analysis/real_plumes/example'
    fig_orn_dyn         = 1
    max_stim_seed       = 3
    orn_fig_name        = '/ORN_lif_dyn_realplume.png'
       
    # stim params
    delay                       = 0    
    stim_params['stim_type']    = 'pl' # 'ts' # 'ss' # 'rs'# 'pl'
    stim_params['t_tot']        = 4300        # ms 
    t_on                        = 700
    stim_params['conc0']        = 1.85e-4
    stim_params['stim_dur']     = np.array([4000, 4000])
    peak                        = np.array([5e-4])         # concentration value for ORN1
   
    # real plumes params
    plume_params = dict([
                        ('whiff_max', 3),
                        ('blank_max', 25),
                        ('rho_t_exp', 0),#[0, 1, 3, 5]
                        ('stim_seed', 10),
                        ])
    stim_params['plume_params'] = plume_params
    
    
peak_ratio      = 1
stim_params['t_on'] = np.array([t_on, t_on+delay])      # ms 


dt_sdf      = params_al_orn['sdf_params']['dt_sdf']
sdf_size    = int(stim_params['t_tot']/dt_sdf)


n_neu       = params_al_orn['orn_layer_params'][0]['n_neu']
n_orns_recep = n_neu*al_params['n_orns_recep']# number of ORNs per each receptor
n_pns_recep = n_neu*al_params['n_pns_recep'] # number of PNs per each receptor

orn_sdf = np.zeros((n_lines, sdf_size, n_orns_recep))
pn_sdf = np.zeros((n_lines, sdf_size, n_pns_recep))


tic = timeit.default_timer()

peaks                       = [1.5e-3] #[1.85e-4, 5e-4, 1.5e-3, 2e-2, 2e-1]
n_peaks = len(peaks)

# INITIALIZE OUTPUT VARIABLES    
conc_s    = np.zeros((n_peaks, 3))
conc_th = np.zeros((n_peaks, 3))
nu_orn_s = np.zeros((n_peaks, 3))
nu_pn_s  = np.zeros((n_peaks, 3))
nu_orn_w = np.zeros((n_peaks, 3))
nu_pn_w = np.zeros((n_peaks, 3))

nu_orn_s_err  = np.zeros((n_peaks, 3))
nu_pn_s_err   = np.zeros((n_peaks, 3))
nu_orn_w_err  = np.zeros((n_peaks, 3))
nu_pn_w_err   = np.zeros((n_peaks, 3))

for id_p, peak in enumerate(peaks):
    stim_params['concs'] = np.array([peak, peak*peak_ratio])

    # FIGURE         
    fig_name    = 'diag_stim_'+ stim_params['stim_type']+'_delay_%d'%delay + \
                    '_nsi%.2f'%nsi_str+'_ln%.1f'%alpha_ln + '_peak%.5f'%peak
    if fig_id == 'ts_a':
        fig_name = fig_name + '_tauln_%d'%pn_ln_params['tau_ln']
    
    
    
    
    if fig_id == 'pl':
        t2plot = -20, stim_params['t_tot']-t_on-500
        rs = 3 # number of rows
        cs = 3 # number of cols
        fig_pn, axs = plt.subplots(rs, cs, figsize=[9, 3])
        orn_al_diag_settings(axs)
        
    else:
        t2plot = -20, stim_params['t_tot']-t_on
        rs = 1 # number of rows
        cs = 3 # number of cols
        fig_pn, axs = plt.subplots(rs, cs, figsize=[9, 3])
        orn_al_settings(axs)
        
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
    
        for id_l in range(n_lines):  
                     
            # Run flynose 
            
            # ORNs layer dynamics
            [t, u_od,  orn_spikes_t, orn_sdf_tmp, orn_sdf_time] = \
                ORNs_layer_dyn.main(params_al_orn, verbose=verbose, )
            
            orn_sdf[id_l,:,:] = orn_sdf_tmp
            
            # AL dynamics
            [t, pn_spike_matrix, pn_sdf_tmp, pn_sdf_time,
                ln_spike_matrix, ln_sdf_tmp, ln_sdf_time,] = \
                AL_dyn.main(params_al_orn, orn_spikes_t, verbose=verbose, )
            pn_sdf[id_l,:,:] = pn_sdf_tmp
            
            
            
        
        data2plot = dict([
                        ('t', t),
                        ('u_od', u_od),
                        ('orn_sdf', orn_sdf),
                        ('orn_sdf_time',orn_sdf_time), 
                        ('pn_sdf', pn_sdf),
                        ('pn_sdf_time',pn_sdf_time), 
                        ])          
    
    
        if fig_id == 'pl':
            orn_al_diag_plot(data2plot, params_al_orn, inh_cond)
        else:
            orn_al_plot(data2plot, params_al_orn, inh_cond)
        
        output2an = dict([
                            ('t', t),
                            ('u_od',u_od),
                            ('orn_sdf', np.mean(orn_sdf, axis=0)),
                            ('orn_sdf_time',orn_sdf_time), 
                            ('pn_sdf', np.mean(pn_sdf, axis=0)),
                            ('pn_sdf_time', pn_sdf_time), 
                            ]) 
        out_olsen       = olsen2010_data(output2an, params_al_orn,)
        conc_th[id_p, id_inh]   = peak
        conc_s[id_p, id_inh]      = out_olsen['conc_s']
        nu_orn_s[id_p, id_inh]    = out_olsen['nu_orn_s']
        nu_orn_w[id_p, id_inh]    = out_olsen['nu_orn_w']
        nu_orn_s_err[id_p, id_inh] = out_olsen['nu_orn_s_err']
        nu_orn_w_err[id_p, id_inh] = out_olsen['nu_orn_w_err']
        
        nu_pn_s[id_p, id_inh]     = out_olsen['nu_pn_s']
        nu_pn_w[id_p, id_inh]     = out_olsen['nu_pn_w']
        nu_pn_s_err[id_p, id_inh] = out_olsen['nu_pn_s_err']
        nu_pn_w_err[id_p, id_inh] = out_olsen['nu_pn_s_err']
        
    
    plt.show()    

    if fig_save:
        print('saving figure in '+fld_analysis)
        fig_pn.savefig(fld_analysis + '/'+ fig_name + '.png')


toc = timeit.default_timer()-tic

print('Diag shade plot elapsed time: %.1f'%(toc))

#%% FIGURE Olsen 2010: ORN vs PN during step stimulus
if olsen_fig:
    rs = 1
    cs = 2
    fig3, axs = plt.subplots(rs,cs, figsize=(11, 4), )
    plt.rc('text', usetex=True)
    
    # PLOT
    for id_inh, inh_cond in enumerate(inh_conds):
        tmp_conc_th = conc_th[:, id_inh]
        tmp_conc_s = conc_s[:, id_inh]
        tmp_nu_orn_s = nu_orn_s[:, id_inh]
        tmp_nu_orn_w = nu_orn_w[:, id_inh]
        tmp_nu_orn_s_err = nu_orn_s_err[:, id_inh]
        tmp_nu_orn_w_err = nu_orn_w_err[:, id_inh]
        
        tmp_nu_pn_s = nu_pn_s[:, id_inh]
        tmp_nu_pn_w = nu_pn_w[:, id_inh]
        tmp_nu_pn_s_err = nu_pn_s_err[:, id_inh]
        tmp_nu_pn_w_err = nu_pn_w_err[:, id_inh]    
        
        
        if id_inh == 0:
            linest = '-'
        elif id_inh == 1:
            linest = '--'
        elif id_inh == 2:
            linest = '-.'
            
        axs[0].errorbar(tmp_nu_orn_s, tmp_nu_pn_s, yerr = tmp_nu_pn_s_err, 
                        fmt='o', label=inh_cond, linestyle=linest)
        
        # strong side
        axs[1].errorbar(tmp_conc_th+1e-5, tmp_nu_orn_s, yerr = tmp_nu_orn_s_err, 
                linewidth=lw, color='blue', ms=15, label='ORNs '+inh_cond, linestyle=linest)
        axs[1].errorbar(tmp_conc_th-1e-5, tmp_nu_pn_s, yerr = tmp_nu_pn_s_err, 
                linewidth=lw, color='orange', ms=15, label='PNs '+inh_cond, linestyle=linest)
     
     
    # SETTINGS
    axs[0].set_ylabel(r'PN (Hz)', fontsize=label_fs)
    axs[0].set_xlabel(r'ORN (Hz)', fontsize=label_fs)
    axs[0].legend(loc='upper left', fontsize=legend_fs)
    
    
    axs[1].legend(loc='upper left', fontsize=legend_fs)

    axs[1].set_ylabel(r'Firing rates (Hz)', fontsize=label_fs)

    axs[1].set_xlabel(r'concentration (au)', fontsize=label_fs)
    axs[1].legend(loc=0, fontsize=legend_fs, frameon=False)
    
    axs[0].text(-.2, 1.0, 'b', transform=axs[0].transAxes, 
         fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    axs[1].text(-.2, 1.0, 'c', transform=axs[1].transAxes,
         fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    axs[1].set_xscale('log')
    
    for j in [0,1]:
        axs[j].tick_params(axis='both', labelsize=label_fs)
        axs[j].spines['right'].set_color('none')
        axs[j].spines['top'].set_color('none')
    
    
    
    ll, bb, ww, hh = axs[0].get_position().bounds
    axs[0].set_position([ll, bb+.1, ww, hh])
    
    ll, bb, ww, hh = axs[1].get_position().bounds
    axs[1].set_position([ll+.1, bb+.05, ww, hh])

    plt.show()
    
    if fig_save:
        fig_name = 'Olsen2010_inh_' + inh_cond +  \
            '_stim_'+ stim_params['stim_type'] +'_dur_%d'%stim_params['stim_dur'][0]
            
        print('saving Olsen2010 PN-ORN figure in '+fld_analysis)
        fig3.savefig(fld_analysis+ fig_name +'.png')


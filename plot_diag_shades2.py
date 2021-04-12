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



# *****************************************************************
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

#%% ****************************************************
def orn_al_settings(axs):

    
    # FIGURE SETTING
    
    dl = .31
    ll_new = .04+np.array([0,.005+ dl,.01+ 2*dl])
    dh = .29
    bb_new = .02+np.array([.01+ 2*dh,.005+ dh,0])
    
    for id_row in range(3):
        for id_col in range(3):
            ll, bb, ww, hh = axs[id_row, id_col].get_position().bounds
            axs[id_row, id_col].set_position([ll_new[id_col], 
                                             bb_new[id_row], dl-.01, dh])
    
    for id_row in range(3):
        for id_col in range(3):
            axs[id_row, id_col].set_xlim(t2plot)
            axs[id_row, id_col].tick_params(axis='both', labelsize=label_fs)
            axs[id_row, id_col].set_xticklabels('')
            axs[id_row, id_col].axis('off')
    
    for id_row in range(3):
        axs[id_row, 1].set_ylim((0, 200))
        axs[id_row, 2].set_ylim((0, 200))
    
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
        axs[2, 1].text(2000, 10, '500 ms', color=black, fontsize=scale_fs)
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
        x4text = 4000
    #%%
    axs[0, 2].text(x4text, 75, 'ctrl\n model', fontsize=scale_fs, fontweight='bold',rotation=90)
    axs[1, 2].text(x4text, 75, 'NSI\n model', fontsize=scale_fs, fontweight='bold',rotation=90)
    axs[2, 2].text(x4text, 75, 'LN\n model', fontsize=scale_fs, fontweight='bold', rotation=90)
        
def orn_al_plot(data_tmp, params_tmp, inh_cond):
    
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
    axs[id_row, 2].plot(pn_sdf_time-t_zero, mu1, linewidth=lw, color=purple)
    axs[id_row, 2].fill_between(pn_sdf_time-t_zero, 
           mu1+sigma1, mu1-sigma1, facecolor=purple, alpha=trsp)
    

#***********************************************
# Standard params
# fld_analysis= 'NSI_analysis/triangle_stim/diag_figs'
inh_conds   = ['nsi', 'ln', 'noin'] 

# %% LOAD PARAMS FROM A FILE
fld_params = 'NSI_analysis/trials/' #Olsen2010
name_data = 'params_al_orn.ini'
params_al_orn = pickle.load(open(fld_params + name_data,  "rb" ))

stim_params         = params_al_orn['stim_params']
orn_layer_params    = params_al_orn['orn_layer_params']
# orn_params          = params_al_orn['orn_params']
sdf_params          = params_al_orn['sdf_params']
al_params           = params_al_orn['al_params']
pn_ln_params        = params_al_orn['pn_ln_params']

n_sens_type         = orn_layer_params.__len__()  # number of type of sensilla

#%%

# ORN NSI params

fld_analysis = 'NSI_analysis/triangle_stim/'
nsi_str     = 0.6
alpha_ln    =  0.6

n_lines     = 10
# Stimulus params 
# fig_id options:  # 'ts_s' #  'ts_a' # 'pl'
fig_id = 'ts_a' 

if fig_id == 'ts_s':
    # stim params
    delay                       = 0    
    stim_params['stim_type']    = 'ts' # 'ss'  # 'ts'
    stim_params['stim_dur']     = np.array([50, 50])
    stim_params['t_tot']        = 1000
    ton1                        = 700
    stim_params['conc0']        = 1.85e-4    # 2.85e-4
    peak                        = 3e-4      
    
elif fig_id == 'ts_a':
    delay                       = 100  
    stim_params['stim_type']    = 'ts' # 'ss'  # 'ts'
    stim_params['stim_dur']     = np.array([50, 50])
    stim_params['t_tot']        = 1000+delay
    ton1                        = 700
    stim_params['conc0']        = 1.85e-4    # 2.85e-4
    peak                        = 3e-4  
    
elif fig_id == 'pl':
    fld_analysis= 'NSI_analysis/real_plumes/example'
    stim_type   = 'pl'          # 'ts'  # 'ss' # 'pl'
    delay       = 0    
    t_tot       = 4300 # ms 
    stim_dur    = 4000
    # real plumes params
    b_max       =  25 # 3, 50, 150
    w_max       =   3 #3, 50, 150
    rho         =   0 #[0, 1, 3, 5]: 
    stim_seed   = 0
    peak        = 1.4 

    
peak_ratio  = 1
stim_params['t_on'] = np.array([ton1, ton1 +delay])      # ms 
stim_params['concs'] = np.array([peak, peak*peak_ratio])


dt_sdf      = params_al_orn['sdf_params']['dt_sdf']
sdf_size    = int(stim_params['t_tot']/dt_sdf)

# figure and output options
fig_save    = 0
data_save   = 0    
verbose     = 0

fig_name    = 'diag_stim_'+ stim_params['stim_type']+'_delay_%d'%delay + \
                    '_nsi%.2f'%nsi_str+'_ln%.1f'%alpha_ln

n_neu       = params_al_orn['orn_layer_params'][0]['n_neu']
n_orns_recep = n_neu*al_params['n_orns_recep']# number of ORNs per each receptor
n_pns_recep = n_neu*al_params['n_pns_recep'] # number of PNs per each receptor

orn_sdf = np.zeros((n_lines, sdf_size, n_orns_recep))
pn_sdf = np.zeros((n_lines, sdf_size, n_pns_recep))


tic = timeit.default_timer()

# *****************************************************************
# FIGURE         
t2plot = -20, stim_params['stim_dur'][1]+150

rs = 3 # number of rows
cs = 3 # number of cols

fig_pn, axs = plt.subplots(rs, cs, figsize=[9, 3])
orn_al_settings(axs)

# simulations and append to figure         
for inh_cond in inh_conds:
                            
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

    for ss in range(n_lines):        
                      
        # Run flynose 
        
        # ORNs layer dynamics
        [t, u_od,  orn_spikes_t, orn_sdf_tmp, orn_sdf_time] = \
            ORNs_layer_dyn.main(params_al_orn, verbose=verbose, )
        
        orn_sdf[ss,:,:] = orn_sdf_tmp
        
        # AL dynamics
        [t, pn_spike_matrix, pn_sdf_tmp, pn_sdf_time,
            ln_spike_matrix, ln_sdf, ln_sdf_time,] = \
            AL_dyn.main(params_al_orn, orn_spikes_t, verbose=verbose, )
        pn_sdf[ss,:,:] = pn_sdf_tmp
        
    data2plot = dict([
                    ('t', t),
                    ('u_od', u_od),
                    ('orn_sdf', orn_sdf),
                    ('orn_sdf_time',orn_sdf_time), 
                    ('pn_sdf', pn_sdf),
                    ('pn_sdf_time',pn_sdf_time), 
                    ])          

    orn_al_plot(data2plot, params_al_orn, inh_cond)
    plt.show()    

if fig_save:
    print('saving figure in '+fld_analysis)
    fig_pn.savefig(fld_analysis + '/'+ fig_name + '.png')

toc = timeit.default_timer()-tic

print('Diag shade plot elapsed time: %.1f'%(toc))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:30:03 2020

plot_diag_shades.py

@author: mario
"""

import numpy as np
import matplotlib.pyplot as plt
import timeit
import flynose
import sdf_krofczik

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

    #%%
    # vertical/horizontal lines for time/Hz scale
    
    if fig_id=='ts_s':
        # horizontal line
        axs[2, 1].plot([100, 150], [50, 50], color=black) 
        axs[2, 1].text(105, 10, '50 ms', color=black, fontsize=scale_fs)
        # vertical line
        axs[2, 1].plot([100, 100], [50, 150], color=black) 
        axs[2, 1].text(80, 140, '100 Hz', color=black,rotation=90, fontsize=scale_fs)
    elif fig_id=='ts_a':
        # horizzontal line
        axs[2, 1].plot([100, 150], [50, 50], color=black) 
        axs[2, 1].text(105, 10, '50 ms', color=black, fontsize=scale_fs)
        # vertical line
        axs[2, 1].plot([100, 100], [50, 150], color=black) 
        axs[2, 1].text(75, 140, '100 Hz', color=black,rotation=90, fontsize=scale_fs)
    elif fig_id=='pl':
        # horizzontal line
        axs[2, 1].plot([2000, 2500], [50, 50], color=black) 
        axs[2, 1].text(2000, 10, '500 ms', color=black, fontsize=scale_fs)
        # vertical line
        axs[2, 1].plot([2000, 2000], [50, 150], color=black) 
        axs[2, 1].text(1600, 130, '100 Hz', color=black,rotation=90, fontsize=scale_fs)
    
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
        
    #%%
        
def orn_al_plot(all_data_tmp, params2an, inh_cond):

    stim_params = params2an[2]
    [stim_type, pts_ms, t_tot, t_on, t_off, 
                       concs, plume_params] = stim_params

    t_zero = t_on[0]
    num_orns_glo = 40
    num_pns_glo = 5
    
    t = all_data_tmp[0]
    u_od =all_data_tmp[1]
    orn_sdf_time = all_data_tmp[2]
    orn_sdf_norm = all_data_tmp[3]        
    pn_sdf_time = all_data_tmp[4] 
    pn_sdf_norm = all_data_tmp[5]    
    
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
        X1 = np.mean(orn_sdf_norm[:,:,:num_orns_glo], axis=2)
        mu1 = X1.mean(axis=0)
        sigma1 = X1.std(axis=0)
        axs[id_row, 1].plot(orn_sdf_time-t_zero, mu1, linewidth=lw, color=green)
        axs[id_row, 1].fill_between(orn_sdf_time-t_zero, 
           mu1+sigma1, mu1-sigma1, facecolor=green, alpha=trsp)
    
        X1 = np.mean(orn_sdf_norm[:,:,num_orns_glo:], axis=2)
        mu1 = X1.mean(axis=0)
        sigma1 = X1.std(axis=0)
        axs[id_row, 1].plot(orn_sdf_time-t_zero, mu1, linewidth=lw, color=purple)
        axs[id_row, 1].fill_between(orn_sdf_time-t_zero, 
           mu1+sigma1, mu1-sigma1, facecolor=purple, alpha=trsp)
        
    X1 = np.mean(pn_sdf_norm[:,:,:num_pns_glo], axis=2)
    mu1 = X1.mean(axis=0)
    sigma1 = X1.std(axis=0)
    axs[id_row, 2].plot(pn_sdf_time-t_zero, mu1, linewidth=lw, color=green)
    axs[id_row, 2].fill_between(orn_sdf_time-t_zero, 
           mu1+sigma1, mu1-sigma1, facecolor=green, alpha=trsp)
    
    X1 = np.mean(pn_sdf_norm[:,:,num_pns_glo:], axis=2)
    mu1 = X1.mean(axis=0)
    sigma1 = X1.std(axis=0)
    axs[id_row, 2].plot(pn_sdf_time-t_zero, mu1, linewidth=lw, color=purple)
    axs[id_row, 2].fill_between(pn_sdf_time-t_zero, 
           mu1+sigma1, mu1-sigma1, facecolor=purple, alpha=trsp)
    

#***********************************************
# Standard params
fld_analysis= 'NSI_analysis/triangle_stim/diag_figs'
inh_conds   = ['nsi', 'ln', 'noin'] 

# ORN NSI params
alpha_ln    = 13.3
nsi_str     = 0.3

# Stimulus params 
fig_id = 'pl' # 'ts_s' #  'ts_a' # 'pl'
if fig_id == 'ts_s':
    stim_type   = 'ts'          # 'ts'  # 'ss' # 'pl'
    delay       = 0    
    t_tot       = 600 +delay       # ms 
    stim_dur    = 50
    peak        = 1.8 
    # real plumes params
    b_max       = np.nan # 3, 50, 150
    w_max       = np.nan #3, 50, 150
    rho         = np.nan #[0, 1, 3, 5]: 
    stim_seed   = np.nan
    
elif fig_id == 'ts_a':
    stim_type   = 'ts'          # 'ts'  # 'ss' # 'pl'
    delay       = 100    
    t_tot       = 600 +delay       # ms 
    stim_dur    = 50
    peak        = 1.8 

    # real plumes params
    b_max       = np.nan # 3, 50, 150
    w_max       = np.nan #3, 50, 150
    rho         = np.nan #[0, 1, 3, 5]: 
    stim_seed   = np.nan
    
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
pts_ms      = 5

t_on        = [300, 300+delay]    # ms
t_off       = np.array(t_on)+stim_dur # ms
concs       = [peak, peak*peak_ratio]
tau_sdf     = 20
dt_sdf      = 5      
sdf_size    = int(t_tot/dt_sdf)       

plume_params = [rho, w_max, b_max, stim_seed]

# figure and output options
orn_fig     = 0
al_dyn      = 1
al_fig      = 0
fig_ui      = 0        
verbose     = 0    
fig_save    = 1
data_save   = 0    

fig_opts    = [orn_fig, al_fig, fig_ui, fig_save, data_save, al_dyn, 
            verbose, fld_analysis] 

stim_params = [stim_type, pts_ms, t_tot, t_on, t_off, 
                       concs, plume_params]

fig_name    = 'diag_stim_'+stim_type+'_delay_%d'%delay


# Plot settings/params
n_lines = 10

num_glo = 2
num_orns = num_glo*40    # number of ORNs per each glomerulus
num_pns = num_glo*5     # number of PNs per each glomerulus

orn_sdf = np.zeros((n_lines, sdf_size, num_orns))
pn_sdf = np.zeros((n_lines, sdf_size, num_pns))

# *****************************************************************

tic = timeit.default_timer()

# *****************************************************************
# FIGURE         
t2plot = -20, stim_dur+150+delay
rs = 3 # number of rows
cs = 3 # number of cols

fig_pn, axs = plt.subplots(rs, cs, figsize=[9, 3])
orn_al_settings(axs)

# *****************************************************************
# simulations and append to figure         
for inh_cond in inh_conds:
                            
    if inh_cond == 'nsi':
        params2an = [nsi_str, .0, stim_params]
    elif inh_cond == 'noin':
        params2an = [0, 0, stim_params]
    elif inh_cond == 'ln':
        params2an = [.0, alpha_ln, stim_params]
    
    for ss in range(n_lines):        
                      
        # RUN SIM
        flynose_out = flynose.main(params2an, fig_opts)
        [t, u_od, orn_spike_matrix, pn_spike_matrix, 
              ln_spike_matrix, ] = flynose_out
                
        # Calculate the SDF for ORNs, PNs and LNs
        orn_sdf_tmp, orn_sdf_time = sdf_krofczik.main(
                    orn_spike_matrix, sdf_size, tau_sdf, dt_sdf)  # (Hz, ms)
        orn_sdf[ss,:,:] = orn_sdf_tmp*1e3
        
        pn_sdf_tmp, pn_sdf_time = sdf_krofczik.main(
                    pn_spike_matrix, sdf_size, tau_sdf, dt_sdf)  # (Hz, ms)
        pn_sdf[ss,:,:] = pn_sdf_tmp*1e3
        
    data2plot = [t, u_od, orn_sdf_time, orn_sdf, 
                  pn_sdf_time, pn_sdf, ]
    orn_al_plot(data2plot, params2an, inh_cond)
        
if fig_save:
    print('saving figure in '+fld_analysis)
    fig_pn.savefig(fld_analysis+ '/'+ fig_name + '.png')

toc = timeit.default_timer()            
print('elapsed time: %.1f'%(toc-tic))
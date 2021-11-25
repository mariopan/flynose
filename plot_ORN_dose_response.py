#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 17:58:19 2021

Dynamic range of the two networks calculatedfor several values of the ratio in 
sensitivity of the two ORNs

plot_ORN_dose_response.py

@author: mario
"""


import numpy as np
import pickle

import matplotlib.pyplot as plt
import string

from scipy.interpolate import interp1d

# STANDARD FIGURE PARAMS
lw = 3
fs = 20
plt.rc('text', usetex=True)  # laTex in the polot
#plt.rc('font', family='serif')
fig_size = [12, 12]
fig_position = 1300,10
title_fs = 20 # font size of ticks
label_fs = 22# font size of labels
panel_fs = 30
black   = 'xkcd:black'
gray    = 'xkcd:gray'
blue    = 'xkcd:blue'
red     = 'xkcd:red'
green   = 'xkcd:green'
purple  = 'xkcd:purple'
orange  = 'xkcd:orange'
pink    = 'xkcd:pink'
cyan    = 'xkcd:cyan'
cmap    = plt.get_cmap('rainbow')
alphabet = string.ascii_lowercase

recep_clrs = ['green','purple','cyan','red']


name_analysis   = 'dose_response'

fld_analysis    = 'NSI_analysis/'+name_analysis+'/30trials/'
fld_analysis    = 'NSI_analysis/'+name_analysis+'/sum_separated/'
fld_analysis    = 'NSI_analysis/'+name_analysis+'/sum_separated_10loops/'

fld_analysis    = 'NSI_analysis/'+name_analysis+'/sum_separated_10loops2/'
n_loops         = 10

# fld_analysis    = 'NSI_analysis/'+name_analysis+'/nsi0.3/'
# fld_analysis    = 'NSI_analysis/'+name_analysis+'/nsi0.4/'
# n_loops         = 3 

c_thr_lw        =  .1
c_thr_hg        = .9

stim_durs       =  [50, ]

fig_save        = 1
fig_dpi=350
dynrng_fig      = 1
thrs_fig        = 1
dose_response_fig       = 1

figure_name = 'stim_ts' + \
        '_durs_%d-%d'%(stim_durs[0], stim_durs[-1],) + \
        '_ctrl_vs_nsi'

dynrng_figure_name = 'dyn_rng_' + figure_name

thrs_figure_name  = 'thrs_' + figure_name

shift_ratios    = [1, .1, .01, .001, .0001]# [1, .01, .0001,] # 
shift_0         = 1e4
n_shift_ratios = len(shift_ratios)
    

inh_conds = ['ctrl', 'NSI']
n_inh_conds = len(inh_conds)
    

thr_hg_sum      = np.zeros((n_inh_conds, n_shift_ratios, n_loops))
thr_lw_sum      = np.zeros((n_inh_conds, n_shift_ratios, n_loops))
dyn_rng_sum     = np.zeros((n_inh_conds, n_shift_ratios, n_loops))

thr_hg_s      = np.zeros((n_inh_conds, n_shift_ratios, n_loops))
thr_lw_s      = np.zeros((n_inh_conds, n_shift_ratios, n_loops))
dyn_rng_s     = np.zeros((n_inh_conds, n_shift_ratios, n_loops))

thr_hg_w      = np.zeros((n_inh_conds, n_shift_ratios, n_loops))
thr_lw_w      = np.zeros((n_inh_conds, n_shift_ratios, n_loops))
dyn_rng_w     = np.zeros((n_inh_conds, n_shift_ratios, n_loops))

thr_lw_ratio   = np.zeros((n_inh_conds, n_shift_ratios, n_loops))
thr_hg_ratio   = np.zeros((n_inh_conds, n_shift_ratios, n_loops))

# np.log10(thr_hg_w[id_inh, id_dur]/ thr_lw_w[id_inh, id_dur] )
measure = 'peak'   # 'avg'

if dose_response_fig:
    fig_dr, axs = plt.subplots(nrows=n_shift_ratios, ncols=n_inh_conds, figsize=[14, 8.5]) # n_inh_conds
    trsp = .4
        
for id_ratio in range(n_shift_ratios):
    alpha_r_1 = int((shift_0*shift_ratios[id_ratio])**0.82206687*12.6228808)
    alpha_r_0 = int(shift_0**0.82206687*12.6228808)
    
    tmp_name = '_stim_ts' + \
        '_durs_%d-%d'%(stim_durs[0], stim_durs[-1],) + \
        '_ctrl_vs_nsi'+\
        '_alphar_0_%d'%alpha_r_0 +\
        '_alphar_1_%d'%alpha_r_1
        
    data = pickle.load(open(
        fld_analysis + name_analysis + tmp_name+'.pickle',  "rb" ))
    
    concs           = data['concs']
    concs_intp                  = 5*np.logspace(-20, -1, 1000) 
    if measure == 'peak':
        orn_w = data['peak_ornw']
        orn_s = data['peak_orns']
    elif measure == 'avg':
        orn_w = data['avg_ornw']
        orn_s = data['avg_orns']
    
    # n_loops = 3# data['n_loops']
    for id_inh, inh_cond in enumerate(inh_conds):
        for id_loop in range(n_loops):
    
            # ANALYSIS of weak odor/ORN 
            nu_intp = interp1d(concs, orn_w[:, id_inh, id_loop], kind=1)
            
            dr_peak = nu_intp(concs_intp)
            
            if c_thr_lw<.5:
                thr_lw_w[id_inh, id_ratio, id_loop] = concs_intp[next(x[0] for x in enumerate(dr_peak) 
                                                     if x[1] > c_thr_lw*np.max(dr_peak))] 
            else:
                thr_lw_w[id_inh, id_ratio, id_loop] = concs_intp[next(x[0] for x in enumerate(dr_peak) 
                                                     if x[1] > c_thr_lw*dr_peak[0])] 
            thr_hg_w[id_inh, id_ratio, id_loop] = concs_intp[next(x[0] for x in enumerate(dr_peak) 
                                                     if x[1] > c_thr_hg*np.max(dr_peak))]
            
            dyn_rng_w[id_inh, id_ratio, id_loop] = np.log10(thr_hg_w[id_inh, id_ratio, id_loop]
                                                / thr_lw_w[id_inh, id_ratio, id_loop] )
            
            # ANALYSIS of strong odor/ORN
            nu_intp = interp1d(concs, orn_s[:, id_inh, id_loop], kind=1)
            
            dr_peak = nu_intp(concs_intp)
            
            if c_thr_lw<.5:
                thr_lw_s[id_inh, id_ratio, id_loop] = concs_intp[next(x[0] for x in enumerate(dr_peak) 
                                                     if x[1] > c_thr_lw*np.max(dr_peak))] 
            else:
                thr_lw_s[id_inh, id_ratio, id_loop] = concs_intp[next(x[0] for x in enumerate(dr_peak) 
                                                     if x[1] > c_thr_lw*dr_peak[0])] 
            thr_hg_s[id_inh, id_ratio, id_loop] = concs_intp[next(x[0] for x in enumerate(dr_peak) 
                                                     if x[1] > c_thr_hg*np.max(dr_peak))]
            
            dyn_rng_s[id_inh, id_ratio, id_loop] = np.log10(thr_hg_s[id_inh, id_ratio, id_loop]
                                                / thr_lw_s[id_inh, id_ratio, id_loop] )
            
            # ANALYSIS of sum odor/ORN
            nu_tmp = (orn_s[:, id_inh, id_loop] + orn_w[:, id_inh, id_loop])*.5
            nu_intp = interp1d(np.log10(concs), nu_tmp, kind=1)
            dr_peak = nu_intp(np.log10(concs_intp))
            
            
            if c_thr_lw<.5:
                thr_lw_tmp  = \
                    concs_intp[next(x[0] for x in enumerate(dr_peak) 
                                                     if x[1] > c_thr_lw*np.max(dr_peak))] 
            else:
                thr_lw_tmp =  concs_intp[next(x[0] for x in enumerate(dr_peak) 
                                                     if x[1] > c_thr_lw*dr_peak[0])] 
            thr_lw_sum[id_inh, id_ratio, id_loop] = thr_lw_tmp
            
            #####
            # dr_pluto = (np.diff(dr_nu)<0) 
            dr_pluto = (np.diff(dr_peak)<0) & (concs_intp[1:]>thr_lw_sum[id_inh, id_ratio, id_loop])
            
            if np.any(dr_pluto):
                tmp_max = next(x[0] for x in enumerate(dr_pluto) if x[1] == True)
                
            else:
                tmp_max = next(x[0] for x in enumerate(dr_peak) if x[1] > 
                               c_thr_hg*np.max(dr_peak))
            
            thr_hg_sum[id_inh, id_ratio, id_loop] = concs_intp[next(x[0] for x in enumerate(dr_peak) 
                                                     if x[1] > c_thr_hg*dr_peak[tmp_max])] 
             
            dyn_rng_sum[id_inh, id_ratio, id_loop] = np.log10(thr_hg_sum[id_inh, id_ratio, id_loop]
                                                / thr_lw_sum[id_inh, id_ratio, id_loop])
            
            
            # Distance between the two low thresholds
            thr_lw_ratio[id_inh, id_ratio, id_loop] = np.log10(thr_lw_s[id_inh, id_ratio, id_loop]
                                                / thr_lw_w[id_inh, id_ratio, id_loop] )
            thr_hg_ratio[id_inh, id_ratio, id_loop] = np.log10(thr_lw_s[id_inh, id_ratio, id_loop]
                                                / thr_lw_w[id_inh, id_ratio, id_loop] )
    
    

        if dose_response_fig:
            
            # PLOT
            mu1 = np.mean(.5*(orn_s[:, id_inh,:]+orn_w[:, id_inh,:]), axis=1)
            sigma1 = np.std(.5*(orn_s[:, id_inh,:]+orn_w[:, id_inh,:]), axis=1)
            axs[id_ratio, id_inh].plot(concs, mu1, '.--', 
                  linewidth=lw, label=r'(ORN$_a$+ORN$_b$)/2', color=blue)
            
            mu1 = np.mean((orn_w[:, id_inh,:]), axis=1)
            sigma1 = np.std((orn_w[:, id_inh,:]), axis=1)
            axs[id_ratio, id_inh].plot(concs, mu1, '.--', 
                  linewidth=lw, label='ORN$_a$', color=green)
            axs[id_ratio, id_inh].fill_between(concs, 
                   mu1+sigma1, mu1-sigma1, facecolor=green, alpha=trsp)
                        
            mu1 = np.mean(orn_s[:, id_inh,:], axis=1)
            sigma1 = np.std(orn_s[:, id_inh,:], axis=1)
            axs[id_ratio, id_inh].plot(concs, mu1, '.--', 
                  linewidth=lw, label='ORN$_b$', color=purple)
            
            # low thrs vertical lines
            thr_tmp = np.mean(thr_lw_w[id_inh, id_ratio, :])
            axs[id_ratio, id_inh].plot([thr_tmp, thr_tmp], [0, 250], 'o:', color=green)
            thr_tmp = np.mean(thr_lw_s[id_inh, id_ratio, :])
            axs[id_ratio, id_inh].plot([thr_tmp, thr_tmp], [0, 250], 'o:', color=purple)
            thr_tmp = np.mean(thr_lw_sum[id_inh, id_ratio, :])
            axs[id_ratio, id_inh].plot([thr_tmp, thr_tmp], [0, 250], 'o:', color=blue)
            
            # high thrs vertical lines
            thr_tmp = np.mean(thr_hg_w[id_inh, id_ratio, :])
            axs[id_ratio, id_inh].plot([thr_tmp, thr_tmp], [0, 250], 'd-.', color=green)
            thr_tmp = np.mean(thr_hg_s[id_inh, id_ratio, :])
            axs[id_ratio, id_inh].plot([thr_tmp, thr_tmp], [0, 250],  'd-.', color=purple)
            thr_tmp = np.mean(thr_hg_sum[id_inh, id_ratio, :])
            axs[id_ratio, id_inh].plot([thr_tmp, thr_tmp], [0, 250],  'd-.', color=blue)
            
            
            # SETTINGS
            if (id_ratio==0)&(id_inh==0):
                axs[id_ratio, id_inh].text(3e-5, 130, r'(ORN$_a$+ORN$_b$)/2',
                                    color=blue, fontsize=label_fs)
                axs[id_ratio, id_inh].text(3e-5, 70, r'ORN$_a$',
                                    color=green, fontsize=label_fs)
                axs[id_ratio, id_inh].text(3e-5, 10, r'ORN$_b$',
                                    color=purple, fontsize=label_fs)
                
            axs[id_ratio, id_inh].set_xscale('log') # 'linear') #'log')
            
            axs[id_ratio, id_inh].spines['right'].set_color('none')
            axs[id_ratio, id_inh].spines['top'].set_color('none')
        
            if id_ratio==(n_shift_ratios-1):
                axs[id_ratio, id_inh].set_xlabel('Odor concentration (a.u.)', fontsize=label_fs-2)
                axs[id_ratio, id_inh].tick_params(axis='both', which='major', labelsize=label_fs-3)
            else:
                axs[id_ratio, id_inh].set_xticks([])
                axs[id_ratio, id_inh].tick_params(axis='y', which='major', labelsize=label_fs-3)
            
            
            if id_ratio==0:
                if id_inh == 0:
                    axs[id_ratio, id_inh].set_title('ctrl', fontsize=label_fs+3)
                elif id_inh == 1:
                    axs[id_ratio, id_inh].set_title('NSI', fontsize=label_fs+3)
            
            if id_inh==0:
                if id_ratio in [1, 3, 5]:
                    axs[id_ratio, id_inh].set_ylabel('Firing rates (Hz)', fontsize=label_fs-2)
            axs[id_ratio, id_inh].set_xlim([1e-10, .5])
            
            axs[0,0].text(-.2, 1.2, 'a', transform=axs[0,0].transAxes,
               color= black, fontsize=panel_fs, fontweight='bold', va='top', ha='right')       
            
            # CHANGE plot position:
            ll, bb, ww, hh = axs[id_ratio, id_inh].get_position().bounds
            axs[id_ratio, id_inh].set_position([ll+.02, bb+.01, ww, hh])        
            plt.show()
            
    if fig_save & dose_response_fig:
        fig_dr.savefig(fld_analysis + figure_name + '.png', 
                       dpi=fig_dpi)    


#%% single duration figure

if dynrng_fig:
    id_dur = 0
    
    rs = 1
    cs = 1
    trsp = .3
    fig, axs    = plt.subplots(rs, cs, figsize=[10, 4])
       
    distance = np.mean(thr_lw_ratio[0, :, :], axis=1)
    
    # FIGURE PLOT                   (n_inh_conds, n_shift_ratios, n_loops)
    mu1 = np.mean(dyn_rng_sum[0, :, :], axis=1)
    sigma1 = np.std(dyn_rng_sum[0, :, :], axis=1)
    x1 = distance+.1
    axs.errorbar(x1, mu1, yerr=sigma1, marker='o', linewidth=lw, linestyle='', color=pink, label='ctrl')
    
    mu1 = np.mean(dyn_rng_sum[1, :, :], axis=1)
    sigma1 = np.std(dyn_rng_sum[1, :, :], axis=1)
    x1 = distance+.2
    axs.errorbar(x1, mu1, yerr=sigma1,marker='o',  linewidth=lw, linestyle='',  color=cyan, label='NSI')
    
    mu1 = np.mean(dyn_rng_w[0, :, :], axis=1)
    sigma1 = np.std(dyn_rng_w[0, :, :], axis=1)
    x1 = distance
    axs.errorbar(x1, mu1, yerr=sigma1, marker='o', linewidth=lw, linestyle='', color=green, label='ORN$_a$')
    
    mu1 = np.mean(dyn_rng_s[0, :, :], axis=1)
    sigma1 = np.std(dyn_rng_s[0, :, :], axis=1)
    x1 = distance-.1
    axs.errorbar(x1, mu1, yerr=sigma1,marker='o',  linewidth=lw, linestyle='', color=purple, label='ORN$_b$')
    
    
    # FIGURE SETTINGS    
    axs.text(3e-5, 6.6, r'NSI, (ORN$_a$+ORN$_b$)/2', color=cyan, fontsize=label_fs)
    axs.text(3e-5, 5.8, r'ctrl, (ORN$_a$+ORN$_b$)/2', color=pink, fontsize=label_fs)
    axs.text(3e-5, 5.0, r'ctrl, ORN$_a$', color=green, fontsize=label_fs)
    axs.text(3e-5, 4.2, r'ctrl, ORN$_b$', color=purple, fontsize=label_fs)
    axs.set_ylim((2.1,7))     
    
    #axs.set_xticklabels('', )
    
    axs.set_ylabel('dyn range  (OM)', fontsize=label_fs, )#verticalalignment='center', )

    labelx = -0.1  # axes coords
    axs.yaxis.set_label_coords(labelx, 0.5)
    
    axs.text(-.2, 1., 'b', transform=axs.transAxes,
            color= black, fontsize=panel_fs, fontweight='bold', va='top', ha='right')       
    axs.tick_params(axis='both', which='major', labelsize=label_fs-3)
    
    axs.spines['right'].set_color('none')
    axs.spines['top'].set_color('none')
    
    axs.set_xlabel('Sensitivity distance (OM)', fontsize=label_fs)
    
    # CHANGE plot position:
    ll, bb, ww, hh = axs.get_position().bounds
    axs.set_position([ll+.075, bb+.08, ww, hh])        
    
    plt.show()
    
    if fig_save:
        fig.savefig(fld_analysis + dynrng_figure_name +  '_dur%d'%stim_durs[id_dur] + '.png', 
                       dpi=fig_dpi) 
        
#%%
if thrs_fig:
    id_dur = 0
    
    rs = 2
    cs = 1
    trsp = .3
    fig, axs    = plt.subplots(rs, cs, figsize=[10, 6])
       
    distance = np.mean(thr_lw_ratio[0, :, :], axis=1)
    
    mu1 = np.mean(thr_lw_sum[0, :, :], axis=1)
    sigma1 = np.std(thr_lw_sum[0, :, :], axis=1)
    x1 = distance+.1
    axs[0].errorbar(x1, mu1, yerr=sigma1, marker='o', linewidth=lw, linestyle='', 
                    color=pink, label='ctrl')
    
    mu1 = np.mean(thr_lw_sum[1, :, :], axis=1)
    sigma1 = np.std(thr_lw_sum[1, :, :], axis=1)
    x1 = distance+.2
    axs[0].errorbar(x1, mu1, yerr=sigma1, marker='o', linewidth=lw, linestyle='', 
                    color=cyan, label='NSI')
    
    mu1 = np.mean(thr_lw_w[0, :, :], axis=1)
    sigma1 = np.std(thr_lw_w[0, :, :], axis=1)
    x1 = distance
    axs[0].errorbar(x1, mu1, yerr=sigma1, marker='o', linewidth=lw, linestyle='', 
                    color=green, label='ORN$_a$')
    
    mu1 = np.mean(thr_lw_s[0, :, :], axis=1)
    sigma1 = np.std(thr_lw_s[0, :, :], axis=1)
    x1 = distance-.1
    
    #### axs[1]
    mu1 = np.mean(thr_hg_sum[0, :, :], axis=1)
    sigma1 = np.std(thr_hg_sum[0, :, :], axis=1)
    x1 = distance+.1
    axs[1].errorbar(x1, mu1, yerr=sigma1, marker='o', linewidth=lw, linestyle='', 
                    color=pink, label='ctrl')
    
    mu1 = np.mean(thr_hg_sum[1, :, :], axis=1)
    sigma1 = np.std(thr_hg_sum[1, :, :], axis=1)
    x1 = distance+.2
    axs[1].errorbar(x1, mu1, yerr=sigma1, marker='o', linewidth=lw, linestyle='', 
                    color=cyan, label='NSI')
    
    mu1 = np.mean(thr_hg_w[0, :, :], axis=1)
    sigma1 = np.std(thr_hg_w[0, :, :], axis=1)
    x1 = distance
    axs[1].errorbar(x1, mu1, yerr=sigma1, marker='o', linewidth=lw, linestyle='', 
                    color=green, label='ORN$_a$')
    
    mu1 = np.mean(thr_hg_s[0, :, :], axis=1)
    sigma1 = np.std(thr_hg_s[0, :, :], axis=1)
    x1 = distance-.1
    
    
    
    # FIGURE SETTINGS    
    axs[1].text(3e-5, 0.0015, r'NSI, (ORN$_a$+ORN$_b$)/2', color=cyan, fontsize=label_fs)
    axs[1].text(3e-5, 0.0010, r'ctrl, (ORN$_a$+ORN$_b$)/2', color=pink, fontsize=label_fs)
    axs[1].text(3e-5, 0.0005, r'ctrl, ORN$_a$', color=green, fontsize=label_fs)
    # axs[0].text(3e-5, .5e-7, r'ctrl, ORN$_b$', color=purple, fontsize=label_fs)
    
    axs[0].set_xticklabels('', )
    
    axs[1].set_xlabel('Sensitivity distance (OM)', fontsize=label_fs)
    
    axs[0].set_ylabel('Low thr  (a.u.)', fontsize=label_fs, )#verticalalignment='center', )    
    axs[1].set_ylabel('High thr  (a.u.)', fontsize=label_fs, )#verticalalignment='center',)  
    
    labelx = -0.1  # axes coords

    for rr in range(rs):
        axs[rr].yaxis.set_label_coords(labelx, 0.5)
    
    # axs[0].set_yscale('log') # 'linear
    # axs[1].set_yscale('log') # 'linear
    
    for id_rs in range(rs): 
        axs[id_rs].text(-.2, 1.08, alphabet[id_rs] , transform=axs[id_rs].transAxes,
                color= black, fontsize=panel_fs, fontweight='bold', va='top', ha='right')       
        axs[id_rs].tick_params(axis='both', which='major', labelsize=label_fs-3)
        
        axs[id_rs].spines['right'].set_color('none')
        axs[id_rs].spines['top'].set_color('none')
        
        # CHANGE plot position:
        ll, bb, ww, hh = axs[id_rs].get_position().bounds
        axs[id_rs].set_position([ll+.075, bb+.04-.05*(id_rs-1), ww, hh])        
    
    plt.show()
    
    if fig_save:
        fig.savefig(fld_analysis + thrs_figure_name +  '_dur%d'%stim_durs[id_dur] + '.png',
                    dpi=fig_dpi)    
    
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 17:58:19 2021

@author: mario
"""


import numpy as np

import matplotlib.pyplot as plt
import string


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

fld_analysis    = 'NSI_analysis/'+name_analysis+'/'

stim_durs   = [10, 20, 50, 100, 200]


fig_save    = 0
muldur_fig  = 0
sindur_fig  = 1

# NSI = 0.4, ctrl has longer dynamic range than NSI 

# ratios = ['1', '.95', '.9', '.7', '0.5', '.2', '0.1', '.05']
# shift_1 = [10000, 9500, 9000,  7000, 5000, 2000, 1000, 500]

ratios = ['1', '.8', '.6', '.4', '.2', '.05']
shift_1 = [10000, 8000, 6000, 4000, 2000, 500]


shift_0 = np.ones(len(ratios))*10000



dyn_rng_res = np.zeros((2,len(stim_durs), len(ratios)))
thr_hg = np.zeros((2, len(stim_durs), len(ratios)))
thr_lw = np.zeros((2, len(stim_durs), len(ratios)))
thr_ratio = np.zeros((len(stim_durs), len(ratios)))

k_ratios = shift_1/ shift_0

import pickle
analysis_name = 'cns_dose_response'

for ii in range(len(ratios)):
    alpha_r_1 = int(shift_1[ii]**0.82206687*12.6228808)
    alpha_r_0 = int(shift_0[ii]**0.82206687*12.6228808)
    
    tmp_name = '_stim_ts' + \
        '_durs_%d-%d'%(stim_durs[0], stim_durs[-1],) + \
        '_ctrl_vs_nsi'+\
        '_alphar_0_%d'%alpha_r_0 +\
        '_alphar_1_%d'%alpha_r_1
        
    data = pickle.load(open(
        fld_analysis+analysis_name+tmp_name+'.pickle',  "rb" ))
    dyn_rng_sum = data['dyn_rng_sum']
    dyn_rng_res[:,:,ii]  = dyn_rng_sum
    
    thr_hg[:, :, ii] = data['thr_hg']
    thr_lw[:, :, ii] = data['thr_lw']
    thr_ratio[:, ii] = data['thr_ratio'][0,:]
    
    if ii == 2:
        dyn_rng = data['dyn_rng']
        dyn_rng_1 = dyn_rng[0, :]



#%% single duration figure

if sindur_fig:
    id_dur = 2
    
    rs = 2
    cs = 1
    
    fig, axs    = plt.subplots(rs, cs, figsize=[8, 5])
    fig_name    = name_analysis + '_dur%d'%stim_durs[id_dur]
       
    distance = k_ratios
    
    distance = thr_ratio[id_dur, :]    
    
    # FIGURE PLOT
    axs[0].plot(distance, dyn_rng_res[0, id_dur, :], 'o--', linewidth=lw, color=pink, label='ctrl')
    axs[0].plot(distance, dyn_rng_res[1, id_dur, :], 'o-', linewidth=lw, color=cyan, label='NSI')
    axs[0].plot(distance, np.ones_like(distance)*dyn_rng_1[id_dur], 'k--', linewidth=lw, color='black', label='single ORN')
    
    axs[1].plot(distance, thr_lw[0, id_dur, :], 'o--', linewidth=lw, color=pink, label='ctrl')
    axs[1].plot(distance, thr_lw[1, id_dur, :], 'o-', linewidth=lw, color=cyan, label='NSI')
    
    # FIGURE SETTINGS    
    axs[0].set_xticklabels('', )
    
    axs[1].set_xlabel('ORNs  similarity (OM)', fontsize=label_fs)
    
    axs[0].legend(frameon=False, fontsize=label_fs-4, loc=4)
    
    axs[0].set_ylabel('dyn range  (OM)', fontsize=label_fs)
    axs[1].set_ylabel('Lower Thr  (au)', fontsize=label_fs)    
    
    # axs[0].set_title('dur: %d ms' %stim_durs[id_dur], fontsize=panel_fs, fontweight='bold', )       
    
    for id_rs in range(rs): 
        axs[id_rs].text(-.2, 1.08, alphabet[id_rs+1] , transform=axs[id_rs].transAxes,
                color= black, fontsize=panel_fs, fontweight='bold', va='top', ha='right')       
        axs[id_rs].tick_params(axis='both', which='major', labelsize=label_fs-3)
        
        axs[id_rs].spines['right'].set_color('none')
        axs[id_rs].spines['top'].set_color('none')
        
        # CHANGE plot position:
        ll, bb, ww, hh = axs[id_rs].get_position().bounds
        axs[id_rs].set_position([ll+.075, bb+.05, ww, hh])        
    
    plt.show()
    
    if fig_save:
        fig.savefig(fld_analysis + '/' + fig_name + '.png')    
    
    
    

#%% multiple durations plot

if muldur_fig:
    rs =  len(stim_durs)
    cs = 1
    fig, axs_all = plt.subplots(rs, cs, figsize=[8, 11])
    fig_name    = name_analysis + '_dur%d-%d'%(stim_durs[0], stim_durs[-1])
       
    for id_dur in range(len(stim_durs)):
    
        
        distance = k_ratios
        distance = thr_ratio[id_dur, :]    
    
        axs = axs_all[id_dur] 
        
        
        # FIGURE PLOT
        axs.plot(distance, dyn_rng_res[0, id_dur, :], '.--', linewidth=lw, color=pink, label='ctrl')
        axs.plot(distance, dyn_rng_res[1, id_dur, :], '.-', linewidth=lw, color=cyan, label='NSI')
        axs.plot(distance, np.ones_like(distance)*dyn_rng_1[id_dur], 'k--', linewidth=lw, color='black', label='single ORN')
        
        # FIGURE SETTINGS
        if id_dur == len(stim_durs)-1:
            axs.set_xlabel('sensitivity distance (a.u.)', fontsize=label_fs)
            axs.legend(frameon=False, fontsize=label_fs)
        if id_dur == 2:
            axs.set_ylabel('dynamic range (OM)', fontsize=label_fs)
        if id_dur == 0:
            axs.text(-.1, 1.08, 'b', transform=axs.transAxes,
              color= black, fontsize=panel_fs, fontweight='bold', va='top', ha='right')       
        
        axs.text(1.05, .8, 'dur: %d ms' %stim_durs[id_dur], transform=axs.transAxes,
                 color= black, fontsize=panel_fs, fontweight='bold', va='top', ha='right')        
        
        axs.tick_params(axis='both', which='major', labelsize=label_fs-3)
                
        axs.spines['right'].set_color('none')
        axs.spines['top'].set_color('none')
        
        # CHANGE plot position:
        ll, bb, ww, hh = axs.get_position().bounds
        bb= bb+.08-.03*id_dur
        axs.set_position([ll, bb, ww, hh])        
    
    plt.show()
    
    if fig_save:
        fig.savefig(fld_analysis + '/' + fig_name + '.png')    

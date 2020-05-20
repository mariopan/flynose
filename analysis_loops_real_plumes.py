#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:49:09 2019

@author: mp525
analysis_loops_real_plumes.py
"""

import numpy as np
import matplotlib.pyplot as plt

import pickle        
from shutil import copyfile

def pars2name(params2an,):
    name_data = ['ORNPNLN' +
                '_stim_' + params2an[7] +
                '_nsi_%.1f'%(params2an[0]) +
                '_ln_%.2f'%(params2an[1]) +
                '_dur2an_%d'%(params2an[2]) +
                '_peak_%.1f'%(params2an[4]) +
                '_rho_%d'%(params2an[6])] 
                
    if params2an[8]<10:
        name_data = [name_data[0] +
                     '_wmax_%.1g'%(params2an[8])]
    else:
        name_data = [name_data[0] +
                     '_wmax_%.2g'%(params2an[8])]
    
    if params2an[9]<10:
        name_data = [name_data[0] +
                     '_bmax_%.1g'%(params2an[9])]
    else:
        name_data = [name_data[0] +
                     '_bmax_%.2g'%(params2an[9])]
            
    return name_data[0]

def pars2name_all(params2an, rhos, w_maxs, b_maxs, ):
    name_data = ['ORNPNLN' +
                '_stim_' + params2an[7] +
                '_dur2an_%d'%(params2an[2]) +
                '_peak_%.1f'%(params2an[4]) +
                '_rho_%d-%d'%(rhos[0],rhos[-1])]
    if w_maxs[-1]<10:
        name_data = [name_data[0] +
                     '_wmax_%.1g-%.1g'%(w_maxs[0],w_maxs[-1])]
    else:
        name_data = [name_data[0] +
                     '_wmax_%.1g-%.2g'%(w_maxs[0],w_maxs[-1])]                
        
    if b_maxs[-1]<10:
        name_data = [name_data[0] +
                     '_bmax_%.1g-%.1g'%(b_maxs[0],b_maxs[-1])]
    elif (b_maxs[0]<10) & (b_maxs[-1]>10):
        name_data = [name_data[0] +
                     '_bmax_%.1g-%.2g'%(b_maxs[0],b_maxs[-1])]                
    else:
        name_data = [name_data[0] +
                     '_bmax_%.2g-%.2g'%(b_maxs[0],b_maxs[-1])]                
    
    return name_data[0]

plt.ion()

# *****************************************************************
# STANDARD FIGURE PARAMS
lw = 2
fs = 20
plt.rc('text', usetex=True)  # laTex in the polot
#plt.rc('font', family='serif')
fig_size = [12, 12]
fig_position = 1300,10
title_fs = 20 # font size of ticks
label_fs = 18 # font size of labels
panel_fs = 30
black   = 'xkcd:black'
blue    = 'xkcd:blue'
red     = 'xkcd:red'
green   = 'xkcd:green'
purple  = 'xkcd:purple'
orange  = 'xkcd:orange'
cmap    = plt.get_cmap('rainbow')

# *****************************************************************

fld_home = 'NSI_analysis/analysis_real_plumes/'

# *******************************************************************
# Fig.PeakPN_wmax and Fig.PeakPN_resumen
fld_analysis = fld_home+'stim_200secs/'     # last longest simulation 200secs 
#fld_analysis = fld_home+'200secs/'     # last longest simulation 200secs 
stim_dur    =  201000
w_maxs      = [.01,.03,.3, 3, 25, 50, ]# max value in the whiff distribution
seeds       = np.arange(1, 30)
data_save   = 0
fig_save    = 0

peak_fig    = 0     # Fig.PeakPNActivity
avg_fig     = 1     # Fig.AverPNActivity
resumen_fig = 0     # Fig.PeakPN_resumen
thrwmax_fig = 0     # Fig.PeakPN_wmax
thr         = 150   # [50, 100, 150]
# *******************************************************************



## ********************************************************************
## simulation 60secs 
#fld_analysis = fld_home+'stim_60s_long_w/'
#stim_dur  =  61000 
#w_maxs  = [.01,.03,.3, 3, 25, ]# max value in the whiff distribution
#seeds = np.arange(1, 44)
#data_save   = 0
#fig_save    = 0
#peak_fig    = 0 # Fig.PeakPNActivity
#avg_fig     = 0 # Fig.AverPNActivity
#resumen_fig = 0 # Fig.PeakPN_resumen
#thrwmax_fig = 1 # Fig.PeakPN_wmax
#thr         = 50 # [50, 100, 150]
## ********************************************************************


## ********************************************************************
#fld_analysis = fld_home+'real_plumes_3m/'
#seeds = np.concatenate([np.arange(0, 4)])  
#w_maxs  = [.01, ]#.03,.3, 3, 25] # [3,50, 150] # max value in the whiff distribution
#w_maxs  = [25, 50]
#stim_dur  =  181000 
#data_save   = 0
#fig_save    = 0
#resumen_fig = 0 # Fig.PeakPN_resumen
#thrwmax_fig = 0 # Fig.PeakPN_wmax
#thr         = 100 # [50, 100, 150]
## ********************************************************************


#fld_analysis = fld_home+'long_stim_60s/'
#seeds = np.concatenate([np.arange(1, 16)])  

#fld_analysis = fld_home+'stim_90s_long_w/'
#fld_analysis = '/home/mario/MEGA/WORK/Code/PYTHON/NSI_analysis/analysis_real_plumes/long_stim/'

inh_conds = ['nsi', 'ln', 'noin']
stim_type = 'pl' # 'ss'   # 'rs'   #  'pl'  # 'ts'

stim_seed = int(0)

n_seeds = np.size(seeds)
b_maxs  = [25] #  max value in the blank distribution
rhos    = [0, 1, 3, 5]
peak    = 1.5

output_names = ['cor_stim', 'overlap_stim', 'cor_whiff', 
                 'interm_th', 'interm_est_1', 'interm_est_2', 'od_avg1', 
                 'od_avg2', 'orn_avg1', 'orn_avg2', 'pn_avg1', 'pn_avg2', 
                 'nu_pn_m50_1', 'nu_pn_m100_1', 'nu_pn_m150_1', 
                 'nu_pn_m50_2', 'nu_pn_m100_2', 'nu_pn_m150_2', ]
      
cor_stim_id = output_names.index('cor_stim')+1
cor_whiff_id = output_names.index('cor_whiff')+1
overlap_stim_id = output_names.index('overlap_stim')+1
interm_th_id     = output_names.index('interm_th')+1
interm_est_1_id     = output_names.index('interm_est_1')+1
interm_est_2_id     = output_names.index('interm_est_2')+1

 
cor_stim = np.zeros((np.size(rhos), np.size(seeds), np.size(w_maxs), np.size(b_maxs),))
cor_whiff = np.zeros((np.size(rhos), np.size(seeds), np.size(w_maxs), np.size(b_maxs),))
overlap_stim = np.zeros((np.size(rhos), np.size(seeds), np.size(w_maxs), np.size(b_maxs),))
interm_th = np.zeros((np.size(rhos), np.size(seeds), np.size(w_maxs), np.size(b_maxs),))
interm_est_1 = np.zeros((np.size(rhos), np.size(seeds), np.size(w_maxs), np.size(b_maxs),))
interm_est_2 = np.zeros((np.size(rhos), np.size(seeds), np.size(w_maxs), np.size(b_maxs),))

od_avg1_id   = output_names.index('od_avg1')+1
od_avg2_id   = output_names.index('od_avg2')+1
orn_avg1_id   = output_names.index('orn_avg1')+1
orn_avg2_id   = output_names.index('orn_avg2')+1
pn_avg1_id    = output_names.index('pn_avg1')+1
pn_avg2_id    = output_names.index('pn_avg2')+1

pn_m50_1_id  = output_names.index('nu_pn_m50_1')+1
pn_m100_1_id = output_names.index('nu_pn_m100_1')+1
pn_m150_1_id = output_names.index('nu_pn_m150_1')+1

pn_m50_2_id  = output_names.index('nu_pn_m50_2')+1
pn_m100_2_id = output_names.index('nu_pn_m100_2')+1
pn_m150_2_id = output_names.index('nu_pn_m150_2')+1


od_avg1 = np.zeros((np.size(rhos), 3, np.size(seeds), np.size(w_maxs), np.size(b_maxs),))
od_avg2 = np.zeros((np.size(rhos), 3, np.size(seeds), np.size(w_maxs), np.size(b_maxs),))
orn_avg1 = np.zeros((np.size(rhos), 3, np.size(seeds), np.size(w_maxs), np.size(b_maxs),))
orn_avg2 = np.zeros((np.size(rhos), 3, np.size(seeds), np.size(w_maxs), np.size(b_maxs),))
pn_avg1 = np.zeros((np.size(rhos), 3, np.size(seeds), np.size(w_maxs), np.size(b_maxs),))
pn_avg2 = np.zeros((np.size(rhos), 3, np.size(seeds), np.size(w_maxs), np.size(b_maxs),))
pn_m50_1 = np.zeros((np.size(rhos), 3, np.size(seeds), np.size(w_maxs), np.size(b_maxs),))
pn_m50_2 = np.zeros((np.size(rhos), 3, np.size(seeds), np.size(w_maxs), np.size(b_maxs),))
pn_m100_1 = np.zeros((np.size(rhos), 3, np.size(seeds), np.size(w_maxs), np.size(b_maxs),))
pn_m100_2 = np.zeros((np.size(rhos), 3, np.size(seeds), np.size(w_maxs), np.size(b_maxs),))
pn_m150_1 = np.zeros((np.size(rhos), 3, np.size(seeds), np.size(w_maxs), np.size(b_maxs),))
pn_m150_2 = np.zeros((np.size(rhos), 3, np.size(seeds), np.size(w_maxs), np.size(b_maxs),))

id_seed = -1
for stim_seed in seeds:
    id_seed = id_seed + 1
    tmp_fld_analysis = fld_analysis+'real_plumes_%d'%stim_seed
    
    id_b_max = -1
    for b_max in b_maxs:
        id_b_max = id_b_max + 1
        id_w_max = -1
        for w_max in w_maxs:
            id_w_max = id_w_max + 1
            id_rho = -1
            for rho in rhos:
                id_rho = id_rho + 1
                for inh_cond in inh_conds:
                    if inh_cond == 'noin':
                        nsi_value, ln_sp_hgt, id_inh = [0.0, 0.00, 0]
                    elif inh_cond == 'nsi':
                        nsi_value, ln_sp_hgt, id_inh = [0.3, 0.00, 1]
                    elif inh_cond == 'ln':
                        nsi_value, ln_sp_hgt, id_inh = [0.0, 0.15, 2]
                    
                    params2an = [nsi_value, ln_sp_hgt, stim_dur, 0, peak, 
                                 1, rho, stim_type,w_max,b_max]
                    name_data = pars2name(params2an,)
                    all_data_tmp = pickle.load(open(tmp_fld_analysis + 
                                '/' + name_data  + '.pickle',  "rb" ))
                    
                    cor_stim[id_rho, id_seed,id_w_max, id_b_max,]  = all_data_tmp[cor_stim_id]
                    cor_whiff[id_rho, id_seed,id_w_max, id_b_max,]  = all_data_tmp[cor_whiff_id]
                    overlap_stim[id_rho, id_seed,id_w_max, id_b_max,]  = all_data_tmp[overlap_stim_id]
                    interm_th[id_rho, id_seed,id_w_max, id_b_max,]  = all_data_tmp[interm_th_id]
                    interm_est_1[id_rho, id_seed,id_w_max, id_b_max,]  = all_data_tmp[interm_est_1_id]
                    interm_est_2[id_rho, id_seed,id_w_max, id_b_max,]  = all_data_tmp[interm_est_2_id]
                    
                    od_avg1[id_rho, id_inh,id_seed,id_w_max, id_b_max,]  = all_data_tmp[od_avg1_id]
                    od_avg2[id_rho, id_inh,id_seed,id_w_max, id_b_max,] = all_data_tmp[od_avg2_id]
                    orn_avg1[id_rho, id_inh,id_seed,id_w_max, id_b_max,]  = all_data_tmp[orn_avg1_id]
                    orn_avg2[id_rho, id_inh,id_seed,id_w_max, id_b_max,] = all_data_tmp[orn_avg2_id]
                    pn_avg1[id_rho, id_inh,id_seed,id_w_max, id_b_max,]  = all_data_tmp[pn_avg1_id]
                    pn_avg2[id_rho, id_inh,id_seed,id_w_max, id_b_max,] = all_data_tmp[pn_avg2_id]
                    
                    pn_m50_1[id_rho, id_inh,id_seed,id_w_max, id_b_max,]  = all_data_tmp[pn_m50_1_id]
                    pn_m100_1[id_rho, id_inh,id_seed,id_w_max, id_b_max,]  = all_data_tmp[pn_m100_1_id]
                    pn_m150_1[id_rho, id_inh,id_seed,id_w_max, id_b_max,]  = all_data_tmp[pn_m150_1_id]
                    
                    pn_m50_2[id_rho, id_inh,id_seed,id_w_max, id_b_max,] = all_data_tmp[pn_m50_2_id]
                    pn_m100_2[id_rho, id_inh,id_seed,id_w_max, id_b_max,] = all_data_tmp[pn_m100_2_id] 
                    pn_m150_2[id_rho, id_inh,id_seed,id_w_max, id_b_max,] = all_data_tmp[pn_m150_2_id] 
                
    
name_data_all = pars2name_all(params2an,rhos, w_maxs, b_maxs, )
if data_save:
    
    # save the newest version of this script:
    copyfile('analysis_loops_real_plumes2.py', fld_analysis+'analysis_loops_real_plumes2.py') 
    
    output_names = ['cor_stim', 'overlap_stim', 'cor_whiff', 
                     'interm_th', 'interm_est_1', 'interm_est_2', 'od_avg1', 
                     'od_avg2', 'orn_avg1', 'orn_avg2', 'pn_avg1', 'pn_avg2', 
                     'pn_m50_1', 'pn_m100_1', 'pn_m150_1', 'pn_m50_2', 
                     'pn_m100_2', 'pn_m150_2', 'seeds', 'w_maxs', 'b_maxs', 'rhos', ]
        
    params2an_names = ['nsi_value', 'ln_spike_height', 'dur2an', 'delay2an', 
                       'peak', 'peak_ratio', 'rho', 'stim_type', 'w_max', 'b_max']
    
    # save the data from all the simulations in a single pickle file
    with open(fld_analysis+name_data_all + '.pickle', 'wb') as f:
        pickle.dump([params2an, cor_stim, overlap_stim, cor_whiff, 
                     interm_th, interm_est_1, interm_est_2, od_avg1, od_avg2, 
                     orn_avg1, orn_avg2, pn_avg1, pn_avg2, 
                     pn_m50_1, pn_m100_1, pn_m150_1, 
                     pn_m50_2, pn_m100_2, pn_m150_2, 
                     seeds, w_maxs, b_maxs, rhos, 
                     params2an_names, output_names], f)
    print('All the data are now save in the folder:')
    print(fld_analysis)
    print('in the file:')
    print(name_data_all)

fig_name = ['_stim_' + params2an[7] +
            '_dur2an_%d'%(params2an[2]) +
            '_peak_%.1f'%(params2an[4]) +
            '_rho_%d-%d'%(rhos[0],rhos[-1]) +
            '_wmax_%.2g-%.2g'%(w_maxs[0],w_maxs[-1]) + 
            '_bmax_%.2g'%(b_maxs[0])]


#%% *********************************************************
## FIGURE Fig.PeakPNActivity
## **********************************************************

if peak_fig:
        
    fig = plt.figure(figsize=(12, 4), ) 
    rs = 1
    cs = 3
    
    ax_m50 = plt.subplot(rs,cs,1)
    ax_m100 = plt.subplot(rs,cs,2)
    ax_m150 = plt.subplot(rs,cs,3)
    corr_obs = np.mean(np.squeeze(cor_stim[:, :,3,0]), axis=1)
      
    ax_m50.errorbar(corr_obs, np.squeeze(np.mean(pn_m50_1[:, 0, :,3,0], axis=1)),
                  yerr=np.squeeze(np.std(pn_m50_1[:, 0, :,3,0], axis=1))/
                  (np.size(pn_m50_1[:, 0, :,3,0],axis=1))**.5, 
                  lw=lw, color='magenta',label='Indep.')
    ax_m50.errorbar(corr_obs, np.squeeze(np.mean(pn_m50_1[:, 2, :,3,0], axis=1)),
                  yerr=np.squeeze(np.std(pn_m50_1[:, 2, :,3,0], axis=1))/
                  (np.size(pn_m50_1[:, 2, :,3,0],axis=1))**.5,
                  lw=lw, color='orange',label='LN ')
    ax_m50.errorbar(corr_obs, np.squeeze(np.mean(pn_m50_1[:, 1, :,3,0], axis=1)),
                  yerr=np.squeeze(np.std(pn_m50_1[:, 1, :,3,0], axis=1))/
                  (np.size(pn_m50_1[:, 1, :,3,0],axis=1))**.5,
                  lw=lw, color='blue',label='NSI ')

    ax_m100.errorbar(corr_obs, np.squeeze(np.mean(pn_m100_1[:, 0, :,3,0], axis=1)),
                  yerr=np.squeeze(np.std(pn_m100_1[:, 0, :,3,0], axis=1))/
                  (np.size(pn_m100_1[:, 0, :,3,0],axis=1))**.5, 
                  lw=lw, color='magenta',label='Indep.')
    ax_m100.errorbar(corr_obs, np.squeeze(np.mean(pn_m100_1[:, 2, :,3,0], axis=1)),
                  yerr=np.squeeze(np.std(pn_m100_1[:, 2, :,3,0], axis=1))/
                  (np.size(pn_m100_1[:, 2, :,3,0],axis=1))**.5,
                  lw=lw, color='orange',label='LN ')
    ax_m100.errorbar(corr_obs, np.squeeze(np.mean(pn_m100_1[:, 1, :,3,0], axis=1)),
                  yerr=np.squeeze(np.std(pn_m100_1[:, 1, :,3,0], axis=1))/
                  (np.size(pn_m100_1[:, 1, :,3,0],axis=1))**.5,
                  lw=lw, color='blue',label='NSI ')

    ax_m150.errorbar(corr_obs, np.squeeze(np.mean(pn_m150_1[:, 0, :,3,0], axis=1)),
                  yerr=np.squeeze(np.std(pn_m150_1[:, 0, :,3,0], axis=1))/
                  (np.size(pn_m150_1[:, 0, :,3,0],axis=1))**.5, 
                  lw=lw, color='magenta',label='Indep.')
    ax_m150.errorbar(corr_obs, np.squeeze(np.mean(pn_m150_1[:, 2, :,3,0], axis=1)),
                  yerr=np.squeeze(np.std(pn_m150_1[:, 2, :,3,0], axis=1))/
                  (np.size(pn_m150_1[:, 2, :,3,0],axis=1))**.5,
                  lw=lw, color='orange',label='LN ')
    ax_m150.errorbar(corr_obs, np.squeeze(np.mean(pn_m150_1[:, 1, :,3,0], axis=1)),
                  yerr=np.squeeze(np.std(pn_m150_1[:, 1, :,3,0], axis=1))/
                  (np.size(pn_m150_1[:, 1, :,3,0],axis=1))**.5,
                  lw=lw, color='blue',label='NSI ')
    
    ax_m50.tick_params(axis='both', which='major', labelsize=label_fs-3)
    ax_m100.tick_params(axis='both', which='major', labelsize=label_fs-3)
    ax_m150.tick_params(axis='both', which='major', labelsize=label_fs-3)
    
    
    ax_m50.set_ylabel('PN activity (ms)', fontsize=label_fs)
    
    ax_m50.set_xlabel('Stim. correlation', fontsize=label_fs)
    ax_m100.set_xlabel('Stim. correlation', fontsize=label_fs)
    ax_m150.set_xlabel('Stim. correlation',fontsize=label_fs)
    
    ax_m50.legend(fontsize=label_fs-3, frameon=False)
    
    ax_m50.set_title('$>50Hz$', fontsize=fs)
    ax_m100.set_title('$>100Hz$', fontsize=fs)
    ax_m150.set_title('$>150Hz$', fontsize=fs)
    
    # original plot position:
    ll, bb, ww, hh = ax_m50.get_position().bounds
    ax_m50.set_position([ll,bb+.04,ww,hh])        
    ll, bb, ww, hh = ax_m100.get_position().bounds
    ax_m100.set_position([ll+.04,bb+.04,ww,hh])        
    ll, bb, ww, hh = ax_m150.get_position().bounds
    ax_m150.set_position([ll+.08,bb+.04,ww,hh])        
    
    ax_m50.text(-.25, 1.1, 'a.', transform=ax_m50.transAxes, color = blue,
              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    ax_m100.text(-.25, 1.1, 'b.', transform=ax_m100.transAxes, color = blue,
              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    ax_m150.text(-.25, 1.1, 'c.', transform=ax_m150.transAxes, color = blue,
              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    
    ax_m50.spines['right'].set_color('none')
    ax_m50.spines['top'].set_color('none')
    ax_m100.spines['right'].set_color('none')
    ax_m100.spines['top'].set_color('none')
    ax_m150.spines['right'].set_color('none')
    ax_m150.spines['top'].set_color('none')
    
    if fig_save:
        fig.savefig(fld_analysis+  '/NSI_HighConc_'+name_data_all+'.png')
        
#%% *********************************************************
## FIGURE Fig.AverPNActivity
## **********************************************************
if avg_fig:
    fig2 = plt.figure(figsize=(12,4), ) 
    rs = 1
    cs = 2
    lw = 4
    
    ax_orn = plt.subplot(rs,cs, 1)
    ax_pn = plt.subplot(rs,cs, 2)
    
    corr_th = np.array(rhos)
    corr_obs = np.mean(np.squeeze(cor_stim[:, :,3,0]), axis=1)
                                    #[id_rho, id_inh,id_seed,id_w_max, id_b_max,]
                                    #[id_rho, id_inh,id_seed] 
     
    ax_orn.errorbar(corr_obs, np.squeeze(np.mean(orn_avg1[:, 0, :,3,0], axis=1)),
                  yerr=np.squeeze(np.std(orn_avg1[:, 0, :,3,0], axis=1))/
                  (np.size(orn_avg1[:, 0, :,3,0],axis=1))**.5, linewidth=lw, ls='-', 
                  color='green',label='Ind Glo 1', fmt='o')
    ax_orn.errorbar(corr_obs, np.squeeze(np.mean(orn_avg1[:, 1, :,3,0], axis=1)),
                  yerr=np.squeeze(np.std(orn_avg1[:, 1, :,3,0], axis=1))/
                  (np.size(orn_avg1[:, 1, :,3,0],axis=1))**.5, linewidth=lw, ls='--', 
                  color='green',label='NSI Glo 1', fmt='*')
    
#    ax_orn.errorbar(corr_obs, np.squeeze(np.mean(orn_avg2[:, 0, :,3,0], axis=1)),
#                  yerr=np.squeeze(np.std(orn_avg2[:, 0, :,3,0], axis=1))/
#                  (np.size(orn_avg2[:, 0, :,3,0],axis=1))**.5, linewidth=lw, 
#                  color='purple',label='Ind Glo 2')
#    ax_orn.errorbar(corr_obs, np.squeeze(np.mean(orn_avg2[:, 1, :,3,0], axis=1)),
#                  yerr=np.squeeze(np.std(orn_avg2[:, 1, :,3,0], axis=1))/
#                  (np.size(orn_avg2[:, 1, :,3,0],axis=1))**.5, linewidth=lw, 
#                  ls='--', color='purple',label='NSI Glo 2')
    
    ax_pn.errorbar(corr_obs, np.squeeze(np.mean(pn_avg1[:, 0, :,3,0], axis=1)),
                  yerr=np.squeeze(np.std(pn_avg1[:, 0, :,3,0], axis=1))/
                  (np.size(pn_avg1[:, 0, :,3,0],axis=1))**.5, linewidth=lw, ls='-', 
                  color='green',label='Ind Glo 1', fmt='o')
    ax_pn.errorbar(corr_obs, np.squeeze(np.mean(pn_avg1[:, 1, :,3,0], axis=1)),
                  yerr=np.squeeze(np.std(pn_avg1[:, 1, :,3,0], axis=1))/
                  (np.size(pn_avg1[:, 1, :,3,0],axis=1))**.5, linewidth=lw, ls='--', 
                  color='green',label='NSI Glo 1', fmt='*')
    ax_pn.errorbar(corr_obs, np.squeeze(np.mean(pn_avg1[:, 2, :,3,0], axis=1)),
                  yerr=np.squeeze(np.std(pn_avg1[:, 2, :,3,0], axis=1))/
                  (np.size(pn_avg1[:, 2, :,3,0],axis=1))**.5, linewidth=lw, ls='-.', 
                  color='green',label='LN Glo 1', fmt='d')

#    ax_pn.errorbar(corr_obs, np.squeeze(np.mean(pn_avg2[:, 0, :,3,0], axis=1)),
#                  yerr=np.squeeze(np.std(pn_avg2[:, 0, :,3,0], axis=1))/
#                  (np.size(pn_avg2[:, 0, :,3,0],axis=1))**.5, linewidth=lw, 
#                  color='purple',label='Ind Glo 2')
#    ax_pn.errorbar(corr_obs, np.squeeze(np.mean(pn_avg2[:, 1, :,3,0], axis=1)),
#                  yerr=np.squeeze(np.std(pn_avg2[:, 1, :,3,0], axis=1))/
#                  (np.size(pn_avg2[:, 1, :,3,0],axis=1))**.5, linewidth=lw, 
#                  ls='--', color='purple',label='NSI Glo 1')
#    ax_pn.errorbar(corr_obs, np.squeeze(np.mean(pn_avg2[:, 2, :,3,0], axis=1)),
#                  yerr=np.squeeze(np.std(pn_avg2[:, 2, :,3,0], axis=1))/
#                  (np.size(pn_avg2[:, 2, :,3,0],axis=1))**.5, linewidth=lw, 
#                  ls='-.', color='purple',label='LN Glo 1')
    
    ax_orn.tick_params(axis='both', which='major', labelsize=label_fs-3)
    ax_pn.tick_params(axis='both', which='major', labelsize=label_fs-3)
    
    ax_orn.set_ylabel('ORN avg (Hz)', fontsize=label_fs)
    ax_pn.set_ylabel('PN avg (Hz)', fontsize=label_fs)
    
    ax_orn.set_xlabel('Obs. correlation', fontsize=label_fs)
    ax_pn.set_xlabel('Obs. correlation', fontsize=label_fs,)
    
    ax_orn.spines['right'].set_color('none')
    ax_orn.spines['top'].set_color('none')
    ax_pn.spines['right'].set_color('none')
    ax_pn.spines['top'].set_color('none')

    ax_orn.text(-.15, 1.15, 'a.', transform=ax_orn.transAxes,color= blue,
              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    ax_pn.text(-.15, 1.15, 'b.', transform=ax_pn.transAxes,color= blue,
              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    # change panels positions and sizes:
    ll, bb, ww, hh = ax_orn.get_position().bounds
    ax_orn.set_position([ll,bb+.04,ww,hh-.04])        
    ll, bb, ww, hh = ax_pn.get_position().bounds
    ax_pn.set_position([ll+.04,bb+.04,ww,hh-.04])        
    
    if fig_save:
        fig2.savefig(fld_analysis+  '/NSI_AverActiv_'+name_data_all+'.png')

#%% *********************************************************
## FIGURE Fig.AverPlumeCorr
## **********************************************************
avgplume_fig = 1
if avgplume_fig:
    fig2 = plt.figure(figsize=(12,4), ) 
    rs = 1
    cs = 2
    lw = 4
    
    ax_conc = plt.subplot(rs,cs,1)
    ax_corr = plt.subplot(rs,cs,2) # Correlation/Overlap between stimuli

    corr_th = np.array(rhos)
    corr_obs = np.mean(np.squeeze(cor_stim[:, :,3,0]), axis=1)
    overlap_obs = np.mean(np.squeeze(overlap_stim[:, :,3,0]), axis=1)
    interm_av_th = np.mean(np.squeeze(interm_th[:, :,3,0]), axis=1)
    interm_obs = np.mean(np.squeeze(interm_est_1[:, :,3,0]), axis=1)
    interm_obs_std = (np.std(np.squeeze(interm_est_1[:, :,3,0]), axis=1)/
                      np.size(np.squeeze(interm_est_1[:, :,3,0]))**0.5)
    
                                    #[id_rho, id_inh,id_seed,id_w_max, id_b_max,]
                                    #[id_rho, id_inh,id_seed] 
    ax_conc.errorbar(corr_obs, np.squeeze(np.mean(od_avg1[:, 0, :,3,0], axis=1)),
                  yerr= .1, linewidth=lw, fmt='o', color='green',label='Glo 1')
    ax_conc.errorbar(corr_obs+.01, 
                     np.squeeze(np.mean(od_avg2[:, 0, :,3,0], axis=1)),
                  yerr= .1, linewidth=lw, color='purple',label='Glo 2')
    ax_conc.set_ylabel('Odorants \n concentration', fontsize=label_fs)
      
    ax_corr.plot(corr_th, corr_obs, '.-', label='corr stim')
    ax_corr.plot(corr_th, overlap_obs, '.-', label='overlap stim')
    ax_corr.plot(corr_th, interm_av_th, '.-', label='interm theor')
    ax_corr.errorbar(corr_th, interm_obs, 
                  yerr=interm_obs_std,  label='interm obs')
     
    ax_conc.tick_params(axis='both', which='major', labelsize=label_fs-3)
    ax_corr.tick_params(axis='both', which='major', labelsize=label_fs-3)
    
    ax_corr.legend(fontsize=label_fs-7, frameon=False)
    
    ax_conc.set_xlabel('Theor. correlation', fontsize=label_fs)
    ax_corr.set_xlabel('Theor. correlation', fontsize=label_fs)
    
    ax_corr.spines['right'].set_color('none')
    ax_corr.spines['top'].set_color('none')
    ax_conc.spines['right'].set_color('none')
    ax_conc.spines['top'].set_color('none')
    
    ax_conc.text(-.15, 1.15, 'a.', transform=ax_conc.transAxes, color= blue,
              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    ax_corr.text(-.15, 1.15, 'b.', transform=ax_corr.transAxes,color= blue,
              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    
    # change panels positions and sizes:
    ll, bb, ww, hh = ax_conc.get_position().bounds
    ax_conc.set_position([ll,bb+.04,ww,hh-.04])        
    ll, bb, ww, hh = ax_corr.get_position().bounds
    ax_corr.set_position([ll+.04,bb+.04,ww,hh-.04])        
        
    if fig_save:
        fig2.savefig(fld_analysis+  '/NSI_PlumeAverActiv_'+name_data_all+'.png')
        
        
#%%**********************************************************
# FIGURE: Fig.PeakPN_resumen
## **********************************************************

if resumen_fig:
    
    cs = 3
    rs = 2
    fig, axs = plt.subplots(rs, cs, figsize=(13,6), )
        
    thr_id = -1
    for thr in [50, 100, 150]:
        thr_id = thr_id + 1
        #pn_m50_1[id_rho, id_inh,id_seed,id_w_max, id_b_max,]
        if thr == 50:
            pn_tmp = .5*(pn_m50_2+pn_m50_1)
        elif thr == 100:
            pn_tmp = .5*(pn_m100_2+pn_m100_1)
        elif thr == 150:
            pn_tmp = .5*(pn_m150_2+pn_m150_1)

            
        pn_tmp0 = np.squeeze(np.mean(pn_tmp[0, :,:,:, 0,], axis=1)) # PN, corr = 0 
        pn_tmp1 = np.squeeze(np.mean(pn_tmp[-1, :,:,:, 0,], axis=1)) # PN, corr = 1
        
        dec_perc_nsi = np.squeeze(pn_tmp0[0,:] - pn_tmp0[1,:])/(
                np.squeeze(pn_tmp1[0,:] - pn_tmp1[1,:]))
        
        dec_perc_ln = np.squeeze(pn_tmp0[0,:] - pn_tmp0[2,:])/(
                np.squeeze(pn_tmp1[0,:] - pn_tmp1[2,:]))
        
        
        delta_ln1 = np.squeeze(pn_tmp1[0,:] - pn_tmp1[2,:])
        delta_ln0 = np.squeeze(pn_tmp0[0,:] - pn_tmp0[2,:])
        delta_nsi1 = np.squeeze(pn_tmp1[0,:] - pn_tmp1[1,:])
        delta_nsi0 = np.squeeze(pn_tmp0[0,:] - pn_tmp0[1,:])
        
        axs[0, thr_id].plot(w_maxs, delta_ln0, 'o--', color=orange, label=r'$x$=LN, $\rho=$0')
        axs[0, thr_id].plot(w_maxs, delta_ln1, '*-', color=orange, label=r'$x$=LN, $\rho=$1')
        axs[0, thr_id].plot(w_maxs, delta_nsi0, 'd--', color=blue, label=r'$x$=NSI, $\rho=$0')
        axs[0, thr_id].plot(w_maxs, delta_nsi1, '.-', color=blue, label=r'$x$=NSI, $\rho=$1')
        axs[0, thr_id].set_xscale('log')
        axs[0, thr_id].set_yscale('log')
        axs[0, thr_id].set_title(r'$\Theta$:%d Hz'%thr, fontsize=fs)
        axs[0, thr_id].spines['right'].set_color('none')
        axs[0, thr_id].spines['top'].set_color('none')
        axs[0, thr_id].text(-.1, 1.15, 'a%d'%thr_id+'.', transform=axs[0,thr_id].transAxes,
              color= blue, fontsize=label_fs, fontweight='bold', va='top', ha='right')
        
        axs[1, thr_id].plot(w_maxs, 1-dec_perc_ln, '*-', color=orange, label='$x$=LN')
        axs[1, thr_id].plot(w_maxs, 1-dec_perc_nsi, 'o-', color=blue,label='$x$=NSI')        
        axs[1, thr_id].set_xscale('log')
        axs[1, thr_id].set_xlabel('$w_{max}$ (s)', fontsize=label_fs)
        axs[1, thr_id].spines['right'].set_color('none')
        axs[1, thr_id].spines['top'].set_color('none')
        axs[1, thr_id].text(-.1, 1.15, 'b%d'%thr_id+'.', transform=axs[1,thr_id].transAxes,
              color= blue, fontsize=label_fs, fontweight='bold', va='top', ha='right')
        
        
        # original plot position:
        ll, bb, ww, hh = axs[0, thr_id].get_position().bounds
        axs[0, thr_id].set_position([ll,bb+.04,ww,hh])        
        
    axs[0, 0].set_ylabel(r'$\nu_{ind} - \nu_x$ (ms)', fontsize=label_fs)
    axs[0, 2].legend(fontsize=label_fs-3, frameon=False)
    
    axs[1, 0].set_ylabel(r'Perf. = 1-$\frac{\nu_{ind}^0 - \nu_x^0}{\nu_{ind}^1 - \nu_x^1}$', fontsize=label_fs)
    axs[1, 0].set_ylim((0,1))
    axs[1, 1].set_ylim((0,1))
    axs[1, 2].set_ylim((0,1))
    axs[1, -1].legend(fontsize=label_fs-3, frameon=False)
    
    for aa in range(cs):
        for bb in range(rs):
            axs[bb,aa].tick_params(axis='both', which='major', labelsize=label_fs)
            
    
    if fig_save:
        fig.savefig(fld_analysis+ '/NSI_Perf_log'+ fig_name[0] + '.png')    



#%% *********************************************************
## FIGURE Fig.PeakPN_wmax
## **********************************************************
if thrwmax_fig:
#    interm_th[id_rho, id_seed,id_w_max, id_b_max,]
    rs = np.size(b_maxs) 
    cs = np.size(w_maxs)    

    wh_tot  = np.ones((np.size(rhos), np.size(w_maxs)))
    wh_tot  = 5*stim_dur*np.squeeze(interm_th[:, 0,:,0]) # id_rho, id_seed,id_w_max, id_b_max,]
    corr_tmp = np.array(rhos) 
    fig, axs = plt.subplots(rs, cs, figsize=(13,2.5), ) 
    
    id_b_max = 0
    b_max = b_maxs[0]
    id_w_max = -1
    for w_max in w_maxs:
        id_w_max = id_w_max + 1
        
        if thr == 50:
            pn_tmp = .5*(pn_m50_2+pn_m50_1)            
            axs[id_w_max].set_ylim((0,90500)) # 250)) # 
            panel_id = 'a%d'%id_w_max+'.'
        elif thr == 100:
            pn_tmp = .5*(pn_m100_2+pn_m100_1)
            axs[id_w_max].set_ylim((0,32500))#60)) # 
            panel_id = 'b%d'%id_w_max+'.'
        elif thr == 150:
            pn_tmp = .5*(pn_m150_2+pn_m150_1)
            axs[id_w_max,].set_ylim((0,6500))#10)) # 
            panel_id = 'c%d'%id_w_max+'.'
            
        y2plot_ln = np.squeeze(pn_tmp[:, 2, :, id_w_max, id_b_max])
        y2plot_nsi = np.squeeze(pn_tmp[:, 1, :, id_w_max, id_b_max])
        y2plot_noin = np.squeeze(pn_tmp[:, 0, :, id_w_max, id_b_max])

        axs[id_w_max].errorbar(corr_tmp, np.mean(y2plot_noin, axis=1),#*100/wh_tot[:, id_w_max],
           yerr=np.std(y2plot_noin, axis=1)/np.sqrt(n_seeds),#*100/wh_tot[:, id_w_max], 
           linewidth=lw, color='magenta',label='Indep')
        axs[id_w_max].errorbar(corr_tmp, np.mean(y2plot_ln, axis=1),#*100/wh_tot[:, id_w_max],
           yerr=np.std(y2plot_ln, axis=1)/np.sqrt(n_seeds),#*100/wh_tot[:, id_w_max], 
           linewidth=lw, color='orange',label='LN')
        axs[id_w_max].errorbar(corr_tmp, np.mean(y2plot_nsi, axis=1),#*100/wh_tot[:, id_w_max],
           yerr=np.std(y2plot_nsi, axis=1)/np.sqrt(n_seeds),#*100/wh_tot[:, id_w_max], 
           linewidth=lw, color='blue',label='NSI')
                
#        if id_w_max == 0:
#            axs[id_w_max].set_title('$w_{max}:%.2g s$'%w_max, fontsize=fs)
#        else:
#            axs[id_w_max].set_title('%.2g s'%w_max, fontsize=fs)
        
#        if id_w_max>0:
#            axs[id_w_max].set_yticks(())
        axs[id_w_max].tick_params(axis='both', which='major', labelsize=label_fs-5)
        axs[id_w_max].set_xticks(corr_tmp[[0, 1, 2]])
        axs[id_w_max].set_xticklabels(np.array(1-np.power(.10, corr_tmp[[0,1,2]])))
        
        axs[id_w_max].spines['right'].set_color('none')
        axs[id_w_max].spines['top'].set_color('none')
    
        if id_w_max==0:
            letter_pos = [-.25, 1.3]
        else:
            letter_pos = [-.1, 1.3]
            
        axs[id_w_max].text(letter_pos[0], letter_pos[1], panel_id, transform=axs[id_w_max].transAxes,color= blue,
              fontsize=panel_fs-10, fontweight='bold', va='top', ha='right')
               
        ll, bb, ww, hh = axs[id_w_max].get_position().bounds
        axs[id_w_max].set_position([ll+.015*id_w_max, bb+.09, ww, hh-.15])   
        
    axs[0].set_ylabel('PN activity (ms)\n' + r'$\Theta$=%d Hz'%thr, fontsize=label_fs)
    if thr == 150:
        axs[0].set_xlabel('Theor. correlation', fontsize=label_fs)
        axs[0].xaxis.set_label_coords(1.1, -.15)
        axs[2].set_xlabel('Theor. correlation', fontsize=label_fs)
        axs[2].xaxis.set_label_coords(1.1, -.15)
        axs[4].set_xlabel('Theor. correlation', fontsize=label_fs)
        axs[4].xaxis.set_label_coords(1.1, -.15)
                             
        axs[0].legend(fontsize=label_fs-3, frameon=False)
#%%        
    if fig_save:
        fig.savefig(fld_analysis+ '/NSI_nuPN_LNvsNSIvsNoIn_%dHz'%thr+ fig_name[0] + '.png')    
    
    
    
    
#%%**********************************************************
## FIGURE: Performance   OLD
### **********************************************************
#fig_name = ['_stim_' + params2an[7] +
#            '_dur2an_%d'%(params2an[2]) +
#            '_peak_%.1f'%(params2an[4]) +
#            '_rho_%d-%d'%(rhos[0],rhos[-1]) +
#            '_wmax_%.2g-%.2g'%(w_maxs[0],w_maxs[-1]) + 
#            '_bmax_%.2g'%(b_maxs[0])]
#
#fig_save = True
#fig, axs = plt.subplots(2, 3, figsize=(12,7), )
#    
#thr_id = -1
#for thr in [50, 100, 150]:
#    thr_id = thr_id + 1
#    #pn_m50_1[id_rho, id_inh,id_seed,id_w_max, id_b_max,]
#    if thr == 50:
#        pn_tmp = .5*(pn_m50_2+pn_m50_1)
#    elif thr == 100:
#        pn_tmp = .5*(pn_m100_2+pn_m100_1)
#    elif thr == 150:
#        pn_tmp = .5*(pn_m150_2+pn_m150_1)
#        
#    pn_tmp0 = np.squeeze(np.mean(pn_tmp[0, :,:,:, 0,], axis=1))
#    pn_tmp1 = np.squeeze(np.mean(pn_tmp[-1, :,:,:, 0,], axis=1))
#    
#    dec_perc = (pn_tmp0 - pn_tmp1)/(pn_tmp0 + pn_tmp1)
#    delta_pn = (pn_tmp0 - pn_tmp1)
#    
#    dec_perc_nsi = np.squeeze(dec_perc[1,:,])
#    dec_perc_ln = np.squeeze(dec_perc[2,:,])
#    delta_pn_nsi = np.squeeze(delta_pn[1,:,])
#    delta_pn_ln = np.squeeze(delta_pn[2,:,])
#    delta_pn_noin = np.squeeze(delta_pn[0,:,])
#    
#    
#    axs[0, thr_id].plot(w_maxs, 100*dec_perc_nsi, 'o--', color=blue,label='NSI, ')
#    axs[0, thr_id].plot(w_maxs, 100*dec_perc_ln, '*-', color=red, label='LN')
#    axs[0, thr_id].set_ylabel('$\Delta$ perc ', fontsize=label_fs)
##    axs[0, thr_id].set_ylim(-20, 100)
#        
#    axs[1, thr_id].plot(w_maxs, delta_pn_nsi, 'o--', color=blue, label='NSI')
#    axs[1, thr_id].plot(w_maxs, delta_pn_ln, '*-', color=red, label='LN')
#    axs[1, thr_id].plot(w_maxs, delta_pn_noin, '.-', color=green, label='No Inh')
#    axs[1, thr_id].set_ylabel('$\Delta$ (ms)', fontsize=label_fs)
#    axs[1, thr_id].set_xlabel('$w_{max}$ (s)', fontsize=label_fs)
#    axs[1, thr_id].text(16, 3000, 'o  NSI', color=blue, fontsize=label_fs)
#    axs[1, thr_id].text(18, 2500, '*  LN', color=red, fontsize=label_fs)
#    axs[1, thr_id].text(20, 2000, '.  No Inhib', color=green, fontsize=label_fs)
#    axs[1, thr_id].legend(fontsize=label_fs)
##    axs[1, thr_id].set_ylim(-2000,5500)
#    axs[0, thr_id].set_title('$\Theta$:%d Hz'%thr)
#    
#if fig_save:
#    fig.savefig(fld_analysis+ '/NSI_Perf_LNvsNSI'+ fig_name[0] + '.png')    
#    
##%% *********************************************************
### FIGURE Example: 1 column per each inhib condition
### **********************************************************
##pn_m50_1[id_rho, id_inh,id_seed,id_w_max, id_b_max,]
#
#pn_tmp_noin = np.mean(np.squeeze(pn_m100_2[:, 0,:, :, -2,]), axis=1)
#pn_tmp_nsi = np.mean(np.squeeze(pn_m100_2[:, 1,:, :, -2,]), axis=1)
#pn_tmp_ln = np.mean(np.squeeze(pn_m100_2[:, 2,:, :, -2,]), axis=1)
#
#pn_tmp_noin_std = np.std(np.squeeze(pn_m100_2[:, 0,:, :, -2,]), axis=1)
#pn_tmp_nsi_std = np.std(np.squeeze(pn_m100_2[:, 1,:, :, -2,]), axis=1)
#pn_tmp_ln_std = np.std(np.squeeze(pn_m100_2[:, 2,:, :, -2,]), axis=1)
#
#corr_tmp = np.array(rhos)
#
#rs = 1 
#cs = 3    
#fig, axs = plt.subplots(rs, cs,figsize=(15,8), )
#id_w_max = -1
#for w_max in [.03, .3, 3]:
#    id_w_max = id_w_max + 1
#    axs[0].errorbar(corr_tmp, pn_tmp_noin[:,id_w_max],
#       yerr=pn_tmp_noin_std[:,id_w_max]/np.sqrt(n_seeds),
#       color=blue, label='w$_{max}%.1g$'%w_max)
#    axs[1].errorbar(corr_tmp, pn_tmp_nsi[:,id_w_max],
#       yerr=pn_tmp_nsi_std[:,id_w_max]/np.sqrt(n_seeds),
#       color=orange, label='w$_{max}%.1g$'%w_max)
#    axs[2].errorbar(corr_tmp, pn_tmp_ln[:,id_w_max],
#       yerr=pn_tmp_ln_std[:,id_w_max]/np.sqrt(n_seeds),
#       color=purple, label='w$_{max}%.1g$'%w_max)
#
#axs[1].legend()
#    
##%% *********************************************************
### FIGURE option 2: 1 column per each pair of bmax and wmax
### **********************************************************
#
##interm_est_2[id_rho, id_seed,id_w_max, id_b_max,]  = all_data_tmp[interm_est_2_id]
##od_avg1[id_rho, id_inh,id_seed,id_w_max, id_b_max,]  = all_data_tmp[od_avg1_id]
#
#fig_highfreq=1
#fig_save = True
#
#   
#rs = 3
#cs = 3
#corr_tmp = np.array(rhos)
#if fig_highfreq:
#    for inh_cond in ['NSI', 'LN']: #['ln', 'noin', 'nsi']:
#        if inh_cond.lower() == 'noin':
#            nsi_value, ln_sp_hgt, id_inh = [0.0, 0.00, 0]
#        elif inh_cond.lower() == 'nsi':
#            nsi_value, ln_sp_hgt, id_inh = [0.3, 0.00, 1]
#        elif inh_cond.lower() == 'ln':
#            nsi_value, ln_sp_hgt, id_inh = [0.0, 0.15, 2]
#        
#        
#        id_b_max = -1
#        for b_max in [.03, .3, 3]:
#            id_b_max = id_b_max + 1
#            fig, axs = plt.subplots(3,3,figsize=(15,8), )
#        
#            id_w_max = -1
#            for w_max in [.03, .3, 3]:
#                id_w_max = id_w_max + 1
#                
#                axs[0,id_w_max].errorbar(corr_tmp, 
#                   np.mean(pn_m50_1[:, id_inh, :,id_w_max, id_b_max,], axis=1),
#                   yerr=np.std(pn_m50_1[:, id_inh, :,id_w_max, 
#                   id_b_max,]/np.sqrt(n_seeds), axis=1),  color='orange',label=inh_cond+' glo 1')
#    
#                axs[0,id_w_max].errorbar(corr_tmp, np.mean(pn_m50_2[:, id_inh, :,id_w_max, id_b_max,], axis=1),
#                              yerr=np.std(pn_m50_2[:, id_inh, :,id_w_max, id_b_max,]/np.sqrt(n_seeds), axis=1),  color=blue, label=inh_cond+' glo 2')
#    
#                axs[1,id_w_max].errorbar(corr_tmp, np.mean(pn_m100_1[:, id_inh, :,id_w_max, id_b_max,], axis=1),
#                              yerr=np.std(pn_m100_1[:, id_inh, :,id_w_max, id_b_max,]/np.sqrt(n_seeds), axis=1),  color='orange',label=inh_cond+' glo 1')
#    
#                axs[1,id_w_max].errorbar(corr_tmp, np.mean(pn_m100_2[:, id_inh, :,id_w_max, id_b_max,], axis=1),
#                              yerr=np.std(pn_m100_2[:, id_inh, :,id_w_max, id_b_max,]/np.sqrt(n_seeds), axis=1),  color=blue,label=inh_cond+' glo 2')
#    
#                axs[2,id_w_max].errorbar(corr_tmp, np.mean(pn_m150_1[:, id_inh, :,id_w_max, id_b_max,], axis=1),
#                              yerr=np.std(pn_m150_1[:, id_inh, :,id_w_max, id_b_max,]/np.sqrt(n_seeds), axis=1),  color='orange',label=inh_cond+' glo 1')
#    
#                axs[2,id_w_max].errorbar(corr_tmp, np.mean(pn_m150_2[:, id_inh, :,id_w_max, id_b_max,], axis=1),
#                              yerr=np.std(pn_m150_2[:, id_inh, :,id_w_max, id_b_max,]/np.sqrt(n_seeds), axis=1),  color=blue,label=inh_cond+' glo 2')
#    
#                
#                axs[2,id_w_max].set_xlabel('Theor. correlation',)
#                
#                axs[0,id_w_max].set_title(inh_cond+
#                       ', $>50Hz$, $w_{max}$:%.2g, $bl_{max}$%.2g'%(w_max, b_max))
#                    
#                axs[1,id_w_max].set_title(inh_cond+', $>100Hz$')
#                axs[2,id_w_max].set_title(inh_cond+', $>150Hz$')
#                axs[2,id_w_max].legend()
#                
#                axs[0,id_w_max].grid()
#                axs[1,id_w_max].grid()
#                axs[2,id_w_max].grid() 
#                
#                
#            if fig_save:
#                fig.savefig(fld_analysis+ 
#                            '/NSI_HighConc_'+ inh_cond+'_'+fig_name[0]+
#                            'bl_max_%.1g'%(b_max) +'.png')
#            

        
#%%******************************************************************
## FIGURE 2, ON THE AVERAGE FIRING RATES        
##******************************************************************        
#fig2 = plt.figure(figsize=(15,8), ) 
#rs = 3
#cs = 3
#
#ax_1 = plt.subplot(rs,cs,3)
#ax_2 = plt.subplot(rs,cs,1)
#
#ax_avg_noin2 = plt.subplot(rs,cs,4)
#ax_avg_nsi2 = plt.subplot(rs,cs,5)
#
#ax_avg_noin3 = plt.subplot(rs,cs, 7)
#ax_avg_nsi3 = plt.subplot(rs,cs, 8)
#ax_avg_ln3 = plt.subplot(rs,cs, 9)
#
#corr_tmp = np.array(rhos)
#ax_1.plot(corr_tmp, np.mean(cor_stim, axis=1), '.-', label='corr stim')
#ax_1.plot(corr_tmp, np.mean(overlap_stim, axis=1), '.-', label='overlap stim')
#ax_1.plot(corr_tmp, np.mean(interm_th, axis=1), '.-', label='interm theor')
#ax_1.errorbar(corr_tmp, np.mean(interm_est_1, axis=1), 
#              yerr=np.std(interm_est_1, axis=1),  label='interm obs')
#        
#ax_1.legend()
#ax_1.set_title('Correlation/Overlap between stimuli', fontsize=fs)
#
#ax_2.errorbar(corr_tmp, np.mean(od_avg1[:, 1, :], axis=1),
#              yerr=np.std(od_avg1[:, 1, :], axis=1), color='orange',label='Odor glo 1')
#ax_2.errorbar(corr_tmp+.01, np.mean(od_avg2[:, 1, :], axis=1),
#              yerr=np.std(od_avg2[:, 1, :], axis=1),  color=blue,label='Odor glo 2')
#ax_2.set_title('Odorants concentration', fontsize=fs)
#
#ax_avg_noin2.errorbar(corr_tmp, np.mean(orn_avg1[:, 0, :], axis=1),
#              yerr=np.std(orn_avg1[:, 0, :], axis=1),  color='orange',label='ORN NOInh glo 1')
#ax_avg_nsi2.errorbar(corr_tmp, np.mean(orn_avg1[:, 1, :], axis=1),
#              yerr=np.std(orn_avg1[:, 1, :], axis=1),  color='orange',label='ORN NSI glo 1')
##ax_avg_ln2.errorbar(corr_tmp, np.mean(orn_avg1[:, 2, :], axis=1),
##              yerr=np.std(orn_avg1[:, 2, :], axis=1),  color='orange',label='ORN LN glo 1')
#
#ax_avg_noin2.errorbar(corr_tmp+.01, np.mean(orn_avg2[:, 0, :], axis=1),
#              yerr=np.std(orn_avg2[:, 0, :], axis=1),  color=blue,label='ORN NoInh glo 2')
#ax_avg_nsi2.errorbar(corr_tmp+.01, np.mean(orn_avg2[:, 1, :], axis=1),
#              yerr=np.std(orn_avg2[:, 1, :], axis=1),  color=blue,label='ORN NSI glo 2')
##ax_avg_ln2.errorbar(corr_tmp+.01, np.mean(orn_avg2[:, 2, :], axis=1),
##              yerr=np.std(orn_avg2[:, 2, :], axis=1),  color=blue,label='ORN LN glo 2')
#
#ax_avg_noin3.errorbar(corr_tmp, np.mean(pn_avg1[:, 0, :], axis=1),
#              yerr=np.std(pn_avg1[:, 0, :], axis=1),  color='orange',label='PN NOInh glo 1')
#ax_avg_nsi3.errorbar(corr_tmp, np.mean(pn_avg1[:, 1, :], axis=1),
#              yerr=np.std(pn_avg1[:, 1, :], axis=1),  color='orange',label='PN NSI glo 1')
#ax_avg_ln3.errorbar(corr_tmp, np.mean(pn_avg1[:, 2, :], axis=1),
#              yerr=np.std(pn_avg1[:, 2, :], axis=1),  color='orange',label='PN LN glo 1')
#
#ax_avg_noin3.errorbar(corr_tmp+.01, np.mean(pn_avg2[:, 0, :], axis=1),
#              yerr=np.std(pn_avg2[:, 0, :], axis=1),  color=blue,label='PN NoInh glo 2')
#ax_avg_nsi3.errorbar(corr_tmp+.01, np.mean(pn_avg2[:, 1, :], axis=1),
#              yerr=np.std(pn_avg2[:, 1, :], axis=1),  color=blue,label='PN NSI glo 2')
#ax_avg_ln3.errorbar(corr_tmp+.01, np.mean(pn_avg2[:, 2, :], axis=1),
#              yerr=np.std(pn_avg2[:, 2, :], axis=1),  color=blue,label='PN LN glo 2')
#
#ax_avg_noin2.set_title('NoInh ORN avg')
#ax_avg_nsi2.set_title('NSI ORN avg')
#
#ax_avg_noin3.set_title('NoInh PN avg')
#ax_avg_nsi3.set_title('NSI PN avg')
#ax_avg_ln3.set_title('LN PN avg')
#
#ax_1.legend()
#ax_2.legend()
#ax_1.set_xlabel('Theor. correlation',)
#ax_avg_noin3.set_xlabel('Theor. correlation',)
#ax_avg_nsi3.set_xlabel('Theor. correlation',)
#ax_avg_ln3.set_xlabel('Theor. correlation',)
#
#
#if fig_save:
#    fig2.savefig(fld_analysis+  '/NSI_AverActiv_'+name_data_all+'.png')
        
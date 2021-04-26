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
pink    = 'xkcd:pink'
cyan    = 'xkcd:cyan'
cmap    = plt.get_cmap('rainbow')




# *****************************************************************
name_analysis   = 'real_plumes'

fld_analysis    = 'NSI_analysis/analysis_'+name_analysis+'/' #'NSI_analysis/analysis_real_plumes_30s_peak1.5/'     

# ORN NSI params
nsi_ln_par      = [[0,0],  [0, .6], [.6, 0],]
seeds           = [1, 15] #np.arange(1, 31)

fld_output      = fld_analysis


# LOAD EXPERIMENT PARAMETERS
batch_params    = pickle.load(open(fld_analysis+name_analysis+'_batch_params.pickle', "rb" ))
[n_loops, nsi_ln_par, stim_dur, w_maxs, b_maxs, rhos, peak, peak_ratio  ] = batch_params 

params_al_orn   = pickle.load(open(fld_analysis +name_analysis+'_params_al_orn.ini',  "rb" ))
stim_params     = params_al_orn['stim_params']


avg_fig         = 1     # Fig.AverPNActivity
peak_fig        = 1     # Figsupp.PeakPN_thr
avgplume_fig    = 1    # FigSupp.AverCorr (Supp materials)
resumen_fig     = 1     # Fig.PeakPN_resumen
thrwmax_fig     = 1     # Fig.PeakPN_wmax

thrs        = [50, 100, 150] # thr
fig_save    = 0
fig_name    = 'dur_%d'%stim_dur + \
            '_nsi_%.2f'%(np.max(nsi_ln_par, axis=0)[0]) +\
            '_ln_%.1f'%(np.max(nsi_ln_par, axis=0)[1])             

#%% *******************************************************************
# DATA ANALYSIS 


b_max = b_maxs[0]
n_seeds = np.size(seeds)
n_rhos = np.size(rhos)
n_wmax = np.size(w_maxs)
 
cor_stim        = np.zeros((n_rhos, n_seeds, n_wmax,))
cor_whiff       = np.zeros((n_rhos, n_seeds, n_wmax,))
overlap_stim    = np.zeros((n_rhos, n_seeds, n_wmax,))
interm_th       = np.zeros((n_rhos, n_seeds, n_wmax,))
interm_est_1    = np.zeros((n_rhos, n_seeds, n_wmax,))
interm_est_2    = np.zeros((n_rhos, n_seeds, n_wmax,))

od_avg1 = np.zeros((n_rhos, 3, n_seeds, n_wmax,))
od_avg2 = np.zeros((n_rhos, 3, n_seeds, n_wmax,))
orn_avg1 = np.zeros((n_rhos, 3, n_seeds, n_wmax,))
orn_avg2 = np.zeros((n_rhos, 3, n_seeds, n_wmax,))
pn_avg1 = np.zeros((n_rhos, 3, n_seeds, n_wmax,))
pn_avg2 = np.zeros((n_rhos, 3, n_seeds, n_wmax,))

# performance thrs measures
pn_m50_1 = np.zeros((n_rhos, 3, n_seeds, n_wmax,))
pn_m50_2 = np.zeros((n_rhos, 3, n_seeds, n_wmax,))
pn_m100_1 = np.zeros((n_rhos, 3, n_seeds, n_wmax,))
pn_m100_2 = np.zeros((n_rhos, 3, n_seeds, n_wmax,))
pn_m150_1 = np.zeros((n_rhos, 3, n_seeds, n_wmax,))
pn_m150_2 = np.zeros((n_rhos, 3, n_seeds, n_wmax,))

pn_av50_1 = np.zeros((n_rhos, 3, n_seeds, n_wmax,))
pn_av100_1 = np.zeros((n_rhos, 3, n_seeds, n_wmax,))
pn_av150_1 = np.zeros((n_rhos, 3, n_seeds, n_wmax,))
pn_av50_2 = np.zeros((n_rhos, 3, n_seeds, n_wmax,))
pn_av100_2 = np.zeros((n_rhos, 3, n_seeds, n_wmax,))
pn_av150_2 = np.zeros((n_rhos, 3, n_seeds, n_wmax,))


for seed_id, stim_seed in enumerate(seeds):
    for w_max_id, w_max in enumerate(w_maxs):            
        for rho_id, rho in enumerate(rhos):
            for [id_inh, [nsi_str, alpha_ln]] in enumerate(nsi_ln_par):
                
                data_name = 'real_plumes' + \
                        '_nsi_%.1f'%(nsi_str) +\
                        '_ln_%.1f'%(alpha_ln) +\
                        '_rho_%d'%(rho) +\
                        '_wmax_%.2f'%(w_max) +\
                        '_seed_%d'%stim_seed
                        
                all_data = pickle.load(open(fld_analysis + 
                            data_name  + '.pickle',  "rb" ))
                [params_al_orn, output2an, ] = all_data
                
   
                cor_stim[rho_id, seed_id, w_max_id,]  = output2an['cor_stim']
                overlap_stim[rho_id, seed_id,w_max_id,]  = output2an['overlap_stim']
                                
                cor_whiff[rho_id, seed_id,w_max_id,]  = output2an['cor_whiff']
                interm_th[rho_id, seed_id,w_max_id,]  = output2an['interm_th']
                interm_est_1[rho_id, seed_id,w_max_id,]  = output2an['interm_est_1']
                interm_est_2[rho_id, seed_id,w_max_id,]  = output2an['interm_est_2']
                
                od_avg1[rho_id, id_inh, seed_id, w_max_id,]  = output2an['od_avg_1']
                od_avg2[rho_id, id_inh, seed_id, w_max_id,] = output2an['od_avg_2']
                orn_avg1[rho_id, id_inh, seed_id, w_max_id,]  = output2an['orn_avg_1']
                orn_avg2[rho_id, id_inh, seed_id, w_max_id,] = output2an['orn_avg_2']
                pn_avg1[rho_id, id_inh, seed_id, w_max_id,]  = output2an['pn_avg_1']
                pn_avg2[rho_id, id_inh, seed_id, w_max_id,] = output2an['pn_avg_2']
                
                # performance measure #1
                perf_time = output2an['perf_time']
                
                pn_m50_1[rho_id, id_inh, seed_id, w_max_id,]  = perf_time[0,0]
                pn_m100_1[rho_id, id_inh, seed_id, w_max_id,]  = perf_time[0,1]
                pn_m150_1[rho_id, id_inh, seed_id, w_max_id,]  = perf_time[0,2]
                
                pn_m50_2[rho_id, id_inh, seed_id, w_max_id,] = perf_time[1,0]
                pn_m100_2[rho_id, id_inh, seed_id, w_max_id,] = perf_time[1,1]
                pn_m150_2[rho_id, id_inh, seed_id, w_max_id,] = perf_time[1,2]
                
                # performance measure #2
                perf_avg = output2an['perf_avg']
                 
                # performance measure #3
                pn_m50_1[rho_id, id_inh, seed_id, w_max_id,]  = perf_avg[0,0]*perf_time[0,0]/1e3
                pn_m100_1[rho_id, id_inh, seed_id, w_max_id,]  = perf_avg[0,1]*perf_time[0,1]/1e3
                pn_m150_1[rho_id, id_inh, seed_id, w_max_id,]  = perf_avg[0,2]*perf_time[0,2]/1e3
                
                pn_m50_2[rho_id, id_inh, seed_id, w_max_id,] = perf_avg[1,0]*perf_time[1,0]/1e3
                pn_m100_2[rho_id, id_inh, seed_id, w_max_id,] = perf_avg[1,1]*perf_time[1,1]/1e3
                pn_m150_2[rho_id, id_inh, seed_id, w_max_id,] = perf_avg[1,2]*perf_time[1,2]/1e3
                


# *********************************************************
## FIGURE Fig.AverPNActivity
if avg_fig:
    w_max_id = 3
    
    
    
    fig2 = plt.figure(figsize=(12,4), ) 
    rs = 1
    cs = 3
    lw = 4
    
    ax_orn = plt.subplot(rs,cs, 1)
    ax_pn = plt.subplot(rs,cs, 2)
    ax_peak = plt.subplot(rs,cs, 3)
    
    corr_th = np.array(rhos)
    corr_obs = np.mean(np.squeeze(cor_stim[:, :, 3,]), axis=1)
                                    #[rho_id, id_inh,seed_id,w_max_id,]
                                    #[rho_id, id_inh,seed_id] 
     
    ax_orn.errorbar(corr_obs, np.squeeze(np.mean(orn_avg1[:, 0, :, w_max_id,], axis=1)),
                  yerr=np.squeeze(np.std(orn_avg1[:, 0, :, w_max_id,], axis=1))/
                  (np.size(orn_avg1[:, 0, :, w_max_id,],axis=1))**.5, linewidth=lw, ls='-.', 
                  color=pink, label='ctrl', fmt='o')
    
    ax_orn.errorbar(corr_obs, np.squeeze(np.mean(orn_avg1[:, 1, :, w_max_id,], axis=1)),
                  yerr=np.squeeze(np.std(orn_avg1[:, 1, :, w_max_id,], axis=1))/
                  (np.size(orn_avg1[:, 1, :, w_max_id,],axis=1))**.5, linewidth=lw, ls='--', 
                  color=cyan, label='NSI', fmt='*')
    
    ax_pn.errorbar(corr_obs, np.squeeze(np.mean(pn_avg1[:, 0, :, w_max_id,], axis=1)),
                  yerr=np.squeeze(np.std(pn_avg1[:, 0, :, w_max_id,], axis=1))/
                  (np.size(pn_avg1[:, 0, :, w_max_id,],axis=1))**.5, linewidth=lw, ls='-.', 
                  color=pink, label='ctrl', fmt='o')
    
    ax_pn.errorbar(corr_obs, np.squeeze(np.mean(pn_avg1[:, 1, :, w_max_id,], axis=1)),
                  yerr=np.squeeze(np.std(pn_avg1[:, 1, :, w_max_id,], axis=1))/
                  (np.size(pn_avg1[:, 1, :, w_max_id,],axis=1))**.5, linewidth=lw, ls='--', 
                  color=cyan, label='NSI', fmt='*')
    
    ax_pn.errorbar(corr_obs, np.squeeze(np.mean(pn_avg1[:, 2, :, w_max_id,], axis=1)),
                  yerr=np.squeeze(np.std(pn_avg1[:, 2, :, w_max_id,], axis=1))/
                  (np.size(pn_avg1[:, 2, :, w_max_id,],axis=1))**.5, linewidth=lw, ls='-', 
                  color='orange', label='LN', fmt='d')
    
    ax_pn.legend(fontsize=label_fs, frameon=False)

    # PEAK PANEL D
    pn_tmp = .5*(pn_m150_2+pn_m150_1)
    y_ticks = [75, 150, 225, 300]
    
    y2plot_ln = np.squeeze(pn_tmp[:, 2, :, w_max_id, ])
    y2plot_nsi = np.squeeze(pn_tmp[:, 1, :, w_max_id, ])
    y2plot_noin = np.squeeze(pn_tmp[:, 0, :, w_max_id, ])

    ax_peak.errorbar(corr_obs, np.mean(y2plot_noin, axis=1),
       yerr=np.std(y2plot_noin, axis=1)/np.sqrt(n_seeds),
       linewidth=lw, ls='-.',color=pink,label='ctrl')
    
    ax_peak.errorbar(corr_obs, np.mean(y2plot_ln, axis=1),
       yerr=np.std(y2plot_ln, axis=1)/np.sqrt(n_seeds),
       linewidth=lw, ls='-',color='orange',label='LN')
    
    ax_peak.errorbar(corr_obs, np.mean(y2plot_nsi, axis=1),
       yerr=np.std(y2plot_nsi, axis=1)/np.sqrt(n_seeds),
       linewidth=lw, ls='--',color=cyan,label='NSI')
           
    # PLOT SETTINGS 
    ax_orn.tick_params(axis='both', which='major', labelsize=label_fs-3)
    ax_pn.tick_params(axis='both', which='major', labelsize=label_fs-3)
    ax_peak.tick_params(axis='both', which='major', labelsize=label_fs-3)
    
    ax_orn.set_ylabel('ORN avg (Hz)', fontsize=label_fs)
    ax_pn.set_ylabel('PN avg (Hz)', fontsize=label_fs)
    ax_peak.set_ylabel('PN activity \n (unitless)', fontsize=label_fs)    

    ax_pn.set_xlabel('Observed correlation (unitless)', fontsize=label_fs,)
    
    ax_orn.spines['right'].set_color('none')
    ax_orn.spines['top'].set_color('none')
    ax_pn.spines['right'].set_color('none')
    ax_pn.spines['top'].set_color('none')
    ax_peak.spines['right'].set_color('none')
    ax_peak.spines['top'].set_color('none')
    
    ax_orn.text(-.15, 1.15, 'b', transform=ax_orn.transAxes,color= black,
              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    ax_pn.text(-.2, 1.15, 'c', transform=ax_pn.transAxes,color= black,
              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    ax_pn.text(-.2, 1.15, 'd', transform=ax_peak.transAxes,color= black,
              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    
    # change panels positions and sizes:
    ll, bb, ww, hh = ax_orn.get_position().bounds
    ax_orn.set_position([ll-.05,bb+.04,ww,hh-.04])        
    ll, bb, ww, hh = ax_pn.get_position().bounds
    ax_pn.set_position([ll+.01,bb+.04,ww,hh-.04])        
    ll, bb, ww, hh = ax_peak.get_position().bounds
    ax_peak.set_position([ll+.07,bb+.04,ww,hh-.04])  
    
    plt.show()
    
    if fig_save:
        fig2.savefig(fld_output+  '/NSI_AverActiv_'+fig_name+'.png')

# *********************************************************
## FIGURE Fig.PeakPNActivity

if peak_fig:

    rs = 1
    cs = 3
    corr_obs = np.mean(np.squeeze(cor_stim[:, :, 3,]), axis=1)
     
    fig, axs = plt.subplots(rs, cs, figsize=(9, 4), ) 
    w_max_id = 3
    
    for thr_id, thr in enumerate(thrs):
        if thr == 50:
            pn_tmp = .5*(pn_m50_2+pn_m50_1)           
            panel_id = 'a'
            y_ticks = [500, 1000, 1500, 2000]
        elif thr == 100:
            pn_tmp = .5*(pn_m100_2+pn_m100_1)
            y_ticks = [300, 600, 900, 1200]
            panel_id = 'b'
        elif thr == 150:
            pn_tmp = .5*(pn_m150_2+pn_m150_1)
            panel_id = 'c'
            y_ticks = [75, 150, 225, 300]
        
        
        y2plot_ln = np.squeeze(pn_tmp[:, 2, :, w_max_id, ])
        y2plot_nsi = np.squeeze(pn_tmp[:, 1, :, w_max_id, ])
        y2plot_noin = np.squeeze(pn_tmp[:, 0, :, w_max_id, ])
        
        axs[thr_id].errorbar(corr_obs, np.mean(y2plot_noin, axis=1),
           yerr=np.std(y2plot_noin, axis=1)/np.sqrt(n_seeds),
           linewidth=lw, color=pink, ls= '-.', label='Indep')
        axs[thr_id].errorbar(corr_obs, np.mean(y2plot_ln, axis=1),
           yerr=np.std(y2plot_ln, axis=1)/np.sqrt(n_seeds),
           linewidth=lw, color='orange',ls= '-',label='LN')
        axs[thr_id].errorbar(corr_obs, np.mean(y2plot_nsi, axis=1),
           yerr=np.std(y2plot_nsi, axis=1)/np.sqrt(n_seeds),
           linewidth=lw, color=cyan,ls= '--',label='NSI')
    
        # PLOT SETTINGS 
        axs[thr_id].tick_params(axis='both', which='major', labelsize=label_fs-3)
        
        axs[thr_id].spines['right'].set_color('none')
        axs[thr_id].spines['top'].set_color('none')
        
        axs[thr_id].text(-.15, 1.15, panel_id, transform=axs[thr_id].transAxes,
                  fontsize=panel_fs, fontweight='bold', va='top', ha='right')
        axs[thr_id].set_title('thr=%d Hz'%thr, fontsize=title_fs)
    axs[0].set_ylabel('PN activity \n (unitless)', fontsize=label_fs)    
    
    axs[1].set_xlabel('Observed correlation (unitless)', fontsize=label_fs,)
        
    # change panels positions and sizes:
    e_sx = 0.01
    ll, bb, ww, hh = axs[0].get_position().bounds
    axs[0].set_position([ll-.0+e_sx,bb+.04,ww,hh-.04])        
    ll, bb, ww, hh = axs[1].get_position().bounds
    axs[1].set_position([ll+ 0.03+e_sx,bb+.04,ww,hh-.04])        
    ll, bb, ww, hh = axs[2].get_position().bounds
    axs[2].set_position([ll+.06+e_sx,bb+.04,ww,hh-.04])  
    
      
    plt.show()
          
    if fig_save:
        fig.savefig(fld_output + '/NSI_HighConc_'+fig_name+'.png')
        
        
#%% *********************************************************
## FIGURE Fig.PeakPN_wmax

if thrwmax_fig:

    rs = np.size(b_maxs) 
    cs = n_wmax    

    corr_tmp = np.array(rhos)
    for thr in thrs:
        fig, axs = plt.subplots(rs, cs, figsize=(9,2.5), ) 
        
        for w_max_id, w_max in enumerate(w_maxs):
            
            if thr == 50:
                pn_tmp = .5*(pn_m50_2+pn_m50_1)           
                y_ticks = [500, 1000, 1500, 2000]
            elif thr == 100:
                pn_tmp = .5*(pn_m100_2+pn_m100_1)
                y_ticks = [300, 600, 900, 1200]
            elif thr == 150:
                pn_tmp = .5*(pn_m150_2+pn_m150_1)
                y_ticks = [75, 150, 225, 300]
                
            y2plot_ln = np.squeeze(pn_tmp[:, 2, :, w_max_id, ])
            y2plot_nsi = np.squeeze(pn_tmp[:, 1, :, w_max_id, ])
            y2plot_noin = np.squeeze(pn_tmp[:, 0, :, w_max_id, ])
    
            axs[w_max_id].errorbar(corr_tmp, np.mean(y2plot_noin, axis=1),
               yerr=np.std(y2plot_noin, axis=1)/np.sqrt(n_seeds),
               linewidth=lw, color=pink, ls='-.', label='Indep')
            axs[w_max_id].errorbar(corr_tmp, np.mean(y2plot_ln, axis=1),
               yerr=np.std(y2plot_ln, axis=1)/np.sqrt(n_seeds),
               linewidth=lw, color='orange',ls='-', label='LN')
            axs[w_max_id].errorbar(corr_tmp, np.mean(y2plot_nsi, axis=1),
               yerr=np.std(y2plot_nsi, axis=1)/np.sqrt(n_seeds),
               linewidth=lw, color=cyan,ls='--', label='NSI')
                    
            # SETTINGS
            #ax_orn.tick_params(axis='both', which='major', labelsize=label_fs-3)
            tick_label_fs = label_fs - 7
            axs[w_max_id].set_title('%.2g'%w_max+' s', fontsize=title_fs)
            axs[w_max_id].tick_params(axis='both', which='major', labelsize=tick_label_fs)
            axs[w_max_id].set_xticks(corr_tmp[[0, 1, 2]])
            axs[w_max_id].set_xticklabels([0, 0.9, 0.999]) # np.array(1-np.power(.10, corr_tmp[[0,1,2]])))
                           
            axs[w_max_id].spines['right'].set_color('none')
            axs[w_max_id].spines['top'].set_color('none')
                   
            # panels positions
            ll, bb, ww, hh = axs[w_max_id].get_position().bounds
            axs[w_max_id].set_position([ll+.01+.015*w_max_id, bb+.1, ww, hh-.15])   
            
            axs[w_max_id].set_yticklabels('', fontsize=tick_label_fs)            
            axs[w_max_id].set_yticks(y_ticks)
            # axs[w_max_id].set_ylim(0,y_ticks[-1]*1.02)                            
            
        letter_pos = [-.5, 1.25]
        axs[0].text(letter_pos[0], letter_pos[1], 'a', transform=axs[0].transAxes, 
                    fontsize=panel_fs, fontweight='bold', va='top', ha='right')
        axs[0].set_yticklabels(y_ticks, fontsize=tick_label_fs)
        axs[0].set_ylabel('PN activity \n (unitless)' , fontsize=label_fs)
        
        axs[2].set_xlabel('Observed correlation (unitless)', fontsize=label_fs)
        axs[2].xaxis.set_label_coords(1.1, -.15)
        
        
        plt.show()
        if fig_save:
            fig.savefig(fld_output+ '/NSI_nuPN_wmax_%dHz'%thr+'_'+ fig_name + '.png')    
            
        
#%%**********************************************************
# FIGURE: Fig.PeakPN_resumen

if resumen_fig:
    
    cs = 2
    rs = 1
    
    for thr_id, thr in enumerate(thrs):
        fig, axs = plt.subplots(rs, cs, figsize=(9,4), )

        # pn_m50_1[rho_id, id_inh,seed_id,w_max_id,]
        if thr == 50:
            pn_tmp = .5*(pn_m50_2+pn_m50_1)
            y_ticks = [250, 500, 750, 1000]
        elif thr == 100:
            pn_tmp = .5*(pn_m100_2+pn_m100_1)
            y_ticks = [150, 300, 450, 600]
        elif thr == 150:
            pn_tmp = .5*(pn_m150_2+pn_m150_1)
            y_ticks = [75, 150, 225, 300]
            
        pn_tmp0 = np.squeeze(np.mean(pn_tmp[0, :,:,:,], axis=1)) # PN, corr = 0 
        pn_tmp1 = np.squeeze(np.mean(pn_tmp[-1, :,:,:,], axis=1)) # PN, corr = 1
        
        delta_ln1 = np.squeeze(pn_tmp1[0,:] - pn_tmp1[2,:])
        delta_ln0 = np.squeeze(pn_tmp0[0,:] - pn_tmp0[2,:])
        delta_nsi1 = np.squeeze(pn_tmp1[0,:] - pn_tmp1[1,:])
        delta_nsi0 = np.squeeze(pn_tmp0[0,:] - pn_tmp0[1,:])
        
        delta_nsi10 = np.squeeze(pn_tmp0[1,:] - pn_tmp1[1,:])
        delta_ln10 = np.squeeze(pn_tmp0[2,:] - pn_tmp1[2,:])
        
        axs[0].plot(w_maxs, delta_ln0, '*-', color=orange, label=r'$x$=LN, $\rho=$0')
        axs[0].plot(w_maxs, delta_nsi0, '.-', color=cyan, label=r'$x$=NSI, $\rho=$0')
        
        axs[1].plot(w_maxs, delta_ln10, '*-', color=orange, label=r'$x$=LN, $\rho=$1')
        axs[1].plot(w_maxs, delta_nsi10, '.-', color=cyan, label=r'$x$=NSI, $\rho=$1')
        
        # SETTINGS        
#        axs[0].set_title(r'$\Theta$:%d Hz'%thr, fontsize=fs)
        
        axs[0].tick_params(axis='both', which='major', labelsize=label_fs-5)
        axs[1].tick_params(axis='both', which='major', labelsize=label_fs-5)
        
        axs[0].text(-.2, 1.1, 'b', transform=axs[0].transAxes,
              color= black, fontsize=panel_fs, fontweight='bold', va='top', ha='right')       
        axs[1].text(-.2, 1.1, 'c', transform=axs[1].transAxes,
              color= black, fontsize=panel_fs, fontweight='bold', va='top', ha='right')
        
        # CHANGE plot position:
        ll, bb, ww, hh = axs[0].get_position().bounds
        axs[0].set_position([ll+.02,bb+.04,ww,hh])        
        ll, bb, ww, hh = axs[1].get_position().bounds
        axs[1].set_position([ll+.06,bb+.04,ww,hh]) 
        
        for id_col in range(cs):
            axs[id_col].spines['right'].set_color('none')
            axs[id_col].spines['top'].set_color('none')
            axs[id_col].set_xlabel('$w_{max}$ (s)', fontsize=label_fs)
            axs[id_col].set_xscale('log')
            axs[id_col].set_yticks(y_ticks)
            axs[id_col].set_yticklabels(y_ticks, fontsize=label_fs-5)
            # axs[id_col].set_ylim(0, y_ticks[-1]*1.05)                    
            axs[id_col].set_xticks([.03, .3, 3, 30])
            axs[id_col].set_xticklabels([.03, .3, 3, 30], fontsize=label_fs-5)
            # make bigger ticks labels 
#            axs[id_col].tick_params(axis='both', which='major', labelsize=label_fs)
            
        axs[0].set_ylabel(r'$p_{ctrl}^0 - p_x^0$' + '  (unitless)', fontsize=label_fs)
        axs[0].text(.03, y_ticks[2]*1.1, 'x=NSI', color=cyan, fontsize=label_fs)
        axs[0].text(.03, y_ticks[1]*1.4, 'x=LN', color=orange, fontsize=label_fs)
        
        axs[1].set_ylabel(r'$p_x^0 - p_x^1$'+ '  (unitless)', fontsize=label_fs)
        axs[1].set_yticklabels('', fontsize=label_fs-5)
         
        
        plt.show()
        if fig_save:
            fig.savefig(fld_output+ '/NSI_Perf_log'+ fig_name + '_%d'%thr+ 'Hz.png')    

       
            
#%% *********************************************************
## FIGURE Fig.AverPlumeCorr

if avgplume_fig:
    fig2 = plt.figure(figsize=(12,4), ) 
    rs = 1
    cs = 2
    lw = 4
    
    ax_conc = plt.subplot(rs,cs,1)
    ax_corr = plt.subplot(rs,cs,2) # Correlation/Overlap between stimuli

    corr_th = np.array(rhos)
    
    corr_obs = np.mean(np.squeeze(cor_stim[:, :,3,]), axis=1)
    corr_obs_err = (np.std(np.squeeze(cor_stim[:, :,3,]), axis=1)/
                        np.size(np.squeeze(cor_stim[:, :,3,]))**0.5)
    overlap_obs = np.mean(np.squeeze(overlap_stim[:, :,3,]), axis=1)
    overlap_obs_err= (np.std(np.squeeze(overlap_stim[:, :,3,]), axis=1)/
                        np.size(np.squeeze(overlap_stim[:, :,3,]))**0.5)
    
    interm_av_th = np.mean(np.squeeze(interm_th[:, :,3,]), axis=1)
    interm_obs = np.mean(np.squeeze(interm_est_1[:, :,3,]), axis=1)
    interm_obs_err = (np.std(np.squeeze(interm_est_1[:, :,3,]), axis=1)/
                      np.size(np.squeeze(interm_est_1[:, :,3,]))**0.5)
    
                                    #[rho_id, id_inh,seed_id,w_max_id,]
                                    #[rho_id, id_inh,seed_id] 
    ax_conc.errorbar(corr_obs, np.squeeze(np.mean(od_avg1[:, 0, :,3,], axis=1)),
                  yerr= .1, linewidth=lw, fmt='o', color='green',label='Glo 1')

    # SETTINGS
    ax_conc.set_ylim((0, .6))
    ax_conc.set_ylabel('Odorants \n concentration', fontsize=label_fs)
    corr_th_v = 1 - np.exp(-corr_th)
    ax_corr.errorbar(corr_th_v, corr_obs, 
                  yerr=corr_obs_err, label='corr stim', fmt='.')
    ax_corr.errorbar(corr_th_v, overlap_obs, 
                  yerr=overlap_obs_err, label='overlap stim', fmt='o')
    ax_corr.plot(corr_th_v, interm_av_th, '-', label='interm theor')
    ax_corr.errorbar(corr_th_v, interm_obs, 
                  yerr=interm_obs_err, label='interm obs', fmt='d')
     
    ax_conc.tick_params(axis='both', which='major', labelsize=label_fs-3)
    ax_corr.tick_params(axis='both', which='major', labelsize=label_fs-3)
    
    ax_corr.legend(fontsize=label_fs-7, frameon=False)
    
    ax_conc.set_xlabel('Observed correlation', fontsize=label_fs)
    ax_corr.set_ylabel('Observed values (au)', fontsize=label_fs)
    ax_corr.set_xlabel('Theor. correlation', fontsize=label_fs)
    
    ax_corr.spines['right'].set_color('none')
    ax_corr.spines['top'].set_color('none')
    ax_conc.spines['right'].set_color('none')
    ax_conc.spines['top'].set_color('none')
    
    ax_conc.text(-.15, 1.15, 'a', transform=ax_conc.transAxes,
              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    ax_corr.text(-.15, 1.15, 'b', transform=ax_corr.transAxes,
              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    
    # change panels positions and sizes:
    ll, bb, ww, hh = ax_conc.get_position().bounds
    ax_conc.set_position([ll,bb+.04,ww,hh-.04])        
    ll, bb, ww, hh = ax_corr.get_position().bounds
    ax_corr.set_position([ll+.04,bb+.04,ww,hh-.04])        
        
    
    plt.show()
    if fig_save:
        fig2.savefig(fld_output+  '/NSI_PlumeAverActiv_'+fig_name+'.png')
                    
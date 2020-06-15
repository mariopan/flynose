#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:49:09 2019

@author: mp525
analysis_ratio_batch2.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle        

# *****************************************************************
# STANDARD FIGURE PARAMS
fs = 20
lw = 2
plt.rc('text', usetex=True)  # laTex in the polot
#plt.rc('font', family='serif')
fig_size = [10, 6]
fig_position = 1300,10
title_fs = 20 # font size of ticks
label_fs = 20 # font size of labels
panel_fs = 30 # font size of panel letters
ticks_fs = label_fs - 3

black   = 'xkcd:black'
blue    = 'xkcd:blue'
red     = 'xkcd:red'
green   = 'xkcd:green'
purple  = 'xkcd:purple'
orange  = 'xkcd:orange'
magenta = 'xkcd:magenta'
# *****************************************************************

# *******************************************************************
# Figure of the average activity for weak and strong input
def fig_pn_distr():
    id_conc = 0
    id_dur = 0
    n_bins  = 20
    for id_ratio in [0, 3, 9]:
        
        # pn_peak_s_noin: (n_ratios, n_concs,n_durs, n_loops)
        noin_s = np.squeeze(pn_peak_s_noin[id_ratio, id_conc, id_dur,:])
        noin_w = np.squeeze(pn_peak_w_noin[id_ratio, id_conc, id_dur,:])
        ln_s = np.squeeze(pn_peak_s_ln[id_ratio, id_conc, id_dur,:])
        ln_w = np.squeeze(pn_peak_w_ln[id_ratio, id_conc, id_dur,:])
        nsi_s = np.squeeze(pn_peak_s_nsi[id_ratio, id_conc, id_dur,:])
        nsi_w = np.squeeze(pn_peak_w_nsi[id_ratio, id_conc, id_dur,:])
        
        rs = 3
        cs = 1
        fig, axs = plt.subplots(rs, cs, figsize=[9,4.5])
        n_tmp, _, _ = axs[0].hist(noin_s, bins=n_bins, label='ctrl s', 
                                 color=green, alpha=.5, density=True,)  
        n_tmp, _, _ = axs[0].hist(noin_w, bins=n_bins, label='ctrl w', 
                                 color=purple, alpha=.5, density=True,)  
        n_tmp, _, _ = axs[1].hist(ln_s, bins=n_bins, label='ctrl s', 
                                 color=green, alpha=.5, density=True,)  
        n_tmp, _, _ = axs[1].hist(ln_w, bins=n_bins, label='ctrl w', 
                                 color=purple, alpha=.5, density=True,)  
        n_tmp, _, _ = axs[2].hist(nsi_s, bins=n_bins, label='ctrl s', 
                                 color=green, alpha=.5, density=True,)  
        n_tmp, _, _ = axs[2].hist(nsi_w, bins=n_bins, label='ctrl w', 
                                 color=purple, alpha=.5, density=True,)  
        
        axs[0].set_title('conc:%.1f'%concs2an[id_conc]+
           ', ratio:%.1f'%conc_ratios[id_ratio]+', dur:%d'%dur2an[id_dur],fontsize=title_fs)
        axs[0].set_ylabel('ctrl',fontsize=label_fs)
        axs[1].set_ylabel('LN',fontsize=label_fs)
        axs[2].set_ylabel('NSI',fontsize=label_fs)
        if fig_save:
            fig.savefig(fld_output+  '/PN_distr_delays0_conc%.1f'%concs2an[id_conc]+\
           '_ratio%.1f'%conc_ratios[id_ratio]+'_dur:%d'%dur2an[id_dur]+'.png')
    
    
# *******************************************************************
# Figure of the average activity for weak and strong input
def fig_activity():
    # pn_peak_s_noin: (n_ratios, n_concs,n_durs, n_loops)
    noin_s = np.squeeze(np.median(pn_peak_s_noin, axis=3))
#    ln_s = np.squeeze(np.median(pn_peak_s_ln, axis=3))
#    nsi_s = np.squeeze(np.median(pn_peak_s_nsi, axis=3))
    
    noin_w = np.squeeze(np.median(pn_peak_w_noin, axis=3))
#    ln_w = np.squeeze(np.median(pn_peak_w_ln, axis=3))
#    nsi_w = np.squeeze(np.median(pn_peak_w_nsi, axis=3))
    
    rs = 1
    cs = 3
    fig, axs = plt.subplots(rs, cs, figsize=(9, 3), ) 
    
    im0 = axs[0].imshow(noin_s, cmap='viridis')
    fig.colorbar(im0, ax=axs[0])
    
    axs[0].set_title('strong, delay=%d'%delay)
    axs[1].set_title('weak')
    axs[2].set_title('ratio')
    
    im1 = axs[1].imshow(noin_w, cmap='viridis')
    fig.colorbar(im1, ax=axs[1])
    
    im2=axs[2].imshow(noin_w/noin_s, cmap='viridis')
    fig.colorbar(im2, ax=axs[2])

# *****************************************************************
fig_save        = 1

id_peak2plot    = 3
measure         = 'peak' # 'avg' # 'peak' # 
delay_fig       = 0 # Fig.ResumeDelayedStimuli
# select a subsample of the params to analyse
nsi_ln_par   = [[0,0],[0.3,0],[0, 2.2]] 
            # [[0,0],[0.3,0],[0,16.6]] 
            # [[0,0],[0.3,0],[0,13.3]] 
            # [[0,0],[0.3,0],[0,10]]

if delay_fig:
    fld_analysis    = 'NSI_analysis/ratio/delays_data'
    fld_output      = 'NSI_analysis/ratio/delays_images_nsi%.1f'%nsi_ln_par[1][0]+\
                        '_ln%.1f'%nsi_ln_par[2][1]
else:
    fld_analysis    = 'NSI_analysis/ratio/ratio_data'
    fld_output      = 'NSI_analysis/ratio/ratio_images/ratio_images_nsi%.1f'%nsi_ln_par[1][0]+\
                        '_ln%.1f'%nsi_ln_par[2][1]
#import os 
#os.mkdir(fld_output)

# LOAD EXPERIMENT PARAMETERS
batch_params    = pickle.load(open(fld_analysis+'/batch_params.pickle', "rb" ))
[n_loops, conc_ratios, concs2an, _, dur2an, delays2an,] = batch_params

if delay_fig==0:
    delays2an=[0,]
n_durs          = np.size(dur2an)
n_delays        = np.size(delays2an)
n_ratios        = np.size(conc_ratios)
n_concs         = np.size(concs2an)


# *****************************************************************
# analysis for zero delay:
ratio_fig       = 1 # Fig.RatioPeak
resumen_chess   = 1 # Fig.ResumeEncodeRatioChess
pn_chess        = 1 # Fig.PNChess
resumen_bar     = 0 # Fig.ResumeEncodeRatioBar
pn_distr        = 1 # Fig.PNdistribution
# *****************************************************************

    
# *****************************************************************
# IMPLEMENT OUTPUT VARIABLES
if delay_fig:

    ratio_avg_noin = np.ones((n_delays,n_durs))
    ratio_peak_noin = np.ones((n_delays,n_durs))
    ratio_avg_noin_err = np.ones((n_delays,n_durs))
    ratio_peak_noin_err = np.ones((n_delays,n_durs))
    
    ratio_avg_ln = np.ones((n_delays,n_durs))
    ratio_avg_nsi = np.ones((n_delays,n_durs))
    ratio_peak_ln = np.ones((n_delays,n_durs))
    ratio_peak_nsi = np.ones((n_delays,n_durs))
    
    ratio_avg_ln_err = np.ones((n_delays,n_durs))
    ratio_avg_nsi_err = np.ones((n_delays,n_durs))
    ratio_peak_ln_err = np.ones((n_delays,n_durs))
    ratio_peak_nsi_err = np.ones((n_delays,n_durs))
    
#*****************************************************
# LOAD DATA AND CALCULATE RATIOS
for delay_id, delay in enumerate(delays2an):
    for [inh_id, [nsi_str, alpha_ln]] in enumerate(nsi_ln_par):
        data_name = 'ratio_nsi_%.2f_ln_%.1f_delay_%d'%(nsi_str, alpha_ln, delay)
#       from the batch_file:
#        pickle.dump([params2an, sdf_params, concs2an, conc_ratios, dur2an,
#                         avg_ornw, avg_orns, avg_pnw, avg_pns, 
#                         peak_ornw, peak_orns, peak_pnw, peak_pns, saved_pars], f)
        
        all_data    = pickle.load(open(fld_analysis+'/' +data_name+'.pickle',  "rb" ) )
        [params2an, sdf_params, _, _, _] = all_data[0:5]
        
        
        if (alpha_ln==0) & (nsi_str==0):
            orn_avg_w_noin   = all_data[5]
            orn_avg_s_noin   = all_data[6]
            pn_avg_w_noin    = all_data[7]
            pn_avg_s_noin    = all_data[8]
            
            orn_peak_w_noin   = all_data[9]
            orn_peak_s_noin   = all_data[10]
            pn_peak_w_noin    = all_data[11]  # minimum value 10Hz
            pn_peak_s_noin    = all_data[12] 
        
        elif (alpha_ln>0) & (nsi_str==0):
            orn_avg_w_ln   = all_data[5]
            orn_avg_s_ln   = all_data[6]
            pn_avg_w_ln    = all_data[7]
            pn_avg_s_ln    = all_data[8]
            
            orn_peak_w_ln = all_data[9]
            orn_peak_s_ln   = all_data[10]
            pn_peak_w_ln    = all_data[11]
            pn_peak_s_ln    = all_data[12]
            
        elif (alpha_ln==0) & (nsi_str>0):
            orn_avg_w_nsi   = all_data[5]
            orn_avg_s_nsi   = all_data[6]
            pn_avg_w_nsi    = all_data[7]
            pn_avg_s_nsi    = all_data[8] #(n_ratios, n_concs,n_durs, n_loops)
            
            orn_peak_w_nsi   = all_data[9]
            orn_peak_s_nsi   = all_data[10]
            pn_peak_w_nsi    = all_data[11]
            pn_peak_s_nsi    = all_data[12]
            
        elif (alpha_ln>0) & (nsi_str>0):
            print(params2an[0:2])
#    if delay_fig:
#       fig_activity()    
    if pn_distr: 
        fig_pn_distr()
    #(n_ratios, n_concs,n_durs, n_loops)
    orn_ratio_avg_nsi   = np.ma.masked_invalid(orn_avg_s_nsi/orn_avg_w_nsi)
    orn_ratio_avg_ln    = np.ma.masked_invalid(orn_avg_s_ln/orn_avg_w_ln)
    orn_ratio_avg_noin  = np.ma.masked_invalid(orn_avg_s_noin/orn_avg_w_noin)
    
    pn_ratio_avg_nsi    = np.ma.masked_invalid(pn_avg_s_nsi/pn_avg_w_nsi)
    pn_ratio_avg_ln     = np.ma.masked_invalid(pn_avg_s_ln/pn_avg_w_ln)
    pn_ratio_avg_noin   = np.ma.masked_invalid(pn_avg_s_noin/pn_avg_w_noin)
    
    orn_ratio_peak_nsi   = np.ma.masked_invalid(orn_peak_s_nsi/orn_peak_w_nsi)
    orn_ratio_peak_ln    = np.ma.masked_invalid(orn_peak_s_ln/orn_peak_w_ln)
    orn_ratio_peak_noin  = np.ma.masked_invalid(orn_peak_s_noin/orn_peak_w_noin)
    
    pn_ratio_peak_nsi    = np.ma.masked_invalid(pn_peak_s_nsi/pn_peak_w_nsi)
    pn_ratio_peak_ln     = np.ma.masked_invalid(pn_peak_s_ln/pn_peak_w_ln)
    pn_ratio_peak_noin   = np.ma.masked_invalid(pn_peak_s_noin/pn_peak_w_noin)
    
    if delay_fig:#(n_ratios, n_concs,n_durs, n_loops)
        # average over the run with identical params
        if measure == 'avg':
            ratio_avg_noin[delay_id, :] = np.median(pn_ratio_avg_noin[0,id_peak2plot ,:,:], axis=1)
            ratio_avg_ln[delay_id, :] = np.median(pn_ratio_avg_ln[0,id_peak2plot ,:,:], axis=1)
            ratio_avg_nsi[delay_id, :] =np.median(pn_ratio_avg_nsi[0,id_peak2plot,:,:], axis=1)
            
            ratio_avg_noin_err[delay_id, :] = np.diff(np.percentile(pn_ratio_avg_noin[0,id_peak2plot,:,:], [25,50])) #np.std(pn_ratio_avg_noin[0,id_peak2plot,:,:], axis=1)
            ratio_avg_ln_err[delay_id, :] = np.diff(np.percentile(pn_ratio_avg_ln[0,id_peak2plot,:,:], [25,50])) #np.std(pn_ratio_avg_ln[0,id_peak2plot ,:,:], axis=1)
            ratio_avg_nsi_err[delay_id, :] =np.diff(np.percentile(pn_ratio_avg_nsi[0,id_peak2plot,:,:], [25,50])) #np.std(pn_ratio_avg_nsi[0,id_peak2plot,:,:], axis=1)
        
        elif measure == 'peak':
            ratio_peak_noin[delay_id, :] = np.median(pn_ratio_peak_noin[0,id_peak2plot,:,:], axis=1)
            ratio_peak_ln[delay_id, :] = np.median(pn_ratio_peak_ln[0,id_peak2plot,:,:], axis=1)
            ratio_peak_nsi[delay_id, :] =np.median(pn_ratio_peak_nsi[0,id_peak2plot,:,:], axis=1)
            
            ratio_peak_noin_err[delay_id, :] = np.diff(np.percentile(pn_ratio_peak_noin[0,id_peak2plot,:,:], [25,50])) #np.std(pn_ratio_peak_noin[0,id_peak2plot,:,:], axis=1)
            ratio_peak_ln_err[delay_id, :] = np.diff(np.percentile(pn_ratio_peak_ln[0,id_peak2plot,:,:], [25,50])) # np.std(pn_ratio_peak_ln[0,id_peak2plot,:,:], axis=1)
            ratio_peak_nsi_err[delay_id, :] = np.diff(np.percentile(pn_ratio_peak_nsi[0,id_peak2plot,:,:], [25,50])) # np.std(pn_ratio_peak_nsi[0,id_peak2plot,:,:], axis=1)
    else:
        if measure == 'avg':
            # average over the run with identical params
            ratio1_noin = np.median(orn_ratio_avg_noin, axis=3)
            ratio1_nsi  = np.median(orn_ratio_avg_nsi, axis=3)
            ratio1_ln   = np.median(orn_ratio_avg_ln, axis=3)
            
            ratio1_err_noin = np.squeeze(np.diff(np.percentile(orn_ratio_avg_noin, [25,50],axis=3), axis=0))
            ratio1_err_nsi = np.squeeze(np.diff(np.percentile(orn_ratio_avg_nsi,  [25,50],axis=3), axis=0)) 
            ratio1_err_ln = np.squeeze(np.diff(np.percentile(orn_ratio_avg_ln,  [25,50],axis=3), axis=0)) 
            
            # average over the run with identical params
            ratio2_noin = np.median(pn_ratio_avg_noin, axis=3)
            ratio2_nsi  = np.median(pn_ratio_avg_nsi, axis=3)
            ratio2_ln   = np.median(pn_ratio_avg_ln, axis=3)
            
            ratio2_err_noin = np.squeeze(np.diff(np.percentile(pn_ratio_avg_noin, [25,50],axis=3), axis=0))#np.std(pn_ratio_peak_noin, axis=3)
            ratio2_err_nsi = np.squeeze(np.diff(np.percentile(pn_ratio_avg_nsi,  [25,50],axis=3), axis=0)) #np.std(pn_ratio_peak_nsi, axis=3)
            ratio2_err_ln = np.squeeze(np.diff(np.percentile(pn_ratio_avg_ln,  [25,50],axis=3), axis=0)) #np.std(pn_ratio_peak_ln, axis=3)  
                
            noin_tmp = ((conc_ratios-pn_ratio_avg_noin.T)/
                        (pn_ratio_avg_noin.T + conc_ratios))**2
            ln_tmp = ((conc_ratios - pn_ratio_avg_ln.T)/
                        (pn_ratio_avg_ln.T + conc_ratios))**2
            nsi_tmp = ((conc_ratios - pn_ratio_avg_nsi.T)/
                        (pn_ratio_avg_nsi.T + conc_ratios))**2
        elif measure == 'peak':
            # average over the run with identical params
            ratio1_noin = np.median(orn_ratio_peak_noin, axis=3)
            ratio1_nsi  = np.median(orn_ratio_peak_nsi, axis=3)
            ratio1_ln   = np.median(orn_ratio_peak_ln, axis=3)
            
            ratio1_err_noin = np.squeeze(np.diff(np.percentile(orn_ratio_peak_noin, [25,50],axis=3), axis=0))#np.std(pn_ratio_peak_noin, axis=3)
            ratio1_err_nsi = np.squeeze(np.diff(np.percentile(orn_ratio_peak_nsi,  [25,50],axis=3), axis=0)) #np.std(pn_ratio_peak_nsi, axis=3)
            ratio1_err_ln = np.squeeze(np.diff(np.percentile(orn_ratio_peak_ln,  [25,50],axis=3), axis=0)) #np.std(pn_ratio_peak_ln, axis=3)  
            
            # average over the run with identical params
            ratio2_noin = np.median(pn_ratio_peak_noin, axis=3)
            ratio2_nsi  = np.median(pn_ratio_peak_nsi, axis=3)
            ratio2_ln   = np.median(pn_ratio_peak_ln, axis=3)
            
            ratio2_err_noin = np.squeeze(np.diff(np.percentile(pn_ratio_peak_noin, [25,50],axis=3), axis=0))#np.std(pn_ratio_peak_noin, axis=3)
            ratio2_err_nsi = np.squeeze(np.diff(np.percentile(pn_ratio_peak_nsi,  [25,50],axis=3), axis=0)) #np.std(pn_ratio_peak_nsi, axis=3)
            ratio2_err_ln = np.squeeze(np.diff(np.percentile(pn_ratio_peak_ln,  [25,50],axis=3), axis=0)) #np.std(pn_ratio_peak_ln, axis=3)  
                
            
            noin_tmp = ((conc_ratios-pn_ratio_avg_noin.T)/
                        (pn_ratio_avg_noin.T + conc_ratios))**2
            ln_tmp = ((conc_ratios - pn_ratio_avg_ln.T)/
                        (pn_ratio_avg_ln.T + conc_ratios))**2
            nsi_tmp = ((conc_ratios - pn_ratio_avg_nsi.T)/
                        (pn_ratio_avg_nsi.T + conc_ratios))**2
            
        # average and std over runs with identical params
        ratio2dist_noin = np.mean(noin_tmp, axis=0).T
        ratio2dist_nsi = np.mean(nsi_tmp, axis=0).T
        ratio2dist_ln = np.mean(ln_tmp, axis=0).T
        
        ratio2dist_err_noin = np.ma.std(noin_tmp, axis=0).T
        ratio2dist_err_nsi = np.ma.std(nsi_tmp, axis=0).T
        ratio2dist_err_ln = np.ma.std(ln_tmp, axis=0).T
        


#%%***********************************************************
## FIGURE ResumeDelayedStimuli
## **********************************************************
if delay_fig:
    y_ticks = np.linspace(0, 2, 5)
    fig, axs = plt.subplots(1, n_durs, figsize=(9.5, 3.5), ) 
    for dur_id in range(n_durs):
        duration = dur2an[dur_id]
        
        if measure=='avg':
            axs[dur_id].errorbar(delays2an, ratio_avg_noin[:, dur_id], 
               yerr=ratio_avg_noin_err[:, dur_id], color='magenta', lw = lw, label= 'ctrl')
            axs[dur_id].errorbar(delays2an, ratio_avg_ln[:, dur_id], 
               yerr=ratio_avg_ln_err[:, dur_id], color=orange, lw = lw, label= 'LN inhib.')
            axs[dur_id].errorbar(delays2an, ratio_avg_nsi[:, dur_id], 
               yerr=ratio_avg_nsi_err[:, dur_id], color=blue, lw = lw, label= 'NSI')    
        elif measure=='peak':        
            axs[dur_id].errorbar(delays2an, ratio_peak_noin[:, dur_id], 
               yerr=ratio_peak_noin_err[:, dur_id], color='magenta', lw = lw, label= 'ctrl')
            axs[dur_id].errorbar(delays2an, ratio_peak_ln[:, dur_id], 
               yerr=ratio_peak_ln_err[:, dur_id], color=orange, lw = lw, label= 'LN inhib.')
            axs[dur_id].errorbar(delays2an, ratio_peak_nsi[:, dur_id],
               yerr=ratio_peak_nsi_err[:, dur_id], color=blue, lw = lw, label= 'NSI')    
        
        # FIGURE SETTINGS
        axs[dur_id].set_title(' %d ms'%(duration), fontsize=title_fs)
        
        axs[dur_id].spines['right'].set_color('none')   
        axs[dur_id].spines['top'].set_color('none')     
#        axs[dur_id].tick_params(axis='both', which='major', labelsize=label_fs-3)
        
        axs[dur_id].set_yticks(y_ticks)
        if dur_id>0:
            axs[dur_id].set_yticklabels('', fontsize=label_fs-5)
        axs[dur_id].set_xticks([0, 250, 500])
        axs[dur_id].set_xticklabels(['0','250','500'], fontsize=label_fs-5)

        axs[dur_id].set_ylim((.0, 1.6))
        
        # original plot position:
        ll, bb, ww, hh = axs[dur_id].get_position().bounds
        axs[dur_id].set_position([ll-.04+.025*dur_id, bb+.1, ww+.025, hh-.15]) 
        
    axs[0].set_yticks([0,.5,1.0,1.5])
    axs[0].set_yticklabels([0,.5,1.0,1.5], fontsize=label_fs-5)
    conc2plot = np.squeeze(concs2an[id_peak2plot]) #  conc_1_r[0,id_peak2plot,0])
    axs[0].set_ylabel(r'$R^{PN} $ (unitless)', fontsize=label_fs)
    axs[2].set_xlabel('Delay (ms)', fontsize=fs)
    

    if measure=='peak':
        axs[0].text(-.2, 1.2, 'b', transform=axs[0].transAxes,
           fontsize=panel_fs, color=black, weight='bold', va='top', ha='right')

    if fig_save:
        if measure == 'avg':
            fig.savefig(fld_output+  '/ratio_avg_delays0-500_dur20-200_conc%.2g'%conc2plot +'.png')
        elif measure == 'peak':
            fig.savefig(fld_output+  '/ratio_peak_delays0-500_dur20-200_conc%.2g'%conc2plot +'.png')
   
#%% *********************************************************
## FIGURE peak
## **********************************************************
if ratio_fig: 
    lw = 3
    rs = 2
    cs = 3
    colors = plt.cm.winter_r
    clr_fct = 30        # color factor
    
    panels_id   = ['a', 'b', 'c', 'd', 'e', 'f', ]
    dur2plot = dur2an[2]
    for dur_id, duration in enumerate(dur2an):
        fig, axs = plt.subplots(rs, cs, figsize=(10,7), ) 
        axs[0,0].set_title(['dur: %d ms'%duration])
        
        dx = .1
        for conc_id, conc_v in enumerate(concs2an): 
            axs[0,0].errorbar(conc_ratios+dx*conc_id, 
               ratio1_noin[:,conc_id, dur_id],
               yerr= ratio1_err_noin[:,conc_id, dur_id], marker='o', 
               label=r'conc1: '+'%.1f'%(conc_v), color=colors(conc_id*clr_fct) )
            
            axs[0,1].errorbar(conc_ratios+dx*conc_id, 
               ratio1_ln[:,conc_id, dur_id],
               yerr= ratio1_err_ln[:,conc_id, dur_id],  marker='o', 
               label=r'conc1: '+'%.1f'%(conc_v), color=colors(conc_id*clr_fct) )
            
            axs[0,2].errorbar(conc_ratios+dx*conc_id, 
               ratio1_nsi[:,conc_id, dur_id],
               yerr= ratio1_err_nsi[:,conc_id, dur_id], marker='o', 
               label=r'conc1: '+'%.1f'%(conc_v), color=colors(conc_id*clr_fct) )
            
            axs[1,0].errorbar(conc_ratios+dx*conc_id, ratio2_noin[:,conc_id, dur_id],
               yerr= ratio2_err_noin[:,conc_id, dur_id], marker='o', 
               label=r''+'%.1f'%(conc_v), 
               color=colors(conc_id*clr_fct) )
            
            axs[1,1].errorbar(conc_ratios+dx*conc_id, ratio2_ln[:,conc_id, dur_id],
               yerr= ratio2_err_ln[:,conc_id, dur_id], marker='o', 
               label=r''+'%.1f'%(conc_v), 
               color=colors(conc_id*clr_fct) )
            
            axs[1,2].errorbar(conc_ratios+dx*conc_id, ratio2_nsi[:,conc_id, dur_id],
               yerr= ratio2_err_nsi[:,conc_id, dur_id], marker='o', 
               label=r''+'%.1f'%(conc_v), 
               color=colors(conc_id*clr_fct) )
            
        
        # FIGURE settings
        axs[0, 0].set_ylabel(r'$R^{ORN} $ (unitless)', fontsize=label_fs)
        axs[1, 0].set_ylabel(r'$R^{PN} $ (unitless)', fontsize=label_fs)      

        axs[1, 1].set_xlabel('Concentration ratio (unitless)', fontsize=label_fs)
        for cc in range(cs):
                
            axs[0, cc].plot(conc_ratios, conc_ratios, '--', lw=lw, color='black', label='expec.')
            axs[1, cc].plot(conc_ratios, conc_ratios, '--', lw=lw, color='black', label='expec.')
            for rr in range(rs):
                axs[rr, cc].tick_params(axis='both', which='major', labelsize=label_fs-3)
                
                axs[rr,cc].set_xticklabels('')
                axs[rr,cc].set_yticklabels('')
                axs[rr,cc].spines['right'].set_color('none')
                axs[rr,cc].spines['top'].set_color('none')
                
                axs[rr,cc].set_xlim((0, 20.5))
                axs[rr,cc].set_ylim((0, 20.5))
                
                axs[rr,cc].set_yticks([0, 5, 10, 15, 20])
                axs[rr,cc].set_xticks([0, 5, 10, 15, 20])
                
                if cc == 0:
                    axs[rr,cc].set_yticklabels(['0','5','10', '15', '20'], fontsize=label_fs-3)
                
                if rr == 1:
                    axs[rr,cc].set_xticklabels(['0','5','10', '15', '20'], fontsize=label_fs-3)
                
                # change plot position:
                ll, bb, ww, hh = axs[rr,cc].get_position().bounds
                axs[rr,cc].set_position([ll+cc*.03, bb+(2-rr)*.03,ww,hh])        

        axs[0,0].set_title('ctrl ', fontsize=label_fs)
        axs[0,1].set_title('LN', fontsize=label_fs)
        axs[0,2].set_title('NSI', fontsize=label_fs)
        

        for cc in [1,2]:
            for rr in range(2):
                axs[rr,cc].text(-.2, 1.2, panels_id[cc*rs+rr], transform=axs[rr,cc].transAxes,
                               fontsize=panel_fs, color=black, weight='bold', va='top', ha='right')
        cc = 0
        for rr in range(2):
            axs[rr,cc].text(-.35, 1.2, panels_id[cc*rs+rr], transform=axs[rr,cc].transAxes,
                           fontsize=panel_fs, color=black, weight='bold', va='top', ha='right')
        axs[1,2].legend(frameon=False)
        if fig_save:
            if measure == 'avg':
                fig.savefig(fld_output+ 
                        '/ratio_stim_avg_dur%d'%duration+'_delay%d'%delay+'.png')        
            elif measure == 'peak':
                fig.savefig(fld_output+  
                        '/ratio_stim_peak_dur%d'%duration+'_delay%d'%delay+'.png')
                    
#%%***********************************************************
# FIGURE PNChess
# Figure of the average activity for weak and strong input
# **********************************************************
if pn_chess:
    ratio2plot = np.round(conc_ratios[::2])
    conc2plot = concs2an[::1]
    
    rs = 3
    cs = 2
    for dur_id, duration in enumerate(dur2an):
        # *******************************************************************
        # pn_peak_s_noin: (n_ratios, n_concs,n_durs, n_loops)
        noin_s = np.median(pn_peak_s_noin[:,:,dur_id, :], axis=2)
        ln_s = np.median(pn_peak_s_ln[:,:,dur_id, :], axis=2)
        nsi_s =np.median(pn_peak_s_nsi[:,:,dur_id, :], axis=2)
        
        noin_w = np.median(pn_peak_w_noin[:,:,dur_id, :], axis=2)
        ln_w = np.median(pn_peak_w_ln[:,:,dur_id, :], axis=2)
        nsi_w = np.median(pn_peak_w_nsi[:,:,dur_id, :], axis=2)
        
        fig, axs = plt.subplots(rs, cs, figsize=(9, 6), ) 
        
        axs[0,0].set_title('PN strong')
        axs[0,1].set_title('PN weak')

        for id_r in range(rs):
            axs[id_r,0].set_ylabel('input (a.u.)', fontsize= label_fs)
        
        axs[0,1].text(12.5, 1.5, 'ctrl', fontsize=label_fs)
        axs[1,1].text(12.5, 1.5, 'LN', fontsize=label_fs)
        axs[2,1].text(12.5, 1.5, 'NSI', fontsize=label_fs)
        im0 = axs[0,0].imshow(noin_s.T, cmap='viridis')
        fig.colorbar(im0, ax=axs[0,0])
        
        im1 = axs[0, 1].imshow(noin_w.T, cmap='viridis')
        fig.colorbar(im1, ax=axs[0,1])
        
        im0 = axs[1, 0].imshow(ln_s.T, cmap='viridis')
        fig.colorbar(im0, ax=axs[1,0])
        
        im1 = axs[1, 1].imshow(ln_w.T, cmap='viridis')
        fig.colorbar(im1, ax=axs[1,1])
        
        im0 = axs[2, 0].imshow(nsi_s.T, cmap='viridis')
        fig.colorbar(im0, ax=axs[2,0])
        
        im1 = axs[2, 1].imshow(nsi_w.T, cmap='viridis')
        fig.colorbar(im1, ax=axs[2,1])
        
        for c_id in range(cs):
            for r_id in range(rs):
                axs[r_id,c_id].set_xticks(range(0,10,2))
                axs[r_id,c_id].set_xticklabels(ratio2plot)
                axs[r_id,c_id].set_yticks(range(4))
                axs[r_id,c_id].set_yticklabels(conc2plot)
                
            axs[2, c_id].set_xlabel('ratio (unitless)', fontsize= label_fs)
            
        if fig_save:
            fig.savefig(fld_output+  '/PN_delays0_dur%d'%duration+'.png')

#%% ************************************************************************
# RESUME BAR FIGURE
if resumen_chess:
    # average over conc.ratio and concentrations
    err_code_nsi = np.mean(ratio2dist_nsi, axis=(0))
    err_code_ln = np.mean(ratio2dist_ln, axis=(0))
    err_code_noin = np.mean(ratio2dist_noin, axis=(0))
    
    rs = 1
    cs = 3
    vmax = 0.4
#    colorbar params
    frac = .1 #.04
    pad = .04
    
    fig, axs = plt.subplots(rs, cs, figsize=(11, 3),) 

    ptns = np.arange(5)
    im0 = axs[0].imshow(err_code_noin, cmap='viridis', vmin=0, vmax=vmax)
    
    im1 = axs[1].imshow(err_code_ln, cmap='viridis', vmin=0, vmax=vmax)
    
    im2 = axs[2].imshow(err_code_nsi, cmap='viridis', vmin=0, vmax=vmax)
    cbar = fig.colorbar(im0, ax=axs[2], fraction=.05,pad=pad,orientation='vertical', 
                 ticks=[0, .2, .4])
    cbar.set_ticklabels([0, .2, .4])
    cbar.set_label('code error', fontsize=label_fs)
    
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=ticks_fs)
    
    # FIGURE SETTINGS
    axs[0].set_title('ctrl', fontsize=title_fs)
    axs[1].set_title('LN', fontsize=title_fs)
    axs[2].set_title('NSI', fontsize=title_fs)
    
    dur2plot = np.round(dur2an[::1])
    conc2plot = concs2an[::1]
    for c_id in range(cs):
        axs[c_id].set_xticks(range(n_durs))
        axs[c_id].set_xticklabels(dur2plot, fontsize= ticks_fs)
        axs[c_id].set_yticks(range(n_concs))
        axs[c_id].set_yticklabels('', fontsize= ticks_fs)      
        
        axs[c_id].set_xlabel('duration (ms)', fontsize= label_fs)
    axs[0].set_yticklabels(conc2plot, fontsize= ticks_fs)  
    axs[0].set_ylabel('input (a.u.)', fontsize= label_fs)
    
    
#    # move plot position:
    ll, bb, ww, hh = axs[0].get_position().bounds
    axs[0].set_position([ll-.05, bb+.07,ww,hh])    
    ll, bb, ww, hh = axs[1].get_position().bounds
    axs[1].set_position([ll-.05, bb+.07,ww,hh])        
    
    db=.005
    ll2, bb2, ww2, hh2 = axs[2].get_position().bounds
    axs[2].set_position([ll+ww, bb+.07, ww+db, hh+db])
        
    if fig_save:
        if measure == 'peak':
            fig.savefig(fld_output + '/ratio_stim_peak_resumechess_durs_delay%d'%delay+'.png')
        elif measure == 'avg':    
            fig.savefig(fld_output + '/ratio_stim_avg_resumechess_durs_delay%d'%delay+'.png')
            
            
#%% ************************************************************************
# RESUME BAR FIGURE
if resumen_bar:
    # average over conc.ratio and concentrations
    avg_ratio_peak_nsi = np.mean(ratio2dist_nsi, axis=(0,1))
    avg_ratio_peak_ln = np.mean(ratio2dist_ln, axis=(0,1))
    avg_ratio_peak_noin = np.mean(ratio2dist_noin, axis=(0,1))
    avg_ratio_peak_nsi_std = np.std(ratio2dist_nsi, axis=(0,1))
    avg_ratio_peak_ln_std = np.std(ratio2dist_ln, axis=(0,1))
    avg_ratio_peak_noin_std = np.std(ratio2dist_noin, axis=(0,1))
    
    width = 0.15
    y_ticks = [0.27, .32,.37]
    rs = 1
    cs = 1
    fig, axs = plt.subplots(rs, cs, figsize=(9,4), ) 

    ptns = np.arange(5)
    axs.bar(ptns-width, avg_ratio_peak_noin, width=width, color='magenta', 
            yerr=avg_ratio_peak_noin_std/np.sqrt(n_ratios*n_concs), 
            label='ctrl', )
    axs.bar(ptns, avg_ratio_peak_ln, width=width, color=orange, 
            yerr=avg_ratio_peak_ln_std/np.sqrt(n_ratios*n_concs), 
            label='LN', )
    axs.bar(ptns+width, avg_ratio_peak_nsi, width=width, color=blue, 
            yerr=avg_ratio_peak_nsi_std/np.sqrt(n_ratios*n_concs), 
            label='NSI', )

    # FIGURE SETTINGS
    axs.spines['right'].set_color('none')   
    axs.spines['top'].set_color('none')                

#    axs.legend(fontsize=label_fs,loc='upper left', frameon=False)
    axs.set_ylabel('coding error (a.u.)', fontsize=label_fs)
    axs.set_xlabel('stimulus duration (ms)', fontsize=label_fs)        
    axs.tick_params(axis='both', which='major', labelsize=label_fs-3)
    axs.set_xticks(ptns)
    axs.set_xticklabels(dur2an, fontsize=label_fs-3)
    axs.set_yticks([0,.1,.2,.3,.4,.5])
    axs.set_yticklabels([0,.1,.2,.3,.4,.5], fontsize=label_fs-3)

    axs.text(-.15, 1.0, 'g', transform=axs.transAxes, 
             fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    axs.text(.3, y_ticks[2], 'ctrl', color='magenta', fontsize=label_fs)
    axs.text(.3, y_ticks[1], 'NSI', color=blue, fontsize=label_fs)
    axs.text(.3, y_ticks[0], 'LN', color=orange, fontsize=label_fs)
    # move plot position:
    ll, bb, ww, hh = axs.get_position().bounds
    axs.set_position([ll+.05,bb+.07,ww,hh])        
    
    if fig_save:
        if measure == 'peak':
            fig.savefig(fld_output + '/ratio_stim_peak_resumebar_durs_delay%d'%delay+'.png')
        elif measure == 'avg':    
            fig.savefig(fld_output + '/ratio_stim_avg_resumebar_durs_delay%d'%delay+'.png')
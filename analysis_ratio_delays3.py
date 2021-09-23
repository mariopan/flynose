#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:49:09 2019

@author: mp525
analysis_ratio_delays.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle        
import scipy.stats
from sklearn.metrics import mutual_info_score
import string 

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

# *****************************************************************
# STANDARD FIGURE PARAMS
fs = 20
lw = 2
plt.rc('text', usetex=True)  # laTex in the plot
# plt.rc('font', family='serif')

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
cyan    = 'xkcd:cyan'
pink    = 'xkcd:pink'
cmap = 'inferno'
alphabet = string.ascii_lowercase
# *****************************************************************



    
# *****************************************************************
fig_save        = 1

delay_fig       = 1             # Fig.ResumeDelayedStimuli

id_peak2plot    = 3             # ONLY for the delays analysis
measure         = 'peak'         # 'avg' # 'peak' # 
corr_mi         = 'corr'        # 'corr' # 'mi'

# analysis for zero delay:
ratio_fig       = 1*(1-delay_fig)   # Response ratios versus concentration ratio 
                                    #   of the two odorants for different 
                                    #   overall concentrations (colours, see 
                                    #   legend in f). Black dashed diagonal is 
                                    #   for peak PN ratios equal to odorant 
                                    #   concentration ratio. Error bars 
                                    #   represent the semi inter-quartile 
                                    #   range calculated over 10 trials.

resumen_chess_lin   = 1*(1-delay_fig)   # Coding error as distance from linear expectation
                                        # of PN for different values of stim duration 
                                        # and concentration values.

resumen_chess_mi   = 1*(1-delay_fig)    # MI (or cross correlation) between 
                                        # concentration ratio and PN ratio

pn_activity     = 0*(1-delay_fig)   # PN activity for weak and strong input

orn_activity    = 0*(1-delay_fig)   # ORN activity for weak and strong input

resumen_bar     = 0*(1-delay_fig)   # Fig.ResumeEncodeRatioBar

# select a subsample of the params to analyse
# nsi_ln_par   = [[0,0],[.4, 0],[0, .4],]

nsi_ln_par   = [[0,0],[0, .6],[.6, .0],[.6, .6], ]

model_colours = [pink, orange, cyan, green]

model_names = ['control', 'LN inhib', 'NSI', 'mix']
model_ls = ['-.', '-', '--', '-.']
    
n_inh2run    = len(nsi_ln_par)


if delay_fig:
    print('Delays analysis, rate measured on '+measure)
    # fld_analysis    = 'NSI_analysis/analysis_delays_tauln250_tausdf41/'
    fld_analysis    = 'NSI_analysis/delays_tauln250_tausdf41/'
    # fld_analysis    = 'NSI_analysis/delays_.4.2/'   # analysis_ratio_tauln25              
    fld_analysis    = 'NSI_analysis/delays_mix/'   # analysis_ratio_tauln25              
    fld_analysis    = 'NSI_analysis/delays_mix_250/'   # analysis_ratio_tauln25              
    fld_output      = fld_analysis
    name_analysis   = 'delays'
    
else:
    print('Ratio analysis, rate measured on '+measure+', ' + corr_mi)
    # fld_analysis    = 'NSI_analysis/ratio_tauln250/'   # analysis_ratio_tauln25              
    # fld_analysis    = 'NSI_analysis/ratio_.4.2/'   # analysis_ratio_tauln25              
    fld_analysis    = 'NSI_analysis/ratio_mix/'   # analysis_ratio_tauln25              
    fld_output      = fld_analysis    
    name_analysis   = 'ratio'

# LOAD EXPERIMENT PARAMETERS
batch_params    = pickle.load(open(fld_analysis+name_analysis+'_batch_params.pickle', "rb" ))
[n_loops, conc_ratios, concs2an, _, dur2an, delays2an,] = batch_params

n_loops = batch_params['n_loops']
conc_ratios = batch_params['conc_ratios']
concs2an = batch_params['concs2an']
# nsi_ln_par = batch_params['nsi_ln_par']
dur2an = batch_params['dur2an']
delays2an = batch_params['delays2an']


params_al_orn   = pickle.load(open(fld_analysis +name_analysis+'_params_al_orn.ini',  "rb" ))
stim_params     = params_al_orn['stim_params']
tau_ln          = params_al_orn['pn_ln_params']['tau_ln']

if delay_fig==0:
    delays2an=[0,]

  

n_durs          = np.size(dur2an)
n_delays        = np.size(delays2an)
n_ratios        = np.size(conc_ratios)
n_concs         = np.size(concs2an)
n_bins_mi       = int(np.sqrt(n_ratios/5))  


# Instantiate output variables for delay figure
if delay_fig:
    pn_ratio_mean = np.ones((n_delays,n_durs, n_inh2run))
    # ratio_peak = np.ones((n_delays,n_durs, n_inh2run))
    pn_ratio_err = np.ones((n_delays,n_durs, n_inh2run))
    # ratio_peak_err = np.ones((n_delays,n_durs, n_inh2run))
else:
    ratio_lin = np.ones((n_ratios, n_concs, n_durs, n_inh2run))
    ratio_lin_err = np.ones((n_ratios, n_concs, n_durs, n_inh2run))
    ratio1 = np.ones((n_ratios, n_concs, n_durs, n_inh2run))
    ratio1_err = np.ones((n_ratios, n_concs, n_durs, n_inh2run))
    ratio2 = np.ones((n_ratios, n_concs, n_durs, n_inh2run))
    ratio2_err = np.ones((n_ratios, n_concs, n_durs, n_inh2run))
    ratio_mi = np.ones((n_concs,n_durs, n_inh2run))
    if pn_activity:
        pn_s = np.zeros((n_ratios, n_concs, n_durs, n_inh2run))
        pn_w = np.zeros((n_ratios, n_concs, n_durs, n_inh2run))
    if orn_activity:
        orn_s = np.zeros((n_ratios, n_concs, n_durs, n_inh2run))
        orn_w = np.zeros((n_ratios, n_concs, n_durs, n_inh2run))
    
# LOAD DATA AND CALCULATE RATIOS
for delay_id, delay in enumerate(delays2an):
    for [id_inh, [nsi_str, alpha_ln]] in enumerate(nsi_ln_par):
        data_name  = name_analysis + \
                '_stim_' + stim_params['stim_type'] +\
                '_nsi_%.1f'%(nsi_str) +\
                '_ln_%.1f'%(alpha_ln) +\
                '_delay2an_%d'%(delay) +\
                '.pickle'        

        all_data    = pickle.load(open(fld_analysis+data_name,  "rb" ) )
        [params_al_orn, output2an, ] = all_data
        
        orn_avg_w = output2an['avg_ornw']
        orn_avg_s = output2an['avg_orns']
        pn_avg_w = output2an['avg_pnw']
        pn_avg_s = output2an['avg_pns']
        
        orn_peak_w = output2an['peak_ornw']
        orn_peak_s = output2an['peak_orns']
        pn_peak_w = output2an['peak_pnw']  # minimum value 10Hz
        pn_peak_s = output2an['peak_pns'] 
            
        #(n_ratios, n_concs,n_durs, n_loops)
        if measure == 'avg':
            orn_ratio_tmp = np.ma.masked_invalid(orn_avg_s/orn_avg_w)
            pn_ratio_tmp = np.ma.masked_invalid(pn_avg_s/pn_avg_w)
        
        elif measure == 'peak':
            orn_ratio_tmp = np.ma.masked_invalid(orn_peak_s/orn_peak_w)
            pn_ratio_tmp = np.ma.masked_invalid(pn_peak_s/pn_peak_w)
    
        if delay_fig:   #(n_ratios, n_concs,n_durs, n_loops)
            # average over the run with identical params
            pn_ratio_mean[delay_id, :, id_inh] = np.median(
                    pn_ratio_tmp[0,id_peak2plot ,:,:], axis=1)
            
            pn_ratio_err[delay_id, :, id_inh] = np.diff(
                np.percentile(pn_ratio_tmp[0,id_peak2plot,:,:], [25,50]))
                
        else:
            # average over the run with identical params
            ratio1[:,:,:,id_inh] = np.median(orn_ratio_tmp, axis=3)
            
            ratio1_err[:,:,:,id_inh] = scipy.stats.iqr(orn_ratio_tmp, axis=3)/2 #np.squeeze(np.diff(np.percentile(orn_ratio_tmp, [25,50],axis=3), axis=0))
            
            # average over the run with identical params
            ratio2[:,:,:,id_inh] = np.median(pn_ratio_tmp, axis=3)
            ratio2_err[:,:,:,id_inh] = scipy.stats.iqr(pn_ratio_tmp, axis=3)/2     #np.squeeze(np.diff(np.percentile(pn_ratio_tmp_noin, [25,50],axis=3), axis=0))#np.std(pn_ratio_tmp_noin, axis=3)
            
            # code error measure
            for iic in range(n_concs):
                for iid in range(n_durs):
                    if corr_mi ==  'corr':
                        ratio_mi[iic,iid, id_inh] = np.corrcoef(conc_ratios, 
                            np.median(pn_ratio_tmp[:,iic,iid,:], axis=1))[0,1]

                    elif corr_mi ==  'mi':
                        ratio_mi[iic,iid, id_inh] = calc_MI(conc_ratios, 
                            np.median(pn_ratio_tmp[:,iic,iid,:], axis=1),
                            n_bins_mi)
    
            delta_tmp = ((conc_ratios - pn_ratio_tmp.T)/
                        (pn_ratio_tmp.T + conc_ratios))**2
                
            # average and std over runs with identical params
            # (n_ratios, n_concs, n_durs, n_inh2run)
            ratio_lin[:,:,:,id_inh] = np.mean(delta_tmp, axis=0).T
            ratio_lin_err[:,:,:,id_inh] = np.ma.std(delta_tmp, axis=0).T
            
            # collect and average over different runs ORN and PN activity
            if pn_activity:
                if measure == 'avg':
                    pn_s[:,:,:,id_inh] = np.median(pn_avg_s, axis= 3)
                    pn_w[:,:,:,id_inh] = np.median(pn_avg_w, axis= 3)
                if measure == 'peak':
                    pn_s[:,:,:,id_inh] = np.median(pn_peak_s, axis= 3)
                    pn_w[:,:,:,id_inh] = np.median(pn_peak_w, axis= 3)
            if orn_activity:
                if measure == 'avg':
                    orn_s[:,:,:,id_inh] = np.median(orn_avg_s, axis= 3)
                    orn_w[:,:,:,id_inh] = np.median(orn_avg_w, axis= 3)
                if measure == 'peak':
                    orn_s[:,:,:,id_inh] = np.median(orn_peak_s, axis= 3)
                    orn_w[:,:,:,id_inh] = np.median(orn_peak_w, axis= 3)
     
            


#%% *********************************************************
## FIGURE ratio 

if ratio_fig: 
    lw = 3
    rs = 2
    cs = n_inh2run
    colors = plt.cm.winter_r
    clr_fct = 30        # color factor
    
    # panels_id   = ['a', 'b', 'c', 'd', 'e', 'f', ]
    dur2plot = dur2an
    for dur_id, duration in enumerate(dur2plot):
        fig, axs = plt.subplots(rs, cs, figsize=(10,7), ) 
        axs[0,0].set_title(['dur: %d ms'%duration])
        
        for [id_inh, [nsi_str, alpha_ln]] in enumerate(nsi_ln_par):
            dx = .1
            
            for conc_id, conc_v in enumerate(concs2an): 
                axs[0, id_inh].errorbar(conc_ratios+dx*conc_id, 
                   ratio1[:,conc_id, dur_id, id_inh],
                   yerr= ratio1_err[:,conc_id, dur_id, id_inh], marker='o', 
                   color=colors(conc_id*clr_fct) )
                
                axs[1, id_inh].errorbar(conc_ratios+dx*conc_id, 
                    ratio2[:,conc_id, dur_id, id_inh],
                    yerr= ratio2_err[:,conc_id, dur_id, id_inh], marker='o', 
                    label=r''+'%.5f'%(conc_v), color=colors(conc_id*clr_fct) )
                
        
        # FIGURE settings
        axs[0, 0].set_ylabel(r'$R^{ORN} $ (unitless)', fontsize=label_fs)
        axs[1, 0].set_ylabel(r'$R^{PN} $ (unitless)', fontsize=label_fs)      

        axs[1, 1].set_xlabel('Concentration ratio (unitless)', x=1.3, fontsize=label_fs)
        for cc in range(cs):
                
            axs[0, cc].plot(conc_ratios, conc_ratios, '--', lw=lw, color='black', label='expec.')
            axs[1, cc].plot(conc_ratios, conc_ratios, '--', lw=lw, color='black', label='expec.')
            for rr in range(rs):
                axs[rr, cc].tick_params(axis='both', which='major', labelsize=label_fs-3)
                
                # axs[rr,cc].set_xticklabels('')
                # axs[rr,cc].set_yticklabels('')
                axs[rr,cc].spines['right'].set_color('none')
                axs[rr,cc].spines['top'].set_color('none')
                
                axs[rr,cc].set_ylim((0.8, 100.5))
                axs[rr,cc].set_xlim((0.8, 20.5))
                
                axs[rr,cc].set_yticks([1, 10, ])
                axs[rr,cc].set_xticks([1, 10, 20])
                
                axs[rr,cc].set_yscale("log")
                axs[rr,cc].set_xscale("log")
                
                
                # change plot position:
                ll, bb, ww, hh = axs[rr,cc].get_position().bounds
                axs[rr,cc].set_position([ll+cc*.03, bb+(2-rr)*.03,ww,hh])       
                
        for cc in range(cs):
            axs[0, cc].set_title(model_names[cc], fontsize=label_fs)        

        for cc in range(1, cs):
            for rr in range(rs):
                axs[rr,cc].text(-.2, 1.2, alphabet[cc*rs+rr], transform=axs[rr,cc].transAxes,
                               fontsize=panel_fs, color=black, weight='bold', va='top', ha='right')
        cc = 0
        for rr in range(rs):
            axs[rr,cc].text(-.35, 1.2, alphabet[cc*rs+rr], transform=axs[rr,cc].transAxes,
                           fontsize=panel_fs, color=black, weight='bold', va='top', ha='right')
        axs[1,1].legend(frameon=False)
        
        if fig_save:
            fig_name = 'ratio_stim_'+ measure+'_dur%d'%duration+'_delay%d'%delay
            fig_name =  'MIX_' + fig_name 
            fig.savefig(fld_output+  fig_name +'.png')
                   
        plt.show()
        
        
#%% ************************************************************************
# RESUME CHESS FIGURE: 
    # analysis of the relative distance between concentration ratio and PN ratio
if resumen_chess_lin:
    
    rs          = 1
    cs          = n_inh2run
    vmin        =  0
    
    vmax        =  0.7 #0.4
#    colorbar params
    frac = .1 #.04
    pad = .04
    
    fig, axs = plt.subplots(rs, cs, figsize=(12, 4.3),) 

    ptns = np.arange(5)
    
    
    for id_inh in range(n_inh2run):
        # average over conc.ratio and concentrations
        err_code = np.mean(ratio_lin[:,:,:,id_inh], axis=(0))
        
        im0 = axs[id_inh].imshow(err_code, cmap=cmap, vmin=vmin, vmax=vmax)
        
    
    # FIGURE SETTINGS
    for id_inh in range(n_inh2run):
        axs[id_inh].set_title(model_names[id_inh], fontsize=title_fs)
    
    dur2plot = np.round(dur2an[::1])
    conc2plot = concs2an[::1]
    for c_id in range(cs):
        axs[c_id].set_xticks(range(n_durs))
        axs[c_id].set_xticklabels(dur2plot, fontsize= ticks_fs)
        axs[c_id].set_yticks(range(n_concs))
        axs[c_id].set_yticklabels('', fontsize= ticks_fs)      
        
        axs[c_id].set_xlabel('duration (ms)', fontsize= label_fs)
    
    axs[0].set_yticklabels([0.00052, 0.00068, 0.00084, 0.001, 0.005, 0.01], fontsize= ticks_fs)  
    # axs[0].set_yticklabels(conc2plot, fontsize= ticks_fs, )  
    axs[0].set_ylabel('input (a.u.)', fontsize= label_fs)    
    # axs[0].ticklabel_format(style='sci',)

    # move plot position:
    ll, bb, ww, hh = axs[0].get_position().bounds
    ww_new = ww*1.0#1.15
    hh_new = hh*1.0#1.15
    bb_new = bb + 0.05#5
    ll_new = ll
    e_sx = 0.05
    axs[0].set_position([ll-.05+e_sx, bb_new, ww_new, hh_new])    
    
    ll, bb, ww, hh = axs[1].get_position().bounds
    axs[1].set_position([ll-.045+e_sx, bb_new, ww_new, hh_new])        
    
    ll, bb, ww, hh = axs[2].get_position().bounds
    axs[2].set_position([ll-.04+e_sx, bb_new, ww_new, hh_new])     
    
    axs[0].text(-.5, 1.2, 'i', transform=axs[0].transAxes, 
             fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    

    cbar = fig.colorbar(im0, ax=axs[3], fraction=.05,pad=pad, 
            orientation='vertical', ticks=[0, vmax/2, vmax])
    cbar.set_label('code error', fontsize=label_fs)
    ticklabs = cbar.ax.get_yticklabels()
    
    #%% to run separately from the other, or it won't plot the ticklabels
    cbar.ax.set_yticklabels(ticklabs, fontsize=ticks_fs)
    
    # adjust bar size and position
    ll, bb, ww, hh = cbar.ax.get_position().bounds
    cbar.ax.set_position([ll-.015+e_sx, bb+.085, ww, hh-.12])
   
    #adjust last chess board size and position
    dwdh =1.0125
    ll_a, bb, ww, hh = axs[3].get_position().bounds
    axs[3].set_position([ll_a -.045+e_sx, bb_new, ww_new*dwdh, hh_new*dwdh])
    
    if fig_save:
        fig_name = 'ratio_stim_'+measure+'_resumechess_delay%d'%delay
        
        fig_name =  'MIX_' + fig_name 
        fig.savefig(fld_output+  fig_name +'.png')
    plt.show()    
    
    
#%% # RESUME CHESS FIGURE with Mutual Information ##########################
# analysis of the MI (or cross correlation) between concentration ratio and PN ratio
if resumen_chess_mi:

    # average over conc.ratio and concentrations

    rs = 1
    cs = n_inh2run
    vmin = .2 # 0
    
    vmax =  1. # 0.4
#    colorbar params
    frac = .1 #.04
    pad = .04
    fig, axs = plt.subplots(rs, cs, figsize=(12, 4.3),) 

    ptns = np.arange(5)
    
    
    for id_inh in range(n_inh2run):
        # average over conc.ratio and concentrations
        err_code = ratio_mi[:,:,id_inh]
        
        im0 = axs[id_inh].imshow(err_code, cmap=cmap, vmin=vmin, vmax=vmax)
        
    
    # FIGURE SETTINGS
    for id_inh in range(n_inh2run):
        axs[id_inh].set_title(model_names[id_inh], fontsize=title_fs)
        
        
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


    # move plot position:
    ll, bb, ww, hh = axs[0].get_position().bounds
    ww_new = ww*1.0#1.15
    hh_new = hh*1.0#1.15
    bb_new = bb + 0.05#5
    ll_new = ll
    e_sx = 0.05
    axs[0].set_position([ll-.05+e_sx, bb_new, ww_new, hh_new])    
    
    ll, bb, ww, hh = axs[1].get_position().bounds
    axs[1].set_position([ll-.045+e_sx, bb_new, ww_new, hh_new])        
    
    ll, bb, ww, hh = axs[2].get_position().bounds
    axs[2].set_position([ll-.04+e_sx, bb_new, ww_new, hh_new])     
    
    cbar = fig.colorbar(im0, ax=axs[-1], fraction=.05,pad=pad, 
            orientation='vertical', ticks=[0, vmax/2, vmax])
    if corr_mi == 'corr':
        cbar.set_label('correlation', fontsize=label_fs)
    elif corr_mi == 'mi':
        cbar.set_label('mutual inform.', fontsize=label_fs)
            
    ticklabs = cbar.ax.get_yticklabels()


#%% to run separately from the other, or it won't plot the ticklabels
if resumen_chess_mi:
    cbar.ax.set_yticklabels(ticklabs, fontsize=ticks_fs)
    
    # adjust bar size and position
    ll, bb, ww, hh = cbar.ax.get_position().bounds
    cbar.ax.set_position([ll-.015+e_sx, bb+.085, ww, hh-.12])
   
    #adjust 3rd chess board size and position
    dwdh =1.0125
    ll_a, bb, ww, hh = axs[-1].get_position().bounds
    axs[-1].set_position([ll_a -.045+e_sx, bb_new, ww_new*dwdh, hh_new*dwdh])

    if fig_save:
        fig_name = 'ratio_stim_'+measure+'_resumechess_' + corr_mi + '_durs_delay%d'%delay
        fig_name =  'MIX_' + fig_name 
        fig.savefig(fld_output+  fig_name +'.png')
        
            
    fig.show()
    
    
#%%***********************************************************
# FIGURE PN activity in a chess plot
# Figure of the average activity for weak and strong input
if pn_activity:

    ratio2plot = np.round(conc_ratios[::3])
    conc2plot = concs2an[::1]
    
    rs = n_inh2run
    cs = 2
    for dur_id, duration in enumerate(dur2an):

        # pn_peak_s_noin: (n_ratios, n_concs,n_durs, n_loops)
        fig, axs = plt.subplots(rs, cs, figsize=(9, 9), ) 
        
        # PLOT 
        for id_inh in range(n_inh2run):
            im0 = axs[id_inh, 0].imshow(pn_s[:,:,dur_id,id_inh].T, 
                                        cmap=cmap, aspect='auto', vmin=0, vmax=300)
            fig.colorbar(im0, ax=axs[id_inh, 0])
        
            im1 = axs[id_inh, 1].imshow(pn_w[:,:,dur_id,id_inh].T, cmap=cmap, aspect='auto', vmin=0, vmax=170)
            fig.colorbar(im1, ax=axs[id_inh,1])
        
        
        # SETTINGS
        axs[0,0].set_title('PN strong')
        axs[0,1].set_title('PN weak')

        for id_r in range(rs):
            axs[id_r,0].set_ylabel('input (a.u.)', fontsize= label_fs)
        
            axs[0,1].text(62.5, 1.5, model_names[id_r], fontsize=label_fs)
            # axs[1,1].text(62.5, 1.5, 'LN', fontsize=label_fs)
            # axs[2,1].text(62.5, 1.5, 'NSI', fontsize=label_fs)
        
        
        for c_id in range(cs):
            axs[2, c_id].set_xlabel('ratio (unitless)', fontsize= label_fs)
            
            for r_id in range(rs):
                # axs[r_id,c_id].set_xticks(np.linspace(1, len(conc_ratios), 15))
                axs[r_id,c_id].set_xticks(range(len(ratio2plot)))
                axs[r_id,c_id].set_xticklabels(ratio2plot)
                axs[r_id,c_id].set_yticks(range(len(conc2plot)))
                axs[r_id,c_id].set_yticklabels(conc2plot)
            
        for c_id in range(cs):
            for r_id in range(rs):
                ll, bb, ww, hh = axs[r_id, c_id].get_position().bounds
                axs[r_id, c_id].set_position([ll, bb, ww, hh])
                
            
    
                
    
        if fig_save:
            fig_name = 'PN_'+measure+'_delays0_dur%d'%duration
            fig_name =  'MIX_' + fig_name 
            fig.savefig(fld_output+  fig_name +'.png')
            

    plt.show()      
            
    
#%%***********************************************************
# FIGURE ORN activity in a chess plot
# Figure of the average activity for weak and strong input
if orn_activity:

    ratio2plot = np.round(conc_ratios[::3])
    conc2plot = concs2an[::1]
    
    rs = n_inh2run
    cs = 2
    for dur_id, duration in enumerate(dur2an):

        fig, axs = plt.subplots(rs, cs, figsize=(9, 9), ) 
        
        # PLOT 
        for id_inh in range(n_inh2run):
            im0 = axs[id_inh, 0].imshow(orn_s[:,:,dur_id,id_inh].T, 
                                        cmap=cmap, aspect='auto', vmin=0, vmax=300)
            fig.colorbar(im0, ax=axs[id_inh, 0])
        
            im1 = axs[id_inh, 1].imshow(orn_w[:,:,dur_id,id_inh].T, cmap=cmap, aspect='auto', vmin=0, vmax=170)
            fig.colorbar(im1, ax=axs[id_inh,1])
        
        
        
        # SETTINGS
        axs[0,0].set_title('ORN strong')
        axs[0,1].set_title('ORN weak')

        for id_r in range(rs):
            axs[id_r,0].set_ylabel('input (a.u.)', fontsize= label_fs)
        
            axs[id_r,1].text(62.5, 1.5, model_names[id_r], fontsize=label_fs)
            
        # axs[1,1].text(62.5, 1.5, 'LN', fontsize=label_fs)
        # axs[2,1].text(62.5, 1.5, 'NSI', fontsize=label_fs)
        
        
        for c_id in range(cs):
            axs[2, c_id].set_xlabel('ratio (unitless)', fontsize= label_fs)
            
            for r_id in range(rs):
                # axs[r_id,c_id].set_xticks(np.linspace(1, len(conc_ratios), 15))
                axs[r_id,c_id].set_xticks(range(len(ratio2plot)))
                axs[r_id,c_id].set_xticklabels(ratio2plot)
                axs[r_id,c_id].set_yticks(range(len(conc2plot)))
                axs[r_id,c_id].set_yticklabels(conc2plot)
            
        for c_id in range(cs):
            for r_id in range(rs):
                ll, bb, ww, hh = axs[r_id, c_id].get_position().bounds
                axs[r_id, c_id].set_position([ll, bb, ww, hh])
                
    
        if fig_save:
            fig_name = 'ORN_'+measure+'_delays0_dur%d'%duration
            fig_name =  'MIX_' + fig_name 
            fig.savefig(fld_output+  fig_name +'.png')
            

    plt.show()      


#%% ************************************************************************
# RESUME BAR FIGURE
if resumen_bar:
    
    width = 0.15
    y_ticks = [0.27, .32,.37, .42]
    rs = 1
    cs = 1
    fig, axs = plt.subplots(rs, cs, figsize=(9,4), ) 

    ptns = np.arange(5)

    for id_inh in range(n_inh2run):
        # average over conc.ratio and concentrations
        err_code_lin = np.mean(ratio_lin[:,:,:,id_inh], axis=(0,1))
        err_code_lin_std = np.std(ratio_lin[:,:,:,id_inh], axis=(0,1))

        axs.bar(ptns+id_inh*width, err_code_lin, width=width, color= model_colours[id_inh], 
                yerr = err_code_lin_std/np.sqrt(n_ratios*n_concs), 
                label=model_names[id_inh], )
    
    # FIGURE SETTINGS
    axs.spines['right'].set_color('none')   
    axs.spines['top'].set_color('none')                

    axs.set_ylabel('coding error (a.u.)', fontsize=label_fs)
    axs.set_xlabel('stimulus duration (ms)', fontsize=label_fs)        
    axs.tick_params(axis='both', which='major', labelsize=label_fs-3)
    axs.set_xticks(ptns)
    axs.set_xticklabels(dur2an, fontsize=label_fs-3)
    axs.set_yticks([0,.1,.2,.3,.4,.5])
    axs.set_yticklabels([0,.1,.2,.3,.4,.5], fontsize=label_fs-3)

    axs.text(-.08, 1.0, 'i', transform=axs.transAxes, 
             fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    
    for id_inh in range(n_inh2run):
        axs.text(4.6, y_ticks[id_inh], model_names[id_inh], 
                 color=model_colours[id_inh], fontsize=label_fs)
        
    # move plot position:
    ll, bb, ww, hh = axs.get_position().bounds
    axs.set_position([ll+.0,bb+.07,ww,hh])        
    
    if fig_save:
        fig_name = 'ratio_stim_'+measure+'_resumebar_durs_delay%d'%delay
        fig_name =  'MIX_' + fig_name 
        fig.savefig(fld_output+  fig_name +'.png')
            
    plt.show()

#%% FIGURE ResumeDelayedStimuli ############################################
if delay_fig:
    
    y_ticks = np.linspace(0, 2, 5)
    fig, axs = plt.subplots(1, n_durs, figsize=(17, 6.3), sharey=True) 
    for dur_id in range(n_durs):
        duration = dur2an[dur_id]
        
        for id_inh in range(n_inh2run):
            axs[dur_id].errorbar(delays2an, pn_ratio_mean[:, dur_id, id_inh], 
                yerr=pn_ratio_err[:, dur_id, id_inh], color=model_colours[id_inh], 
                lw = lw, label= model_names[id_inh])
        
        # FIGURE SETTINGS
        axs[dur_id].set_title(' %d ms'%(duration), fontsize=title_fs)
        
        axs[dur_id].spines['right'].set_color('none')   
        axs[dur_id].spines['top'].set_color('none')     
        
        if dur_id>0:
            axs[dur_id].set_yticklabels('', fontsize=label_fs-5)
        axs[dur_id].set_xticks([0, 250, 500])
        axs[dur_id].set_xticklabels(['0','250','500'], fontsize=label_fs-5)

        axs[dur_id].set_ylim((.3, 1.7))
        
        # original plot position:
        ll, bb, ww, hh = axs[dur_id].get_position().bounds
        axs[dur_id].set_position([ll-.06+.025*dur_id, bb+.1, ww+.025, hh-.15]) 
        
    axs[0].set_yticks([0,.5,1.0,1.5])
    axs[0].set_yticklabels([0,.5,1.0,1.5], fontsize=label_fs-5)
    axs[0].set_ylim((.3, 1.7))
    
    
    conc2plot = np.squeeze(concs2an[id_peak2plot]) #  conc_1_r[0,id_peak2plot,0])
    axs[0].set_ylabel(r'$R^{PN} $ (unitless)', fontsize=label_fs)
    axs[2].set_xlabel('Delay (ms)', fontsize=fs)
    
    if tau_ln == 25:
        panel_id = 'b'
        axs[1].legend(fontsize=fs-2, frameon=False)
    elif tau_ln == 250:
        panel_id = 'a'
    
    axs[0].text(-.2, 1.2, panel_id, transform=axs[0].transAxes,
           fontsize=panel_fs, color=black, weight='bold', va='top', ha='right')
    plt.show()
    
    
    if fig_save:
        fig_name = 'delays_'+measure+'_delays0-500_dur20-200_conc%.2g'%conc2plot +'_tauln%d'%tau_ln
        fig_name =  'MIX_' + fig_name 
        fig.savefig(fld_output+  fig_name +'.png')
        
#%% FIGURE ResumeDelayedStimuli, one figure one duration ##################################

if delay_fig:
    y_ticks = np.linspace(0, 2, 5)
    for dur_id in range(n_durs):
        fig, axs = plt.subplots(1, 1, figsize=(9, 5), sharey=True) # figsize=(17, 6.3), sharey=True) 
        duration = dur2an[dur_id]
        
        
        for id_inh in range(n_inh2run):
            axs.errorbar(delays2an, pn_ratio_mean[:, dur_id, id_inh], 
                yerr=pn_ratio_err[:, dur_id, id_inh], color=model_colours[id_inh], 
                lw = lw, label= model_names[id_inh])
            
            
        
        
        # FIGURE SETTINGS
        axs.set_title(' %d ms'%(duration), fontsize=title_fs)
        
        axs.spines['right'].set_color('none')   
        axs.spines['top'].set_color('none')     
        
        # if dur_id>0:
        axs.set_yticklabels('', fontsize=label_fs-5)
        axs.set_xticks([0, 250, 500])
        axs.set_xticklabels(['0','250','500'], fontsize=label_fs-5)

        axs.set_ylim((.3, 1.7))
        
        # original plot position:
        ll, bb, ww, hh = axs.get_position().bounds
        axs.set_position([ll, bb+.1, ww+.025, hh-.15]) 
        
        axs.set_yticks([0,.5,1.0,1.5])
        axs.set_yticklabels([0,.5,1.0,1.5], fontsize=label_fs-5)
        axs.set_ylim((.55, 1.7))
    
    
        conc2plot = np.squeeze(concs2an[id_peak2plot]) #  conc_1_r[0,id_peak2plot,0])
        axs.set_ylabel(r'$R^{PN} $ (unitless)', fontsize=label_fs)
        axs.set_xlabel('Delay (ms)', fontsize=fs)
        axs.legend(fontsize=fs-3, frameon=False)
    
        axs.text(-.08, 1.2, 'b', transform=axs.transAxes,
               fontsize=panel_fs, color=black, weight='bold', va='top', ha='right')
    
    
        if fig_save:
            fig_name = 'delays_'+measure+'_delays0-500_dur%d'%duration+'_conc%.2g'%conc2plot +'_tauln%d'%tau_ln
            fig_name =  'MIX_' + fig_name 
            fig.savefig(fld_output+ fig_name+'.png')
            
    plt.show()            
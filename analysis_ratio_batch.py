#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:49:09 2019

@author: mp525
analysis_loops_script_corr.py
"""


import numpy as np
#import system_ORNPNLN_corr
import matplotlib.pyplot as plt

import pickle        
from os import path
from os import mkdir
from shutil import copyfile

plt.ion()
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
black   = 'xkcd:black'
blue    = 'xkcd:blue'
red     = 'xkcd:red'
green   = 'xkcd:green'
purple  = 'xkcd:purple'
orange  = 'xkcd:orange'
magenta = 'xkcd:magenta'
# *****************************************************************

fld_py_h    = '/home/m/mp/mp525/MEGA/WORK/Code/PYTHON/'
fld_py_w    = '/home/mario/MEGA/WORK/Code/PYTHON/'

if path.isdir(fld_py_h):
    fld_py = fld_py_h
elif path.isdir(fld_py_w):
    fld_py = fld_py_w
else:
    print('ERROR: no python folder!')
    
fld_home = fld_py+'NSI_analysis/analysis_ratio/'

##*****************************************************
## # Fig.ResumeDelayedStimuli: ratio analysis with delayes
#fld_analysis    = fld_home+'ratio1_delay_analysis/'
#n_loops         = 70
#durs            = [20, 50, 100,200,]
#id_peak2plot    = 2
#delays2an       = [0,10, 20, 50, 100, 200,500,] 
#peak_fig        = 0
#avg_fig         = 0
#delay_fig       = 1 # Fig.ResumeDelayedStimuli
#n_ratios        = 1
#resumen_fig     = 0
#fig_save        = 1 
##*****************************************************


##*****************************************************
## # Fig.ResumeEncodeRatio: ratio analysis with delayes
#fld_analysis = fld_home+'ratio_short_stimuli/'
#n_loops     = 10
#fld_analysis = fld_home+'ratio_short_stimuli2/'
#n_ratios    = 5
#fld_analysis = fld_home+'ratio_short_stimuli3/'
fld_analysis = fld_home+'ratio_short_stimuli4/' # Fig.ResumeEncodeRatio
#fld_analysis = fld_home+'ratio_short_stimuli5/'
fld_analysis = fld_home+'ratio_short_stimuli6/'
#fld_analysis = fld_home+'ratio_batch_ln13.3/'

n_ratios    = 10

n_loops     = 50

durs        =  [10, 20, 50, 100, 200,]#200,]
delays2an   = [0,] 
peak_fig    = 0 # Fig.RatioPeak
resumen_fig = 0 # Fig.ResumeEncodeRatio
resumen_bar = 1 # Fig.ResumeEncodeRatioBar
avg_fig     = 0
delay_fig   = 0
fig_save        = 1 

#*****************************************************

n_durs      = np.size(durs)
n_delays    = np.size(delays2an)
n_concs     = 4


if delay_fig:
    id_peak2plot = 3

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

if peak_fig:
    ratio2_peak_noin = np.zeros((n_ratios, n_concs,n_durs))
    ratio2_peak_nsi = np.zeros((n_ratios, n_concs,n_durs))
    ratio2_peak_ln = np.zeros((n_ratios, n_concs,n_durs))
    ratio2_peak_err_noin = np.zeros((n_ratios, n_concs, n_durs))
    ratio2_peak_err_nsi = np.zeros((n_ratios, n_concs, n_durs))
    ratio2_peak_err_ln = np.zeros((n_ratios, n_concs,n_durs))
    
if resumen_bar | resumen_fig | peak_fig:
    ratio2dist_peak_noin = np.zeros((n_ratios, n_concs,n_durs))
    ratio2dist_peak_nsi = np.zeros((n_ratios, n_concs,n_durs))
    ratio2dist_peak_ln = np.zeros((n_ratios, n_concs,n_durs))
    ratio2dist_peak_err_noin = np.zeros((n_ratios, n_concs, n_durs))
    ratio2dist_peak_err_nsi = np.zeros((n_ratios, n_concs, n_durs))
    ratio2dist_peak_err_ln = np.zeros((n_ratios, n_concs,n_durs))   
    
for id_dur in range(n_durs):
    duration = durs[id_dur]
    dk = -1
    for delay in delays2an:
        dk += 1
        fld_analysis_tmp = fld_analysis+'ratio_stim_dur_%d'%duration+'_delay_%d'%delay        
        
        #pickle.dump([params2an, peak_1, ratio_peak, 
        #                                 avg_orn1, avg_orn2, avg_pn1, avg_pn2,
        #                                 peak_orn1, peak_orn2, peak_pn1, peak_pn2]    
        data_name = 'NoIn_ratio_an'
        all_data    = pickle.load( open( fld_analysis_tmp+'/' +data_name+'.pickle',  "rb" ) )
        stim_params = all_data[0]
        conc_1_r    = all_data[1]
        conc_ratio  = all_data[2]
        
        n_peak_ratio, n_peak, n_loops = np.shape(all_data[3])
        if n_peak_ratio!= n_ratios:
            print('number of ratio not as planned')
        if n_peak!= n_concs:
            print('number of concentrations not as planned')
        print('effective n_loops:%d'%n_loops)
        orn_avg1_noin   = all_data[3]
        orn_avg2_noin   = all_data[4]
        pn_avg1_noin    = all_data[5]
        pn_avg2_noin    = all_data[6]
        
        orn_peak1_noin   = all_data[7]
        orn_peak2_noin   = all_data[8]
        pn_peak1_noin    = all_data[9]
        pn_peak2_noin    = all_data[10]
        
        data_name = 'LN_ratio_an'
        all_data    = pickle.load( open( fld_analysis_tmp+'/' +data_name+'.pickle',  "rb" ) )
        stim_params = all_data[0]
        print('LN inhib conditions: ')
        print(stim_params[0:2])
        
        orn_avg1_ln   = all_data[3]
        orn_avg2_ln   = all_data[4]
        pn_avg1_ln    = all_data[5]
        pn_avg2_ln    = all_data[6]
        
        orn_peak1_ln = all_data[7]
        orn_peak2_ln   = all_data[8]
        pn_peak1_ln    = all_data[9]
        pn_peak2_ln    = all_data[10]
        
        data_name = 'NSI_ratio_an'
        all_data    = pickle.load( open( fld_analysis_tmp+'/' +data_name+'.pickle',  "rb" ) )
        stim_params = all_data[0]
        print('NSI inhib conditions: ')
        print(stim_params[0:2])
        
        orn_avg1_nsi   = all_data[3]
        orn_avg2_nsi   = all_data[4]
        pn_avg1_nsi    = all_data[5]
        pn_avg2_nsi    = all_data[6]
        
        orn_peak1_nsi   = all_data[7]
        orn_peak2_nsi   = all_data[8]
        pn_peak1_nsi    = all_data[9]
        pn_peak2_nsi    = all_data[10]
        
        orn_ratio_avg_nsi   = orn_avg2_nsi/orn_avg1_nsi
        orn_ratio_avg_ln    = orn_avg2_ln/orn_avg1_ln
        orn_ratio_avg_noin  = orn_avg2_noin/orn_avg1_noin
        
        pn_ratio_avg_nsi    = np.ma.masked_invalid(pn_avg2_nsi/pn_avg1_nsi)
        pn_ratio_avg_ln     = np.ma.masked_invalid(pn_avg2_ln/pn_avg1_ln)
        pn_ratio_avg_noin   = np.ma.masked_invalid(pn_avg2_noin/pn_avg1_noin)
        
        orn_ratio_peak_nsi   = orn_peak2_nsi/orn_peak1_nsi
        orn_ratio_peak_ln    = orn_peak2_ln/orn_peak1_ln
        orn_ratio_peak_noin  = orn_peak2_noin/orn_peak1_noin
        
        pn_ratio_peak_nsi    = np.ma.masked_invalid(pn_peak2_nsi/pn_peak1_nsi)
        pn_ratio_peak_ln     = np.ma.masked_invalid(pn_peak2_ln/pn_peak1_ln)
        pn_ratio_peak_noin   = np.ma.masked_invalid(pn_peak2_noin/pn_peak1_noin)
        
        if delay_fig:
            ratio_avg_noin[dk, id_dur] = np.mean(pn_ratio_avg_noin[:,id_peak2plot ,:])
            ratio_peak_noin[dk, id_dur] = np.mean(pn_ratio_peak_noin[0,id_peak2plot,:])
            ratio_avg_ln[dk, id_dur] = np.mean(pn_ratio_avg_ln[:,id_peak2plot ,:])
            ratio_avg_nsi[dk, id_dur] =np.mean(pn_ratio_avg_nsi[:,id_peak2plot,:])
            ratio_peak_ln[dk, id_dur] = np.mean(pn_ratio_peak_ln[0,id_peak2plot,:])
            ratio_peak_nsi[dk, id_dur] =np.mean(pn_ratio_peak_nsi[0,id_peak2plot,:])
            
            ratio_avg_ln_err[dk, id_dur] = np.std(pn_ratio_avg_ln[0,id_peak2plot ,:])
            ratio_avg_nsi_err[dk, id_dur] =np.std(pn_ratio_avg_nsi[0,id_peak2plot,:])
            ratio_peak_ln_err[dk, id_dur] = np.std(pn_ratio_peak_ln[0,id_peak2plot,:])
            ratio_peak_nsi_err[dk, id_dur] =np.std(pn_ratio_peak_nsi[0,id_peak2plot,:])
            ratio_peak_noin_err[dk, id_dur] =np.std(pn_ratio_peak_noin[0,id_peak2plot,:])
            ratio_avg_noin_err[dk, id_dur] =np.std(pn_ratio_avg_noin[:,id_peak2plot,:])
            
        conc_ratio_t = np.squeeze(conc_ratio[:,0,0])
        conc_ratio_mat = np.squeeze(conc_ratio[:,0,:])
        
        if peak_fig:
              
            for pk in range(4): #4
                noin_tmp = pn_ratio_peak_noin[:,pk,:]
                ln_tmp = pn_ratio_peak_ln[:,pk,:]
                nsi_tmp = pn_ratio_peak_nsi[:,pk,:]
                
                ratio2_peak_noin[:,pk, id_dur] = np.mean(noin_tmp, axis=1)
                ratio2_peak_nsi[:,pk, id_dur] = np.mean(nsi_tmp, axis=1)
                ratio2_peak_ln[:,pk, id_dur] = np.mean(ln_tmp, axis=1)
                
                ratio2_peak_err_noin[:,pk, id_dur] = np.std(noin_tmp, axis=1)
                ratio2_peak_err_nsi[:,pk, id_dur] = np.std(nsi_tmp, axis=1)
                ratio2_peak_err_ln[:,pk, id_dur] = np.std(ln_tmp, axis=1)  
                
        if resumen_fig | resumen_bar | peak_fig:
            for pk in [1, 2,3]:#range(4): #4
                noin_tmp = (conc_ratio_mat/pn_ratio_peak_noin[:,pk,:]-1)**2
                ln_tmp = (conc_ratio_mat/pn_ratio_peak_ln[:,pk,:]-1)**2
                nsi_tmp = (conc_ratio_mat/pn_ratio_peak_nsi[:,pk,:]-1)**2
                
                ratio2dist_peak_noin[:,pk, id_dur] = np.mean(noin_tmp, axis=1)
                ratio2dist_peak_nsi[:,pk, id_dur] = np.mean(nsi_tmp, axis=1)
                ratio2dist_peak_ln[:,pk, id_dur] = np.mean(ln_tmp, axis=1)
                
                ratio2dist_peak_err_noin[:,pk, id_dur] = np.ma.std(noin_tmp, axis=1)
                ratio2dist_peak_err_nsi[:,pk, id_dur] = np.ma.std(nsi_tmp, axis=1)
                ratio2dist_peak_err_ln[:,pk, id_dur] = np.ma.std(ln_tmp, axis=1)
      

#%% **********************************************************
## FIGURE peak
## **********************************************************
#durs = [50]
if peak_fig: 
    
    panels_id   = ['a.', 'b.', 'c.', 'd.', 'e.', 'f.', 'g.', 'h.', 'i.', ]
    for id_dur, duration in enumerate(durs):
        lw = 3
        rs = 3
        cs = 3
        fig, axs = plt.subplots(rs, cs, figsize=(10,7), ) 
        
        colors = plt.cm.winter_r
        clr_fct = 30
        
        conc_1 = conc_1_r[0,:,0]
        dx = .1
        for pk in range(4): 
            axs[0,0].errorbar(conc_ratio_t+dx*pk, np.squeeze(np.mean(orn_ratio_peak_noin[:,pk,:], axis=1)),
               yerr= np.squeeze(np.std(orn_ratio_peak_noin[:,pk,:], axis=1))/np.sqrt(n_loops), marker='o', 
               label=r'conc1: '+'%.1f'%(conc_1[pk]), color=colors(pk*clr_fct) )
            axs[0,1].errorbar(conc_ratio_t+dx*pk, np.squeeze(np.mean(orn_ratio_peak_ln[:,pk,:], axis=1)),
               yerr= np.squeeze(np.std(orn_ratio_peak_ln[:,pk,:], axis=1))/np.sqrt(n_loops),  marker='o', 
               label=r'conc1: '+'%.1f'%(conc_1[pk]), color=colors(pk*clr_fct) )
            axs[0,2].errorbar(conc_ratio_t+dx*pk, np.squeeze(np.mean(orn_ratio_peak_nsi[:,pk,:], axis=1)),
               yerr= np.squeeze(np.std(orn_ratio_peak_nsi[:,pk,:], axis=1))/np.sqrt(n_loops), marker='o', 
               label=r'conc1: '+'%.1f'%(conc_1[pk]), color=colors(pk*clr_fct) )
            
            axs[1,0].errorbar(conc_ratio_t+dx*pk, ratio2_peak_noin[:,pk, id_dur],
               yerr= ratio2_peak_err_noin[:,pk, id_dur]/np.sqrt(n_loops), marker='o', 
               label=r'conc1: '+'%.1f'%(conc_1[pk]), 
               color=colors(pk*clr_fct) )
            
            axs[1,1].errorbar(conc_ratio_t+dx*pk, ratio2_peak_ln[:,pk, id_dur],
               yerr= ratio2_peak_err_ln[:,pk, id_dur]/np.sqrt(n_loops), marker='o', 
               label=r'conc1: '+'%.1f'%(conc_1[pk]), 
               color=colors(pk*clr_fct) )
            
            axs[1,2].errorbar(conc_ratio_t+dx*pk, ratio2_peak_nsi[:,pk, id_dur],
               yerr= ratio2_peak_err_nsi[:,pk, id_dur]/np.sqrt(n_loops),
                marker='o', label=r'conc1: '+'%.1f'%(conc_1[pk]), 
               color=colors(pk*clr_fct) )
            
            axs[2,0].errorbar(conc_ratio_t+dx*pk, conc_ratio_t/ratio2_peak_noin[:,pk, id_dur],
               yerr= ratio2dist_peak_err_noin[:,pk, id_dur]/np.sqrt(n_loops),
               marker='o', label=r'conc1: '+'%.1f'%(conc_1[pk]),
               color=colors(pk*clr_fct) )
            
            axs[2,1].errorbar(conc_ratio_t+dx*pk, conc_ratio_t/ratio2_peak_ln[:,pk, id_dur],
               yerr= ratio2dist_peak_err_ln[:,pk, id_dur]/np.sqrt(n_loops),
               marker='o', label=r'conc1: '+'%.1f'%(conc_1[pk]), 
               color=colors(pk*clr_fct) )
            
            axs[2,2].errorbar(conc_ratio_t+dx*pk, conc_ratio_t/ratio2_peak_nsi[:,pk, id_dur],
               yerr= ratio2dist_peak_err_nsi[:,pk, id_dur]/np.sqrt(n_loops),
                marker='o', label=r'conc1: '+'%.1f'%(conc_1[pk]), 
               color=colors(pk*clr_fct) )
        
        # FIGURE settings
        axs[0,0].set_ylabel('ratio\nORN response', fontsize=fs)
        axs[1,0].set_ylabel('ratio\nPN response', fontsize=fs)
        axs[2,0].set_ylabel('PN ratio \n error', fontsize=fs)
        

        for cc in range(cs):
            axs[2, cc].set_xlabel('conc ratio', fontsize=fs)
                
            axs[0, cc].plot(conc_ratio_t, conc_ratio_t, '--', lw=lw, color='black', label='expec.')
            axs[1, cc].plot(conc_ratio_t, conc_ratio_t, '--', lw=lw, color='black', label='expec.')
            axs[2, cc].plot(conc_ratio_t, np.ones_like(conc_ratio_t), '--', 
               lw=lw, color='black', label='expec.')
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
                    axs[rr,cc].set_yticklabels(['0','5','10', '15', '20'], fontsize=fs)
                
                if rr == 2:
                    axs[rr,cc].set_xticklabels(['0','5','10', '15', '20'], fontsize=fs)
                    axs[rr,cc].set_ylim((0, 15.5))
                
                # change plot position:
                ll, bb, ww, hh = axs[rr,cc].get_position().bounds
                axs[rr,cc].set_position([ll+cc*.04, bb+(2-rr)*.04,ww,hh])        

        
        #axs[0,0].set_title('Independent \n peak, dur:%d ms'%duration+', delay:%d ms'%delay, fontsize=fs)
        axs[0,0].set_title('Independent ', fontsize=fs)
        axs[0,1].set_title('AL lateral Inhib.', fontsize=fs)
        axs[0,2].set_title('NSI mechanism', fontsize=fs)
        

        for cc in [1,2]:
            for rr in range(3):
                axs[rr,cc].text(-.2, 1.2, panels_id[cc*3+rr], transform=axs[rr,cc].transAxes,
                               fontsize=panel_fs, color=blue, weight='bold', va='top', ha='right')
        cc = 0
        for rr in range(3):
            axs[rr,cc].text(-.35, 1.2, panels_id[cc*3+rr], transform=axs[rr,cc].transAxes,
                           fontsize=panel_fs, color=blue, weight='bold', va='top', ha='right')
        
        
        if fig_save:
            fig.savefig(fld_analysis+  '/ratio_stim_peak_dur%d'%duration+'_delay%d'%delay+'.png')


#%% RESUME BAR FIGURE
if resumen_bar:
                
    avg_ratio_peak_nsi = np.mean(ratio2dist_peak_nsi, axis=(0,1))
    avg_ratio_peak_ln = np.mean(ratio2dist_peak_ln, axis=(0,1))
    avg_ratio_peak_noin = np.mean(ratio2dist_peak_noin, axis=(0,1))
    avg_ratio_peak_nsi_std = np.std(ratio2dist_peak_nsi, axis=(0,1))
    avg_ratio_peak_ln_std = np.std(ratio2dist_peak_ln, axis=(0,1))
    avg_ratio_peak_noin_std = np.std(ratio2dist_peak_noin, axis=(0,1))
    
    width = 0.3
    
    rs = 1
    cs = 1
    fig, axs = plt.subplots(rs, cs, figsize=(9,4), ) 

    ptns = np.arange(5)
    axs.bar(ptns-width, avg_ratio_peak_noin, width=width, color='magenta', 
            yerr=avg_ratio_peak_noin_std/np.sqrt(n_ratios*n_concs), 
            label='Indep.', )
    axs.bar(ptns, avg_ratio_peak_ln, width=width, color='orange', 
            yerr=avg_ratio_peak_ln_std/np.sqrt(n_ratios*n_concs), 
            label='AL inh.', )
    axs.bar(ptns+width, avg_ratio_peak_nsi, width=width, color='blue', 
            yerr=avg_ratio_peak_nsi_std/np.sqrt(n_ratios*n_concs), 
            label='NSI', )
#%%
    # FIGURE SETTINGS
    axs.spines['right'].set_color('none')   
    axs.spines['top'].set_color('none')                
                
    axs.legend(fontsize=label_fs,loc='upper left', frameon=False)
    axs.set_ylabel('avg coding error (au)', fontsize=label_fs)
    axs.set_xlabel('stimulus duration (ms)', fontsize=label_fs)        
    axs.tick_params(axis='both', which='major', labelsize=label_fs-3)
    axs.set_xticks(ptns)
    axs.set_xticklabels(durs, fontsize=fs)
#    axs.set_yscale('log')    
    axs.text(-.15, 1.0, 'j.', transform=axs.transAxes, color= blue,
             fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    
    # original plot position:
    ll, bb, ww, hh = axs.get_position().bounds
    axs.set_position([ll+.05,bb+.07,ww,hh])        
    
    if fig_save:
        fig.savefig(fld_analysis+  '/ratio_stim_peak_resumebar_durs_delay%d'%delay+'.png')

#%% RESUME FIGURE
if resumen_fig:
                
    avg_ratio_peak_nsi = np.mean(ratio2dist_peak_nsi, axis=(0,1))
    avg_ratio_peak_ln = np.mean(ratio2dist_peak_ln, axis=(0,1))
    avg_ratio_peak_noin = np.mean(ratio2dist_peak_noin, axis=(0,1))
    avg_ratio_peak_nsi_std = np.std(ratio2dist_peak_nsi, axis=(0,1))
    avg_ratio_peak_ln_std = np.std(ratio2dist_peak_ln, axis=(0,1))
    avg_ratio_peak_noin_std = np.std(ratio2dist_peak_noin, axis=(0,1))
    
    rs = 1
    cs = 1
    fig, axs = plt.subplots(rs, cs, figsize=(12, 4), ) 

    axs.errorbar(durs, avg_ratio_peak_noin, yerr=avg_ratio_peak_noin_std/np.sqrt(n_ratios*n_concs), 
                 color='magenta', lw=lw, label='Indep.')
    axs.errorbar(durs, avg_ratio_peak_ln, yerr=avg_ratio_peak_ln_std/np.sqrt(n_ratios*n_concs),
                 color='orange', lw=lw, label='AL inh.')
    axs.errorbar(durs, avg_ratio_peak_nsi, yerr=avg_ratio_peak_nsi_std/np.sqrt(n_ratios*n_concs),
                 color='blue', lw=lw,  label='NSI')
    
    axs.spines['right'].set_color('none')   
    axs.spines['top'].set_color('none')                
                
    axs.legend(fontsize=label_fs,loc='upper left', frameon=False)
    axs.set_ylabel('avg coding error (au)', fontsize=label_fs)
    axs.set_xlabel('stimulus duration (ms)', fontsize=label_fs)        
    axs.tick_params(axis='both', which='major', labelsize=label_fs-3)
    axs.text(-.05, 1.0, 'j.', transform=axs.transAxes,
                               fontsize=30, fontweight='bold', va='top', ha='right')
    # original plot position:
    ll, bb, ww, hh = axs.get_position().bounds
    axs.set_position([ll,bb+.07,ww,hh])        
    
    if fig_save:
        fig.savefig(fld_analysis+  '/ratio_stim_peak_resume_durs_delay%d'%delay+'.png')

#%%***********************************************************
## FIGURE ResumeDelayedStimuli
## **********************************************************

if delay_fig:
    fig, axs = plt.subplots(1, n_durs, figsize=(10, 3.5), ) 
#    fig.tight_layout(pad=.6)
    for id_dur in range(n_durs):
        duration = durs[id_dur]
            
#        axs[id_dur].errorbar(delays2an, ratio_avg_noin[:, id_dur], 
#           yerr=ratio_avg_noin_err[:, id_dur]/n_loops**.5, label= 'noin avg')    
#        axs[id_dur].errorbar(delays2an, ratio_avg_ln[:, id_dur], 
#           yerr=ratio_avg_ln_err[:, id_dur]/n_loops**.5, label= 'ln avg')
#        axs[id_dur].errorbar(delays2an, ratio_avg_nsi[:, id_dur], 
#           yerr=ratio_avg_nsi_err[:, id_dur]/n_loops**.5, label= 'nsi avg')    
#        
        
        axs[id_dur].errorbar(delays2an, ratio_peak_noin[:, id_dur], 
           yerr=ratio_peak_noin_err[:, id_dur]/n_loops**.5, color='magenta', lw = lw, label= 'Indep.')
        axs[id_dur].errorbar(delays2an, ratio_peak_ln[:, id_dur], 
           yerr=ratio_peak_ln_err[:, id_dur]/n_loops**.5, color='orange', lw = lw, label= 'LN inhib.')
        axs[id_dur].errorbar(delays2an, ratio_peak_nsi[:, id_dur],
           yerr=ratio_peak_nsi_err[:, id_dur]/n_loops**.5, color='blue', lw = lw, label= 'NSI')    
        
        axs[id_dur].set_xlabel('Delay (ms)', fontsize=fs)
        axs[id_dur].set_title(' %d ms'%(duration), fontsize=fs)
    
        axs[id_dur].set_ylim((.5, 1.2))
        axs[id_dur].spines['right'].set_color('none')   
        axs[id_dur].spines['top'].set_color('none')     
        axs[id_dur].tick_params(axis='both', which='major', labelsize=label_fs-3)
        axs[id_dur].set_yticklabels('')
        
        # original plot position:
        ll, bb, ww, hh = axs[id_dur].get_position().bounds
        axs[id_dur].set_position([ll,bb+.1,ww,hh-.15]) 
        
        axs[id_dur].set_xticks([0, 250, 500])
        axs[id_dur].set_xticklabels(['0','250','500'], fontsize=label_fs-3)
    
    axs[0].set_yticks([.6, .8, 1., 1.2])
    axs[0].set_yticklabels(['0.6', '0.8', '1.0', '1.2'], fontsize=label_fs-3)
    axs[0].set_title('Duration: %d ms'%(durs[0]), fontsize=fs)
    
    conc2plot = np.squeeze(conc_1_r[0,id_peak2plot,0])
    axs[0].set_ylabel('freq PN ratio', fontsize=fs)
#    axs[0].set_title('Duration: %d ms, \nconc: %.2g'%(durs[0], conc2plot), fontsize=fs)    
    axs[1].legend(bbox_to_anchor=(.3, .45), frameon=False, fontsize=label_fs-5)
    
    if fig_save:
        fig.savefig(fld_analysis+  '/ratio1_delays0-500_dur20-200_conc%.2g'%conc2plot +'.png')
   

#%% **********************************************************
## FIGURE average activity
## **********************************************************
if avg_fig:
    lw = 3
    rs = 3
    cs = 3
    fig, axs = plt.subplots(rs, cs, figsize=(15, 9), ) 
    
    colors = plt.cm.winter_r
    clr_fct = 30
    
    conc_1 = conc_1_r[0,:,0]
    dx = .1
    for pk in [0,1,2,3,]: #range(np.size(conc_1)):# [0,1,3,5,7,]:#
        axs[0,0].errorbar(conc_ratio_t+dx*pk, np.squeeze(np.mean(orn_ratio_avg_noin[:,pk,:], axis=1)),
           yerr= np.squeeze(np.std(orn_ratio_avg_noin[:,pk,:], axis=1))/np.sqrt(n_loops), marker='o', 
           label=r'conc1: '+'%.1f'%(conc_1[pk]), color=colors(pk*clr_fct) )
        axs[0,1].errorbar(conc_ratio_t+dx*pk, np.squeeze(np.mean(orn_ratio_avg_ln[:,pk,:], axis=1)),
           yerr= np.squeeze(np.std(orn_ratio_avg_ln[:,pk,:], axis=1))/np.sqrt(n_loops),  marker='o', 
           label=r'conc1: '+'%.1f'%(conc_1[pk]), color=colors(pk*clr_fct) )
        axs[0,2].errorbar(conc_ratio_t+dx*pk, np.squeeze(np.mean(orn_ratio_avg_nsi[:,pk,:], axis=1)),
           yerr= np.squeeze(np.std(orn_ratio_avg_nsi[:,pk,:], axis=1))/np.sqrt(n_loops), marker='o', 
           label=r'conc1: '+'%.1f'%(conc_1[pk]), color=colors(pk*clr_fct) )
        
        axs[1,0].errorbar(conc_ratio_t+dx*pk, np.squeeze(np.mean(pn_ratio_avg_noin[:,pk,:], axis=1)),
           yerr= np.squeeze(np.std(pn_ratio_avg_noin[:,pk,:], axis=1))/np.sqrt(n_loops),  marker='o', 
           label=r'conc1: '+'%.1f'%(conc_1[pk]), 
           color=colors(pk*clr_fct) )
        axs[1,1].errorbar(conc_ratio_t+dx*pk, np.squeeze(np.mean(pn_ratio_avg_ln[:,pk,:], axis=1)),
           yerr= np.squeeze(np.std(pn_ratio_avg_ln[:,pk,:], axis=1))/np.sqrt(n_loops),  marker='o', 
           label=r'conc1: '+'%.1f'%(conc_1[pk]), 
           color=colors(pk*clr_fct) )
        axs[1,2].errorbar(conc_ratio_t+dx*pk, np.squeeze(np.mean(pn_ratio_avg_nsi[:,pk,:], axis=1)),
           yerr= np.squeeze(np.std(pn_ratio_avg_nsi[:,pk,:], axis=1))/np.sqrt(n_loops),
            marker='o', label=r'conc1: '+'%.1f'%(conc_1[pk]), 
           color=colors(pk*clr_fct) )
        
        axs[2,0].errorbar(conc_ratio_t+dx*pk, conc_ratio_t/np.squeeze(np.mean(pn_ratio_avg_noin[:,pk,:], axis=1)),
           yerr= np.squeeze(np.std(pn_ratio_avg_noin[:,pk,:], axis=1))/np.sqrt(n_loops)/conc_ratio_t,  marker='o', 
           label=r'conc1: '+'%.1f'%(conc_1[pk]), 
           color=colors(pk*clr_fct) )
        axs[2,1].errorbar(conc_ratio_t+dx*pk, conc_ratio_t/np.squeeze(np.mean(pn_ratio_avg_ln[:,pk,:], axis=1)),
           yerr= np.squeeze(np.std(pn_ratio_avg_ln[:,pk,:], axis=1))/np.sqrt(n_loops)/conc_ratio_t,  marker='o', 
           label=r'conc1: '+'%.1f'%(conc_1[pk]), 
           color=colors(pk*clr_fct) )
        axs[2,2].errorbar(conc_ratio_t+dx*pk, conc_ratio_t/np.squeeze(np.mean(pn_ratio_avg_nsi[:,pk,:], axis=1)),
           yerr= np.squeeze(np.std(pn_ratio_avg_nsi[:,pk,:], axis=1))/np.sqrt(n_loops)/conc_ratio_t,
            marker='o', label=r'conc1: '+'%.1f'%(conc_1[pk]), 
           color=colors(pk*clr_fct) )
    
    
    axs[0,0].errorbar(conc_ratio_t, conc_ratio_t,lw=lw, color='r', label='theor')
    axs[0,1].errorbar(conc_ratio_t, conc_ratio_t, lw=lw, color='r', label='theor')
    axs[0,2].errorbar(conc_ratio_t, conc_ratio_t, lw=lw, color='r', label='theor')

    axs[1,0].errorbar(conc_ratio_t, conc_ratio_t, lw=lw, color='r', label='theor')
    axs[1,1].errorbar(conc_ratio_t, conc_ratio_t, lw=lw, color='r', label='theor')
    axs[1,2].errorbar(conc_ratio_t, conc_ratio_t, lw=lw, color='r', label='theor')
    
    axs[2,0].errorbar(conc_ratio_t, np.ones_like(conc_ratio_t),lw=lw, color='r', label='theor')
    axs[2,1].errorbar(conc_ratio_t, np.ones_like(conc_ratio_t), lw=lw, color='r', label='theor')
    axs[2,2].errorbar(conc_ratio_t, np.ones_like(conc_ratio_t), lw=lw, color='r', label='theor')
    
    axs[0,0].set_ylabel('ratio\nORN response', fontsize=fs)
    axs[1,0].set_ylabel('ratio\nPN response', fontsize=fs)
    axs[2,0].set_ylabel('PN Ratio \n error', fontsize=fs)
    
    axs[0,0].text(2, 3.5, 'ORNs', fontsize=fs)
    axs[1,0].text(2, 5, 'PNs', fontsize=fs)
    axs[2,0].text(2, 5, 'PNs', fontsize=fs)
    
    axs[2,0].set_xlabel('conc ratio', fontsize=fs)
    axs[2,1].set_xlabel('conc ratio', fontsize=fs)
    axs[2,2].set_xlabel('conc ratio', fontsize=fs) 
    
    axs[2,0].set_ylim((0, 5))
    axs[2,1].set_ylim((0, 5))
    axs[2,2].set_ylim((0, 5))

    
    axs[0,2].legend()
    axs[0,0].set_title('Independent\n avg, dur:%d ms'%stim_params[2]+', delay:%d'%delay, fontsize=fs)
    #axs[0,1].legend()
    axs[0,1].set_title('AL lateral Inhib.', fontsize=fs)
    #axs[0,2].legend()
    axs[0,2].set_title('NSI mechanism', fontsize=fs)
    
    
    
    if fig_save:
        fig.savefig(fld_analysis+  '/ratio_stim_avg_dur%d'%stim_params[2]+'_delay%d'%delay+'.png')        
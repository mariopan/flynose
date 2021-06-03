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
# *****************************************************************


def fig_pn_distr():
    """ Figure of the average activity for weak and strong input
    """
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
    
    
# *****************************************************************
fig_save        = 0

id_peak2plot    = 3             # ONLY for the delays analysis
measure         = 'avg'         # 'avg' # 'peak' # 
corr_mi         = 'corr'        # 'corr' # 'mi'
delay_fig       = 0             # Fig.ResumeDelayedStimuli

# analysis for zero delay:
ratio_fig       = 1*(1-delay_fig)   # Response ratios versus concentration ratio 
                                    #   of the two odorants for different 
                                    #   overall concentrations (colours, see 
                                    #   legend in f). Black dashed diagonal is 
                                    #   for peak PN ratios equal to odorant 
                                    #   concentration ratio. Error bars 
                                    #   represent the semi inter-quartile 
                                    #   range calculated over 10 trials.

resumen_chess   = 0*(1-delay_fig)   # Analysis of the coding error for different values of 
                                    #   stimulus duration and concentration values.

resumen_chess_mi   = 0*(1-delay_fig)  # Analysis of the MI (or cross correlation) between 
                                    #   concentration ratio and PN ratio

pn_activity     = 1*(1-delay_fig)   # PN activity for weak and strong input

orn_activity    = 1*(1-delay_fig)   # ORN activity for weak and strong input

resumen_bar     = 0*(1-delay_fig)   # Fig.ResumeEncodeRatioBar
pn_distr        = 0*(1-delay_fig)   # Fig.PNdistribution
  


# select a subsample of the params to analyse
# nsi_ln_par   = [[0,0],[.4, 0],[0, .4],]
nsi_ln_par   = [[0,0],[.6, 0],[0, .6],]

if delay_fig:
    print('Delays analysis, rate measured on '+measure)
    # fld_analysis    = 'NSI_analysis/analysis_delays_tauln250_tausdf41/'
    fld_analysis    = 'NSI_analysis/delays_tauln250_tausdf41/'
    # fld_analysis    = 'NSI_analysis/delays_.4.2/'   # analysis_ratio_tauln25              
    fld_output      = fld_analysis
    name_analysis   = 'delays'
    
else:
    print('Ratio analysis, rate measured on '+measure+', ' + corr_mi)
    fld_analysis    = 'NSI_analysis/ratio_tauln250/'   # analysis_ratio_tauln25              
    # fld_analysis    = 'NSI_analysis/ratio_.4.2/'   # analysis_ratio_tauln25              
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

ratio_mi_noin = np.ones((n_concs,n_durs))
ratio_mi_ln = np.ones((n_concs,n_durs))
ratio_mi_nsi = np.ones((n_concs,n_durs))
    
# LOAD DATA AND CALCULATE RATIOS
for delay_id, delay in enumerate(delays2an):
    for [inh_id, [nsi_str, alpha_ln]] in enumerate(nsi_ln_par):
        data_name  = name_analysis + \
                '_stim_' + stim_params['stim_type'] +\
                '_nsi_%.1f'%(nsi_str) +\
                '_ln_%.1f'%(alpha_ln) +\
                '_delay2an_%d'%(delay) +\
                '.pickle'        

        all_data    = pickle.load(open(fld_analysis+data_name,  "rb" ) )
        [params_al_orn, output2an, ] = all_data
        
        
        if (alpha_ln==0) & (nsi_str==0):
            orn_avg_w_noin   = output2an['avg_ornw']
            orn_avg_s_noin   = output2an['avg_orns']
            pn_avg_w_noin    = output2an['avg_pnw']
            pn_avg_s_noin    = output2an['avg_pns']
            
            orn_peak_w_noin   = output2an['peak_ornw']
            orn_peak_s_noin   = output2an['peak_orns']
            pn_peak_w_noin    = output2an['peak_pnw']  # minimum value 10Hz
            pn_peak_s_noin    = output2an['peak_pns'] 
        
        elif (alpha_ln>0) & (nsi_str==0):
            orn_avg_w_ln   = output2an['avg_ornw']
            orn_avg_s_ln   = output2an['avg_orns']
            pn_avg_w_ln    = output2an['avg_pnw']
            pn_avg_s_ln    = output2an['avg_pns']
            
            orn_peak_w_ln   = output2an['peak_ornw']
            orn_peak_s_ln   = output2an['peak_orns']
            pn_peak_w_ln    = output2an['peak_pnw']
            pn_peak_s_ln    = output2an['peak_pns']
            
        elif (alpha_ln==0) & (nsi_str>0):
            orn_avg_w_nsi   = output2an['avg_ornw']
            orn_avg_s_nsi   = output2an['avg_orns']
            pn_avg_w_nsi    = output2an['avg_pnw']
            pn_avg_s_nsi    = output2an['avg_pns'] #(n_ratios, n_concs,n_durs, n_loops)
            
            orn_peak_w_nsi   = output2an['peak_ornw']
            orn_peak_s_nsi   = output2an['peak_orns']
            pn_peak_w_nsi    = output2an['peak_pnw']
            pn_peak_s_nsi    = output2an['peak_pns']
            
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
            
            # ratio_avg_noin_err[delay_id, :] = scipy.stats.iqr(pn_ratio_avg_noin, axis=3)/2 #np.diff(np.percentile(pn_ratio_avg_noin[0,id_peak2plot,:,:], [25,50])) #np.std(pn_ratio_avg_noin[0,id_peak2plot,:,:], axis=1)
            # ratio_avg_ln_err[delay_id, :] = scipy.stats.iqr(pn_ratio_avg_ln, axis=3)/2 #np.diff(np.percentile(pn_ratio_avg_ln[0,id_peak2plot,:,:], [25,50])) #np.std(pn_ratio_avg_ln[0,id_peak2plot ,:,:], axis=1)
            # ratio_avg_nsi_err[delay_id, :] = scipy.stats.iqr(pn_ratio_avg_nsi, axis=3)/2 #np.diff(np.percentile(pn_ratio_avg_nsi[0,id_peak2plot,:,:], [25,50])) #np.std(pn_ratio_avg_nsi[0,id_peak2plot,:,:], axis=1)
            ratio_avg_noin_err[delay_id, :] = np.diff(np.percentile(pn_ratio_avg_noin[0,id_peak2plot,:,:], [25,50]))
            #scipy.stats.iqr(pn_ratio_peak_noin, axis=3)/2                  
            #np.std(pn_ratio_peak_noin[0,id_peak2plot,:,:], axis=1)
            ratio_avg_ln_err[delay_id, :] = np.diff(np.percentile(pn_ratio_avg_ln[0,id_peak2plot,:,:], [25,50])) # 
            # scipy.stats.iqr(pn_ratio_peak_ln, axis=3)/2 #np.std(pn_ratio_peak_ln[0,id_peak2plot,:,:], axis=1)
            ratio_avg_nsi_err[delay_id, :] =  np.diff(np.percentile(pn_ratio_avg_nsi[0,id_peak2plot,:,:], [25,50])) #

        
        elif measure == 'peak':
            ratio_peak_noin[delay_id, :] = np.median(pn_ratio_peak_noin[0,id_peak2plot,:,:], axis=1)
            ratio_peak_ln[delay_id, :] = np.median(pn_ratio_peak_ln[0,id_peak2plot,:,:], axis=1)
            ratio_peak_nsi[delay_id, :] =np.median(pn_ratio_peak_nsi[0,id_peak2plot,:,:], axis=1)
            
            ratio_peak_noin_err[delay_id, :] = np.diff(np.percentile(pn_ratio_peak_noin[0,id_peak2plot,:,:], [25,50]))
            #scipy.stats.iqr(pn_ratio_peak_noin, axis=3)/2                  
            #np.std(pn_ratio_peak_noin[0,id_peak2plot,:,:], axis=1)
            ratio_peak_ln_err[delay_id, :] = np.diff(np.percentile(pn_ratio_peak_ln[0,id_peak2plot,:,:], [25,50])) # 
            # scipy.stats.iqr(pn_ratio_peak_ln, axis=3)/2 #np.std(pn_ratio_peak_ln[0,id_peak2plot,:,:], axis=1)
            ratio_peak_nsi_err[delay_id, :] =  np.diff(np.percentile(pn_ratio_peak_nsi[0,id_peak2plot,:,:], [25,50])) #
            # scipy.stats.iqr(pn_ratio_peak_nsi, axis=3)/2 # np.std(pn_ratio_peak_nsi[0,id_peak2plot,:,:], axis=1)
    else:
        if measure == 'avg':
            # average over the run with identical params
            ratio1_noin = np.median(orn_ratio_avg_noin, axis=3)
            ratio1_ln   = np.median(orn_ratio_avg_ln, axis=3)
            ratio1_nsi  = np.median(orn_ratio_avg_nsi, axis=3)
            
            ratio1_err_noin = scipy.stats.iqr(orn_ratio_avg_noin, axis=3)/2 #np.squeeze(np.diff(np.percentile(orn_ratio_avg_noin, [25,50],axis=3), axis=0))
            ratio1_err_ln = scipy.stats.iqr(orn_ratio_avg_ln, axis=3)/2 #np.squeeze(np.diff(np.percentile(orn_ratio_avg_ln,  [25,50],axis=3), axis=0)) 
            ratio1_err_nsi = scipy.stats.iqr(orn_ratio_avg_nsi, axis=3)/2 #np.squeeze(np.diff(np.percentile(orn_ratio_avg_nsi,  [25,50],axis=3), axis=0)) 
            
            # average over the run with identical params
            ratio2_noin = np.median(pn_ratio_avg_noin, axis=3)
            ratio2_ln   = np.median(pn_ratio_avg_ln, axis=3)
            ratio2_nsi  = np.median(pn_ratio_avg_nsi, axis=3)
            
            
            ratio2_err_noin = scipy.stats.iqr(pn_ratio_avg_noin, axis=3)/2     #np.squeeze(np.diff(np.percentile(pn_ratio_avg_noin, [25,50],axis=3), axis=0))#np.std(pn_ratio_peak_noin, axis=3)
            ratio2_err_ln = scipy.stats.iqr(pn_ratio_avg_ln, axis=3)/2     #np.squeeze(np.diff(np.percentile(pn_ratio_avg_ln,  [25,50],axis=3), axis=0)) #np.std(pn_ratio_peak_ln, axis=3)  
            ratio2_err_nsi = scipy.stats.iqr(pn_ratio_avg_nsi, axis=3)/2     #np.squeeze(np.diff(np.percentile(pn_ratio_avg_nsi,  [25,50],axis=3), axis=0)) #np.std(pn_ratio_peak_nsi, axis=3)
            
            # code error measure
            for iic in range(n_concs):
                for iid in range(n_durs):
                    if corr_mi ==  'corr':
                        ratio_mi_noin[iic,iid,] = np.corrcoef(conc_ratios, 
                            np.median(pn_ratio_avg_noin[:,iic,iid,:], axis=1))[0,1]
                        
                        ratio_mi_ln[iic,iid,] = np.corrcoef(conc_ratios, 
                            np.median(pn_ratio_avg_ln[:,iic,iid,:], axis=1))[0,1]
                        
                        ratio_mi_nsi[iic,iid,] = np.corrcoef(conc_ratios, 
                            np.median(pn_ratio_avg_nsi[:,iic,iid,:], axis=1))[0,1]
                    elif corr_mi ==  'mi':
                        ratio_mi_noin[iic,iid,] = calc_MI(conc_ratios, 
                            np.median(pn_ratio_avg_noin[:,iic,iid,:], axis=1),
                            n_bins_mi)
                        
                        ratio_mi_ln[iic,iid,] = calc_MI(conc_ratios, 
                            np.median(pn_ratio_avg_ln[:,iic,iid,:], axis=1), 
                            n_bins_mi)
                        
                        ratio_mi_nsi[iic,iid,] = calc_MI(conc_ratios, 
                            np.median(pn_ratio_avg_nsi[:,iic,iid,:], axis=1), 
                            n_bins_mi)
    
            noin_tmp = ((conc_ratios-pn_ratio_avg_noin.T)/
                        (pn_ratio_avg_noin.T + conc_ratios))**2
            ln_tmp = ((conc_ratios - pn_ratio_avg_ln.T)/
                        (pn_ratio_avg_ln.T + conc_ratios))**2
            nsi_tmp = ((conc_ratios - pn_ratio_avg_nsi.T)/
                        (pn_ratio_avg_nsi.T + conc_ratios))**2
        elif measure == 'peak':
            # average over the run with identical params
            ratio1_noin = np.median(orn_ratio_peak_noin, axis=3)
            ratio1_ln   = np.median(orn_ratio_peak_ln, axis=3)
            ratio1_nsi  = np.median(orn_ratio_peak_nsi, axis=3)
            
            ratio1_err_noin = scipy.stats.iqr(orn_ratio_peak_noin, axis=3)/2     # np.squeeze(np.diff(np.percentile(orn_ratio_peak_noin, [25,50],axis=3), axis=0))#np.std(pn_ratio_peak_noin, axis=3)
            ratio1_err_ln = scipy.stats.iqr(orn_ratio_peak_ln, axis=3)/2     #np.squeeze(np.diff(np.percentile(orn_ratio_peak_ln,  [25,50],axis=3), axis=0)) #np.std(pn_ratio_peak_ln, axis=3)  
            ratio1_err_nsi = scipy.stats.iqr(orn_ratio_peak_nsi, axis=3)/2     # np.squeeze(np.diff(np.percentile(orn_ratio_peak_nsi,  [25,50],axis=3), axis=0)) #np.std(pn_ratio_peak_nsi, axis=3)
            
            # average over the run with identical params
            ratio2_noin = np.median(pn_ratio_peak_noin, axis=3)
            ratio2_ln   = np.median(pn_ratio_peak_ln, axis=3)
            ratio2_nsi  = np.median(pn_ratio_peak_nsi, axis=3)
            
            ratio2_err_noin = scipy.stats.iqr(pn_ratio_peak_noin, axis=3)/2     #np.squeeze(np.diff(np.percentile(pn_ratio_peak_noin, [25,50],axis=3), axis=0))#np.std(pn_ratio_peak_noin, axis=3)
            ratio2_err_ln = scipy.stats.iqr(pn_ratio_peak_ln, axis=3)/2     #np.squeeze(np.diff(np.percentile(pn_ratio_peak_ln,  [25,50],axis=3), axis=0)) #np.std(pn_ratio_peak_ln, axis=3)  
            ratio2_err_nsi = scipy.stats.iqr(pn_ratio_peak_nsi, axis=3)/2     #np.squeeze(np.diff(np.percentile(pn_ratio_peak_nsi,  [25,50],axis=3), axis=0)) #np.std(pn_ratio_peak_nsi, axis=3)
                
            # code error measure
            for iic in range(n_concs):
                for iid in range(n_durs):
                    if corr_mi ==  'corr':
                        ratio_mi_noin[iic,iid,] = np.corrcoef(conc_ratios, 
                            np.median(pn_ratio_peak_noin[:,iic,iid,:], axis=1))[0,1]
                        
                        ratio_mi_ln[iic,iid,] = np.corrcoef(conc_ratios, 
                            np.median(pn_ratio_peak_ln[:,iic,iid,:], axis=1))[0,1]
                        
                        ratio_mi_nsi[iic,iid,] = np.corrcoef(conc_ratios, 
                            np.median(pn_ratio_peak_nsi[:,iic,iid,:], axis=1))[0,1]
                    elif corr_mi ==  'mi':
                        ratio_mi_noin[iic,iid,] = calc_MI(conc_ratios, 
                            np.median(pn_ratio_peak_noin[:,iic,iid,:], axis=1),
                            n_bins_mi)
                        
                        ratio_mi_ln[iic,iid,] = calc_MI(conc_ratios, 
                            np.median(pn_ratio_peak_ln[:,iic,iid,:], axis=1), 
                            n_bins_mi)
                        
                        ratio_mi_nsi[iic,iid,] = calc_MI(conc_ratios, 
                            np.median(pn_ratio_peak_nsi[:,iic,iid,:], axis=1), 
                            n_bins_mi)

            noin_tmp = ((conc_ratios - pn_ratio_peak_noin.T)/
                        (pn_ratio_peak_noin.T + conc_ratios))**2
            ln_tmp = ((conc_ratios - pn_ratio_peak_ln.T)/
                        (pn_ratio_peak_ln.T + conc_ratios))**2
            nsi_tmp = ((conc_ratios - pn_ratio_peak_nsi.T)/
                        (pn_ratio_peak_nsi.T + conc_ratios))**2
            
        # average and std over runs with identical params
        ratio2dist_noin = np.mean(noin_tmp, axis=0).T
        ratio2dist_nsi = np.mean(nsi_tmp, axis=0).T
        ratio2dist_ln = np.mean(ln_tmp, axis=0).T
        
        ratio2dist_err_noin = np.ma.std(noin_tmp, axis=0).T
        ratio2dist_err_nsi = np.ma.std(nsi_tmp, axis=0).T
        ratio2dist_err_ln = np.ma.std(ln_tmp, axis=0).T
        


#%% FIGURE ResumeDelayedStimuli ############################################
if delay_fig:
    y_ticks = np.linspace(0, 2, 5)
    fig, axs = plt.subplots(1, n_durs, figsize=(17, 6.3), sharey=True) 
    for dur_id in range(n_durs):
        duration = dur2an[dur_id]
        
        if measure=='avg':
            axs[dur_id].errorbar(delays2an, ratio_avg_noin[:, dur_id], 
               yerr=ratio_avg_noin_err[:, dur_id], color=pink, 
               lw = lw, label= 'ctrl')
            axs[dur_id].errorbar(delays2an, ratio_avg_ln[:, dur_id], 
               yerr=ratio_avg_ln_err[:, dur_id], color=orange, 
               lw = lw, label= 'LN inhib.')
            axs[dur_id].errorbar(delays2an, ratio_avg_nsi[:, dur_id], 
               yerr=ratio_avg_nsi_err[:, dur_id], color=cyan, ls='--',
               lw = lw, label= 'NSI')    
        elif measure=='peak':        
            axs[dur_id].errorbar(delays2an, ratio_peak_noin[:, dur_id], 
               yerr=ratio_peak_noin_err[:, dur_id], color= pink, ls='-.',
               lw = lw, label= 'ctrl')
            axs[dur_id].errorbar(delays2an, ratio_peak_ln[:, dur_id], 
               yerr=ratio_peak_ln_err[:, dur_id], color=orange, ls='-',
               lw = lw, label= 'LN inhib.')
            axs[dur_id].errorbar(delays2an, ratio_peak_nsi[:, dur_id],
               yerr=ratio_peak_nsi_err[:, dur_id], color=cyan, ls='--',
               lw = lw, label= 'NSI')    
        
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
    axs[1].legend(fontsize=fs-2, frameon=False)

    axs[0].text(-.2, 1.2, 'b', transform=axs[0].transAxes,
           fontsize=panel_fs, color=black, weight='bold', va='top', ha='right')
    plt.show()
    
    
    if fig_save:
        if measure == 'avg':
            fig.savefig(fld_output+  '/delays_avg_delays0-500_dur20-200_conc%.2g'%conc2plot +'_tauln%d'%tau_ln+'.png')
        elif measure == 'peak':
            fig.savefig(fld_output+  '/delays_peak_delays0-500_dur20-200_conc%.2g'%conc2plot +'_tauln%d'%tau_ln+'.png')
   

#%% *********************************************************
## FIGURE peak
if ratio_fig: 
    lw = 3
    rs = 2
    cs = 3
    colors = plt.cm.winter_r
    clr_fct = 30        # color factor
    
    panels_id   = ['a', 'b', 'c', 'd', 'e', 'f', ]
    # dur2plot = dur2an[2]
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
               label=r''+'%.5f'%(conc_v), 
               color=colors(conc_id*clr_fct) )
            
            axs[1,1].errorbar(conc_ratios+dx*conc_id, ratio2_ln[:,conc_id, dur_id],
               yerr= ratio2_err_ln[:,conc_id, dur_id], marker='o', 
               label=r''+'%.5f'%(conc_v), 
               color=colors(conc_id*clr_fct) )
            
            axs[1,2].errorbar(conc_ratios+dx*conc_id, ratio2_nsi[:,conc_id, dur_id],
               yerr= ratio2_err_nsi[:,conc_id, dur_id], marker='o', 
               label=r''+'%.5f'%(conc_v), 
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
                
                # axs[rr,cc].set_xticklabels('')
                # axs[rr,cc].set_yticklabels('')
                axs[rr,cc].spines['right'].set_color('none')
                axs[rr,cc].spines['top'].set_color('none')
                
                axs[rr,cc].set_ylim((0.8, 100.5))
                axs[rr,cc].set_xlim((0.8, 20.5))
                
                axs[rr,cc].set_yticks([1, 10, ])
                axs[rr,cc].set_xticks([1, 10, 20])
                
                axs[rr,cc].set_yscale("log")
                
                # if cc == 0:
                #     axs[rr,cc].set_yticklabels(['0','5','10', '15', '20'], fontsize=label_fs-3)
                
                # if rr == 1:
                #     axs[rr,cc].set_xticklabels(['0','5','10', '15', '20'], fontsize=label_fs-3)
                
                # change plot position:
                ll, bb, ww, hh = axs[rr,cc].get_position().bounds
                axs[rr,cc].set_position([ll+cc*.03, bb+(2-rr)*.03,ww,hh])       
                


        axs[0,0].set_title('ctrl-model', fontsize=label_fs)
        axs[0,1].set_title('LN-model', fontsize=label_fs)
        axs[0,2].set_title('NSI-model', fontsize=label_fs)
        

        for cc in [1,2]:
            for rr in range(2):
                axs[rr,cc].text(-.2, 1.2, panels_id[cc*rs+rr], transform=axs[rr,cc].transAxes,
                               fontsize=panel_fs, color=black, weight='bold', va='top', ha='right')
        cc = 0
        for rr in range(2):
            axs[rr,cc].text(-.35, 1.2, panels_id[cc*rs+rr], transform=axs[rr,cc].transAxes,
                           fontsize=panel_fs, color=black, weight='bold', va='top', ha='right')
        axs[1,1].legend(frameon=False)
        
        if fig_save:
            if measure == 'avg':
                fig.savefig(fld_output+ 
                        '/ratio_stim_avg_dur%d'%duration+'_delay%d'%delay+'.png')        
            elif measure == 'peak':
                fig.savefig(fld_output+  
                        '/ratio_stim_peak_dur%d'%duration+'_delay%d'%delay+'.png')
                   
        plt.show()
        
#%% ************************************************************************
# RESUME CHESS FIGURE: 
    # analysis of the relative distance between concentration ratio and PN ratio
if resumen_chess:
    # average over conc.ratio and concentrations
    err_code_nsi = np.mean(ratio2dist_nsi, axis=(0))
    err_code_ln = np.mean(ratio2dist_ln, axis=(0))
    err_code_noin = np.mean(ratio2dist_noin, axis=(0))
    
    
    rs = 1
    cs = 3
    vmin =  0
    
    vmax =  0.7 #0.4
#    colorbar params
    frac = .1 #.04
    pad = .04
    
    fig, axs = plt.subplots(rs, cs, figsize=(12, 4.3),) 

    ptns = np.arange(5)
    im0 = axs[0].imshow(err_code_noin, cmap=cmap, vmin=vmin, vmax=vmax)
    
    im1 = axs[1].imshow(err_code_ln, cmap=cmap, vmin=vmin, vmax=vmax)
    
    im2 = axs[2].imshow(err_code_nsi, cmap=cmap, vmin=vmin, vmax=vmax)
    
    
    # FIGURE SETTINGS
    axs[0].set_title('ctrl-model', fontsize=title_fs)
    axs[1].set_title('LN-model', fontsize=title_fs)
    axs[2].set_title('NSI-model', fontsize=title_fs)
    
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
    
    axs[0].text(-.2, 1.1, 'g', transform=axs[0].transAxes, 
             fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    

    cbar = fig.colorbar(im0, ax=axs[2], fraction=.05,pad=pad, 
            orientation='vertical', ticks=[0, vmax/2, vmax])
    cbar.set_label('code error', fontsize=label_fs)
    ticklabs = cbar.ax.get_yticklabels()
    
    #%% to run separately from the other, or it won't plot the ticklabels
    cbar.ax.set_yticklabels(ticklabs, fontsize=ticks_fs)
    
    # adjust bar size and position
    ll, bb, ww, hh = cbar.ax.get_position().bounds
    cbar.ax.set_position([ll-.015+e_sx, bb+.085, ww, hh-.12])
   
    #adjust 3rd chess board size and position
    dwdh =1.0125
    ll_a, bb, ww, hh = axs[2].get_position().bounds
    axs[2].set_position([ll_a -.045+e_sx, bb_new, ww_new*dwdh, hh_new*dwdh])
    
    if fig_save:
        if measure == 'peak':
            fig.savefig(fld_output + '/ratio_stim_peak_resumechess_durs_delay%d'%delay+'.png')
        elif measure == 'avg':    
            fig.savefig(fld_output + '/ratio_stim_avg_resumechess_durs_delay%d'%delay+'.png')        
    
    plt.show()    
    
    
    
#%% # RESUME CHESS FIGURE with Mutual Information ##########################
# analysis of the MI (or cross correlation) between concentration ratio and PN ratio
if resumen_chess_mi:
    # average over conc.ratio and concentrations
    err_code_nsi = ratio_mi_nsi
    err_code_ln = ratio_mi_ln
    err_code_noin = ratio_mi_noin
    
    rs = 1
    cs = 3
    vmin = .2 # 0
    
    vmax =  1. # 0.4
#    colorbar params
    frac = .1 #.04
    pad = .04
    
    fig, axs = plt.subplots(rs, cs, figsize=(12, 4.3),) 

    ptns = np.arange(5)
    im0 = axs[0].imshow(err_code_noin, cmap=cmap, vmin=vmin, vmax=vmax)
    
    im1 = axs[1].imshow(err_code_ln, cmap=cmap, vmin=vmin, vmax=vmax)
    
    im2 = axs[2].imshow(err_code_nsi, cmap=cmap, vmin=vmin, vmax=vmax)
    
    
    # FIGURE SETTINGS
    axs[0].set_title('ctrl-model mi', fontsize=title_fs)
    axs[1].set_title('LN-model', fontsize=title_fs)
    axs[2].set_title('NSI-model', fontsize=title_fs)
    
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
    
    cbar = fig.colorbar(im0, ax=axs[2], fraction=.05,pad=pad, 
            orientation='vertical', ticks=[0, vmax/2, vmax])
    cbar.set_label('code error', fontsize=label_fs)
    ticklabs = cbar.ax.get_yticklabels()

#%% to run separately from the other, or it won't plot the ticklabels
if resumen_chess_mi:
    cbar.ax.set_yticklabels(ticklabs, fontsize=ticks_fs)
    
    # adjust bar size and position
    ll, bb, ww, hh = cbar.ax.get_position().bounds
    cbar.ax.set_position([ll-.015+e_sx, bb+.085, ww, hh-.12])
   
    #adjust 3rd chess board size and position
    dwdh =1.0125
    ll_a, bb, ww, hh = axs[2].get_position().bounds
    axs[2].set_position([ll_a -.045+e_sx, bb_new, ww_new*dwdh, hh_new*dwdh])

    if fig_save:
        if measure == 'peak':
            fig.savefig(fld_output + '/ratio_stim_peak_resumechess_' + 
                        corr_mi + '_durs_delay%d'%delay+'.png')
        elif measure == 'avg':    
            fig.savefig(fld_output + '/ratio_stim_avg_resumechess_' + 
                        corr_mi + '_durs_delay%d'%delay+'.png')
            
    plt.show()
    



#%%***********************************************************
# FIGURE PN activity in a chess plot
# Figure of the average activity for weak and strong input
if pn_activity:

    ratio2plot = np.round(conc_ratios[::3])
    conc2plot = concs2an[::1]
    
    rs = 3
    cs = 2
    for dur_id, duration in enumerate(dur2an):

        # pn_peak_s_noin: (n_ratios, n_concs,n_durs, n_loops)
        noin_s = np.median(pn_peak_s_noin[:,:,dur_id, :], axis=2)
        ln_s = np.median(pn_peak_s_ln[:,:,dur_id, :], axis=2)
        nsi_s =np.median(pn_peak_s_nsi[:,:,dur_id, :], axis=2)
        
        noin_w = np.median(pn_peak_w_noin[:,:,dur_id, :], axis=2)
        ln_w = np.median(pn_peak_w_ln[:,:,dur_id, :], axis=2)
        nsi_w = np.median(pn_peak_w_nsi[:,:,dur_id, :], axis=2)
        
        fig, axs = plt.subplots(rs, cs, figsize=(9, 6), ) 
        
        # PLOT 
        im0 = axs[0,0].imshow(noin_s.T, cmap=cmap, aspect='auto', vmin=0, vmax=300)
        fig.colorbar(im0, ax=axs[0,0])
        
        im1 = axs[0, 1].imshow(noin_w.T, cmap=cmap, aspect='auto', vmin=0, vmax=170)
        fig.colorbar(im1, ax=axs[0,1])
        
        im0 = axs[1, 0].imshow(ln_s.T, cmap=cmap, aspect='auto', vmin=0, vmax=300)
        fig.colorbar(im0, ax=axs[1,0])
        
        im1 = axs[1, 1].imshow(ln_w.T, cmap=cmap, aspect='auto', vmin=0, vmax=170)
        fig.colorbar(im1, ax=axs[1,1])
        
        im0 = axs[2, 0].imshow(nsi_s.T, cmap=cmap, aspect='auto', vmin=0, vmax=300)
        fig.colorbar(im0, ax=axs[2,0])
        
        im1 = axs[2, 1].imshow(nsi_w.T, cmap=cmap, aspect='auto', vmin=0, vmax=170)
        fig.colorbar(im1, ax=axs[2,1])
        
        
        # SETTINGS
        axs[0,0].set_title('PN strong')
        axs[0,1].set_title('PN weak')

        for id_r in range(rs):
            axs[id_r,0].set_ylabel('input (a.u.)', fontsize= label_fs)
        
        axs[0,1].text(62.5, 1.5, 'ctrl', fontsize=label_fs)
        axs[1,1].text(62.5, 1.5, 'LN', fontsize=label_fs)
        axs[2,1].text(62.5, 1.5, 'NSI', fontsize=label_fs)
        
        
        for c_id in range(cs):
            axs[2, c_id].set_xlabel('ratio (unitless)', fontsize= label_fs)
            
            for r_id in range(rs):
                axs[r_id,c_id].set_xticks(np.linspace(1, len(conc_ratios), 15))
                axs[r_id,c_id].set_xticklabels(ratio2plot)
                axs[r_id,c_id].set_yticks(range(len(conc2plot)))
                axs[r_id,c_id].set_yticklabels(conc2plot)
            
        for c_id in range(cs):
            for r_id in range(rs):
                ll, bb, ww, hh = axs[r_id, c_id].get_position().bounds
                axs[r_id, c_id].set_position([ll, bb, ww, hh])
                
            
    
                
    
        if fig_save:
            fig.savefig(fld_output+  '/PN_delays0_dur%d'%duration+'.png')

    plt.show()      
            
#%%***********************************************************
# FIGURE ORN activity in a chess plot
# Figure of the average activity for weak and strong input
if orn_activity:

    ratio2plot = np.round(conc_ratios[::3])
    conc2plot = concs2an[::1]
    
    rs = 3
    cs = 2
    for dur_id, duration in enumerate(dur2an):

        # pn_peak_s_noin: (n_ratios, n_concs,n_durs, n_loops)
        noin_s = np.median(orn_peak_s_noin[:,:,dur_id, :], axis=2)
        ln_s = np.median(orn_peak_s_ln[:,:,dur_id, :], axis=2)
        nsi_s =np.median(orn_peak_s_nsi[:,:,dur_id, :], axis=2)
        
        noin_w = np.median(orn_peak_w_noin[:,:,dur_id, :], axis=2)
        ln_w = np.median(orn_peak_w_ln[:,:,dur_id, :], axis=2)
        nsi_w = np.median(orn_peak_w_nsi[:,:,dur_id, :], axis=2)
        
        fig, axs = plt.subplots(rs, cs, figsize=(9, 6), ) 
        
        # PLOT 
        im0 = axs[0,0].imshow(noin_s.T, cmap=cmap, aspect='auto', vmin=0, vmax=300)
        fig.colorbar(im0, ax=axs[0,0])
        
        im1 = axs[0, 1].imshow(noin_w.T, cmap=cmap, aspect='auto', vmin=0, vmax=170)
        fig.colorbar(im1, ax=axs[0,1])
        
        im0 = axs[1, 0].imshow(ln_s.T, cmap=cmap, aspect='auto', vmin=0, vmax=300)
        fig.colorbar(im0, ax=axs[1,0])
        
        im1 = axs[1, 1].imshow(ln_w.T, cmap=cmap, aspect='auto', vmin=0, vmax=170)
        fig.colorbar(im1, ax=axs[1,1])
        
        im0 = axs[2, 0].imshow(nsi_s.T, cmap=cmap, aspect='auto', vmin=0, vmax=300)
        fig.colorbar(im0, ax=axs[2,0])
        
        im1 = axs[2, 1].imshow(nsi_w.T, cmap=cmap, aspect='auto', vmin=0, vmax=170)
        fig.colorbar(im1, ax=axs[2,1])
        
        
        # SETTINGS
        axs[0,0].set_title('ORN strong')
        axs[0,1].set_title('ORN weak')

        for id_r in range(rs):
            axs[id_r,0].set_ylabel('input (a.u.)', fontsize= label_fs)
        
        axs[0,1].text(62.5, 1.5, 'ctrl', fontsize=label_fs)
        axs[1,1].text(62.5, 1.5, 'LN', fontsize=label_fs)
        axs[2,1].text(62.5, 1.5, 'NSI', fontsize=label_fs)
        
        
        for c_id in range(cs):
            axs[2, c_id].set_xlabel('ratio (unitless)', fontsize= label_fs)
            
            for r_id in range(rs):
                axs[r_id,c_id].set_xticks(np.linspace(1, len(conc_ratios), 15))
                axs[r_id,c_id].set_xticklabels(ratio2plot)
                axs[r_id,c_id].set_yticks(range(len(conc2plot)))
                axs[r_id,c_id].set_yticklabels(conc2plot)
            
        for c_id in range(cs):
            for r_id in range(rs):
                ll, bb, ww, hh = axs[r_id, c_id].get_position().bounds
                axs[r_id, c_id].set_position([ll, bb, ww, hh])
                
    
        if fig_save:
            fig.savefig(fld_output+  '/ORN_delays0_dur%d'%duration+'.png')

    plt.show()      



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
    axs.bar(ptns-width, avg_ratio_peak_noin, width=width, color=pink, 
            yerr=avg_ratio_peak_noin_std/np.sqrt(n_ratios*n_concs), 
            label='ctrl', )
    axs.bar(ptns, avg_ratio_peak_ln, width=width, color=orange, 
            yerr=avg_ratio_peak_ln_std/np.sqrt(n_ratios*n_concs), 
            label='LN', )
    axs.bar(ptns+width, avg_ratio_peak_nsi, width=width, color=cyan, 
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
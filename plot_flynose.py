#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:30:03 2020

plot_flynose.py

@author: mario
"""


import numpy as np
#import scipy.stats as spst
#from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#import timeit
#from os import getcwd as pwd
#from scipy.integrate import quad

import pickle        
from os import path
from os import mkdir
from shutil import copyfile
import matplotlib as mpl

def flynose_plot(all_data_tmp, params2an, id_c):

    t_on =500
    num_orns_glo = 40
    num_pns_glo = 5
    num_lns_glo = 3
    
    t = all_data_tmp[1]
    u_od =all_data_tmp[2]
    orn_sdf_norm = all_data_tmp[3]
    orn_sdf_time = all_data_tmp[4]
    
    pn_sdf_norm = all_data_tmp[5]
    pn_sdf_time = all_data_tmp[6]
    ln_sdf_norm = all_data_tmp[7]
    ln_sdf_time = all_data_tmp[8]    
    
    ax_conc.plot(t-t_on, 100*u_od[:,0], color=greenmap.to_rgba(id_c + 1), linewidth=lw+2, 
                  label='glom : '+'%d'%(1))
    
    ax_orn.plot(orn_sdf_time-t_on, np.mean(orn_sdf_norm[:,:num_orns_glo], axis=1), 
                 color=greenmap.to_rgba(id_c + 1), linewidth=lw+1,label='sdf glo 1')
    
    ax_pn.plot(pn_sdf_time-t_on, np.mean(pn_sdf_norm[:,:num_pns_glo], axis=1), 
               color=greenmap.to_rgba(id_c + 1), linewidth=lw+1, label='PN, glo 1')
        
    ll = 0
    ax_ln.plot(ln_sdf_time-t_on, ln_sdf_norm[:,ll], 
        color=greenmap.to_rgba(id_c + 1), linewidth=lw+1, label='LN, glo 2')
    
    if params2an[3]>0:
        ax_conc.plot(t-t_on, 100*u_od[:,1], '--',color=purplemap.to_rgba(id_c + 1), 
             linewidth=lw+1, label='glom : '+'%d'%(2))
        ax_orn.plot(orn_sdf_time-t_on, np.mean(orn_sdf_norm[:,num_orns_glo:], axis=1), 
                 '--',color=purplemap.to_rgba(id_c + 1), linewidth=lw,label='sdf glo 2')
        ax_pn.plot(pn_sdf_time-t_on, np.mean(pn_sdf_norm[:,num_pns_glo:], axis=1), 
               '--', color=purplemap.to_rgba(id_c + 1), linewidth=lw, label='PN, glo 2')
        ll = num_lns_glo
        ax_ln.plot(ln_sdf_time-t_on, ln_sdf_norm[:,ll], '--',
                   color=purplemap.to_rgba(id_c + 1), linewidth=lw, label='LN, glo 1')
    
    ax_conc.spines['right'].set_color('none')
    ax_conc.spines['top'].set_color('none')
    ax_orn.spines['right'].set_color('none')
    ax_orn.spines['top'].set_color('none')
    ax_pn.spines['right'].set_color('none')
    ax_pn.spines['top'].set_color('none')
    ax_ln.spines['right'].set_color('none')
    ax_ln.spines['top'].set_color('none')
        

def olsen_orn_pn(nu_orn, sigma, nu_max):
    nu_pn = nu_max * np.power(nu_orn, 1.5)/(np.power(nu_orn, 1.5) + np.power(sigma,1.5))
    return nu_pn

    
def olsen2010_data(all_data_tmp, ):

    pts_ms  = 5
    t_on    =500
    t_off   = 1000
    stim_on  = t_on*pts_ms 
    stim_off = t_off*pts_ms 
    num_orns_glo = 40
    num_pns_glo = 5
    num_lns_glo = 5
    u_od = all_data_tmp[2]
    orn_sdf_norm = all_data_tmp[3]
    orn_sdf_time = all_data_tmp[4]
    
    pn_sdf_norm = all_data_tmp[5]
    pn_sdf_time = all_data_tmp[6]
    ln_sdf_norm = all_data_tmp[7]
    ln_sdf_time = all_data_tmp[8]
    
#pickle.dump([params2an, t, u_od, orn_sdf_norm, orn_sdf_time, 
#                 pn_sdf_norm, pn_sdf_time, 
#                 ln_sdf_norm, ln_sdf_time, 
#                 params2an_names, output_names], f)    
#    orn_id_stim = np.flatnonzero((orn_sdf_time>t_on) & (orn_sdf_time<t_off))
    pn_id_stim = np.flatnonzero((pn_sdf_time>t_on) & (pn_sdf_time<t_off))
    ln_id_stim = np.flatnonzero((ln_sdf_time>t_on) & (ln_sdf_time<t_off))
    nu_orn = np.mean(orn_sdf_norm[pn_id_stim, :num_orns_glo])
    nu_pn = np.mean(pn_sdf_norm[pn_id_stim, :num_pns_glo])
    nu_ln = np.mean(ln_sdf_norm[ln_id_stim, :num_lns_glo])
    conc = np.mean(u_od[stim_on:stim_off, 0], axis=0)
    
    nu_orn_err = np.std(orn_sdf_norm[pn_id_stim, :num_orns_glo])/np.sqrt(num_orns_glo)
    nu_pn_err = np.std(pn_sdf_norm[pn_id_stim, :num_pns_glo])/np.sqrt(num_pns_glo)
    nu_ln_err = np.std(ln_sdf_norm[ln_id_stim, :num_lns_glo])/np.sqrt(num_lns_glo)
#    conc = np.mean(u_od[stim_on:stim_off, 0], axis=0)
    
    out_olsen = np.zeros((7))
    out_olsen[0] = conc
    out_olsen[1] = nu_orn
    out_olsen[2] = nu_pn
    out_olsen[3] = nu_ln
    out_olsen[4] = nu_orn_err
    out_olsen[5] = nu_pn_err
    out_olsen[6] = nu_ln_err
    return [out_olsen] 

# *****************************************************************
# STANDARD FIGURE PARAMS
lw = 4
fs = 13
plt.rc('text', usetex=True)  # laTex in the polot
#plt.rc('font', family='serif')
fig_position = 1300,10
title_fs = 20 # font size of ticks
label_fs = 20 # font size of labels
legend_fs = 12

black   = 'xkcd:black'
blue    = 'xkcd:blue'
red     = 'xkcd:red'
green   = 'xkcd:green'
purple  = 'xkcd:purple'
orange  = 'xkcd:orange'
#cmap    = plt.get_cmap('rainbow')

#*****************************************************
# FIG: Olsen-Wilson 2010
peaks = np.linspace(0, 5,11)
stim_dur = 50#20#500
    

n_lines = np.size(peaks)
c = np.arange(1, n_lines + 1)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
greenmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Greens)
purplemap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Purples)
#greenmap.set_array([])

fig_save = 1
data_save = True
# *****************************************************************


for stim_seed in [0]:
    #fig_ow_tc_name       = 'ORN-PN_Olsen2010_timecourse.png'
    fld_analysis = '../Olsen2010'
    #fld_analysis = '../NSI_analysis/triangle_stim'
    copyfile('flynose.py', fld_analysis+'/flynose.py') 
    
    inh_conds = ['noin'] #['nsi', 'ln', 'noin'] #
    stim_type = 'ss' # 'ss'   # 'rs'   #  'pl'  # 'ts'
    delay2an = 0
    peak_ratio = 1
    b_max = [3] # 3, 50, 150
    w_max = [3] # 3, 50, 150
    rho = [0] #[0, 1, 3, 5]: 
    for inh_cond in inh_conds:
        
        #******************************************************
        # FIGURE ORN, PN, LN
        t2plot = -50, 800 # 000-t_on, t2simulate-t_on
        rs = 4 # number of rows
        cs = 1 # number of cols
        
        fig_pn = plt.figure(figsize=[7, 7])
        #        fig_pn.canvas.manager.window.wm_geometry("+%d+%d" % fig_position )
        #        fig_pn.tight_layout()
        ax_conc = plt.subplot(rs, cs, 1)
        ax_orn = plt.subplot(rs, cs, 1+cs)
        ax_pn = plt.subplot(rs, cs, 1+cs*2)
        ax_ln = plt.subplot(rs, cs, 1+cs*3)
        
        ax_conc.set_xlim(t2plot)
        ax_orn.set_xlim(t2plot)
        ax_pn.set_xlim(t2plot)
        ax_ln.set_xlim(t2plot)
        
        ax_orn.set_ylim((0, 250))
        ax_pn.set_ylim((0, 230))
        ax_ln.set_ylim((0, 250))
    
        ax_conc.tick_params(axis='both', labelsize=label_fs)
        ax_orn.tick_params(axis='both', labelsize=label_fs)
        ax_pn.tick_params(axis='both', labelsize=label_fs)
        ax_ln.tick_params(axis='both', labelsize=label_fs)
        
        ax_conc.set_xticklabels('')
        ax_orn.set_xticklabels('')
        ax_pn.set_xticklabels('')
        
        ax_conc.set_ylabel('Input ORN ', fontsize=label_fs)
        ax_orn.set_ylabel(r' ORN  (Hz)', fontsize=label_fs)
        ax_pn.set_ylabel(r' PN  (Hz)', fontsize=label_fs)
        ax_ln.set_ylabel(r' LN  (Hz)', fontsize=label_fs)
        ax_ln.set_xlabel('Time  (ms)', fontsize=label_fs)
        
        #******************************************************
        conc    = np.zeros_like(peaks)
        nu_orn  = np.zeros_like(peaks)
        nu_pn   = np.zeros_like(peaks)
        nu_ln   = np.zeros_like(peaks)
        nu_orn_err  = np.zeros_like(peaks)
        nu_pn_err   = np.zeros_like(peaks)
        nu_ln_err   = np.zeros_like(peaks)
        
        id_c = 0
        for peak in peaks:
            if inh_cond == 'nsi':
                params2an = [.3, .0, stim_dur, delay2an, peak, 
                             peak_ratio, rho, stim_type,w_max,b_max]
            elif inh_cond == 'noin':
                params2an = [0, 0, stim_dur, delay2an, peak, 
                             peak_ratio, rho, stim_type,w_max,b_max]
            elif inh_cond == 'ln':
                params2an = [.0, .15, stim_dur, delay2an, peak, 
                             peak_ratio, rho, stim_type,w_max,b_max]
            
            print(params2an)
            
            # LOAD DATA
            tmp_name_data = ['_stim_' + params2an[7] +
                '_nsi_%.1f'%(params2an[0]) +
                '_lnspH_%.2f'%(params2an[1]) +
                '_dur2an_%d'%(params2an[2]) +
                '_delays2an_%d'%(params2an[3]) +
                '_peak_%.2f'%(params2an[4]) +
                '_peakratio_%.1f'%(params2an[5]) + # 
                '.pickle'] #'_rho_%.1f'%(params2an[6]) +  
            all_name_data = ['/data/ORNALrate' + tmp_name_data[0]]
            
            # ORN_data output_names = ['t', 'u_od', 'orn_sdf_norm', 'orn_sdf_time', 
            #                       'pn_sdf_norm', 'pn_sdf_time', 
            #                       'ln_sdf_norm', 'ln_sdf_time', ]
            # AL_data output_names = ['pn_sdf_norm', 'pn_sdf_time', 
            #                        'ln_sdf_norm', 'ln_sdf_time', ]
            all_data_tmp = pickle.load(open(fld_analysis+ all_name_data[0],  "rb" ))
            
            
            [out_olsen] = olsen2010_data(all_data_tmp, )
            conc[id_c] = out_olsen[0]
            nu_orn[id_c] = out_olsen[1]
            nu_pn[id_c] = out_olsen[2]
            nu_ln[id_c] = out_olsen[3]
            nu_orn_err[id_c] = out_olsen[4]
            nu_pn_err[id_c] = out_olsen[5]
            nu_ln_err[id_c] = out_olsen[6]
            flynose_plot(all_data_tmp, params2an, id_c)
            id_c +=1
        if fig_save:
            print('saving figure in '+fld_analysis)
            fig_ow_tc_name = 'ORN-PN_Olsen2010_timecourse_dur_%d.png'%params2an[2]
            fig_pn.savefig(fld_analysis+  '/images/' +fig_ow_tc_name)
                           
#                           ORNPNLN_steps_' +
#                    'nsi_%.1f'%(params2an[0]) +
#                    '_lnspH_%.1f'%(params2an[1]) +
#                    '_dur2an_%d'%(params2an[2]) +
#                    '_delay2an_%d'%(params2an[3]) +
#                    '_peak_%.1f'%(params2an[4]) +
#                    '_peakratio_%.1f'%(params2an[5]) +
#                    '.png')

        
        #%% *********************************************************************
        # FIGURE Olsen 2010: ORN vs PN during step stimulus

        
        fig_ow_name = 'ORN-PN_Olsen2010_dur_%d.png'%params2an[2]

        # Constrain the optimization region
        popt, pcov = curve_fit(olsen_orn_pn, 
                               nu_orn, nu_pn, bounds=(0,[250, 300])) # , bounds=(0, [3., 1., 0.5])
    #    popt_peak, pcov = curve_fit(olsen_orn_pn, 
    #                                nu_orn_stim_peak, nu_pn_stim_peak, bounds=(0,[250, 300])) # , bounds=(0, [3., 1., 0.5])
        
        rs = 2
        cs = 1
        fig3, axs = plt.subplots(rs,cs, figsize=(7,8), )
        
        plt.rc('text', usetex=True)
        
        axs[0].errorbar(nu_orn, nu_pn, yerr=nu_pn_err, linewidth=lw, label='simulation')
        axs[0].plot(nu_orn, olsen_orn_pn(nu_orn, *popt), '--', linewidth=lw, 
                label=r'fit: $\sigma$=%5.0f, $\nu_{max}$=%5.0f' % tuple(popt))
        
        axs[0].set_ylabel(r'PN (Hz)', fontsize=label_fs)
        axs[0].set_xlabel(r'ORN (Hz)', fontsize=label_fs)
        
        axs[1].errorbar(conc, nu_orn, yerr=nu_orn_err, linewidth=lw, 
           markersize=15, label='ORNs ')
        axs[1].errorbar(conc, nu_pn, yerr=nu_pn_err, linewidth=lw, 
           markersize=15, label='PNs ')
        axs[1].errorbar(conc, nu_ln, yerr=nu_ln_err, linewidth=lw, 
           markersize=15, label='LNs ')
        axs[1].legend(loc=0, fontsize=legend_fs)

        axs[1].set_ylabel(r'Firing rates (Hz)', fontsize=label_fs)

        axs[1].set_xlabel(r'concentration [au]', fontsize=label_fs)
        axs[1].legend(loc=0, fontsize=legend_fs)
 #%%       
    #    axs[1,0].plot(nu_orn_stim_peak, nu_pn_stim_peak, '*', label='simulation')
    #    axs[1,0].plot(nu_orn_stim_peak, olsen_orn_pn(nu_orn_stim_peak, *popt_peak), '--', 
    #            label=r'fit: $\sigma$=%5.0f, $\nu_{max}$=%5.0f' % tuple(popt_peak))
        
    #        axs[1,0].set_ylabel(r'$\nu \, $ PN peak (Hz)', fontsize=label_fs)
    #        axs[1,0].set_xlabel(r'$\nu \,$ ORN peak (Hz)', fontsize=label_fs)
       
    #    axs[1,1].plot(conc2an, nu_orn_stim_peak, '*', label='ORN ')
    #    axs[1,1].plot(conc2an, nu_pn_stim_peak, 'o', label='PN ')
        
        
        for j in [0,1]:
            axs[j].tick_params(axis='both', labelsize=label_fs)
            axs[j].spines['right'].set_color('none')
            axs[j].spines['top'].set_color('none')
        
        if fig_save:
            fig3.savefig(fld_analysis+  '/images/'+fig_ow_name)
    
        print('')


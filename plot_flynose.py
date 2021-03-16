#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:30:03 2020

plot_flynose.py

@author: mario
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import pickle        
from shutil import copyfile
import matplotlib as mpl


# *****************************************************************
# STANDARD FIGURE PARAMS
lw = 4
fs = 13
plt.rc('text', usetex=True)  # laTex in the polot
#plt.rc('font', family='serif')
fig_position = 1300,10
title_fs = 20 # font size of ticks
label_fs = 20 # font size of labels
panel_fs = 30 # font size of panel' letters
legend_fs = 12
ticks_fs = label_fs - 3

black   = 'xkcd:black'
blue    = 'xkcd:blue'
red     = 'xkcd:red'
green   = 'xkcd:green'
purple  = 'xkcd:purple'
orange  = 'xkcd:orange'
#cmap    = plt.get_cmap('rainbow')

def martelli_plot(all_data_tmp, analysis_par, id_c):
    
    num_orns_glo = 40
    t = all_data_tmp[1]
    t2simulate = t[-1]
    u_od =all_data_tmp[2]
    orn_sdf_norm = all_data_tmp[3]
    orn_sdf_time = all_data_tmp[4]
    
    orn2plot = np.mean(orn_sdf_norm[:,:num_orns_glo], axis=1)
    st_name = analysis_par[7]
    
    
    # SETTINGS
    if len(st_name) == 2:
        print('normalized to the peak') 
        orn2plot = orn2plot/np.max(orn2plot)
        panel_letters = ['e', 'f']
    else:
        
        st_name = st_name[:-1]
        print(st_name)
        if st_name == 'parabola_':
            panel_letters = ['e', 'f']
        elif st_name == 'step_':
            panel_letters = ['a', 'b']
        elif st_name == 'ramp_':
            panel_letters = ['c', 'd']
    
    
    ax_conc_m.set_xlim(t2plot)
    ax_orn_m.set_xlim(t2plot)

    if id_c==0:
        ax_conc_m.tick_params(axis='both', labelsize=ticks_fs)
        ax_orn_m.tick_params(axis='both', labelsize=ticks_fs)
        
        ax_conc_m.set_xticklabels('')
        
        if (len(st_name) == 2):
            ax_conc_m.set_ylabel('Input (a.u.)', fontsize=label_fs)
            ax_orn_m.set_ylabel(' Norm. firing rates \n (unitless)', fontsize=label_fs)
            
        if (st_name == 'step_'):
            ax_conc_m.set_ylabel('Input (a.u.)', fontsize=label_fs)
            ax_orn_m.set_ylabel(' Firing rates \n (unitless)', fontsize=label_fs)
            
        if len(st_name) > 2:
            ax_conc_m.set_title(st_name[:-1], fontsize = title_fs+10)
            
            ax_conc_m.set_xticks(np.linspace(0, t2simulate, 8))
            ax_conc_m.set_xticklabels('')
            ax_orn_m.set_xticks(np.linspace(0, t2simulate, 7))
            ax_orn_m.set_xticklabels(['0', '', '', '1500', '', '', '3000'])
            
            ax_conc_m.set_yticks(np.linspace(0,400, 5))
            ax_orn_m.set_yticks(np.linspace(0,250, 6))
            
            ax_conc_m.set_ylim((-5, 400))
            ax_orn_m.set_ylim((0, 250))            
            
            ax_conc_m.grid(True, which='both',lw=1, ls=':')
            ax_orn_m.grid(True, which='both',lw=1, ls=':')
        else:
            ax_conc_m.set_xticks(np.linspace(0, t2simulate, 5))
            ax_orn_m.set_xticks(np.linspace(0, t2simulate, 5))
            
            ax_conc_m.spines['right'].set_color('none')
            ax_conc_m.spines['top'].set_color('none')
            ax_orn_m.spines['right'].set_color('none')
            ax_orn_m.spines['top'].set_color('none')
        ax_orn_m.set_xlabel('Time  (ms)', fontsize=label_fs)

        ll, bb, ww, hh = ax_conc_m.get_position().bounds
        ax_conc_m.set_position([ll+.075, bb, ww, hh])
        ll, bb, ww, hh = ax_orn_m.get_position().bounds
        ax_orn_m.set_position([ll+.075, bb, ww, hh])
    
        ax_conc_m.text(-.15, 1.1, panel_letters[0], transform=ax_conc_m.transAxes,
              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
        ax_orn_m.text(-.15, 1.1, panel_letters[1], transform=ax_orn_m.transAxes, 
              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            
    # PLOT
    id_col = id_c + 3
    ax_conc_m.plot(t-t_on, 100*u_od[:,0], color=greenmap.to_rgba(id_col), linewidth=lw, 
              label='glom : '+'%d'%(1))
    
    ax_orn_m.plot(orn_sdf_time-t_on, orn2plot, 
             color=greenmap.to_rgba(id_col), linewidth=lw-1,label='sdf glo 1')
    
    

def orn_al_settings(ax_ornal):

    # ****************************************************
    # FIGURE SETTING
    
    dx = 0.075
    dy = np.linspace(0.075, 0, 4, )
    for id_ax in range(4):
        ll, bb, ww, hh = ax_ornal[id_ax].get_position().bounds
        ax_ornal[id_ax].set_position([ll+dx, bb+dy[id_ax], ww-dx, hh])

    
    ax_ornal[0].set_xlim(t2plot)
    ax_ornal[1].set_xlim(t2plot)
    ax_ornal[2].set_xlim(t2plot)
    ax_ornal[3].set_xlim(t2plot)
    
    ax_ornal[1].set_ylim((0, 280))
    ax_ornal[2].set_ylim((0, 250))
    ax_ornal[3].set_ylim((0, 250))

    ax_ornal[0].tick_params(axis='both', labelsize=label_fs)
    ax_ornal[1].tick_params(axis='both', labelsize=label_fs)
    ax_ornal[2].tick_params(axis='both', labelsize=label_fs)
    ax_ornal[3].tick_params(axis='both', labelsize=label_fs)
    
    ax_ornal[0].set_xticklabels('')
    ax_ornal[1].set_xticklabels('')
    ax_ornal[2].set_xticklabels('')
    
    ax_ornal[0].set_ylabel('Input ORN ', fontsize=label_fs)
    ax_ornal[1].set_ylabel(r' ORN  (Hz)', fontsize=label_fs)
    ax_ornal[2].set_ylabel(r' PN  (Hz)', fontsize=label_fs)
    ax_ornal[3].set_ylabel(r' LN  (Hz)', fontsize=label_fs)
    ax_ornal[3].set_xlabel('Time  (ms)', fontsize=label_fs)
    ax_ornal[0].text(-.2, 1.25, 'a', transform=ax_ornal[0].transAxes, 
                 fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    ax_ornal[1].text(-.2, 1.25, 'b', transform=ax_ornal[1].transAxes, 
                 fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    ax_ornal[2].text(-.2, 1.25, 'c', transform=ax_ornal[2].transAxes, 
                 fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    ax_ornal[3].text(-.2, 1.25, 'd', transform=ax_ornal[3].transAxes, 
                 fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    
    ax_ornal[0].spines['right'].set_color('none')
    ax_ornal[0].spines['top'].set_color('none')
    ax_ornal[1].spines['right'].set_color('none')
    ax_ornal[1].spines['top'].set_color('none')
    ax_ornal[2].spines['right'].set_color('none')
    ax_ornal[2].spines['top'].set_color('none')
    ax_ornal[3].spines['right'].set_color('none')
    ax_ornal[3].spines['top'].set_color('none')
        
def orn_al_plot(all_data_tmp, analysis_par, id_c):

    t_zero =300
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
    
    # ****************************************************
    # PLOTTING DATA
    id_col = id_c + 3
    ax_ornal[0].plot(t-t_zero, 100*u_od[:,0], color=greenmap.to_rgba(id_col), linewidth=lw, 
                  label='glom : '+'%d'%(1))
    
    ax_ornal[1].plot(orn_sdf_time-t_zero, np.mean(orn_sdf_norm[:,:num_orns_glo], axis=1), 
                 color=greenmap.to_rgba(id_col), linewidth=lw-1,label='sdf glo 1')
    
    ax_ornal[2].plot(pn_sdf_time-t_zero, np.mean(pn_sdf_norm[:,:num_pns_glo], axis=1), 
               color=greenmap.to_rgba(id_col), linewidth=lw-1, label='PN, glo 1')
        
    ll = 0
    ax_ornal[3].plot(ln_sdf_time-t_zero, ln_sdf_norm[:,ll], 
        color=greenmap.to_rgba(id_col), linewidth=lw-1, label='LN, glo 2')
    
    if analysis_par[3]>0:
        ax_ornal[0].plot(t-t_zero, 100*u_od[:,1], '--',color=purplemap.to_rgba(id_col), 
             linewidth=lw-1, label='glom : '+'%d'%(2))
        ax_ornal[1].plot(orn_sdf_time-t_zero, np.mean(orn_sdf_norm[:,num_orns_glo:], axis=1), 
                 '--',color=purplemap.to_rgba(id_col), linewidth=lw-2,label='sdf glo 2')
        ax_ornal[2].plot(pn_sdf_time-t_zero, np.mean(pn_sdf_norm[:,num_pns_glo:], axis=1), 
               '--', color=purplemap.to_rgba(id_col), linewidth=lw-2, label='PN, glo 2')
        ll = num_lns_glo
        ax_ornal[3].plot(ln_sdf_time-t_zero, ln_sdf_norm[:,ll], '--',
                   color=purplemap.to_rgba(id_col), linewidth=lw-2, label='LN, glo 1')
    
def olsen_orn_pn(nu_orn, sigma, nu_max):
    nu_pn = nu_max * np.power(nu_orn, 1.5)/(np.power(nu_orn, 1.5) + np.power(sigma,1.5))
    return nu_pn
       
def olsen2010_data(all_data_tmp, analysis_par):
    pts_ms  =   5
#    t_on    =   500
    t_off   =   100+t_on+analysis_par[2]
    num_orns_glo = 40
    num_pns_glo = 5
    num_lns_glo = 5
    
    stim_on  = t_on*pts_ms 
    stim_off = t_off*pts_ms 
#    pickle.dump([[params2an, t, u_od, 
#     orn_sdf_norm, orn_sdf_time, 
#     pn_sdf_norm, pn_sdf_time, 
#     ln_sdf_norm, ln_sdf_time, 
#     params2an_names, output_names], f)
    u_od = all_data_tmp[2]
    orn_sdf_norm = all_data_tmp[3]
    orn_sdf_time = all_data_tmp[4]
    
    pn_sdf_norm = all_data_tmp[5]
    pn_sdf_time = all_data_tmp[6]
    ln_sdf_norm = all_data_tmp[7]
    ln_sdf_time = all_data_tmp[8]
    
    orn_id_stim = np.flatnonzero((orn_sdf_time>t_on) & (orn_sdf_time<t_off))
    pn_id_stim = np.flatnonzero((pn_sdf_time>t_on) & (pn_sdf_time<t_off))
    ln_id_stim = np.flatnonzero((ln_sdf_time>t_on) & (ln_sdf_time<t_off))
    nu_orn = np.mean(orn_sdf_norm[orn_id_stim, :num_orns_glo])
    nu_pn = np.mean(pn_sdf_norm[pn_id_stim, :])
    nu_ln = np.mean(ln_sdf_norm[ln_id_stim, :])
    conc = np.mean(u_od[stim_on:stim_off, 0], axis=0)
    
    nu_orn_err = np.std(orn_sdf_norm[orn_id_stim, :num_orns_glo])/np.sqrt(num_orns_glo)
    nu_pn_err = np.std(pn_sdf_norm[pn_id_stim, :])/np.sqrt(2*num_pns_glo)
    nu_ln_err = np.std(ln_sdf_norm[ln_id_stim, :])/np.sqrt(2*num_lns_glo)
#    conc = np.mean(u_od[stim_on:stim_off, 0], axis=0)
    
    out_olsen = np.zeros((7))
    out_olsen[0] = conc
    out_olsen[1] = nu_orn
    out_olsen[2] = nu_pn
    out_olsen[3] = nu_ln
    out_olsen[4] = nu_orn_err
    out_olsen[5] = nu_pn_err
    out_olsen[6] = nu_ln_err
    
    return out_olsen
    
#*****************************************************
# FIG.Laz
# t_on = 0
# peaks       = [1, 2, 3]#np.linspace(0, 7, 11)
# stim_dur    = 1000  # 50 # 100 #500
# inh_conds   = ['nsi'] #['nsi', 'ln', 'noin'] #
# stim_type_tmp = 'step_' #'ramp_' #'step_' # 'parabola_' #
# stim_type   = stim_type_tmp
# delay2an    = 0
# peak_ratio  = 1
# b_max       = np.nan # 3, 50, 150
# w_max       = np.nan # 3, 50, 150
# rho         = np.nan #[0, 1, 3, 5]: 
# alpha_ln    = 13.3 # alpha_ln  = 0.4
# nsi_str     = 0.3        
# n_lines     = np.size(peaks)
# fld_analysis = 'NSI_analysis/lazar_sim/'

# martelli_fig= 1 # Fig.laz
# orn_al_fig  = 0
# olsen_fig   = 0

# fig_martelli_name   = stim_type_tmp
# fig_orn_al_name = ''
# fig_olsen_fit_name = ''

# fig_save    = 0




#*****************************************************
# FIG: Olsen-Wilson 2010 / Martelli 2013
peaks       = np.linspace(0.7, 7, 10)
stim_dur    = 500  # 50 # 100 #500
t_on        = 300  #  500
inh_conds   = ['noin'] #['nsi', 'ln', 'noin'] #
stim_type   = 'ss' # 'ss'   # 'ts'
delay2an    = 0
peak_ratio  = 1
b_max       = 3 # 3, 50, 150
w_max       = 3 # 3, 50, 150
rho         = 0 #[0, 1, 3, 5]: 
alpha_ln    = 0.6
nsi_str     = 0.3        
n_lines     = np.size(peaks)
fld_analysis        = 'Olsen2010_Martelli2013/data'
fig_orn_al_name     = '/../images/' +'ORN-PN_Olsen2010_timecourse_dur_%d'%stim_dur
fig_olsen_fit_name  = '/../images/' +'ORN-PN_Olsen2010_dur_%d'%stim_dur
fig_martelli_name   = '/../images/' +'ORN-Martelli2013_dur_%d'%stim_dur
orn_al_fig      = 1 # PNORNactivity, time course side
olsen_fig       = 1 # PNORNactivity, Olsen side
martelli_fig    = 1 # ORN_response, Martelli side

fig_save        = 0
data_save       = 0

stim_type_tmp = '' #'ramp_' #'step_' # 'parabola_' #


c = np.arange(1, n_lines + 4)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
greenmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Greens)
purplemap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Purples)
#greenmap.set_array([])

# *****************************************************************


for stim_seed in [0]:
    
    for inh_cond in inh_conds:
        
        #******************************************************
        # FIGURE Olsen 2010
        if orn_al_fig:
            t2plot = -200, stim_dur+300
            rs = 4 # number of rows
            cs = 1 # number of cols
            
            fig_pn, ax_ornal = plt.subplots(rs, cs, figsize=[8, 8])
            orn_al_settings(ax_ornal)
            
        #******************************************************
        # FIGURE Martelli 2013
        if martelli_fig:
            t2plot = -200,stim_dur+300
            if len(stim_type) > 2:
                t2plot = 0, 3500
            rs = 2 # number of rows
            cs = 1 # number of cols
            
            fig_pn_m = plt.figure(figsize=[5.71, 8])
            ax_conc_m = plt.subplot(rs, cs, 1)
            ax_orn_m = plt.subplot(rs, cs, 1+cs)
        
        #******************************************************
        conc    = np.zeros_like(peaks)
        conc_th = np.zeros_like(peaks)
        nu_orn  = np.zeros_like(peaks)
        nu_pn   = np.zeros_like(peaks)
        nu_ln   = np.zeros_like(peaks)
        nu_orn_err  = np.zeros_like(peaks)
        nu_pn_err   = np.zeros_like(peaks)
        nu_ln_err   = np.zeros_like(peaks)
        
        for id_c, peak in enumerate(peaks):
            if len(stim_type)>2:
                stim_type = stim_type_tmp+np.str(peak)
            if inh_cond == 'nsi':
                analysis_par = [nsi_str, .0, stim_dur, delay2an, peak, 
                             peak_ratio, rho, stim_type,w_max,b_max]
            elif inh_cond == 'noin':
                analysis_par = [0, 0, stim_dur, delay2an, peak, 
                             peak_ratio, rho, stim_type,w_max,b_max]
            elif inh_cond == 'ln':
                analysis_par = [.0, alpha_ln, stim_dur, delay2an, peak, 
                             peak_ratio, rho, stim_type,w_max,b_max]
                      
            # LOAD DATA
            if len(stim_type)==2:
                tmp_name_data = ['/ORNALrate_stim_' + analysis_par[7] +
                    '_nsi_%.1f'%(analysis_par[0]) +
                    '_ln_%.2f'%(analysis_par[1]) +
                    '_dur2an_%d'%(analysis_par[2]) +
                    '_delay2an_%d'%(analysis_par[3]) +
                    '_peak_%.1f'%(analysis_par[4]) +
                    '_peakratio_%.1f'%(analysis_par[5]) + # 
                    '.pickle'] #'_rho_%.1f'%(analysis_par[6]) +  
            else:
                tmp_name_data = ['/ORNrate_stim_' + analysis_par[7] + 
                                 '_nsi_%.1f'%(analysis_par[0]) +
                                 '_ln_%.2f'%(analysis_par[1]) +
                                '.pickle']
            all_name_data = tmp_name_data[0]
            
            all_data_tmp = pickle.load(open(fld_analysis+ all_name_data,  "rb" ))

            if orn_al_fig:
                orn_al_plot(all_data_tmp, analysis_par, id_c)
            if martelli_fig:
                martelli_plot(all_data_tmp, analysis_par, id_c)

            if olsen_fig:
                out_olsen = olsen2010_data(all_data_tmp, analysis_par)
                conc[id_c] = out_olsen[0]
                conc_th[id_c] = analysis_par[4]
                nu_orn[id_c] = out_olsen[1]
                nu_pn[id_c] = out_olsen[2]
                nu_ln[id_c] = out_olsen[3]
                nu_orn_err[id_c] = out_olsen[4]
                nu_pn_err[id_c] = out_olsen[5]
                nu_ln_err[id_c] = out_olsen[6]
                
        if fig_save:
            print('saving figure in '+fld_analysis)
            if orn_al_fig:
                fig_pn.savefig(fld_analysis+  fig_orn_al_name+'_'+inh_cond+'.png')
            if martelli_fig:
                fig_pn_m.savefig(fld_analysis+  fig_martelli_name+'_'+inh_cond+'.png')
        


        
        #%% *********************************************************************
        # FIGURE Olsen 2010: ORN vs PN during step stimulus

        if olsen_fig:
            
            # Constrain the optimization region
            popt, pcov = curve_fit(olsen_orn_pn, nu_orn, nu_pn, 
                        bounds=(0,[250, 300])) # , bounds=(0, [3., 1., 0.5])
            nuorn_fit = np.linspace(2, nu_orn[-1]*1.1, 100)
            
            rs = 2
            cs = 1
            fig3, axs = plt.subplots(rs,cs, figsize=(8,8), )
            
            plt.rc('text', usetex=True)
            
            axs[0].errorbar(nu_orn, nu_pn, yerr=nu_pn_err, label='simulation', fmt='o')
            axs[0].plot(nuorn_fit , olsen_orn_pn(nuorn_fit , *popt), '--', linewidth=lw, 
                    label=r'fit: $\sigma$=%5.0f, $\nu_{max}$=%5.0f' % tuple(popt))
            
            axs[0].set_ylabel(r'PN (Hz)', fontsize=label_fs)
            axs[0].set_xlabel(r'ORN (Hz)', fontsize=label_fs)
            
            axs[1].errorbar(conc_th+.05, nu_orn, yerr=nu_orn_err, linewidth=lw, 
               markersize=15, label='ORNs ')
            axs[1].errorbar(conc_th-.05, nu_pn, yerr=nu_pn_err, linewidth=lw, 
               markersize=15, label='PNs ')
            axs[1].errorbar(conc_th, nu_ln, yerr=nu_ln_err, linewidth=lw, 
               markersize=15, label='LNs ')
            axs[1].legend(loc='upper left', fontsize=legend_fs)
    
            axs[1].set_ylabel(r'Firing rates (Hz)', fontsize=label_fs)
    
            axs[1].set_xlabel(r'concentration [au]', fontsize=label_fs)
            axs[1].legend(loc=0, fontsize=legend_fs, frameon=False)
            
#            if stim_dur == 500:
            axs[0].text(-.2, 1.1, 'e', transform=axs[0].transAxes, 
                 fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            axs[1].text(-.2, 1.1, 'f', transform=axs[1].transAxes,
                 fontsize=panel_fs, fontweight='bold', va='top', ha='right')
        
            for j in [0,1]:
                axs[j].tick_params(axis='both', labelsize=label_fs)
                axs[j].spines['right'].set_color('none')
                axs[j].spines['top'].set_color('none')
            
            dx = 0.1
            ll, bb, ww, hh = axs[0].get_position().bounds
            axs[0].set_position([ll+dx, bb+.05, ww-dx, hh])
            ll, bb, ww, hh = axs[1].get_position().bounds
            axs[1].set_position([ll+dx, bb, ww-dx, hh])

            if fig_save:
                fig3.savefig(fld_analysis+  fig_olsen_fit_name+'_'+inh_cond+'.png')
        print('')


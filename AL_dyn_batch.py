#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:33:34 2020

AL_dyn_batch.py

This script is the raw model for flynose2.0. It runs ORNs_layer_dyn.py to 
generate ORN activity and then AL_dyn.py to run the AL dynamics

@author: mario
"""

#%% Setting parameters and define functions

import numpy as np
import timeit
from scipy.optimize import curve_fit
import pickle        
from os import path
import matplotlib.pyplot as plt
import matplotlib as mpl

import AL_dyn
import ORNs_layer_dyn
import plot_al_orn
import set_orn_al_params


# STANDARD FIGURE PARAMS 
lw = 2
fs = 13
plt.rc('text', usetex=True)  # laTex in the polot
#plt.rc('font', family='serif')
fig_size = [12, 12]
fig_position = 1300,10
title_fs = 20 # font size of ticks
label_fs = 20 # font size of labels
ticks_fs = label_fs - 3
panel_fs = 30 # font size of panel's letter
legend_fs = 12
black   = 'xkcd:black'
blue    = 'xkcd:blue'
red     = 'xkcd:red'
green   = 'xkcd:green'
purple  = 'xkcd:purple'
orange  = 'xkcd:orange'


cmap    = plt.get_cmap('rainbow')
recep_clrs = ['green','purple','cyan','red']
np.set_printoptions(precision=2)


def tictoc():
    return timeit.default_timer()


def fig_npeaks_settings(ax_ornal, params_tmp):
    """ ORN and PN Firing rates figure settings """
    
    dx = 0.075
    dy = np.linspace(0.075, 0, 4, )
    for id_ax in range(4):
        ll, bb, ww, hh = ax_ornal[id_ax].get_position().bounds
        ax_ornal[id_ax].set_position([ll+dx, bb+dy[id_ax], ww-dx, hh])

    stim_dur   =   params_tmp['stim_params']['stim_dur'][0]
    
    t2plot = -200, stim_dur+300+delay
            
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
        
def fig_npeaks(data_tmp, params_tmp, id_c):
    """ plot time course for ONR and PNS for a single concentration value """
    n_orns_recep = params_tmp['orn_layer_params'][0]['n_orns_recep']
    
    t           = data_tmp['t']
    u_od        = data_tmp['u_od']
    orn_sdf     = data_tmp['orn_sdf']
    orn_sdf_time = data_tmp['orn_sdf_time']
    pn_sdf = data_tmp['pn_sdf']
    pn_sdf_time = data_tmp['pn_sdf_time']
    ln_sdf = data_tmp['ln_sdf']
    ln_sdf_time = data_tmp['ln_sdf_time']
    
    t_zero    =   params_tmp['stim_params']['t_on'][0]    
    n_orns_recep = params_tmp['al_params']['n_orns_recep']
    n_pns_recep = params_tmp['al_params']['n_pns_recep']
    n_lns_recep = params_tmp['al_params']['n_lns_recep']
        
    # ****************************************************
    # PLOTTING DATA
    id_col = id_c + 3
    # Weak ORNs/PNs
    ax_ornal[0].plot(t-t_zero, 100*u_od[:,0], color=greenmap.to_rgba(id_col), 
                     linewidth=lw, label='glom : '+'%d'%(1))
    
    ax_ornal[1].plot(orn_sdf_time-t_zero, np.mean(orn_sdf[:,:n_orns_recep], axis=1), 
                 color=greenmap.to_rgba(id_col), linewidth=lw, label='sdf glo w')
    
    ax_ornal[2].plot(pn_sdf_time-t_zero, np.mean(pn_sdf[:,:n_pns_recep], axis=1), 
               color=greenmap.to_rgba(id_col), linewidth=lw, label='PN, glo w')
        
    
    ax_ornal[3].plot(ln_sdf_time-t_zero, np.mean(ln_sdf[:, :n_lns_recep], axis=1), 
                     color=greenmap.to_rgba(id_col), linewidth=lw, label='LN, glo w')
    
    # # Strong ORNs/PNs
    # ax_ornal[0].plot(t-t_zero, 100*u_od[:,1], color=purplemap.to_rgba(id_col), 
    #                   linewidth=lw, )#label='glom : '+'%d'%(2))
    
    # ax_ornal[1].plot(orn_sdf_time-t_zero, np.mean(orn_sdf[:, n_orns_recep:], axis=1), 
    #                   color=purplemap.to_rgba(id_col), linewidth=lw, )#label='sdf glo s')
    
    # ax_ornal[2].plot(pn_sdf_time-t_zero, np.mean(pn_sdf[:, n_pns_recep:], axis=1), 
    #                   color=purplemap.to_rgba(id_col), linewidth=lw, )#label='PN, glo s')
    # ax_ornal[3].plot(ln_sdf_time-t_zero, np.mean(ln_sdf[:, n_lns_recep:], axis=1), 
    #             color=purplemap.to_rgba(id_col), linewidth=lw, )#label='LN, glo w')
    
    
    # ax_ornal[3].legend()
    
def olsen_orn_pn(nu_orn, sigma, nu_max):
    """ fitting curve used by Olsen et al. 2010 for the relation between PN and ORN rates """
    nu_pn = nu_max * np.power(nu_orn, 1.5)/(np.power(nu_orn, 1.5) + np.power(sigma,1.5))
    return nu_pn
       
def olsen2010_data(data_tmp, params_tmp):
    pts_ms  =   params_tmp['stim_params']['pts_ms']

    t_on    =   params_tmp['stim_params']['t_on']
    stim_dur   =   params_tmp['stim_params']['stim_dur']
    t_off   = t_on + stim_dur
    
    n_orns_recep = params_tmp['al_params']['n_orns_recep']
    n_pns_recep = params_tmp['al_params']['n_pns_recep']
    n_lns_recep = params_tmp['al_params']['n_lns_recep']
    
    stim_on  = t_on*pts_ms 
    stim_off = t_off*pts_ms 
    
    u_od        = data_tmp['u_od']
    orn_sdf     = data_tmp['orn_sdf']
    orn_sdf_time = data_tmp['orn_sdf_time']
    pn_sdf      = data_tmp['pn_sdf']
    pn_sdf_time = data_tmp['pn_sdf_time']
    ln_sdf      = data_tmp['ln_sdf']
    ln_sdf_time = data_tmp['ln_sdf_time']
    
    # if stim_dur[0] == 500:
    orn_id_stim_s = np.flatnonzero((orn_sdf_time>t_on[1]) & (orn_sdf_time<t_off[1]))
    pn_id_stim_s = np.flatnonzero((pn_sdf_time>t_on[1]) & (pn_sdf_time<t_off[1]))
    ln_id_stim_s = np.flatnonzero((ln_sdf_time>t_on[1]) & (ln_sdf_time<t_off[1]))
    
    orn_id_stim_w = np.flatnonzero((orn_sdf_time>t_on[0]) & (orn_sdf_time<t_off[0]))
    pn_id_stim_w = np.flatnonzero((pn_sdf_time>t_on[0]) & (pn_sdf_time<t_off[0]))
    ln_id_stim_w = np.flatnonzero((ln_sdf_time>t_on[0]) & (ln_sdf_time<t_off[0]))
    # else:
    #     orn_id_stim_s = np.flatnonzero((orn_sdf_time>t_on[1]) & (orn_sdf_time<t_on[1]+time2analyse))
    #     pn_id_stim_s = np.flatnonzero((pn_sdf_time>t_on[1]) & (pn_sdf_time<t_on[1]+time2analyse))
    #     ln_id_stim_s = np.flatnonzero((ln_sdf_time>t_on[1]) & (ln_sdf_time<t_on[1]+time2analyse))
    #     orn_id_stim_w = np.flatnonzero((orn_sdf_time>t_on[0]) & (orn_sdf_time<t_on[0]+time2analyse))
    #     pn_id_stim_w = np.flatnonzero((pn_sdf_time>t_on[0]) & (pn_sdf_time<t_on[0]+time2analyse))
    #     ln_id_stim_w = np.flatnonzero((ln_sdf_time>t_on[0]) & (ln_sdf_time<t_on[0]+time2analyse))
        
        
    nu_orn_w = np.mean(orn_sdf[orn_id_stim_w, :n_orns_recep])
    nu_pn_w = np.mean(pn_sdf[pn_id_stim_w, :n_pns_recep])
    
    nu_orn_s = np.mean(orn_sdf[orn_id_stim_s, n_orns_recep:])
    nu_pn_s = np.mean(pn_sdf[pn_id_stim_s, n_pns_recep:])
    nu_ln_s = np.mean(ln_sdf[ln_id_stim_s, n_lns_recep:])
    nu_ln_w = np.mean(ln_sdf[ln_id_stim_w, :n_lns_recep])
    conc_s = np.mean(u_od[stim_on[1]:stim_off[1], 1], axis=0)
    conc_w = np.mean(u_od[stim_on[0]:stim_off[0], 0], axis=0)
    
    nu_orn_s_err = np.std(orn_sdf[orn_id_stim_s, :n_orns_recep])/np.sqrt(n_orns_recep)
    nu_orn_w_err = np.std(orn_sdf[orn_id_stim_w, n_orns_recep:])/np.sqrt(n_orns_recep)
    nu_pn_s_err = np.std(pn_sdf[pn_id_stim_s, :n_pns_recep])/np.sqrt(n_pns_recep)
    nu_pn_w_err = np.std(pn_sdf[pn_id_stim_w, n_pns_recep:])/np.sqrt(n_pns_recep)
    nu_ln_s_err = np.std(ln_sdf[ln_id_stim_s, :n_lns_recep])/np.sqrt(n_lns_recep)
    nu_ln_w_err = np.std(ln_sdf[ln_id_stim_w, n_lns_recep:])/np.sqrt(n_lns_recep)
    
    out_olsen = dict([
        ('conc_s', conc_s),
        ('conc_w', conc_w),
        ('nu_orn_s', nu_orn_s),
        ('nu_orn_w', nu_orn_w),
        ('nu_pn_s', nu_pn_s),
        ('nu_pn_w', nu_pn_w),
        ('nu_ln_s', nu_ln_s),
        ('nu_ln_w', nu_ln_w),
        ('nu_orn_s_err', nu_orn_s_err),
        ('nu_orn_w_err', nu_orn_w_err),
        ('nu_pn_s_err', nu_pn_s_err),
        ('nu_pn_w_err', nu_pn_w_err),
        ('nu_ln_s_err', nu_ln_s_err),
        ('nu_ln_w_err', nu_ln_w_err),
        ])
    return out_olsen
    
   
# %% LOAD PARAMS FROM A FILE
params_al_orn = set_orn_al_params.main(n_od=2,n_orn=2)

# fld_analysis = 'NSI_analysis/trials/' #Olsen2010
# name_data = 'params_al_orn.ini'
# params_al_orn = pickle.load(open(fld_analysis+ name_data,  "rb" ))

stim_params = params_al_orn['stim_params']
orn_layer_params= params_al_orn['orn_layer_params']
orn_params= params_al_orn['orn_params']
sdf_params= params_al_orn['sdf_params']
al_params= params_al_orn['al_params']
pn_ln_params= params_al_orn['pn_ln_params']

n_sens_type       = orn_layer_params.__len__()  # number of type of sensilla



# %%
# Stimulus parameters
delay                       = 0
t0                          = 1000
stim_name                   = ''
stim_params['pts_ms']       = 10


stim_params['t_on']         = np.array([t0, t0+delay])
stim_params['conc0']        = 1.85e-6    # fitted value: 2.85e-4

# PNs average activity during 500ms stimulation (see Olsen et al. 2010)
nu_pn_obs                   = [8, 75, 130, 150, ]
nu_orn_obs                  = [3, 21, 55, 125, ]

# nsi params
nsi_str                     = .6
alpha_ln                    = .6

# output params
data_save                   = 0

# time2analyse                = 50#200

# fig4 options
stim_durs                   = [500]      # [ms]
stim_params['stim_type']    = 'ss'      # 'ss'  # 'ts' # 'rs' # 'pl'
peak_ratios                 = np.linspace(1, 20, 1,) 
peaks                       = [1.85e-4, 3e-4, .8e-3, 1.5e-3, 1.5e-2, .1]#[1.85e-4, 3e-4, .8e-3, 3e-3,]#np.logspace(-3.3, -2.6, 4)
inh_conds                   = ['noin']  # ['nsi', 'noin', 'ln']


# figs/data flags
dataratio_save              = 0
al_orn_1r_fig               = 0     # single run figure 
npeaks_fig                  = 1     # multiple peaks PN and ORN time course 
olsen_fig                   = 1     # PN vs ORN activity, like Olsen 2010
figs_save                   = 0
fld_analysis                = 'NSI_analysis/Olsen2010/'
data_save                   = 0



tic = tictoc()

# OLSEN 2010 FIGURES 
n_lines     = np.size(peaks)

c = np.arange(1, n_lines + 4)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
greenmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Greens)
purplemap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Purples)


# FIGURE TIME COURSE Olsen 2010
if npeaks_fig:
    rs = 4 # number of rows
    cs = 1 # number of cols
    fig_pn, ax_ornal = plt.subplots(rs, cs, figsize=[8, 8])
    
if path.isdir(fld_analysis):
    print('Analysis fld: ' + fld_analysis)    
else:
    print('no fld analysis, please create one. thanks')

for stim_dur in stim_durs:
    stim_params['stim_dur']     = np.array([stim_dur, stim_dur])
    stim_params['t_tot']        = t0+delay+stim_dur+300

    for inh_cond in inh_conds:
        
        nu_pn_ratio = np.zeros((len(peaks), len(peak_ratios)))
        for id_r, peak_ratio in enumerate(peak_ratios):
            # INITIALIZE OUTPUT VARIABLES    
            conc_s    = np.zeros_like(peaks)
            conc_th = np.zeros_like(peaks)
            nu_orn_s = np.zeros_like(peaks)
            nu_pn_s  = np.zeros_like(peaks)
            nu_orn_w = np.zeros_like(peaks)
            nu_pn_w = np.zeros_like(peaks)
            
            nu_orn_s_err  = np.zeros_like(peaks)
            nu_pn_s_err   = np.zeros_like(peaks)
            nu_orn_w_err  = np.zeros_like(peaks)
            nu_pn_w_err   = np.zeros_like(peaks)
            
            nu_ln_s   = np.zeros_like(peaks)
            nu_ln_w     = np.zeros_like(peaks)
            nu_ln_s_err   = np.zeros_like(peaks)
            nu_ln_w_err   = np.zeros_like(peaks)
        
        
            tic = tictoc() 
                
            for id_c, peak in enumerate(peaks):
                if stim_params['stim_type'] == 'ext':
                    stim_params['stim_data_name'] = stim_params['stim_data_name'][:-1]+str(peak)
                    
                    print(stim_params['stim_data_name'])
                else:
                    stim_params['concs'] = np.array([peak, peak_ratio*peak])
                
                for sst in range(n_sens_type):
                    if inh_cond == 'nsi':
                        w_nsi = nsi_str    
                        orn_layer_params[sst]['w_nsi']  = nsi_str    
                        pn_ln_params['alpha_ln']        = 0
                    elif inh_cond == 'noin':
                        w_nsi = 0    
                        orn_layer_params[sst]['w_nsi']  = 0
                        pn_ln_params['alpha_ln']        = 0
                    elif inh_cond == 'ln':
                        w_nsi = 0    
                        orn_layer_params[sst]['w_nsi']  = 0    
                        pn_ln_params['alpha_ln']        = alpha_ln
                
                # SAVE SDF OF ORN  and PNs, LNs FIRING RATE and the params
                if stim_params['stim_type'] == 'ext':
                    name_data = 'AL_ORN_rate' +\
                            '_stim_' + stim_name + str(peak) +\
                            '_nsi_%.1f'%(orn_layer_params[0]['w_nsi']) + \
                            '_ln_%.1f'%(pn_ln_params['alpha_ln']) + \
                            '.pickle'
                else:
                    name_data = 'AL_ORN_rate' +\
                            '_stim_' + stim_params['stim_type'] +\
                            '_nsi_%.1f'%(orn_layer_params[0]['w_nsi']) +\
                            '_ln_%.1f'%(pn_ln_params['alpha_ln']) + \
                            '_dur2an_%d'%(stim_params['stim_dur'][0]) +\
                            '_delay2an_%d'%(delay) +\
                            '_peak_%.2f'%(np.log10(peak)) +\
                            '_peakratio_%.1f'%(peak_ratio) +\
                            '.pickle'
                        
                #### RUN SIMULATIONS #####################################################
                # ORNs layer dynamics
                output_orn = ORNs_layer_dyn.main(params_al_orn)
                [t, u_od,  orn_spikes_t, orn_sdf,orn_sdf_time] = output_orn 
                
                # AL dynamics
                output_al = AL_dyn.main(params_al_orn, orn_spikes_t)
                [t, pn_spike_matrix, pn_sdf, pn_sdf_time,
                              ln_spike_matrix, ln_sdf, ln_sdf_time,] = output_al
            
                output2an = dict([
                            ('t', t),
                            ('u_od',u_od),
                            ('orn_sdf', orn_sdf),
                            ('orn_sdf_time',orn_sdf_time), 
                            ('pn_sdf', pn_sdf),
                            ('pn_sdf_time', pn_sdf_time), 
                            ('ln_sdf', ln_sdf),
                            ('ln_sdf_time', ln_sdf_time), 
                            ])                            
            
                if data_save:
                    with open(fld_analysis+name_data, 'wb') as f:
                        pickle.dump([params_al_orn, output2an], f)
                            
                # FIGURE ORN, PN, LN
                if al_orn_1r_fig:
                    fig_al_orn = plot_al_orn.main(params_al_orn, output_orn, output_al)
                    if figs_save:
                        fig_name  = 'ORN_AL_timecourse_inh_' + inh_cond +  \
                    '_stim_'+ stim_params['stim_type'] +'_dur_%d'%stim_dur + '_p' + str(peak) 
                        print('saving single run time-course figure in '+fld_analysis)
                        fig_al_orn.savefig(fld_analysis+fig_name+ '.png')   
                
                
                
                if npeaks_fig:
                    if (id_c == 0):
                        fig_npeaks_settings(ax_ornal, params_al_orn)
                    fig_npeaks(output2an, params_al_orn, id_c)
                    plt.show()
                    
                out_olsen       = olsen2010_data(output2an, params_al_orn,)
                conc_th[id_c]   = peak
                conc_s[id_c]      = out_olsen['conc_s']
                nu_orn_s[id_c]    = out_olsen['nu_orn_s']
                nu_orn_w[id_c]    = out_olsen['nu_orn_w']
                nu_orn_s_err[id_c] = out_olsen['nu_orn_s_err']
                nu_orn_w_err[id_c] = out_olsen['nu_orn_w_err']
                
                nu_pn_s[id_c]     = out_olsen['nu_pn_s']
                nu_pn_w[id_c]     = out_olsen['nu_pn_w']
                nu_pn_s_err[id_c] = out_olsen['nu_pn_s_err']
                nu_pn_w_err[id_c] = out_olsen['nu_pn_s_err']
                
                nu_ln_s[id_c]     = out_olsen['nu_ln_s']
                nu_ln_w[id_c]     = out_olsen['nu_ln_w']
                nu_ln_s_err[id_c] = out_olsen['nu_ln_s_err']
                nu_ln_w_err[id_c] = out_olsen['nu_ln_w_err']
            
            toc = tictoc()
            print('sim time: %.2f s' %(toc-tic))
            
            print('conc ratio: %d'%peak_ratio)
            # print('nu PN strong avg:')        
            # print(nu_pn_s)
            print('nu PN weak avg:')        
            print(nu_pn_w)
            # print('nu PN ratio avg:')        
            # print(nu_pn_s/nu_pn_w)
            nu_pn_ratio[:, id_r] = nu_pn_s/nu_pn_w
            
            
              
            # print(nu_orn)
            print('nu ORN weak avg:')        
            print(nu_orn_w)
            # print('nu ORN ratio avg:')        
            # print(nu_orn/nu_orn_w)
            # print('')
            # print('nu LN avg:')        
            # print(nu_ln)
            
            # dnu = nu_pn_obs - nu_pn_s
            # cost_est = 0.5 * np.sum(dnu**2)
            # print('estimated cost: %.2f'%cost_est)
            print('')
            
            if figs_save & npeaks_fig:
                print('saving Olsen2010 time-course figure in '+fld_analysis)
                fig_name = 'Olsen2010_timecourse_inh_' + inh_cond +  \
                        '_stim_'+ stim_params['stim_type'] +'_dur_%d'%stim_dur 
                fig_pn.savefig(fld_analysis+  fig_name+'_'+inh_cond+'.png')    
            
        
            #%% FIGURE Olsen 2010: ORN vs PN during step stimulus
            if olsen_fig:
                
                # Constrain the optimization region
                popt, pcov = curve_fit(olsen_orn_pn, nu_orn_s, nu_pn_s, 
                            bounds=(0,[250, 300])) # , bounds=(0, [3., 1., 0.5])
                nuorn_fit = np.linspace(2, nu_orn_s[-1]*1.1, 100)
                # print('sigma: %.2f'%popt[0])
                # print('nu max: %.2f'%popt[1])
                
                rs = 2
                cs = 1
                fig3, axs = plt.subplots(rs,cs, figsize=(5,8), )
                
                plt.rc('text', usetex=True)
                
                # if len(nu_pn_obs)==len(nu_orn_s):
                if stim_dur==500:
                    axs[0].plot(nu_orn_obs, nu_pn_obs, 'k*')
                axs[0].errorbar(nu_orn_s, nu_pn_s, yerr=nu_pn_s_err, fmt='o')
                axs[0].plot(nuorn_fit , olsen_orn_pn(nuorn_fit , *popt), '--', linewidth=lw, 
                        label=r'fit: $\sigma$=%5.0f, $\nu_{max}$=%5.0f' % tuple(popt))
                
                
                axs[0].set_ylabel(r'PN (Hz)', fontsize=label_fs)
                axs[0].set_xlabel(r'ORN (Hz)', fontsize=label_fs)
                
                # strong side
                axs[1].errorbar(conc_th+1e-5, nu_orn_s, yerr=nu_orn_s_err, 
                        linewidth=lw, color='blue', ms=15, label='ORNs ')
                axs[1].errorbar(conc_th-1e-5, nu_pn_s, yerr=nu_pn_s_err, 
                        linewidth=lw, color='orange', ms=15, label='PNs ')
                axs[1].errorbar(conc_th, nu_ln_s, yerr=nu_ln_s_err, 
                        linewidth=lw, color='green', ms=15, label='LNs ')
                # # weak side
                # axs[1].errorbar(conc_th+1e-5, nu_orn_w, yerr=nu_orn_w_err, ls='--', 
                #         linewidth=lw, color='blue', ms=15, label='ORNs weak')
                # axs[1].errorbar(conc_th-1e-5, nu_pn_w, yerr=nu_pn_w_err, ls='--', 
                #         linewidth=lw, color='orange', ms=15, label='PNs weak')
                # axs[1].errorbar(conc_th, nu_ln_w, yerr=nu_ln_w_err, ls='--', 
                #         linewidth=lw, color='green', ms=15, label='LNs weak')
                
                
                axs[1].legend(loc='upper left', fontsize=legend_fs)
            
                axs[1].set_ylabel(r'Firing rates (Hz)', fontsize=label_fs)
            
                axs[1].set_xlabel(r'concentration (au)', fontsize=label_fs)
                axs[1].legend(loc=0, fontsize=legend_fs, frameon=False)
                
                axs[0].text(-.2, 1.1, 'e', transform=axs[0].transAxes, 
                     fontsize=panel_fs, fontweight='bold', va='top', ha='right')
                axs[1].text(-.2, 1.1, 'f', transform=axs[1].transAxes,
                     fontsize=panel_fs, fontweight='bold', va='top', ha='right')
                axs[1].set_xscale('log')
                
                for j in [0,1]:
                    axs[j].tick_params(axis='both', labelsize=label_fs)
                    axs[j].spines['right'].set_color('none')
                    axs[j].spines['top'].set_color('none')
                
                dx = 0.1
                ll, bb, ww, hh = axs[0].get_position().bounds
                axs[0].set_position([ll+dx, bb+.05, ww-dx, hh])
                ll, bb, ww, hh = axs[1].get_position().bounds
                axs[1].set_position([ll+dx, bb, ww-dx, hh])
            
                plt.show()
                
                if figs_save:
                    fig_name = 'Olsen2010_inh_' + inh_cond +  \
                        '_stim_'+ stim_params['stim_type'] +'_dur_%d'%stim_dur 
                        
                    print('saving Olsen2010 PN-ORN figure in '+fld_analysis)
                    fig3.savefig(fld_analysis+ fig_name +'.png')
                    if data_save:
                        with open(fld_analysis+ 'params_al_orn'+'_dur_%d'%stim_dur+'.pickle', 'wb') as f:
                            pickle.dump(params_al_orn, f)
                    
        if inh_cond == 'nsi':
            nsi_ratio = nu_pn_ratio
        elif inh_cond == 'noin':
            noin_ratio = nu_pn_ratio
        elif inh_cond == 'ln':
            ln_ratio = nu_pn_ratio
            
          
    if dataratio_save:
        ratio_out = dict([
            ('peak_ratios', peak_ratios),
            ('peaks', peaks),
            ('params_al_orn', params_al_orn),
            ('noin_ratio', noin_ratio),
            ('nsi_ratio', nsi_ratio),
            ('ln_ratio', ln_ratio),
            ])
        
        with open(fld_analysis+ 'ratio_dur%d'%stim_dur+'.pickle', 'wb') as f:
            pickle.dump(ratio_out, f)

       

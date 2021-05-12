#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:21:20 2021

ORN_dyn_plot.py

This script run NSI_ORN_LIF.py one or multiple times and saves the data 
the following figures of the NSI paper:
    fig.3 ORN dynamics of all its components (ORN_response)
    fig.3 ORN firing rates for several values of the stimulations (martelli2013)
    fig.3s ORN dynamics for stimuli a la Lazar (lazar)   

@author: mario
"""

import numpy as np
import matplotlib.pyplot as plt
import timeit
import pickle        
import matplotlib as mpl

from scipy.interpolate import interp1d


import NSI_ORN_LIF
# import plot_orn  
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
black   = 'xkcd:black'
blue    = 'xkcd:blue'
red     = 'xkcd:red'
green   = 'xkcd:green'
purple  = 'xkcd:purple'
orange  = 'xkcd:orange'
cmap    = plt.get_cmap('rainbow')
recep_clrs = ['green','purple','cyan','red']



# tic toc
def tictoc():
    return timeit.default_timer()         

def figure_multipeaks(data2an, params2an, id_p):
    
    n_orns_recep = params2an['sens_params']['n_orns_recep']
    
    t           = data2an['t']
    t2simulate  = t[-1]
    u_od        = data2an['u_od']
    orn_sdf     = data2an['orn_sdf']
    orn_sdf_time = data2an['orn_sdf_time']
    
    t_on        = params2an['stim_params']['t_on'][0]            
    
    
    # PLOT
    id_pol = id_p + 3
    ax_conc_m.plot(t-t_on, 100*u_od[:,0], color=greenmap.to_rgba(id_pol), linewidth=lw, 
              label='glom : '+'%d'%(1))
    
    orn2plot    = np.mean(orn_sdf[:, :n_orns_recep], axis=1)
    # print('normalized to the peak') 
    # orn2plot = orn2plot/np.max(orn2plot)
    ax_orn_m.plot(orn_sdf_time-t_on, orn2plot, 
             color=greenmap.to_rgba(id_pol), linewidth=lw-1,)
    
    # second ORNs
    # ax_conc_m.plot(t-t_on, 100*u_od[:,1], color=purplemap.to_rgba(id_pol), linewidth=lw, 
              # label='glom : '+'%d'%(1))
    
    orn2plot    = np.mean(orn_sdf[:, n_orns_recep:], axis=1)
    ax_orn_m.plot(orn_sdf_time-t_on, orn2plot, 
                  color=purplemap.to_rgba(id_pol), linewidth=lw-1,)
    
    # SETTINGS
    ax_conc_m.set_xlim(t2plot)
    ax_orn_m.set_xlim(t2plot)

    if id_p==0:
        ax_conc_m.tick_params(axis='both', labelsize=ticks_fs)
        ax_orn_m.tick_params(axis='both', labelsize=ticks_fs)
        
        ax_conc_m.set_ylabel('Input (a.u.)', fontsize=label_fs)
        ax_orn_m.set_ylabel('Firing rates (Hz)', fontsize=label_fs)
        
        
        ax_conc_m.set_xticks(np.linspace(0, t2simulate, 5))
        ax_orn_m.set_xticks(np.linspace(0, t2simulate, 5))
        
        ax_conc_m.spines['right'].set_color('none')
        ax_conc_m.spines['top'].set_color('none')
        ax_orn_m.spines['right'].set_color('none')
        ax_orn_m.spines['top'].set_color('none')
        
        ax_conc_m.set_xlabel('Time  (ms)', fontsize=label_fs)
        ax_orn_m.set_xlabel('Time  (ms)', fontsize=label_fs)

        ll, bb, ww, hh = ax_conc_m.get_position().bounds
        ax_conc_m.set_position([ll-.05, bb+.05, ww, hh])
        
        ll, bb, ww, hh = ax_orn_m.get_position().bounds
        ax_orn_m.set_position([ll, bb+.05, ww, hh])
    
        ax_conc_m.text(-.15, 1.1, 'a', transform=ax_conc_m.transAxes,
              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
        ax_orn_m.text(-.15, 1.1, 'b', transform=ax_orn_m.transAxes, 
              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            
    
    
# LOAD PARAMS FROM A FILE
params_al_orn = set_orn_al_params.main(2)

# stimulus params
stim_params         = params_al_orn['stim_params']
sens_params         = params_al_orn['orn_layer_params'][0]
orn_params          = params_al_orn['orn_params']
sdf_params          = params_al_orn['sdf_params']
al_params           = params_al_orn['al_params']
# pn_ln_params        = params_al_orn['pn_ln_params']

# nsi params
inh_conds           = ['ctrl', 'nsi']  #['nsi', 'ctrl'] #
nsi_strs            = [0, 0.6] #[0.4, 0]

n_pns_recep         = al_params['n_pns_recep'] # number of PNs per each glomerulus
n_orns_recep        = sens_params['n_orns_recep']   # number of ORNs per each glomerulus
    
shift_0 = 10000
shift_1 = 9000

# stim params
delay                       = 0
peak_ratio                  = 1
peaks                       = 5*np.logspace(-10, -1, 20) 
peaks_intp                  = 5*np.logspace(-10, -1, 100) 
stim_durs                   = [20, 50, 100, 200]

# stim_dur                    = 200
stim_type                   = 'ts'  #'ts' # 'ss'
t_on                        = 1000

stim_params['stim_type']    = stim_type
stim_params['t_tot']        = t_on+max(stim_durs)+500        # ms 
stim_params['t_on']         =  np.array([t_on, t_on+delay])
stim_params['conc0']        = 1.85e-20
    

# Output options
fig_save                    = 0
data_save                   = 0
fig_multipeaks              = 0
fig_doseres                 = 1

t2an_on                     = t_on
t2an_off                    = t_on+ 200
fld_analysis                = 'NSI_analysis/dose_response/'   #'NSI_analysis/tuning_curves/'

n_r                         =   sens_params['transd_params'][0]['n'][0]
alpha_r_0 = shift_0**n_r*12.6228808
alpha_r_1 = shift_1**n_r*12.6228808

measure                     = 'peak'  # 'avg' 'peak'


##############################################################################
# Transduction parameters
od_pref = np.array([[1,0], [1, 0],]) # ORNs' sensibilities to each odours
     
transd_vect_3A = od_pref[0,:]
transd_vect_3B = od_pref[1,:]

ab3A_params = dict([
                    ('n', .822066870*transd_vect_3A), 
                    ('alpha_r', alpha_r_0*transd_vect_3A), 
                    ('beta_r', 7.6758436748e-02*transd_vect_3A),
                    ])

ab3B_params = dict([
                    ('n', .822066870*transd_vect_3B), 
                    ('alpha_r', alpha_r_1*transd_vect_3B), 
                    ('beta_r', 7.6758436748e-02*transd_vect_3B),
                    ])

transd_params = (ab3A_params, ab3B_params)
    
sens_params['transd_params'] = transd_params
sens_params['od_pref'] = od_pref
##############################################################################


n_peaks     = np.size(peaks)

c = np.arange(1, n_peaks + 4)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
greenmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Greens)
purplemap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Purples)

params_1sens   = dict([
                ('stim_params', stim_params),
                ('sens_params', sens_params),
                ('orn_params', orn_params),
                ('sdf_params', sdf_params),
                ])
n_peaks = len(peaks)

avg_ornw = np.zeros((n_peaks,  len(inh_conds)))
avg_orns = np.zeros((n_peaks,  len(inh_conds))) 
avg_pnw = np.zeros((n_peaks,  len(inh_conds)))
avg_pns = np.zeros((n_peaks,  len(inh_conds)))

peak_ornw = np.zeros((n_peaks,  len(inh_conds)))
peak_orns  = np.zeros((n_peaks,  len(inh_conds)))
peak_pnw = np.zeros((n_peaks,  len(inh_conds)))
peak_pns = np.zeros((n_peaks,  len(inh_conds)))
thr_lw = np.zeros((len(inh_conds), len(stim_durs)), dtype=int)
thr_hg = np.zeros((len(inh_conds),len(stim_durs)), dtype=int)
dyn_rng= np.zeros((len(inh_conds),len(stim_durs)))

tic = tictoc()

# RUN SIMS AND PLOT

if fig_doseres:
    fig_dr, axs = plt.subplots(nrows=len(stim_durs), ncols=1, figsize=[8, 5]) # len(inh_conds)

for id_dur, stim_dur in enumerate(stim_durs):
    stim_params['stim_dur']     = np.array([stim_dur, stim_dur+delay])
    t2plot                      = -200,stim_dur+300

    # ax2 = plt.subplot(rs, cs, 3)  
    for id_inh, inh_cond in enumerate(inh_conds):
        
        nsi_str = nsi_strs[id_inh]
        sens_params['w_nsi']    = nsi_str
        # FIGURE dose response
        if fig_doseres:
            # if (len(stim_durs)==1) & (len(inh_conds)>1):
            #     ax2 = axs[id_inh]
            # el
            if (len(stim_durs)==1):# & (len(inh_conds)==1):
                ax2 = axs
            else:
                ax2 = axs[id_dur]#, id_inh]
        
        # FIGURE multiple concentrations
        if fig_multipeaks:
    
            rs = 1 # number of rows
            cs = 2 # number of cols
            
            fig_pn_m = plt.figure(figsize=[9, 4])
            ax_conc_m = plt.subplot(rs, cs, 1)
            ax_orn_m = plt.subplot(rs, cs, 2)
            
    
        
        for id_p, peak in enumerate(peaks):
            
            stim_params['concs'] = np.array([peak, peak*peak_ratio])
            
            # RUN SIM
            orn_lif_out = NSI_ORN_LIF.main(params_1sens, )
            [t, u_od, r_orn, v_orn, y_orn, num_spikes, orn_spikes_t, 
                  orn_sdf, orn_sdf_time,] = orn_lif_out
    
            output2an = dict([
                        ('t', t),
                        ('u_od',u_od),
                        ('orn_sdf', orn_sdf),
                        ('orn_sdf_time',orn_sdf_time), ])   
            
            # Calculate avg and peak SDF for ORNs
            if orn_spikes_t.size >0:
                t_peak = 500
                id_stim_w = np.flatnonzero((orn_sdf_time>t2an_on) 
                                        & (orn_sdf_time<t2an_off))
                
                id_stim_s = np.flatnonzero((orn_sdf_time>t2an_on+delay) 
                                            & (orn_sdf_time<t2an_off+delay))
                    
                peak_ornw[id_p, id_inh] = np.max(np.mean(orn_sdf[id_stim_w, :n_orns_recep], axis=1)) # using average PN
                avg_ornw[id_p, id_inh]  = np.mean(orn_sdf[id_stim_w, :n_orns_recep])
                
                peak_orns[id_p, id_inh] = np.max(np.mean(orn_sdf[id_stim_s, n_orns_recep:], axis=1)) # using average PN
                avg_orns[id_p, id_inh]  = np.mean(orn_sdf[id_stim_s, n_orns_recep:])
                
                
            
            
            
        
            
            # FIGURE ORN DYNAMICS OR MULTIPLTE PEAKS 
            if fig_multipeaks:
                figure_multipeaks(output2an, params_1sens, id_p)
                if fig_save:
            
                    tmp_name = '_stim_' + stim_params['stim_type'] +\
                            '_nsi_%.1f'%(sens_params['w_nsi']) +\
                            '_dur2an_%d'%(stim_params['stim_dur'][0]) +\
                            '_alphar_0_%d'%alpha_r_0 +\
                            '_alphar_1_%d'%alpha_r_1
                    fig_name = 'orn_tuning' + tmp_name
                    
                    fig_pn_m.savefig(fld_analysis + fig_name + '.png')
                    if data_save:
                        params_name = 'param_1sens_'+ measure + tmp_name                
                        with open(fld_analysis+params_name+'.pickle', 'wb') as f:
                            pickle.dump([params_1sens, ], f)  
        
        
        nu_intp = interp1d(peaks, peak_ornw[:,id_inh]+peak_orns[:,id_inh], kind=1)
        dr_peak = nu_intp(peaks_intp)
        
        thr_lw[id_inh, id_dur] = next(x[0] for x in enumerate(dr_peak) if x[1] > 0.05*np.max(dr_peak))

        thr_hg[id_inh, id_dur] = next(x[0] for x in enumerate(dr_peak) if x[1] > 0.95*np.max(dr_peak))
        
        dyn_rng[id_inh, id_dur] = -np.log10(peaks_intp[thr_hg[id_inh, id_dur]] - peaks_intp[thr_lw[id_inh, id_dur]] )
        
        if fig_doseres:
            if measure == 'avg':
                ax2.plot(peaks, avg_ornw[:,id_inh], '.-', linewidth=lw+1, color='green')
                ax2.plot(peaks, avg_orns[:,id_inh], '.-', linewidth=lw+1, color='purple')
                # ax2.plot(peaks, (avg_orns+avg_ornw)[:,id_inh], '.-', linewidth=lw+1, color='blue')
            elif measure == 'peak':
                # ax2.plot(peaks, np.max(peak_ornw[:,id_inh])-peak_ornw[:,id_inh], '.-', linewidth=lw+2*id_inh, color='green')
                # ax2.plot(peaks, np.max(peak_orns[:,id_inh])-peak_orns[:,id_inh], '.-', linewidth=lw+2*id_inh, color='purple')
                # ax2.plot(peaks, peak_ornw[:,id_inh], '.-', linewidth=lw+2*id_inh, color='green')
                # ax2.plot(peaks, peak_orns[:,id_inh], '.-', linewidth=lw+2*id_inh, color='purple')
                ax2.plot(peaks, (peak_orns+peak_ornw)[:,id_inh], '.-', linewidth=lw+2*id_inh, color='blue')
                ax2.plot(peaks_intp, dr_peak, '--',linewidth=lw+id_inh, color='red')
                
            if id_dur==0:
                ax2.set_title(inh_cond, fontsize=title_fs+2)
            elif id_dur==(len(stim_durs)-1):
                ax2.set_xlabel('odour conc.', fontsize=label_fs-2)
            
            if id_inh==0:
                ax2.set_ylabel('Avg FR (Hz)', fontsize=label_fs-2)
            
            if id_inh==len(inh_conds)-1:
                ax2.text(1.2, 1,'%d ms'%stim_dur, transform=ax2.transAxes,rotation='vertical',
                         fontsize=label_fs-2, fontweight='bold', va='top', ha='right')
            
            ax2.set_xscale('log') # 'linear') #'log')
            
            ax2.spines['right'].set_color('none')
            ax2.spines['top'].set_color('none')
    
        plt.show()
        
        if fig_save:
            if fig_save:
            
                if fig_doseres:
                    tmp_name = '_stim_' + stim_params['stim_type'] +\
                        '_durs_%d-%d_'%(stim_durs[0], stim_durs[-1],) +\
                        inh_conds[0]+'_vs_'+inh_conds[1]+\
                        '_alphar_0_%d'%alpha_r_0 +\
                        '_alphar_1_%d'%alpha_r_1
                        
                    fig_name = 'orn_dose-response' + tmp_name
                    
                    fig_dr.savefig(fld_analysis + fig_name + '.png')
                
              
print(dyn_rng)        
toc = tictoc()-tic

print('Diag shade plot elapsed time: %.1f'%(toc))
                        



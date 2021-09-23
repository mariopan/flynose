#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:28:17 2021

ORN_dose_response.py

This script run sensillum_dyn.py one or multiple times and saves the data.

Useful to create the following figure(s) of the NSI paper:
    
    fig.10a Dose response curves of two neurons for the control network 
        (dashed lines) and for the NSI network (solid lines).
    
    fig.10b-c Lower threshold and dynamic range of two neurons for the control network 
        (dashed lines) and for the NSI network (solid lines). 
        See plot_dose_reponse.py
    
@author: mario
"""

import numpy as np
import matplotlib.pyplot as plt
import timeit
import pickle        
import matplotlib as mpl
import string

# internal submodules
import sensillum_dyn
import set_orn_al_params
import stim_fcn_oop
            


    
# STANDARD FIGURE PARAMS
lw = 3
fs = 13
plt.rc('text', usetex=True)  # laTex in the polot
#plt.rc('font', family='serif')
fig_size = [12, 12]
fig_position = 1300,10
title_fs = 20 # font size of ticks
label_fs = 22 # font size of labels
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
alphabet = string.ascii_lowercase


# tic toc
def tictoc():
    return timeit.default_timer()         

def figure_multipeaks(data2an, params2an, id_conc):
    
    n_orns_recep = params2an['sens_params']['n_orns_recep']
    
    t           = data2an['t']
    t2simulate  = t[-1]
    u_od        = data2an['u_od']
    orn_sdf     = data2an['orn_sdf']
    orn_sdf_time = data2an['orn_sdf_time']
    
    t_on        = params2an['stim_params']['t_on'][0]            
    
    
    # PLOT
    id_pol = id_conc + 3
    ax_conc_m.plot(t-t_on, 100*u_od[:,0], color=greenmap.to_rgba(id_pol), linewidth=lw, 
              label='glom : '+'%d'%(1))
    
    orn2plot    = np.mean(orn_sdf[:, :n_orns_recep], axis=1)
    # print('normalized to the conc') 
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

    if id_conc==0:
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
nsi_strs            = [0, 0.4] #[0.4, 0]
n_inh_conds         = len(inh_conds)

n_pns_recep         = al_params['n_pns_recep'] # number of PNs per each glomerulus
n_orns_recep        = sens_params['n_orns_recep']   # number of ORNs per each glomerulus
    

analysis_name       = 'dose_response' 



# stim params
delay                       = 0
concs_ratio                 = 1
concs                       = 5*np.logspace(-20, -1, 30)#50) #9
stim_durs                   = [50] #[10, 20, 50, 100, 200]

n_durs                      = len(stim_durs)
stim_type                   = 'ts'  #'ts' # 'ss'
t_on                        = 1000

stim_params['stim_type']    = stim_type
stim_params['t_tot']        = t_on+max(stim_durs)+500        # ms 
stim_params['t_on']         =  np.array([t_on, t_on+delay])
stim_params['conc0']        = 1.85e-20
    

# Output options
fig_save                    = 0
data_save                   = 1
fig_multipeaks              = 0

t2an_on                     = t_on
t2an_off                    = t_on+ 200
fld_analysis                = 'NSI_analysis/dose_response/nsi0.4_thr10p/'   #'NSI_analysis/tuning_curves/'
# thresholds for detection and ceiling
# c_thr_hg                    = .90
# c_thr_lw                    = .10 #3

# ratios = [1, .8, .6, .4, .2, .05] #['1', '.8', '.6', '.4', '.2', '.05']
# shift_ratios = [1,.1,.01,.001,.0001,] #['1', '.8', '.6', '.4', '.2', '.05']
shift_ratios                = [1, .1, .01, .001, .0001,]

n_loops = 3

for shift_ratio in shift_ratios:
    
    
    # shift_ratio = float(sys.argv[1])
    print('shift ratio: %.2f'%shift_ratio)
    shift_0         = 1e4
    shift_1         = shift_ratio*shift_0
    
    
    n_r                         =   sens_params['transd_params'][0]['n'][0]
    alpha_r_0 = shift_0**n_r*12.6228808
    alpha_r_1 = shift_1**n_r*12.6228808
    
    measure                     = 'conc'  # 'avg' 'conc'
    
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
    
    
    
    n_peaks     = np.size(concs)
    
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
    n_peaks = len(concs)
    
    avg_ornw = np.zeros((n_peaks,  n_inh_conds, n_loops))
    avg_orns = np.zeros((n_peaks,  n_inh_conds, n_loops)) 
    
    peak_ornw = np.zeros((n_peaks,  n_inh_conds, n_loops))
    peak_orns = np.zeros((n_peaks,  n_inh_conds, n_loops))
    
    
    # RUN SIMS AND PLOT
    tic = tictoc()
    
    for id_dur, stim_dur in enumerate(stim_durs):
        
        stim_params['stim_dur']     = np.array([stim_dur, stim_dur+delay])
        t2plot                      = -200,stim_dur+300
    
        for id_inh, inh_cond in enumerate(inh_conds):
            
            nsi_str = nsi_strs[id_inh]
            sens_params['w_nsi']    = nsi_str
            
            # FIGURE multiple concentrations
            if fig_multipeaks:
        
                rs = 1 # number of rows
                cs = 2 # number of cols
                
                fig_pn_m = plt.figure(figsize=[9, 4])
                ax_conc_m = plt.subplot(rs, cs, 1)
                ax_orn_m = plt.subplot(rs, cs, 2)
                
        
            
            for id_conc, conc in enumerate(concs):
                
                stim_params['concs'] = np.array([conc, conc*concs_ratio])
                
                # RUN SIM
                # GENERATE ODOUR STIMULUS/I and UPDATE STIM PARAMS
                
                for id_loop in range(n_loops):
                    
                    plume           = stim_fcn_oop.main(stim_params, verbose=False)
                    u_od            = plume.u_od
        
                    output_orn     = sensillum_dyn.main(params_1sens, u_od)
                    
                    tmp_ks = ['t', 'u_od', 'r_orn', 'v_orn', 'y_orn', 'spikes_orn', 'spike_matrix', 
                         'orn_sdf', 'orn_sdf_time',]
                    
                    [t, u_od, r_orn, v_orn, y_orn, spikes_orn, spike_matrix, 
                         orn_sdf, orn_sdf_time,] = [output_orn[x] for x in tmp_ks] 
            
                    output2an = dict([
                                ('t', t),
                                ('u_od',u_od),
                                ('orn_sdf', orn_sdf),
                                ('orn_sdf_time',orn_sdf_time), ])   
                    
                    # Calculate avg and conc SDF for ORNs
                    if spike_matrix.size >0:
                        t_peak = 500
                        id_stim_w = np.flatnonzero((orn_sdf_time>t2an_on) 
                                                & (orn_sdf_time<t2an_off))
                        
                        id_stim_s = np.flatnonzero((orn_sdf_time>t2an_on+delay) 
                                                    & (orn_sdf_time<t2an_off+delay))
                            
                        peak_ornw[id_conc, id_inh, id_loop] = np.max(np.mean(orn_sdf[id_stim_w, :n_orns_recep], axis=1)) # using average PN
                        avg_ornw[id_conc, id_inh, id_loop]  = np.mean(orn_sdf[id_stim_w, :n_orns_recep])
                        
                        peak_orns[id_conc, id_inh, id_loop] = np.max(np.mean(orn_sdf[id_stim_s, n_orns_recep:], axis=1)) # using average PN
                        avg_orns[id_conc, id_inh, id_loop]  = np.mean(orn_sdf[id_stim_s, n_orns_recep:])
                        
                        
                    else:
                        print('what the f**k')
                    
                # FIGURE ORN DYNAMICS OR MULTIPLTE concs 
                if fig_multipeaks:
                    figure_multipeaks(output2an, params_1sens, id_conc)
                    if fig_save:
                
                        tmp_name = '_stim_' + stim_params['stim_type'] +\
                                '_nsi_%.1f'%(sens_params['w_nsi']) +\
                                '_dur2an_%d'%(stim_params['stim_dur'][0]) +\
                                '_alphar_0_%d'%alpha_r_0 +\
                                '_alphar_1_%d'%alpha_r_1
                        fig_name = 'orn_tuning' + tmp_name
                        
                        fig_pn_m.savefig(fld_analysis + fig_name + '.png')                       
                    plt.show()     
    
    
    tmp_name = '_stim_' + stim_params['stim_type'] +\
            '_durs_%d-%d_'%(stim_durs[0], stim_durs[-1],) +\
            inh_conds[0]+'_vs_'+inh_conds[1]+\
            '_alphar_0_%d'%alpha_r_0 +\
            '_alphar_1_%d'%alpha_r_1
                    
    if data_save:
        data2save = dict([
            ('avg_orns', avg_orns),            
            ('avg_ornw', avg_ornw),
            ('peak_orns', peak_orns),            
            ('peak_ornw', peak_ornw),
            ('shift_ratios', shift_ratio),
            ('stim_durs', stim_durs),
            ('concs', concs),
            ('n_loops', n_loops),
            ('nsi_strs', nsi_strs), 
            ('params_1sens', params_1sens), 
            ])
        with open(fld_analysis+analysis_name+tmp_name+'.pickle', 'wb') as f:
                pickle.dump(data2save, f)  
    
                    
    np.set_printoptions(precision=3)
    
    toc = tictoc()-tic
    
    print('Dose-response analysis: elapsed time %.1f s'%(toc))
                            
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:21:20 2021

ORN_dyn_plot.py

This script run NSI_ORN_LIF.py one or multiple times and saves the data 
the following figures of the NSI paper:
    fig.3 ORN dynamics of all its components (ORN_response)
    fig.3 ORN firing rates for several values of the stimulations (martelli2013)
    fig.3s ORN dynamics for stimuli a la Lazar (lazar: ramp, parabola and step)   

@author: mario
"""



import numpy as np
import matplotlib.pyplot as plt
import timeit
# from scipy import signal

import pickle        
from os import path
from os import mkdir
# from shutil import copyfile

import NSI_ORN_LIF
# import sdf_krofczik

# tic toc
def tictoc():
    return timeit.default_timer()

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


# Standard params
alpha_ln        = 13.3# ln spike h=0.4
nsi_str         = 0.3


# stimulus params
stim_params     = dict([
                    ('stim_type' , 'ss'),   # 'rs' # 'ts'  # 'ss' # 'pl'
                    ('pts_ms' , 5),         # simulated pts per ms 
                    ('n_od', 2), 
                    ('t_tot', 2000),        # ms  
                    ('conc0', [1.9e-04]),    # 2.854e-04
                    ('od_noise', 5), #3.5
                    ('od_filter_frq', 0.002), #.002
                    ('r_noise', .50), #6.0
                    ('r_filter_frq', 0.002), # 0.002
                    ])

n_od = stim_params['n_od']
if n_od == 1:
    concs_params    = dict([
                    ('stim_dur' , np.array([500])),     # ms
                    ('t_on', np.array([300])),          # ms
                    ('concs', np.array([0.003])),
                    ])
elif n_od == 2:
    concs_params    = dict([
                    ('stim_dur' , np.array([500, 500])),  # ms
                    ('t_on', np.array([50, 50])), # ms
                    ('concs', np.array([.50, .00])),
                    ])

stim_params.update(concs_params)
    

# Transduction parameters
od_pref = np.array([[1,0], [0,1],]) # ORNs' sensibilities to each odours
     
transd_vect_3A = od_pref[0,:]
transd_vect_3B = od_pref[1,:]

# TEMP: Each ORN will have its transduction properties based on DoOR
ab3A_params = dict([
                    ('n', .822066870*transd_vect_3A), 
                    ('alpha_r', 12.6228808*transd_vect_3A), 
                    ('beta_r', 7.6758436748e-02*transd_vect_3A),
                    ])

ab3B_params = dict([
                    ('n', .822066870*transd_vect_3B), 
                    ('alpha_r', 12.6228808*transd_vect_3B), 
                    ('beta_r', 7.6758436748e-02*transd_vect_3B),
                    ])

# Sensilla/network parameters
transd_params       = (ab3A_params, )#ab3B_params)

n_orns_recep        = 20         # number of ORNs per each receptor
n_neu               = transd_params.__len__()         # number of ORN cohoused in the sensillum



# TEMP: Each sensillum will have its properties based on DoOR
sens_params     = dict([
                    ('n_neu', n_neu),
                    ('n_orns_recep', n_orns_recep),
                    ('od_pref' , od_pref),
    # NSI params
                    ('w_nsi', 0.),  # 0.3
                    ('transd_params', transd_params),
                    ])
    
# ORN Parameters 
orn_params  = dict([
    # LIF params
                    ('t_ref', 2*stim_params['pts_ms']), # ms; refractory period 
                    ('theta', 1),                   # [mV] firing threshold
                    # fitted values
                    ('tau_v', 2.26183540),          # [ms]
                    ('vrest', -0.969461053),        # [mV] resting potential
                    ('vrev', 21),  #25wnsi.2 30wnsi.5         # 21.1784081 [mV] reversal potential
                    # ('v_k', vrest),
                    ('g_y', .5853575783),       
                    ('g_r', .864162073), 
                    # initial values of y anr r
                    ('r0', 0.15), 
                    ('y0', 1), 
    # Adaptation params
                    ('alpha_y', .45310619), 
                    ('beta_y', 3.467184e-03), 
                    ])


# analysis params
sdf_params      = dict([
                    ('tau_sdf', 41),
                    ('dt_sdf', 5),
                    ])

#***********************************************
# 'martelli2013' # 'orn_response' # 'ramp' # 'parabola' # 'step'
fig2plot ='orn_response'

# %%  ORN dynamic response
if fig2plot == 'orn_response':
    
    fld_analysis = 'NSI_analysis/ORN_LIF_dynamics' #/sdf_test
    orn_fig_name = '/ORN_lif_dyn.png'
    
    # stim params
    stim_params['stim_type'] = 'ss' # 'ts' # 'ss' # 'rp'# '
    stim_params['t_tot'] = 2000        # ms 
    delay       = 0
    stim_params['t_on'] =  np.array([1000, 1000])
    stim_params['stim_dur'] = np.array([500, 500])
    peaks       = np.array([.001])         # concentration value for ORN1
    peak_ratio  = 1e-6         # concentration ratio: ORN2/ORN1    
    
    # nsi params
    inh_conds   = ['noin'] 
    
    fig_orn_dyn = 1
    fig_save    = 0
    data_save   = 1  
    n_loops     = 1

# Martelli 2013 figure
elif fig2plot == 'martelli2013':

    fld_analysis = 'NSI_analysis/Martelli2013'

    # stim params
    stim_params['stim_type'] = 'ss' # 'ss'  # 'ts'
    stim_params['stim_dur'] = np.array([500, 500])
    delay       = 0
    peak_ratio  = 1
    peaks       = np.linspace(0,.5,11)

    # nsi params
    inh_conds   = ['noin'] #['nsi', 'ln', 'noin'] #
    
    fig_orn_dyn = 0
    fig_save    = 0
    data_save   = 1  
    n_loops     = 1
    

# Lazar and Kim data reproduction
elif (fig2plot == 'ramp') | (fig2plot == 'parabola') | (fig2plot == 'step'):
    
    fld_analysis    = 'NSI_analysis/lazar_sim2/'
    
    # stim params 
    stim_params['stim_type'] = 'ext'
    stim_params['stim_data_name'] = 'lazar_data_hr/'+fig2plot+'_1'#.dat
    stim_name = fig2plot
    peaks       = np.array([1, 2, 3])
    
    # nsi params 
    inh_conds       = ['noin', ] #'ln', 'noin'

    fig_orn_dyn = 0
    fig_save    = 0
    data_save   = 1    
    n_loops     = 1

    # tau_sdf     = 60
    # dt_sdf      = 5      
    # sdf_params      = [tau_sdf, dt_sdf]

print(fig2plot)
params2an   = dict([
                ('stim_params', stim_params),
                ('sens_params', sens_params),
                ('orn_params', orn_params),
                ('sdf_params', sdf_params),
                ])
  
# %% RUN SIMULATIONS
for stim_seed in range(1):
    
    if path.isdir(fld_analysis):
        print('OLD analysis fld: ' + fld_analysis)    
    else:
        print('NEW analysis fld: ' + fld_analysis)    
        mkdir(fld_analysis)
    
    for peak in peaks:
        
        
        if stim_params['stim_type'] == 'ext':
            stim_params['stim_data_name'] = stim_params['stim_data_name'][:-1]+str(peak)
            
            print(stim_params['stim_data_name'])
        else:
            stim_params['concs'] = np.array([peak, peak*peak_ratio])
        
        tic = tictoc() #timeit.default_timer()
        for inh_cond in inh_conds:
            if inh_cond == 'nsi':
                sens_params['w_nsi'] = nsi_str    #params2an[0:2] = [nsi_str, .0, ]
            elif inh_cond == 'noin':
                sens_params['w_nsi'] = 0    #params2an[0:2] = [0, 0, ]
            elif inh_cond == 'ln':
                sens_params['w_nsi'] = 0    #params2an[0:2] = [.0, alpha_ln,]
            
            for id_loop in range(n_loops):
                orn_lif_out = NSI_ORN_LIF.main(params2an, )
                [t, u_od, r_orn, v_orn, y_orn, 
                   num_spikes, orn_spike_matrix, orn_sdf, orn_sdf_time,] = orn_lif_out
                
                # SAVE SDF OF ORN FIRING RATE
                if data_save:
                    if stim_params['stim_type'] == 'ext':
                        name_data = '/ORNrate' +\
                                '_stim_' + stim_name + str(peak) +\
                                '_nsi_%.1f'%(sens_params['w_nsi']) +\
                                '.pickle'
                    else:
                        name_data = '/ORNrate' +\
                                '_stim_' + stim_params['stim_type'] +\
                                '_nsi_%.1f'%(sens_params['w_nsi']) +\
                                '_dur2an_%d'%(stim_params['stim_dur'][0]) +\
                                '_delay2an_%d'%(delay) +\
                                '_peak_%.1f'%(peak) +\
                                '_peakratio_%.1f'%(peak_ratio) +\
                                '.pickle'
                        
                    output2an = dict([
                                ('t', t),
                                ('u_od',u_od),
                                ('orn_sdf', orn_sdf),
                                ('orn_sdf_time',orn_sdf_time), ])                            
                    
                    with open(fld_analysis+name_data, 'wb') as f:
                        pickle.dump([params2an, output2an], f)
                        
            toc = timeit.default_timer()
        print('time to run %d sims: %.1fs'%(np.size(inh_conds),toc-tic))
        print('')
        
#%%

if fig_orn_dyn:
    t_on    = np.min(stim_params['t_on'])
    stim_dur = stim_params['stim_dur'][0]
    t_tot   = stim_params['t_tot']
    pts_ms  = stim_params['pts_ms']
    vrest   = orn_params['vrest']
    vrev    = orn_params['vrev']
    n_neu   = sens_params['n_neu']
    
    n_neu_tot       = n_neu*n_orns_recep
    recep_clrs = ['purple','green','cyan','red']
    
    # Create Transduction Matrix to plot odour     
    transd_mat = np.zeros((n_neu, n_od))
    for pp in range(n_neu):
        transd_mat[pp,:] = sens_params['od_pref'][pp,:]
    

    t2plot = -t_on, t_tot-t_on#np.min([1000-t_on, t_tot-t_on])
    panels_id = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
    
    rs = 5      # number of rows
    cs = n_neu  #  number of cols
                    
    fig_orn, ax_orn = plt.subplots(rs, cs, figsize=[8.5, 9])
    fig_orn.tight_layout()
        
    if n_neu == 1:
        weight_od = u_od*transd_mat[0,:]
        
        # PLOT
        ax_orn[0].plot(t-t_on, weight_od, linewidth=lw+1, )
        for rr in range(1, rs):
            X0 = t-t_on
            trsp = .3
            if rr == 1:
                X1 = r_orn
                trsp = .3            
            elif rr == 2:
                X1 = y_orn
            elif rr == 3:
                X1 = v_orn
                ax_orn[3].plot([t[0]-t_on, t[-1]-t_on], [vrest, vrest], 
                               '--', linewidth=lw, color=black,)
                # ax_orn[3].plot([t[0]-t_on, t[-1]-t_on], [vrev, vrev], 
                #                '-.', linewidth=lw, color=black,)
            elif rr == 4:
                X1 = orn_sdf
                X0 = orn_sdf_time-t_on
            mu1 = X1.mean(axis=1)
            sigma1 = X1.std(axis=1)
            
            ax_orn[rr].plot(X0, mu1,  
                          linewidth=lw+1, color=recep_clrs[0],)
            for nn in range(n_orns_recep):
                ax_orn[rr].plot(X0, X1[:, nn], 
                                linewidth=lw-1, color=recep_clrs[0], alpha=trsp)
        
        # SETTINGS
        # ax_orn[4].set_ylim(0, 30)
        for rr in range(rs):
            ax_orn[rr].tick_params(axis='both', which='major', labelsize=ticks_fs)
            ax_orn[rr].text(-.15, 1.25, panels_id[rr], transform=ax_orn[0].transAxes, 
                         fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            ax_orn[rr].spines['right'].set_color('none')
            ax_orn[rr].spines['top'].set_color('none')
            ax_orn[rr].set_xlim((t2plot))
            
        # for rr in range(rs-1):
        #     ax_orn[rr].set_xticklabels('')
                 
        ax_orn[0].set_ylabel('Input (a.u.)', fontsize=label_fs)
        ax_orn[3].set_ylabel(r'V (a.u.)', fontsize=label_fs)
        ax_orn[4].set_ylabel('firing rates (Hz)', fontsize=label_fs)   
        ax_orn[1].set_ylabel(r'r (a.u.) ', fontsize=label_fs, )
        ax_orn[2].set_ylabel(r'y adapt (a.u.)', fontsize=label_fs, )
        ax_orn[4].set_xlabel('Time  (ms)', fontsize=label_fs) 
        
        ll, bb, ww, hh = ax_orn[0].get_position().bounds
        ww_new = ww - 0.08
        bb_plus = 0.015
        ll_new = ll + 0.075
        hh_new = hh - 0.05
        ax_orn[0].set_position([ll_new, bb+2.1*bb_plus, ww_new, hh_new])
        
        ll, bb, ww, hh = ax_orn[1].get_position().bounds
        ax_orn[1].set_position([ll_new, bb+2.0*bb_plus, ww_new, hh])
        
        ll, bb, ww, hh = ax_orn[2].get_position().bounds
        ax_orn[2].set_position([ll_new, bb+1.9*bb_plus, ww_new, hh])
        
        ll, bb, ww, hh = ax_orn[3].get_position().bounds
        ax_orn[3].set_position([ll_new, bb+1.8*bb_plus, ww_new, hh])
        
        ll, bb, ww, hh = ax_orn[4].get_position().bounds
        ax_orn[4].set_position([ll_new, bb+1.7*bb_plus, ww_new, hh])
        
        
    else:
        for id_neu in range(n_neu):
            
            # PLOT
            weight_od = u_od*transd_mat[id_neu,:]
            ax_orn[0, id_neu].plot(t-t_on, weight_od, linewidth=lw+1, 
                                   color=black,) 
            
            for rr in range(1, rs):
                X0 = t-t_on
                trsp = .1
                if rr == 1:
                    X1 = r_orn[:, id_neu*n_orns_recep:((id_neu+1)*n_orns_recep)]
                elif rr == 2:
                    trsp = .1
                    X1 = y_orn[:, id_neu*n_orns_recep:((id_neu+1)*n_orns_recep)]
                elif rr == 3:
                    X1 = v_orn[:, id_neu*n_orns_recep:((id_neu+1)*n_orns_recep)]
                    ax_orn[3, id_neu].plot([t[0]-t_on, t[-1]-t_on], [vrest, vrest], 
                                   '--', linewidth=lw, color=red,)
                    # ax_orn[3, id_neu].plot([t[0]-t_on, t[-1]-t_on], [vrev, vrev], 
                    #                '-.', linewidth=lw, color=red,)
                elif rr == 4:
                    X1 = orn_sdf[:, id_neu*n_orns_recep:((id_neu+1)*n_orns_recep)] 
                    X0 = orn_sdf_time-t_on
                mu1 = X1.mean(axis=1)
                sigma1 = X1.std(axis=1)
                
                ax_orn[rr, id_neu].fill_between(X0, mu1+sigma1, mu1-sigma1, 
                            facecolor=recep_clrs[id_neu], alpha=trsp)
                
                ax_orn[rr, id_neu].plot(X0, mu1,  
                               linewidth=lw+1, color=recep_clrs[id_neu],)
                # for nn in range(n_orns_recep):
                    # ax_orn[rr, id_neu].plot(X0, X1[:, nn], 
                              # linewidth=lw-1, color=recep_clrs[id_neu], alpha=trsp)
                
        
            # FIGURE SETTINGS
            for rr in range(rs):
                ax_orn[rr, id_neu].tick_params(axis='both', which='major', labelsize=ticks_fs)
                ax_orn[rr, id_neu].set_xlim((t2plot))      
                ax_orn[rr, id_neu].spines['top'].set_color('none')
                ax_orn[rr, id_neu].spines['right'].set_color('none')
                            
            ax_orn[4, id_neu].set_xlabel('Time  (ms)', fontsize=label_fs) 
        
            # LABELING THE PANELS
            # ax_orn[0, id_neu].text(-.15, 1.25, panels_id[0+id_neu], 
            #                        transform=ax_orn[0, id_neu].transAxes, 
            #                   fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            # ax_orn[1, id_neu].text(-.15, 1.25, panels_id[0+id_neu], transform=ax_orn[0, id_neu].transAxes, 
            #                   fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            # ax_orn[2, id_neu].text(-.15, 1.25, panels_id[0+id_neu], transform=ax_orn[0, id_neu].transAxes, 
            #                   fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            # ax_orn[3, id_neu].text(-.15, 1.25, panels_id[n_neu+id_neu], transform=ax_orn[3, id_neu].transAxes, 
            #                   fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            # ax_orn[4, id_neu].text(-.15, 1.25, panels_id[(n_neu*2)+id_neu], transform=ax_orn[4, id_neu].transAxes, 
            #                   fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            
            for rr in range(rs-1):
                ax_orn[rr, id_neu].set_xticklabels('')
    
            if id_neu == 0:
                ax_orn[0, id_neu].set_ylabel('Input (a.u.)', fontsize=label_fs)
                ax_orn[1, id_neu].set_ylabel(r'r (a.u.) ', fontsize=label_fs, )
                ax_orn[2, id_neu].set_ylabel(r'y adapt (a.u.)', fontsize=label_fs)
                ax_orn[3, id_neu].set_ylabel(r'V (a.u.)', fontsize=label_fs)
                ax_orn[4, id_neu].set_ylabel('firing rates (Hz)', fontsize=label_fs)        
                                         
                ll, bb, ww, hh = ax_orn[0, id_neu].get_position().bounds
                ww_new = ww - 0.08
                bb_plus = 0.015
                ll_new = ll + 0.075
                hh_new = hh - 0.05
                ax_orn[0, id_neu].set_position([ll_new, bb+2.1*bb_plus, ww_new, hh_new])
                
                ll, bb, ww, hh = ax_orn[1, id_neu].get_position().bounds
                ax_orn[1, id_neu].set_position([ll_new, bb+2.0*bb_plus, ww_new, hh])
                
                ll, bb, ww, hh = ax_orn[2, id_neu].get_position().bounds
                ax_orn[2, id_neu].set_position([ll_new, bb+1.9*bb_plus, ww_new, hh])
                
                ll, bb, ww, hh = ax_orn[3, id_neu].get_position().bounds
                ax_orn[3, id_neu].set_position([ll_new, bb+1.8*bb_plus, ww_new, hh])
                
                ll, bb, ww, hh = ax_orn[4, id_neu].get_position().bounds
                ax_orn[4, id_neu].set_position([ll_new, bb+1.7*bb_plus, ww_new, hh])
                
            else:
                ll, bb, ww, hh = ax_orn[0, id_neu].get_position().bounds
                ww_new = ww - 0.08
                bb_plus = 0.015
                ll_new = ll + (0.075-(0.03*id_neu))
                hh_new = hh - 0.05
                ax_orn[0, id_neu].set_position([ll_new, bb+2.1*bb_plus, ww_new, hh_new])
                
                ll, bb, ww, hh = ax_orn[1, id_neu].get_position().bounds
                ax_orn[1, id_neu].set_position([ll_new, bb+2.0*bb_plus, ww_new, hh])
                
                ll, bb, ww, hh = ax_orn[2, id_neu].get_position().bounds
                ax_orn[2, id_neu].set_position([ll_new, bb+1.9*bb_plus, ww_new, hh])
                
                ll, bb, ww, hh = ax_orn[3, id_neu].get_position().bounds
                ax_orn[3, id_neu].set_position([ll_new, bb+1.8*bb_plus, ww_new, hh])
                
                ll, bb, ww, hh = ax_orn[4, id_neu].get_position().bounds
                ax_orn[4, id_neu].set_position([ll_new, bb+1.7*bb_plus, ww_new, hh])
                
                  
    
    fig_orn.align_labels() 
    plt.show()
    if fig_save:
        fig_orn.savefig(fld_analysis + orn_fig_name)
                        
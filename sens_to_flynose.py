#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:33:34 2020

sens_to_flynose.py

This script is the raw model for flynose2.0. It runs NSI_ORN_LIF.py to 
generate ORN activity and then run the AL dynamics.

@author: mario
"""
#%% Setting parameters and define functions

import numpy as np
import timeit

import AL_dyn
import ORNs_layer_dyn
import sdf_krofczik

import matplotlib.pyplot as plt

# *****************************************************************
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


def tictoc():
    return timeit.default_timer()

   
# Stimulus params
stim_params     = dict([
                    ('stim_type' , 'rs'),   # 'ts'  # 'ss' # 'pl'
                    ('pts_ms' , 5),         # simulated pts per ms 
                    ('n_od', 2),            # number of odours
                    ('t_tot', 4000),        # ms 
                    ('conc0', [2.85e-04]),    # 2.854e-04
                    ('od_noise', 5), #3.5
                    ('od_filter_frq', 0.002), #.002
                    ('r_noise', .50), #6.0
                    ('r_filter_frq', 0.002), # 0.002
                    ])

n_od = stim_params['n_od']
if n_od == 1:
    concs_params    = dict([
                    ('stim_dur' , np.array([500])),
                    ('t_on', np.array([300])),          # ms
                    ('concs', np.array([0.01])),
                    ])
elif n_od == 2:
    concs_params    = dict([
                    ('stim_dur' , np.array([50, 50])),
                    ('t_on', np.array([3900, 3900])),          # ms
                    ('concs', np.array([.003, .003])),
                    ])

stim_params.update(concs_params)

# ORN Parameters 
orn_params  = dict([
    # LIF params
                    ('t_ref', 2*stim_params['pts_ms']), # ms; refractory period 
                    ('theta', 1),                 # [mV] firing threshold
                    ('tau_v', 2.26183540),        # [ms]
                    ('vrest', -0.969461053),      # [mV] resting potential
                    ('vrev', 21.1784081),  # [mV] reversal potential
                    # ('v_k', vrest),
                    ('g_y', .5853575783),       
                    ('g_r', .864162073),  
                    ('r0', 0.15), 
                    ('y0', 2), 
    # Adaptation params
                    ('alpha_y', .45310619), 
                    ('beta_y', 3.467184e-03), 
                    ])

# SDF/Analysis params
sdf_params      = dict([
                    ('tau_sdf', 41),
                    ('dt_sdf', 5),
                    ])

# ***************************************************************************
# TEMP: THIS DESCRIPTION SHOULD BE CREATED PER EACH DIFFERENT SENSILLUM/ORN
#       IT SHOULD CHANGE DIRECTLY THE VALUES OF TRANSDUCTION PARAMS, 
#       NOT THE TRANSDUCTION VECTORS

# Sensilla/network parameters
n_orns_recep        = 20         # number of ORNs per each receptor

# Transduction parameters
od_pref = np.array([[1,0], [0,1],]) # ORNs' sensibilities to each odours
               #  [0, 1], [1,0], 
                    # [0,0], [1,0], [0,1], [1,0]
     
transd_vect_3A = od_pref[0,:]
transd_vect_3B = od_pref[1,:]
transd_vect_3B = od_pref[1,:]

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

ornXXC_params = dict([
                    ('n', .822066870*transd_vect_3A), 
                    ('alpha_r', 12.6228808*transd_vect_3A), 
                    ('beta_r', 7.6758436748e-02*transd_vect_3A),
                    ])

# sensillum 0
transd_params0 = (ab3A_params, ab3B_params)
sens_params0     = dict([
                    ('n_neu', transd_params0.__len__()), # number of ORN cohoused in the sensillum
                    ('n_orns_recep', n_orns_recep),
                    ('od_pref' , od_pref),
    # NSI params
                    ('w_nsi', .000001), 
                    ('transd_params', transd_params0),
                    ])

# sensillum 1
transd_params1 = (ab3A_params, ab3B_params, ornXXC_params)
sens_params1   = dict([
                    ('n_neu', transd_params1.__len__()),
                    ('n_orns_recep', n_orns_recep),
                    ('od_pref' , od_pref),
    # NSI params
                    ('w_nsi', .00000002), 
                    ('transd_params', transd_params1),
                    ])

# sensillum 2
transd_params2 = (ab3A_params, ab3B_params, )
sens_params2   = dict([
                    ('n_neu', transd_params2.__len__()),
                    ('n_orns_recep', n_orns_recep),
                    ('od_pref' , od_pref),
    # NSI params
                    ('w_nsi', .2), 
                    ('transd_params', transd_params2),
                    ])

orn_layer = []
orn_layer.append(sens_params0)
# orn_layer.append(sens_params1)
orn_layer.append(sens_params2)
# orn_layer.append(sens_params1)

#################### END PARAMS SETTINGS FOR ORNS #############################


#################### AL + ORNs NETWORK PARAMS #################################

n_sens_type       = orn_layer.__len__()  # number of type of sensilla
n_recep_list      = np.zeros(n_sens_type, dtype=int)
for st in range(n_sens_type):
    n_recep_list[st]      = orn_layer[st]['n_neu'] #[n_neu, n_neu]    # number of ORNs per sensilla


# AL DYNAMICS PARAMETERS 

al_params  = dict([
                    ('n_pns_recep', 5),
                    ('n_lns_recep', 3),
                    ('theta', orn_params['theta']),
                    ('t_ref', orn_params['t_ref']),
                    ('n_recep_list', n_recep_list),
                    ('n_sens_type', n_sens_type),                    
                    ('n_orns_recep', n_orns_recep),                    
                        ])


pn_ln_params = dict([
                    ('alpha_ln', 0), # [ms]
                    ('tau_v', .5), # [ms]
                    ('tau_s', 10),
                    ('alpha_orn', 12), 
                    ('vrest_pn', -6.5), # [mV] resting potential
                    ('vrev_pn', 15),    # [mV] reversal potential
                    ('vpn_noise', 6),   # extra noise input to PNs
                    ('alpha_x', 9.6),   # ORN input coeff for adaptation variable x_pn
                    ('tau_x', 600),     # [ms] time scale for dynamics of adaptation    
                                        # variable x_pn
                    ('alpha_pn', 12),
                    # LN params
                    ('vrest_ln', -3),   # [mV] resting potential
                    ('vrev_ln', 15),
                    ('vln_noise', 1),
                    ('tau_y', 600),
                    ])

# %% ORNS layer SIMULATION
tic = tictoc()
params_all_sens   = dict([
                ('stim_params', stim_params),
                ('orn_layer', orn_layer),
                ('orn_params', orn_params),
                ('sdf_params', sdf_params),
                ])

[t, u_od,  orn_spikes_t, orn_sdf,orn_sdf_time] = ORNs_layer_dyn.main(params_all_sens)


# %%  AL DYNAMICS SIMULATION 

fig_al_save     = 1
fld_analysis    = 'NSI_analysis/trials'
fig_al_name     = '/AL_dyn' + '.png'


al_output = AL_dyn.main(al_params, pn_ln_params, stim_params, sdf_params, orn_spikes_t)


[t, pn_spike_matrix, pn_sdf, pn_sdf_time,
              ln_spike_matrix, ln_sdf, ln_sdf_time,] = al_output

   
    
# %% FIGURE ORN, PN, LN
# Generate a figure per each sensillum type

al_fig = 1
stim_type = stim_params['stim_type']

panels_id = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
rs = 4 # number of rows
cs = 1 # number of cols
fig_size = [7, 8] 
trsp = .3
    
t_tot               = stim_params['t_tot']
t_on    = np.min(stim_params['t_on'])
t2plot = -t_on, t_tot-t_on, 
n_lns_recep = al_params['n_lns_recep']
n_pns_recep = al_params['n_pns_recep']

if al_fig:
       
    # Calculate the mean for PNs and LNs                
    pn_avg = np.mean(pn_sdf)
    ln_avg = np.mean(ln_sdf)
    print('FR avg PNs: %.2f Hz' %pn_avg)
    print('FR avg LNs: %.2f Hz' %ln_avg)
    
    
    recep_id = 0
    for qq in range(n_sens_type):
        num_recep = n_recep_list[qq]

        fig_al = plt.figure(figsize=fig_size)
        ax_conc = plt.subplot(rs, cs, 1)
        ax_orn = plt.subplot(rs, cs, 2)
        ax_pn = plt.subplot(rs, cs, 3)
        ax_ln = plt.subplot(rs, cs, 4)
        
        
        ax_conc.plot(t-t_on, 100*u_od[:,0], color=green, linewidth=lw+2, 
                           label='glom : '+'%d'%(1))
        ax_conc.plot(t-t_on, 100*u_od[:,1], color=purple, linewidth=lw+1, 
                          label='glom : '+'%d'%(2))
        
        for rr in range(num_recep):
            X1 = orn_sdf[:, recep_id*n_orns_recep:((recep_id+1)*n_orns_recep)] # np.mean(orn_sdf_norm[:,:,num_orns_glo:], axis=2)
            mu1 = X1.mean(axis=1)
            sigma1 = X1.std(axis=1)
            ax_orn.plot(orn_sdf_time-t_on, mu1, 
                        color=recep_clrs[rr], linewidth=lw-1, )
            ax_orn.fill_between(orn_sdf_time-t_on, mu1+sigma1, mu1-sigma1, 
                                facecolor=recep_clrs[rr], alpha=trsp)
                    
            # ax_pn.plot(t-t_on, 
            #            u_orn[:, recep_id*n_pns_recep:((recep_id+1)*n_pns_recep)], '--', #pn_sdf
            #            color=recep_clrs[rr], linewidth=lw,)
            
            # # scatter plot PNs
            # for nn1 in range(n_pns_recep):
            #     pn_t_spikes = pn_spike_matrix[pn_spike_matrix[:,1] == rr*n_pns_recep+nn1, 0]
            #     ax_pn.scatter(pn_t_spikes-t_on, np.ones_like(pn_t_spikes)*(rr*n_pns_recep+nn1),
            #                     color=recep_clrs[rr], s=10)
            # ax_ln.plot(t-t_on, 
            #             v_pn[:, recep_id*n_pns_recep:((recep_id+1)*n_pns_recep)], '--', #pn_sdf
            #             color=recep_clrs[rr], linewidth=lw,)
                
            ax_pn.plot(pn_sdf_time-t_on, 
                    pn_sdf[:, recep_id*n_pns_recep:((recep_id+1)*n_pns_recep)], 
                    color=recep_clrs[rr], linewidth=lw,)
            
            ax_ln.plot(ln_sdf_time-t_on, 
                        ln_sdf[:,recep_id*n_lns_recep:((recep_id+1)*n_lns_recep)], 
                        color=recep_clrs[rr], linewidth=lw, )
            
            # # scatter plot LNs
            # for nn1 in range(n_lns_recep):
            #     ln_t_spikes = ln_spike_matrix[ln_spike_matrix[:,1] == rr*n_lns_recep+nn1, 0]
            #     ax_pn.scatter(ln_t_spikes-t_on, np.ones_like(ln_t_spikes)*(rr*n_lns_recep+nn1),
            #                     color=recep_clrs[rr], s=10)
            
            recep_id = recep_id+1
            
        ax_conc.set_xlim(t2plot)
        ax_orn.set_xlim(t2plot)
        ax_pn.set_xlim(t2plot)
        ax_ln.set_xlim(t2plot)
        

        ax_orn.set_ylim((0, 30))
        # ax_pn.set_ylim((0, 30))
        # ax_ln.set_ylim((0, 30))

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
        if stim_type == 'pl':
            ax_orn.set_ylim((0, 150))
            ax_pn.set_ylim((0, 180))
            ax_ln.set_ylim((0, 250))
            ax_pn.set_xticks(np.linspace(0, t2plot[1], 6))
            ax_ln.set_xticks(np.linspace(0, t2plot[1], 6))
            ax_pn.set_xticklabels(np.linspace(0, t2plot[1], 6)/1e3)
            ax_ln.set_xticklabels(np.linspace(0, t2plot[1], 6)/1e3)
            ax_pn.set_xlabel('Time  (ms)', fontsize=label_fs)
            ax_conc.text(-.15, 1.15, 'a.', transform=ax_conc.transAxes,
                color=blue, fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            ax_orn.text(-.15, 1.15, 'b.', transform=ax_orn.transAxes,
                color=blue, fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            ax_pn.text(-.15, 1.15, 'c.', transform=ax_pn.transAxes,
                color=blue, fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            ax_ln.text(-.15, 1.15, 'd.', transform=ax_ln.transAxes,
                color=blue, fontsize=panel_fs, fontweight='bold', va='top', ha='right')

        # tmp
        if not(stim_type == 'pl'):
            title_fs = 30
            # if (params2an[1] ==0) & (params2an[1] ==0):
            #     ax_conc.set_title('a. Independent', fontsize=title_fs)
            # elif (params2an[1] >0):
            #     ax_conc.set_title('b. LN inhib.', fontsize=title_fs)
            # else:
            #      ax_conc.set_title('c. NSI', fontsize=title_fs)   
             
        ax_conc.spines['right'].set_color('none')
        ax_conc.spines['top'].set_color('none')
        ax_orn.spines['right'].set_color('none')
        ax_orn.spines['top'].set_color('none')
        ax_pn.spines['right'].set_color('none')
        ax_pn.spines['top'].set_color('none')
        ax_ln.spines['right'].set_color('none')
        ax_ln.spines['top'].set_color('none')
        
        if (stim_type == 'pl'):
            dx = 0
        else:
            dx = 0.05
        dy = 0.05
            
        ll, bb, ww, hh = ax_conc.get_position().bounds
        ax_conc.set_position([ll+dx, bb+dy, ww, hh])
        ll, bb, ww, hh = ax_pn.get_position().bounds
        ax_pn.set_position([ll+dx, bb+dy, ww, hh])
        ll, bb, ww, hh = ax_orn.get_position().bounds
        ax_orn.set_position([ll+.05, bb+dy, ww, hh])
        ll, bb, ww, hh = ax_ln.get_position().bounds
        ax_ln.set_position([ll+.05, bb+dy, ww, hh])
            
        fig_al.align_labels() 
        plt.show()
        if fig_al_save:
            fig_al.savefig(fld_analysis + fig_al_name + '.png')
        
# #%% ORN FIGURE

# orn_fig = 0
# vrest   = orn_params['vrest']
# vrev    = orn_params['vrev']

# trsp    = .1

# if orn_fig:          
#     # One figure per each sensilla type
#     for st in range(n_sens_type):
#         n_neu   = orn_layer[st]['n_neu'] # TEMP: THIS SHOULD CHANGE PER EACH TYPE OF SENSILLA
        
#         rs = 5      # number of rows
#         cs = n_neu  #  number of cols
                        
#         fig_orn, ax_orn = plt.subplots(rs, cs, figsize=[8.5, 9])
#         fig_orn.tight_layout()
        
        
#         # Create Transduction Matrix to plot odour 
#         transd_mat = np.zeros((n_neu, n_od))
#         for pp in range(n_neu):
#             transd_mat[pp,:] = orn_layer[st]['od_pref'][pp,:]
        
#         if n_neu == 1:
#             weight_od = u_od*transd_mat[0,:]
            
#             # PLOT
#             ax_orn[0].plot(t-t_on, weight_od, linewidth=lw+1, )
#             for rr in range(1, rs):
#                 X0 = t-t_on
#                 if rr == 1:
#                     X1 = r_orn
#                 elif rr == 2:
#                     X1 = y_orn
#                 elif rr == 3:
#                     X1 = v_orn
#                     ax_orn[3].plot([t[0]-t_on, t[-1]-t_on], [vrest, vrest], 
#                                    '--', linewidth=lw, color=red,)
#                     # ax_orn[3].plot([t[0]-t_on, t[-1]-t_on], [vrev, vrev], 
#                     #                '-.', linewidth=lw, color=red,)
#                 elif rr == 4:
#                     X1 = orn_sdf
#                     X0 = orn_sdf_time-t_on
#                 mu1 = X1.mean(axis=1)
#                 sigma1 = X1.std(axis=1)
                
#                 ax_orn[rr].plot(X0, mu1,  
#                               linewidth=lw+1, color=recep_clrs[0],)
#                 for nn in range(n_orns_recep):
#                     ax_orn[rr].plot(X0, X1[:, nn], 
#                                     linewidth=lw-1, color=recep_clrs[0], alpha=trsp)
                
#             # SETTINGS
#             for rr in range(rs):
#                 ax_orn[rr].tick_params(axis='both', which='major', labelsize=ticks_fs)
#                 ax_orn[rr].text(-.15, 1.25, panels_id[rr], transform=ax_orn[0].transAxes, 
#                              fontsize=panel_fs, fontweight='bold', va='top', ha='right')
#                 ax_orn[rr].spines['right'].set_color('none')
#                 ax_orn[rr].spines['top'].set_color('none')
#                 ax_orn[rr].set_xlim((t2plot))
                
#             for rr in range(rs-1):
#                 ax_orn[rr].set_xticklabels('')
                     
#             ax_orn[0].set_ylabel('Input (a.u.)', fontsize=label_fs)
#             ax_orn[3].set_ylabel(r'V (a.u.)', fontsize=label_fs)
#             ax_orn[4].set_ylabel('firing rates (Hz)', fontsize=label_fs)   
#             ax_orn[1].set_ylabel(r'r (a.u.) ', fontsize=label_fs, )
#             ax_orn[2].set_ylabel(r'y adapt (a.u.)', fontsize=label_fs, )
#             ax_orn[4].set_xlabel('Time  (ms)', fontsize=label_fs) 
            
#             ll, bb, ww, hh = ax_orn[0].get_position().bounds
#             ww_new = ww - 0.08
#             bb_plus = 0.015
#             ll_new = ll + 0.075
#             hh_new = hh - 0.05
#             ax_orn[0].set_position([ll_new, bb+2.1*bb_plus, ww_new, hh_new])
            
#             ll, bb, ww, hh = ax_orn[1].get_position().bounds
#             ax_orn[1].set_position([ll_new, bb+2.0*bb_plus, ww_new, hh])
            
#             ll, bb, ww, hh = ax_orn[2].get_position().bounds
#             ax_orn[2].set_position([ll_new, bb+1.9*bb_plus, ww_new, hh])
            
#             ll, bb, ww, hh = ax_orn[3].get_position().bounds
#             ax_orn[3].set_position([ll_new, bb+1.8*bb_plus, ww_new, hh])
            
#             ll, bb, ww, hh = ax_orn[4].get_position().bounds
#             ax_orn[4].set_position([ll_new, bb+1.7*bb_plus, ww_new, hh])
            
#             plt.show()
            
#         else:
#             trsp = .3
#             for id_neu in range(n_neu):
                
#                 # PLOT
#                 weight_od = u_od*transd_mat[id_neu,:]
#                 ax_orn[0, id_neu].plot(t-t_on, weight_od, linewidth=lw+1, 
#                                        color=black,) 
                
#                 for rr in range(1, rs):
#                     X0 = t-t_on
#                     if rr == 1:
#                         X1 = r_orn[:, id_neu*n_orns_recep:((id_neu+1)*n_orns_recep)]
#                     elif rr == 2:
#                         X1 = y_orn[:, id_neu*n_orns_recep:((id_neu+1)*n_orns_recep)]
#                     elif rr == 3:
#                         X1 = v_orn[:, id_neu*n_orns_recep:((id_neu+1)*n_orns_recep)]
#                         ax_orn[3, id_neu].plot([t[0]-t_on, t[-1]-t_on], [vrest, vrest], 
#                                        '--', linewidth=lw, color=red,)
#                         # ax_orn[3, id_neu].plot([t[0]-t_on, t[-1]-t_on], [vrev, vrev], 
#                         #                '-.', linewidth=lw, color=red,)
#                     elif rr == 4:
#                         X1 = orn_sdf[:, id_neu*n_orns_recep:((id_neu+1)*n_orns_recep)] 
#                         X0 = orn_sdf_time-t_on
#                     mu1 = X1.mean(axis=1)
#                     sigma1 = X1.std(axis=1)
                    
#                     ax_orn[rr, id_neu].plot(X0, mu1,  
#                                   linewidth=lw+1, color=recep_clrs[id_neu],)
#                     for nn in range(n_orns_recep):
#                         ax_orn[rr, id_neu].plot(X0, X1[:, nn], 
#                                   linewidth=lw-1, color=recep_clrs[id_neu], alpha=trsp)
            
#                 # FIGURE SETTINGS
#                 for rr in range(rs):
#                     ax_orn[rr, id_neu].tick_params(axis='both', which='major', labelsize=ticks_fs)
#                     ax_orn[rr, id_neu].set_xlim((t2plot))      
#                     ax_orn[rr, id_neu].spines['top'].set_color('none')
#                     ax_orn[rr, id_neu].spines['right'].set_color('none')
                                
#                 ax_orn[4, id_neu].set_xlabel('Time  (ms)', fontsize=label_fs) 
            
#                 # LABELING THE PANELS
#                 # ax_orn[0, id_neu].text(-.15, 1.25, panels_id[0+id_neu], 
#                 #                        transform=ax_orn[0, id_neu].transAxes, 
#                 #                   fontsize=panel_fs, fontweight='bold', va='top', ha='right')
#                 # ax_orn[1, id_neu].text(-.15, 1.25, panels_id[0+id_neu], transform=ax_orn[0, id_neu].transAxes, 
#                 #                   fontsize=panel_fs, fontweight='bold', va='top', ha='right')
#                 # ax_orn[2, id_neu].text(-.15, 1.25, panels_id[0+id_neu], transform=ax_orn[0, id_neu].transAxes, 
#                 #                   fontsize=panel_fs, fontweight='bold', va='top', ha='right')
#                 # ax_orn[3, id_neu].text(-.15, 1.25, panels_id[n_neu+id_neu], transform=ax_orn[3, id_neu].transAxes, 
#                 #                   fontsize=panel_fs, fontweight='bold', va='top', ha='right')
#                 # ax_orn[4, id_neu].text(-.15, 1.25, panels_id[(n_neu*2)+id_neu], transform=ax_orn[4, id_neu].transAxes, 
#                 #                   fontsize=panel_fs, fontweight='bold', va='top', ha='right')
                
#                 for rr in range(rs-1):
#                     ax_orn[rr, id_neu].set_xticklabels('')
        
#                 if id_neu == 0:
#                     ax_orn[0, id_neu].set_ylabel('Input (a.u.)', fontsize=label_fs)
#                     ax_orn[1, id_neu].set_ylabel(r'r (a.u.) ', fontsize=label_fs, )
#                     ax_orn[2, id_neu].set_ylabel(r'y adapt (a.u.)', fontsize=label_fs)
#                     ax_orn[3, id_neu].set_ylabel(r'V (a.u.)', fontsize=label_fs)
#                     ax_orn[4, id_neu].set_ylabel('firing rates (Hz)', fontsize=label_fs)        
                                             
#                     ll, bb, ww, hh = ax_orn[0, id_neu].get_position().bounds
#                     ww_new = ww - 0.08
#                     bb_plus = 0.015
#                     ll_new = ll + 0.075
#                     hh_new = hh - 0.05
#                     ax_orn[0, id_neu].set_position([ll_new, bb+2.1*bb_plus, ww_new, hh_new])
                    
#                     ll, bb, ww, hh = ax_orn[1, id_neu].get_position().bounds
#                     ax_orn[1, id_neu].set_position([ll_new, bb+2.0*bb_plus, ww_new, hh])
                    
#                     ll, bb, ww, hh = ax_orn[2, id_neu].get_position().bounds
#                     ax_orn[2, id_neu].set_position([ll_new, bb+1.9*bb_plus, ww_new, hh])
                    
#                     ll, bb, ww, hh = ax_orn[3, id_neu].get_position().bounds
#                     ax_orn[3, id_neu].set_position([ll_new, bb+1.8*bb_plus, ww_new, hh])
                    
#                     ll, bb, ww, hh = ax_orn[4, id_neu].get_position().bounds
#                     ax_orn[4, id_neu].set_position([ll_new, bb+1.7*bb_plus, ww_new, hh])
                    
#                 else:
#                     ll, bb, ww, hh = ax_orn[0, id_neu].get_position().bounds
#                     ww_new = ww - 0.08
#                     bb_plus = 0.015
#                     ll_new = ll + (0.075-(0.03*id_neu))
#                     hh_new = hh - 0.05
#                     ax_orn[0, id_neu].set_position([ll_new, bb+2.1*bb_plus, ww_new, hh_new])
                    
#                     ll, bb, ww, hh = ax_orn[1, id_neu].get_position().bounds
#                     ax_orn[1, id_neu].set_position([ll_new, bb+2.0*bb_plus, ww_new, hh])
                    
#                     ll, bb, ww, hh = ax_orn[2, id_neu].get_position().bounds
#                     ax_orn[2, id_neu].set_position([ll_new, bb+1.9*bb_plus, ww_new, hh])
                    
#                     ll, bb, ww, hh = ax_orn[3, id_neu].get_position().bounds
#                     ax_orn[3, id_neu].set_position([ll_new, bb+1.8*bb_plus, ww_new, hh])
                    
#                     ll, bb, ww, hh = ax_orn[4, id_neu].get_position().bounds
#                     ax_orn[4, id_neu].set_position([ll_new, bb+1.7*bb_plus, ww_new, hh])
                    
                      
        
#             plt.show()
     
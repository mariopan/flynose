#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:08:05 2021


plot_orn.py

@author: mario
"""

#%% Setting parameters and define functions

import numpy as np
import timeit
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


def main(params_1sens, output_orn, ):
    
    stim_params     = params_1sens['stim_params']
    orn_params      = params_1sens['orn_params']
    sens_params     = params_1sens['sens_params']
    #sdf_params     = params_al_orn['sdf_params']
    
    # ORN layer dynamics output
    # [t, u_od, r_orn, v_orn, y_orn, spikes_orn, spike_matrix, 
    #      orn_sdf, orn_sdf_time,]  = output_orn
    tmp_ks = ['t', 'u_od', 'r_orn', 'v_orn', 'y_orn', 'spikes_orn', 'spike_matrix', 
         'orn_sdf', 'orn_sdf_time',]
    
    [t, u_od, r_orn, v_orn, y_orn, spikes_orn, spike_matrix, 
         orn_sdf, orn_sdf_time,] = [output_orn[x] for x in tmp_ks] 
    
    
    
    chunked=0
    # Check the size of the vectors 
    if len(t) != len(v_orn):
        print('NOTE: The simulation is run in chunks. To inspect it and plot the variables, '
              ' please, modify the simulation duration (t_tot) or the chunk size (t_part).')
        chunked=1
        # return
    
    # figure params
    # stim_type       = stim_params['stim_type']    
    t_on    = np.min(stim_params['t_on'])
    stim_dur = np.max(stim_params['stim_dur'])
    t_tot   = stim_params['t_tot']
    n_od    = stim_params['n_od']
    
    vrest   = orn_params['vrest']
    # vrev    = orn_params['vrev']
    n_neu   = sens_params['n_neu']
    n_orns_recep= sens_params['n_orns_recep']
    
    # Create Transduction Matrix to plot odour 
    # transd_mat = np.zeros((n_neu, n_od))
    # for pp in range(n_neu):
        # transd_mat[pp,:] = sens_params['od_pref'][pp,:]
    

    t2plot = -150, stim_dur+300 ##np.min([1000-t_on, t_tot-t_on])
#    t2plot = -t_on, t_tot-t_on
    panels_id = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
    
    rs = 5      # number of rows
    cs = n_neu  #  number of cols
                    
    fig_orn, ax_orn = plt.subplots(rs, cs, figsize=[8.5, 9])
    fig_orn.tight_layout()
        
    
        
    if n_neu == 1:
        # weight_od = u_od#*transd_mat[0,:]
        
        # PLOT
        if n_od == 1:
            ax_orn[0].plot(t-t_on, u_od, color='blue', linewidth=lw+1, )
        elif n_od == 2:
            ax_orn[0].plot(t-t_on, u_od[:,0], color='green', linewidth=lw+1, )
            ax_orn[0].plot(t-t_on, u_od[:,1], color='purple', linewidth=lw+1, )
            
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
            elif rr == 4:
                X1 = orn_sdf
                X0 = orn_sdf_time-t_on
            mu1 = X1.mean(axis=1)
            sigma1 = X1.std(axis=1)
            
            if not(any([rr==1,rr==2,rr==3]) & (chunked==1)):
                ax_orn[rr].plot(X0, mu1,  
                          linewidth=lw+1, color=recep_clrs[0],)
                for nn in range(n_orns_recep):
                    ax_orn[rr].plot(X0, X1[:, nn], 
                                linewidth=lw-1, color=recep_clrs[0], alpha=trsp)
        
        # SETTINGS
        for rr in range(rs):
            ax_orn[rr].tick_params(axis='both', which='major', labelsize=ticks_fs)
            ax_orn[rr].text(-.15, 1.25, panels_id[rr], transform=ax_orn[rr].transAxes, 
                         fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            ax_orn[rr].spines['right'].set_color('none')
            ax_orn[rr].spines['top'].set_color('none')
            ax_orn[rr].set_xlim((t2plot))
                 
        # ax_orn[0].set_yticklabels([], fontsize=label_fs)
        # ax_orn[1].set_yticklabels([], fontsize=label_fs)
        # ax_orn[2].set_yticklabels([], fontsize=label_fs)
        ax_orn[0].set_ylabel('Input (a.u.)', fontsize=label_fs)
        ax_orn[1].set_ylabel(r'r (a.u.) ', fontsize=label_fs, )
        ax_orn[2].set_ylabel(r'y adapt (a.u.)', fontsize=label_fs, )
        ax_orn[3].set_ylabel(r'V (mV)', fontsize=label_fs)
        ax_orn[4].set_ylabel('firing rates (Hz)', fontsize=label_fs)   
        ax_orn[4].set_xlabel('Time  (ms)', fontsize=label_fs) 
        
        ll, bb, ww, hh = ax_orn[0].get_position().bounds
        ww_new = ww - 0.12
        bb_plus = 0.015
        ll_new = ll + 0.075 + .025
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
            if n_od == 1:
                ax_orn[0, 0].plot(t-t_on, u_od, color=green, linewidth=lw+1, )
            elif n_od == 2:
                ax_orn[0, 0].plot(t-t_on, u_od[:,0], color=green, linewidth=lw+1, )
                ax_orn[0, 0].plot(t-t_on, u_od[:,1], color=purple, linewidth=lw+1, )
            
            # weight_od = u_od#*transd_mat[id_neu,:]
            # ax_orn[0, id_neu].plot(t-t_on, u_od, linewidth=lw+1) 
            
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
                if not(any([rr==1,rr==2,rr==3]) & (chunked==1)):
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
        
            # ax_orn[4, id_neu].set_ylim(0, 30)
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
                ww_new = ww - 0.15
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
                ww_new = ww - 0.15
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
#    plt.tight_layout()
     
    return fig_orn
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 10:32:42 2021
figure_al_orn.py

It plots the dynamics of a single simulation from the output of 
ORNs_layer_dyn.py and AL_dyn.py 

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


def main(params_al_orn, output_orn_layer, output_al):

    stim_params     = params_al_orn['stim_params']
    orn_layer_params= params_al_orn['orn_layer_params']
    #orn_params     = params_al_orn['orn_params']
    #sdf_params     = params_al_orn['sdf_params']
    al_params       = params_al_orn['al_params']
    pn_ln_params    = params_al_orn['pn_ln_params']
    
    n_sens_type     = orn_layer_params.__len__()  # number of type of sensilla
    n_recep_list    = al_params['n_recep_list']
    stim_type       = stim_params['stim_type']    
    
    # NSI - LN inhib condition
    nsi_str = orn_layer_params[0]['w_nsi']
    alpha_ln = pn_ln_params['alpha_ln']
    
    # ORN layer dynamics output
    [t, u_od,  orn_spikes_t, orn_sdf, orn_sdf_time] = output_orn_layer
                    
    # AL dynamics outputs
    [t, pn_spike_matrix, pn_sdf, pn_sdf_time,
        ln_spike_matrix, ln_sdf, ln_sdf_time,] = output_al
    
    # panels_id   = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
    rs          = 4 # number of rows
    cs          = 1 # number of cols
    fig_size    = [7, 8] 
    trsp        = .3   # transparency for error shade on ORN rates
    fig_al_save = 0
    
    t_tot               = stim_params['t_tot']
    t_on    = np.min(stim_params['t_on'])
    t2plot = -t_on, t_tot-t_on, 
    n_orns_recep = al_params['n_orns_recep']
    n_lns_recep = al_params['n_lns_recep']
    n_pns_recep = al_params['n_pns_recep']
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
        
    
        # ax_orn.set_ylim((0, 30))
        # ax_pn.set_ylim((0, 30))
        # ax_ln.set_ylim((0, 30))
    
        ax_conc.tick_params(axis='both', labelsize=label_fs)
        ax_orn.tick_params(axis='both', labelsize=label_fs)
        ax_pn.tick_params(axis='both', labelsize=label_fs)
        ax_ln.tick_params(axis='both', labelsize=label_fs)
        
        ax_conc.set_xticklabels('')
        ax_orn.set_xticklabels('')
        ax_pn.set_xticklabels('')
        
        ax_conc.set_title(['NSI: %.1f, LN:%.0f'%(nsi_str, alpha_ln)], fontsize=label_fs)
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
        # if not(stim_type == 'pl'):
        #     title_fs = 30
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
            fld_analysis = '/'
            fig_al_name  = 'AL_ORN_idrec%d'%qq 
            fig_al.savefig(fld_analysis + fig_al_name + '.png')
    return fig_al
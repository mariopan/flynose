#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:33:34 2020

AL_dyn_batch.py

This script is the raw model for flynose2.0. It runs NSI_ORN_LIF.py to 
generate ORN activity and then run the AL dynamics

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
import figure_al_orn

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
legend_fs = 12
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


def fig_npeaks_settings(ax_ornal, params_tmp):
    # ORN and PN Firing rates figure settings
    
    dx = 0.075
    dy = np.linspace(0.075, 0, 4, )
    for id_ax in range(4):
        ll, bb, ww, hh = ax_ornal[id_ax].get_position().bounds
        ax_ornal[id_ax].set_position([ll+dx, bb+dy[id_ax], ww-dx, hh])

    t_on    =   params_tmp['stim_params']['t_on'][0]
    stim_dur   =   params_tmp['stim_params']['stim_dur'][0]
    t_off   = t_on + stim_dur
    
    t2plot = -200+t_on, t_off+300
            
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
    # plot time course for ONR and PNS for a single concentration value
    n_orns_recep = params_tmp['orn_layer_params'][0]['n_orns_recep']
    
    t           = data_tmp['t']
    u_od        = data_tmp['u_od']
    orn_sdf     = data_tmp['orn_sdf']
    orn_sdf_time = data_tmp['orn_sdf_time']
    pn_sdf = data_tmp['pn_sdf']
    pn_sdf_time = data_tmp['pn_sdf_time']
    ln_sdf = data_tmp['ln_sdf']
    ln_sdf_time = data_tmp['ln_sdf_time']
    
    t_zero      =    00
    n_orns_recep = params_tmp['al_params']['n_orns_recep']
    n_pns_recep = params_tmp['al_params']['n_pns_recep']
    n_lns_recep = params_tmp['al_params']['n_lns_recep']
        
    # ****************************************************
    # PLOTTING DATA
    id_col = id_c + 3
    ax_ornal[0].plot(t-t_zero, 100*u_od[:,0], color=greenmap.to_rgba(id_col), linewidth=lw, 
                  label='glom : '+'%d'%(1))
    
    ax_ornal[1].plot(orn_sdf_time-t_zero, np.mean(orn_sdf[:,:n_orns_recep], axis=1), 
                 color=greenmap.to_rgba(id_col), linewidth=lw-1,label='sdf glo 1')
    
    ax_ornal[2].plot(pn_sdf_time-t_zero, np.mean(pn_sdf[:,:n_pns_recep], axis=1), 
               color=greenmap.to_rgba(id_col), linewidth=lw-1, label='PN, glo 1')
        
    ll = 0
    ax_ornal[3].plot(ln_sdf_time-t_zero, ln_sdf[:,ll], 
                     color=greenmap.to_rgba(id_col), linewidth=lw-1, label='LN, glo 1')
    
    ax_ornal[0].plot(t-t_zero, 100*u_od[:,1], color=purplemap.to_rgba(id_col), 
                     linewidth=lw-1, label='glom : '+'%d'%(2))
    
    ax_ornal[1].plot(orn_sdf_time-t_zero, np.mean(orn_sdf[:,n_orns_recep:], axis=1), 
                     color=purplemap.to_rgba(id_col), linewidth=lw-1,label='sdf glo 2')
    
    ax_ornal[2].plot(pn_sdf_time-t_zero, np.mean(pn_sdf[:,n_pns_recep:], axis=1), 
                     color=purplemap.to_rgba(id_col), linewidth=lw-1, label='PN, glo 2')
    
    ll = n_lns_recep
    ax_ornal[3].plot(ln_sdf_time-t_zero, ln_sdf[:,ll], 
                color=purplemap.to_rgba(id_col), linewidth=lw-1, label='LN, glo 2')
    
def olsen_orn_pn(nu_orn, sigma, nu_max):
    # fitting curve used by Olsen et al. 2010 for the relation between PN and ORN rates
    nu_pn = nu_max * np.power(nu_orn, 1.5)/(np.power(nu_orn, 1.5) + np.power(sigma,1.5))
    return nu_pn
       
def olsen2010_data(data_tmp, params_tmp):
    pts_ms  =   params_tmp['stim_params']['pts_ms']

    t_on    =   params_tmp['stim_params']['t_on'][0]
    stim_dur   =   params_tmp['stim_params']['stim_dur'][0]
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
    
    orn_id_stim = np.flatnonzero((orn_sdf_time>t_on) & (orn_sdf_time<t_off))
    pn_id_stim = np.flatnonzero((pn_sdf_time>t_on) & (pn_sdf_time<t_off))
    ln_id_stim = np.flatnonzero((ln_sdf_time>t_on) & (ln_sdf_time<t_off))
    
    nu_orn_w = np.mean(orn_sdf[orn_id_stim, :n_orns_recep])
    nu_pn_w = np.mean(pn_sdf[pn_id_stim, :n_pns_recep])
    # nu_ln_w = np.mean(ln_sdf[ln_id_stim, :n_lns_recep])
    # conc_w = np.mean(u_od[stim_on:stim_off, 1], axis=0)
    
    nu_orn_s = np.mean(orn_sdf[orn_id_stim, n_orns_recep:])
    nu_pn_s = np.mean(pn_sdf[pn_id_stim, n_pns_recep:])
    nu_ln_s = np.mean(ln_sdf[ln_id_stim, n_lns_recep:])
    conc_s = np.mean(u_od[stim_on:stim_off, 0], axis=0)
    
    nu_orn_err = np.std(orn_sdf[orn_id_stim, :n_orns_recep])/np.sqrt(n_orns_recep)
    nu_pn_err = np.std(pn_sdf[pn_id_stim, :n_pns_recep])/np.sqrt(n_pns_recep)
    nu_ln_err = np.std(ln_sdf[ln_id_stim, :n_lns_recep])/np.sqrt(n_lns_recep)
    
    out_olsen = np.zeros((9))
    out_olsen[0] = conc_s
    out_olsen[1] = nu_orn_s
    out_olsen[2] = nu_pn_s
    out_olsen[3] = nu_ln_s
    out_olsen[4] = nu_orn_err
    out_olsen[5] = nu_pn_err
    out_olsen[6] = nu_ln_err
    out_olsen[7] = nu_pn_w
    out_olsen[8] = nu_orn_w
    
    return out_olsen
    
   
#%% Stimulus params
stim_params     = dict([
                    ('stim_type' , 'ss'),   # 'ts'  # 'ss' # 'pl'
                    ('pts_ms' , 5),         # simulated pts per ms 
                    ('n_od', 2),            # number of odours
                    ('t_tot', 1500),        # ms 
                    ('conc0', 2.85e-04),    # 2.854e-04
                    ('od_noise', 2),        # 5
                    ('od_filter_frq', 0.002), #.002
                    ('r_noise', .50),       # .5
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
                    ('stim_dur' , np.array([500, 500])),
                    ('t_on', np.array([500, 500])),          # ms
                    ('concs', np.array([.003, .000003])),
                    ])

stim_params.update(concs_params)

# ORN Parameters 
orn_params  = dict([
    # LIF params
                    ('t_ref', 2*stim_params['pts_ms']), # timesteps; refractory period 
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
                    ('tau_sdf', 21),
                    ('dt_sdf', 1),
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

# ornXXC_params = dict([
#                     ('n', .822066870*transd_vect_3A), 
#                     ('alpha_r', 12.6228808*transd_vect_3A), 
#                     ('beta_r', 7.6758436748e-02*transd_vect_3A),
#                     ])

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

# # sensillum 1
# transd_params1 = (ab3A_params, ab3B_params, ornXXC_params)
# sens_params1   = dict([
#                     ('n_neu', transd_params1.__len__()),
#                     ('n_orns_recep', n_orns_recep),
#                     ('od_pref' , od_pref),
#     # NSI params
#                     ('w_nsi', .00000002), 
#                     ('transd_params', transd_params1),
#                     ])

# # sensillum 2
# transd_params2 = (ab3A_params, ab3B_params, )
# sens_params2   = dict([
#                     ('n_neu', transd_params2.__len__()),
#                     ('n_orns_recep', n_orns_recep),
#                     ('od_pref' , od_pref),
#     # NSI params
#                     ('w_nsi', .2), 
#                     ('transd_params', transd_params2),
#                     ])

orn_layer_params = []
orn_layer_params.append(sens_params0)
# orn_layer_params.append(sens_params1)
# orn_layer_params.append(sens_params2)
# orn_layer_params.append(sens_params1)

#################### END PARAMS SETTINGS FOR ORNS #############################


#################### AL + ORNs NETWORK PARAMS #################################

n_sens_type       = orn_layer_params.__len__()  # number of type of sensilla
n_recep_list      = np.zeros(n_sens_type, dtype=int)
for st in range(n_sens_type):
    n_recep_list[st]      = orn_layer_params[st]['n_neu'] #[n_neu, n_neu]    # number of ORNs per sensilla


# AL DYNAMICS PARAMETERS 

al_params  = dict([
                    ('n_pns_recep', 5),
                    ('n_lns_recep', 3),
                    ('theta', -30),                     # 1
                    ('t_ref', orn_params['t_ref']),
                    ('n_recep_list', n_recep_list),
                    ('n_sens_type', n_sens_type),                    
                    ('n_orns_recep', n_orns_recep),                    
                        ])

pn_ln_params = dict([
                    # CHANGED params
                    ('vrev_pn',     0),    # 15 [mV] reversal potential
                    ('vrest_pn',  -65),    # -6.5 [mV] resting potential
                    
                    ('tau_s',       2* 10),    # 10 [ms]
                    ('alpha_orn',  .75* 3*4),   # 3  coeff for the ORN input to PNs                    
                    ('tau_v',      20*.5),    # .5 [ms]
                    
                    ('g_s',        1*1), # 1                                        
                    ('g_l',        1e-10*1), #  1 
                    
                    ('alpha_x',    0*2.4*3), # 2.4 ORN input coeff for adaptation variable x_pn
                    ('tau_x',      1*600),    # 600 [ms] time scale for dynamics of adaptation    
                                            # variable x_pn
                    
                    ('vpn_noise',  .0061*10),  # NEW # extra noise input to PNs
                    
                    # LN params
                    ('vln_noise',   0*350),    # NEW
                    ('alpha_pn',    3*.6),  # 3  # coeff for the PN input to LNs
                    
                    # LN params
                    ('vrest_ln', -55),   # -3[mV] resting potential
                    ('vrev_ln', -80),        # 15  [mV] reversal potential
                    ('tau_y', 600),
                    # LN to PN
                    ('alpha_ln', 0), # [ms]
                    ])

# ORNS layer SIMULATION
params_al_orn = dict([
                ('stim_params', stim_params),
                ('orn_layer_params', orn_layer_params),
                ('orn_params', orn_params),
                ('sdf_params', sdf_params),
                ('al_params', al_params),
                ('pn_ln_params',pn_ln_params),
                ])

fld_analysis = 'NSI_analysis/trials/' #Olsen2010

params_file = 'params_al_orn.ini'
# SAVE PARAMS IN THE STANDARD FOLDER AND FILE
with open(fld_analysis+params_file, 'wb') as f:
    pickle.dump(params_al_orn, f)


# %% LOAD PARAMS FROM A FILE

# fld_analysis = 'NSI_analysis/trials/' #Olsen2010
# name_data = 'params_al_orn.ini'
# params_al_orn = pickle.load(open(fld_analysis+ name_data,  "rb" ))

# stim_params = params_al_orn['stim_params']
# orn_layer_params= params_al_orn['orn_layer_params']
# orn_params= params_al_orn['orn_params']
# sdf_params= params_al_orn['sdf_params']
# al_params= params_al_orn['al_params']
# pn_ln_params= params_al_orn['pn_ln_params']

# n_sens_type       = orn_layer_params.__len__()  # number of type of sensilla



# %%
# stim params
delay                       = 0
t0                          = 500
stim_name                   = ''

stim_params['stim_type']    = 'ss' # 'ss'  # 'ts' # 'rs' # 'pl'
stim_dur                    = 500        # 10, 50, 200

stim_params['stim_dur']     = np.array([stim_dur, stim_dur])
stim_params['t_tot']        = t0+delay+stim_dur+300
stim_params['t_on']         = np.array([t0, t0+delay])

stim_params['conc0']        = 1.85e-4    # 1.85e-4  # fitting value: 2.85e-4
peak_ratio                  = 1
peaks                       = [1.85e-4, 3e-4, 2e-3, 2e-2,]#*np.logspace(-4, -2.5, 10)  # np.array([3e-4, 0.0006,0.0012, 0.0025,])#.005])#np.array([0.0001, 0.0006,0.0012, 0.0025, 0.005]) # np.logspace(-4, -1, 5) 

# nsi params
inh_cond                    = 'noin'    #['nsi', 'ln', 'noin'] #
nsi_str                     = .6
alpha_ln                    = 100

# output params
run_sims                    = 1     # Run sims or just load the data
data_save                   = 0
al_orn_1r_fig               = 1     # single run figure flag
fig_orn_al_name_1r             = 'ORN_AL_timecourse_' +inh_cond+\
                        '_stim_'+ stim_params['stim_type'] +'_dur_%d'%stim_dur

npeaks_fig                  = 0     # multiple peaks PN and ORN time course 
fig_orn_al_name             = 'Olsen2010_timecourse_' +inh_cond+\
                        '_stim_'+ stim_params['stim_type'] +'_dur_%d'%stim_dur

olsen_fig                   = 1     # PN vs ORN activity, like Olsen 2010
fig_olsen_fit_name          = 'Olsen2010' +inh_cond+\
                        '_stim_'+ stim_params['stim_type'] +'_dur_%d'%stim_dur

figs_save                   = 0


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
    
# INITIALIZE OUTPUT VARIABLES    
conc    = np.zeros_like(peaks)
conc_th = np.zeros_like(peaks)
nu_orn  = np.zeros_like(peaks)
nu_pn   = np.zeros_like(peaks)
nu_ln   = np.zeros_like(peaks)
nu_orn_err  = np.zeros_like(peaks)
nu_pn_err   = np.zeros_like(peaks)
nu_ln_err   = np.zeros_like(peaks)
nu_ln_err   = np.zeros_like(peaks)
nu_orn_w = np.zeros_like(peaks)
nu_pn_w = np.zeros_like(peaks)


if path.isdir(fld_analysis):
    print('Analysis fld: ' + fld_analysis)    
else:
    print('no fld analysis, please create one. thanks')

for id_c, peak in enumerate(peaks):
    if stim_params['stim_type'] == 'ext':
        stim_params['stim_data_name'] = stim_params['stim_data_name'][:-1]+str(peak)
        
        print(stim_params['stim_data_name'])
    else:
        stim_params['concs'] = np.array([peak, peak_ratio*peak])
    
    tic = tictoc() 
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
    if run_sims:
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
        fig_al_name     = fig_orn_al_name_1r + '_p' + str(peak) 
        if al_orn_1r_fig:
            fig_al_orn = figure_al_orn.main(params_al_orn, output_orn, output_al)
            if figs_save:
                print('saving single run time-course figure in '+fld_analysis)
                fig_al_orn.savefig(fld_analysis+fig_al_name+ '.png')  
    ###########################################################################

    #### LOAD DATA ############################################################
    else:
        if npeaks_fig | olsen_fig:
            data_params     = pickle.load(open(fld_analysis+ name_data,  "rb" ))
            params_al_orn   = data_params[0]
            output2an       = data_params[1]
    
    if npeaks_fig:
        if (id_c == 0):
            fig_npeaks_settings(ax_ornal, params_al_orn)
        fig_npeaks(output2an, params_al_orn, id_c)
        plt.show()
        
    out_olsen       = olsen2010_data(output2an, params_al_orn,)
    conc_th[id_c]   = peak
    conc[id_c]      = out_olsen[0]
    nu_orn[id_c]    = out_olsen[1]
    nu_pn[id_c]     = out_olsen[2]
    nu_ln[id_c]     = out_olsen[3]
    nu_orn_err[id_c] = out_olsen[4]
    nu_pn_err[id_c] = out_olsen[5]
    nu_ln_err[id_c] = out_olsen[6]
    
    nu_pn_w[id_c]     = out_olsen[7]
    nu_orn_w[id_c]    = out_olsen[8]
    

print('conc ratio: %d'%peak_ratio)
print('nu PN avg:')        
print(nu_pn)
print('nu PN weak avg:')        
print(nu_pn_w)
print('nu PN ratio avg:')        
print(nu_pn/nu_pn_w)
print('')
print('nu ORN avg:')        
print(nu_orn)
print('nu ORN weak avg:')        
print(nu_orn_w)
print('nu ORN ratio avg:')        
print(nu_orn/nu_orn_w)
# print('nu LN avg:')        
# print(nu_ln)

if figs_save & npeaks_fig:
    print('saving Olsen2010 time-course figure in '+fld_analysis)
    fig_pn.savefig(fld_analysis+  fig_orn_al_name+'_'+inh_cond+'.png')    
    

#%% FIGURE Olsen 2010: ORN vs PN during step stimulus
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
    
    axs[1].errorbar(conc_th+1e-5, nu_orn, yerr=nu_orn_err, linewidth=lw, 
       markersize=15, label='ORNs ')
    axs[1].errorbar(conc_th-1e-5, nu_pn, yerr=nu_pn_err, linewidth=lw, 
       markersize=15, label='PNs ')
    axs[1].errorbar(conc_th, nu_ln, yerr=nu_ln_err, linewidth=lw, 
       markersize=15, label='LNs ')
    axs[1].legend(loc='upper left', fontsize=legend_fs)

    axs[1].set_ylabel(r'Firing rates (Hz)', fontsize=label_fs)

    axs[1].set_xlabel(r'concentration [au]', fontsize=label_fs)
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
        print('saving Olsen2010 PN-ORN figure in '+fld_analysis)
        fig3.savefig(fld_analysis+  fig_olsen_fit_name+'_'+inh_cond+'.png')

       

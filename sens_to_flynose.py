#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:33:34 2020

sens_to_flynose.py

This script converts the output from NSI_ORN_LIF.py to 
the AL dynamics of flynose.py

@author: mario
"""

import numpy as np
import timeit

import NSI_ORN_LIF

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


def tictoc():
    return timeit.default_timer()

def pn2ln_v_ex(x0,t, s, ln_params, ):
#    ln_params = np.array([tau_s, tau_v, a_s_ln, vrev_ln, vrest_ln])
    tau_v = ln_params[1]
    
    vrev = ln_params[3]
    vrest = ln_params[4]
    
    # PN -> LN equations:
    # ORN -> PN equations:
    dt = t[1]-t[0]
    b = -(1 + s)/tau_v
    a = (vrest + s*vrev)/tau_v
    y = (x0 + a/b)*np.exp(b*dt)-a/b
    #dvdt = ((vrest-v) + s*(vrev-v) )/tau_v
    return y

def pn2ln_s_ex(x0,t, u_pn, ln_params, ):
    #    ln_params = np.array([tau_s, tau_v, a_s_ln, vrev_ln, vrest_ln])
    tau_s = ln_params[0]
    a_s = ln_params[2]
    
    # PN -> LN equation of s:
    b = (-1-a_s*u_pn)/tau_s
    a = a_s*u_pn/tau_s
    dt = t[1]-t[0]
    y = (x0 + a/b)*np.exp(b*dt)-a/b
#    dsdt = (a_s*u_pn*(1-s) - s)/tau_s       
    return y


def y_ln_fun_ex(y0, t, u_ln, tau_y, alpha_ln,):
    b = (-alpha_ln*u_ln-1)/tau_y
    a = alpha_ln*u_ln/tau_y
    dt = t[1]-t[0]
    y = (y0 + a/b)*np.exp(b*dt)-a/b
    return y

def orn2pn_s_ex(x0,t, u_orn, x_pn,y_ln,pn_params,):
    #    pn_params  = np.array([tau_s, tau_v, a_s_pn, vrev_pn, vrest_pn])
    tau_s = pn_params[0]
    a_s = pn_params[2]
    
    # ORN -> PN equations:
    b = (-1-a_s*u_orn*(1-x_pn)*(1-y_ln))/tau_s
    a = a_s*u_orn*(1-x_pn)*(1-y_ln)/tau_s
    dt = t[1]-t[0]
    y = (x0 + a/b)*np.exp(b*dt)-a/b
    return y

def orn2pn_v_ex(x0,t, s, pn_params,):
#    pn_params  = np.array([tau_s, tau_v, a_s_pn, vrev_pn, vrest_pn])
    tau_v = pn_params[1]
    
    vrev = pn_params[3]
    vrest = pn_params[4]
    
    # ORN -> PN equations:
    dt = t[1]-t[0]
    b = -(1 + s)/tau_v
    a = (vrest + s*vrev)/tau_v
    y = (x0 + a/b)*np.exp(b*dt)-a/b
#    dvdt = (vrest + s*vrev)/tau_v  - v*(1 + g*s)/tau_v
    return y

def x_adapt_ex(x0,t,u_orn, tau, a_ad,):
    b = (-a_ad*u_orn-1)/tau
    a = a_ad*u_orn/tau
    dt = t[1]-t[0]
    y = (x0 + a/b)*np.exp(b*dt)-a/b
    return y

#%% Simulation INI

# # Save params 
# stim_data_fld = ''
# fld_analysis    = 'NSI_analysis/trials'
   
# Stimulus params
stim_params     = dict([
                    ('stim_type' , 'ss'),   # 'ts'  # 'ss' # 'pl'
                    ('pts_ms' , 5),         # simulated pts per ms 
                    ('n_od', 2),            # number of odours
                    ('t_tot', 2000),        # ms 
                    ('conc0', [2.853669391e-04]),
                    ('r_noise', 0.1),
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
                    ('t_on', np.array([800, 800])),          # ms
                    ('concs', np.array([.01, .0001])),
                    ])

stim_params.update(concs_params)

# Sensilla/network parameters
n_sens              = 40
n_neu               = 2

# Transduction parameters
od_pref = np.array([[1,0], [0,1], [0, 1], [1,0], 
                    [0,0], [1,0], [0,1], [1,0]])
     
transd_vect_3A = od_pref[0,:]
transd_vect_3B = od_pref[1,:]

ab3A_params = dict([
    # Transduction params
                    ('n', .822066870*transd_vect_3A), 
                    ('alpha_r', 12.6228808*transd_vect_3A), 
                    ('beta_r', 7.6758436748e-02*transd_vect_3A),
                    ])

ab3B_params = dict([
    # Transduction params
                    ('n', .822066870*transd_vect_3B), 
                    ('alpha_r', 12.6228808*transd_vect_3B), 
                    ('beta_r', 7.6758436748e-02*transd_vect_3B),
                    ])

transd_params = (ab3A_params, ab3B_params)

sens_params     = dict([
                    ('n_neu', n_neu),
                    ('n_sens', n_sens),
                    ('od_pref' , od_pref),
    # NSI params
                    ('w_nsi', .000000015), 
                    ('transd_params', transd_params),
                    ])

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
    # Adaptation params
                    ('alpha_y', .45310619), 
                    ('beta_y', 3.467184e-03), 
                    ])


# SDF/Analysis params
tau_sdf             = 41
dt_sdf              = 5
sdf_params          = [tau_sdf, dt_sdf, ]

#**************************************
# ORN to AL CONNECTIVITY MATRIX
num_glo_list        = [n_neu, n_neu]#[4,3,2,1]     # number of glomeruli per sensilla
num_sens_type       = len(num_glo_list)  # number of type of sensilla
num_glo_tot         = sum(num_glo_list) # number of glomeruli in total

# NETWORK PARAMETERS 
num_orns_pn         = 18    # number of ORNs per each PN in each glomerulus
num_orns_glo        = 40    # number of ORNs per each glomerulus
num_orns_tot        = num_orns_glo*num_glo_tot  # total number of ORNs 
num_pns_glo         = 5     # number of PNs per each glomerulus
num_lns_glo         = 3     # number of LNs per each glomerulus
num_pns_tot         = num_pns_glo*num_glo_tot # number of total PNs
num_lns_tot         = num_lns_glo*num_glo_tot # number of total LNs 

pts_ms              = stim_params['pts_ms']
t_tot               = stim_params['t_tot']
n2sim               = pts_ms*t_tot       # number of time points
u_orn               = np.zeros((n2sim, num_pns_tot))
sdf_size            = int(t_tot/dt_sdf)
    
# ORN, PN and LN PARAMETERS
spike_length        = int(4*pts_ms)     # [ms]
t_ref               = 2*pts_ms          # ms; refractory period 
theta               = 1                 # [mV] firing threshold

orn_spike_height    = .3
pn_spike_height     = .3
ln_spike_height     = .3

# Each PN belongs to ONLY one of the glomeruli
ids_glo     = np.arange(num_glo_tot)
ids_pn_glo  = np.repeat(ids_glo, num_pns_glo)

# Connectivity matrices between ORNs and PNs 
orn_pn_mat          = np.zeros((num_orns_tot, num_pns_tot))  
for pp in range(num_pns_tot):
    rnd_ids             = np.random.permutation(num_orns_glo) 
    tmp_ids             = rnd_ids[:num_orns_pn] + \
        num_orns_glo*ids_pn_glo[pp]
    # ids_orn_pn[pp,:]    = tmp_ids
    orn_pn_mat[tmp_ids, pp] = orn_spike_height

# Connectivity matrices between PNs and LNs
pn_ln_mat           = np.zeros((num_pns_tot, num_lns_tot))
for pp in range(num_glo_tot):
    pn_ln_mat[pp*num_pns_glo:(pp+1)*num_pns_glo,
              pp*num_lns_glo:(pp+1)*num_lns_glo] = pn_spike_height

glo_id = 0        
ln_pn_mat           = np.zeros((num_lns_tot,num_pns_tot))
for pp in range(num_sens_type):
    num_glo = num_glo_list[pp]
    # Inhibitory LN connectivity within glomeruli cluster
    ln_pn_mat[(glo_id*num_lns_glo):((glo_id+num_glo)*num_lns_glo),
              (glo_id*num_pns_glo):((glo_id+num_glo)*num_pns_glo)] = ln_spike_height
    for qq in range(num_glo):
        # PN innervating LN are not inhibited
        ln_pn_mat[((glo_id+qq)*num_lns_glo):((glo_id+qq+1)*num_lns_glo),
                  ((glo_id+qq)*num_pns_glo):((glo_id+qq+1)*num_pns_glo)] = 0
    glo_id = glo_id + num_glo

#%%*********************************************************************
# SIMULATIONS

# ORN LIF SIMULATION
num_spikes_orn_tot = np.zeros((n2sim, num_orns_tot))
orn_sdf_tot = np.zeros((sdf_size, num_orns_tot))

for id_sens, n_neu in enumerate(num_glo_list):
    tic = tictoc()
    orn_lif_out   = NSI_ORN_LIF.main(orn_params, stim_params, 
                                 sdf_params, sens_params)
    toc = tictoc()
    [t, u_od, r_orn, v_orn, y_orn, 
     num_spikes_orn, spike_matrix, orn_sdf, orn_sdf_time] = orn_lif_out 
    
    ids_orn = np.arange(n_neu*num_orns_glo)+id_sens*num_orns_glo
    num_spikes_orn_tot[:, ids_orn] = num_spikes_orn
    orn_sdf_tot[:, ids_orn] = orn_sdf



#%%
orn_spike_all = num_spikes_orn_tot.dot(orn_pn_mat) 
u_orn =  orn_spike_all
t_zeros = np.zeros((1, num_pns_tot))
for tt in range(spike_length-1):
    orn_spike_all = np.concatenate((t_zeros, orn_spike_all[:-1,:]))
    u_orn = u_orn + orn_spike_all
    
#%%
tic = tictoc()

al_dyn = 1

# *****************************************************************
# AL SIMULATION 
# *****************************************************************
  
alpha_ln = 0.03                           
# *****************************************************************
# PN and LN PARAMETERS and OUTPUT VECTORS

tau_v               = .5        # [ms]
tau_s               = 10        # [ms]

#**************************************
# PN PARAMETERS
a_s_pn              = 2.5       #     
vrest_pn            = -6.5      # [mV] resting potential
vrev_pn             = 15.0      # [mV] reversal potential

alpha_x             = 2.         # ORN input coeff for adaptation variable x_pn
tau_x               = 600    # [ms] time scale for dynamics of adaptation variable x_pn
x_pn0               = 0.48*np.ones(num_pns_tot)     # 0.27

pn_params  = np.array([tau_s, tau_v, a_s_pn, vrev_pn, vrest_pn])

#**************************************
# LN PARAMETERS
a_s_ln              = 2.5       #     
vrest_ln            = -3.0      # -1.5 [mV] resting potential
vrev_ln             = 15.0      # [mV] reversal potential


tau_y               = 600    # [ms] time scale for dynamics of adaptation variable y_ln
y_ln0               = 0.025*np.ones(num_pns_tot) # 0.2
ln_params = np.array([tau_s, tau_v, a_s_ln, vrev_ln, vrest_ln])
#**************************************

# INITIALIZE LN to PN output vectors
x_pn            = np.zeros((n2sim, num_pns_tot))
u_pn            = np.zeros((n2sim, num_lns_tot))
u_ln            = np.zeros((n2sim, num_pns_tot))
y_ln            = np.zeros((n2sim, num_pns_tot))

# INITIALIZE PN output vectors
num_spike_pn    = np.zeros((n2sim, num_pns_tot))

# INITIALIZE LN output vectors
s_ln            = np.zeros((n2sim, num_lns_tot))
v_ln            = np.zeros((n2sim, num_lns_tot))
num_spike_ln    = np.zeros((n2sim, num_lns_tot))  

# PN and LN params initial conditions
x_pn[0, :]      = x_pn0
s_pn            = np.zeros((n2sim, num_pns_tot))
v_pn            = np.ones((n2sim, num_pns_tot))*vrest_pn
pn_ref_cnt      = np.zeros(num_pns_tot) # Refractory period counter starts from 0

y_ln[0, :]      = y_ln0
s_ln            = np.zeros((n2sim, num_lns_tot))
v_ln            = np.ones((n2sim, num_lns_tot))*vrest_ln
ln_ref_cnt      = np.zeros(num_lns_tot) # initially the ref period cnter is equal to 0
        


if al_dyn:
    # *****************************************************************
    # solve ODE for PN and LN
    for tt in range(1, n2sim-t_ref-1):
        # span for next time step
        tspan = [t[tt-1],t[tt]]
        
        pp_rnd = np.arange(num_pns_tot) # np.random.permutation(num_pns_tot)
        
        # ******************************************************************
        # Vectorized and fast UPDATE PNS 
        # ******************************************************************
        # adaptation variable of PN neuron
        x_pn[tt, pp_rnd] = x_adapt_ex(x_pn[tt-1,pp_rnd],tspan, 
                u_orn[tt, pp_rnd], tau_x, alpha_x, )        
    
        # Inhibitory input to PNs
        y_ln[tt, pp_rnd] = y_ln_fun_ex(y_ln[tt-1, pp_rnd],tspan, 
                u_ln[tt, pp_rnd], tau_y, alpha_ln, )
    
        # *********************************
        # ORN -> PN synapses
        
        # *********************************
        # PNs whose ref_cnt is equal to zero:
        pn_ref_0 = pn_ref_cnt==0
        s_pn[tt, pn_ref_0] = orn2pn_s_ex(s_pn[tt-1, pn_ref_0],tspan, 
            u_orn[tt, pn_ref_0], x_pn[tt-1, pn_ref_0], y_ln[tt-1, pn_ref_0], pn_params, )
        v_pn[tt, pn_ref_0] = orn2pn_v_ex(v_pn[tt-1, pn_ref_0],tspan, 
                s_pn[tt-1, pn_ref_0], pn_params, )
        
        # *********************************
        # PNs whose ref_cnt is different from zero:
        pn_ref_no0 = pn_ref_cnt!=0
        # Refractory period count down
        pn_ref_cnt[pn_ref_no0] = pn_ref_cnt[pn_ref_no0] - 1  
        
        # PNs whose Voltage is above threshold AND whose ref_cnt is equal to zero:
        pn_above_thr = (v_pn[tt, :] >= theta) & (pn_ref_cnt==0)
        num_spike_pn[tt, pn_above_thr] = num_spike_pn[tt, pn_above_thr] + 1
        u_pn[tt:tt+spike_length, :] = (u_pn[tt:tt+spike_length, :] + 
                np.sum(pn_ln_mat[pn_above_thr,:], axis=0))
        pn_ref_cnt[pn_above_thr] = t_ref
        
        # *********************************
        # PN -> LN synapses        
        
        # *********************************
        # LNs whose ref_cnt is equal to zero:
        ln_ref_0 = ln_ref_cnt==0
        s_ln[tt, ln_ref_0] = pn2ln_s_ex(s_ln[tt-1, ln_ref_0], tspan, 
                    u_pn[tt, ln_ref_0], ln_params, )
        v_ln[tt, ln_ref_0] = pn2ln_v_ex(v_ln[tt-1, ln_ref_0], tspan, 
                    s_ln[tt-1, ln_ref_0], ln_params, )
        
        # *********************************
        # LNs whose ref_cnt is different from zero:
        ln_ref_no0 = ln_ref_cnt!=0
        # Refractory period count down
        ln_ref_cnt[ln_ref_no0] = ln_ref_cnt[ln_ref_no0] - 1  
        
        # LNs whose Voltage is above threshold AND whose ref_cnt is equal to zero:
        ln_above_thr = (v_ln[tt, :] >= theta) & (ln_ref_cnt==0)
        num_spike_ln[tt, ln_above_thr] = num_spike_ln[tt, ln_above_thr] + 1
        u_ln[tt:tt+spike_length, :] = (u_ln[tt:tt+spike_length, :] + 
                    np.sum(ln_pn_mat[ln_above_thr,:], axis=0))
        ln_ref_cnt[ln_above_thr] = t_ref
        # ******************************************************************
        
    # *****************************************************************
    # Calculate the spike matrix of PNs and LNs
    pn_spike_matrix = np.asarray(np.where(num_spike_pn))
    pn_spike_matrix[0,:] = pn_spike_matrix[0,:]/pts_ms
    pn_spike_matrix = np.transpose(pn_spike_matrix)
    
    ln_spike_matrix = np.asarray(np.where(num_spike_ln))
    ln_spike_matrix[0,:] = ln_spike_matrix[0,:]/pts_ms
    ln_spike_matrix = np.transpose(ln_spike_matrix)
    
    

else:
    [pn_spike_matrix, ln_spike_matrix, ] = [np.nan, np.nan]

    
toc = tictoc()

print(toc-tic)


# %%******************************************
# FIGURE ORN, PN, LN
# Generate a figure per each sensillum type

al_fig = 1
stim_type = 'ss'

t_on = stim_params['t_on']
if al_dyn & al_fig:
    # *****************************************************************
    # Calculate the SDF for PNs and LNs
    pn_sdf, pn_sdf_time = sdf_krofczik.main(pn_spike_matrix, sdf_size,
                                                 tau_sdf, dt_sdf)  # (Hz, ms)
    pn_sdf= pn_sdf*1e3

    ln_sdf, ln_sdf_time = sdf_krofczik.main(ln_spike_matrix, sdf_size,
                                                 tau_sdf, dt_sdf)  # (Hz, ms)
    ln_sdf= ln_sdf*1e3
    
    t2plot = -100, 300#t_tot 
    rs = 4 # number of rows
    cs = 1 # number of cols
    fig_size = [7, 8] 
    fig_color = ['purple','green','cyan','red']
    
    if stim_type == 'pl':
        #lw = 1.1
        t2plot = 0, 4000
        rs = 2 # number of rows
        cs = 2 # number of cols
        fig_size = [10, 5]

    glo_id = 0
    for qq in range(num_sens_type):
        num_glo = num_glo_list[qq]
        
        ax_conc = plt.subplot(rs, cs, 1)
        ax_orn = plt.subplot(rs, cs, 2)
        ax_pn = plt.subplot(rs, cs, 3)
        ax_ln = plt.subplot(rs, cs, 4)
        fig_pn = plt.figure(figsize=fig_size)
        # ax_conc.plot(t-t_on[0], 100*u_od[:,0], color=green, linewidth=lw+2, 
        #                   label='glom : '+'%d'%(1))
        # ax_conc.plot(t-t_on[0], 100*u_od[:,1], '--',color=purple, linewidth=lw+1, 
        #                   label='glom : '+'%d'%(2))
        
        for ll in range(num_glo):
            ax_orn.plot(orn_sdf_time-t_on[0], np.mean(orn_sdf_tot[:,glo_id*num_orns_glo:((glo_id+1)*num_orns_glo)], axis=1),
                                                  color=fig_color[ll], linewidth=lw+1,label='sdf glo')
            
            ax_pn.plot(pn_sdf_time-t_on[0], pn_sdf[:, 
                        glo_id*num_pns_glo:((glo_id+1)*num_pns_glo)], 
                       '--', color=fig_color[ll], linewidth=lw, label='PN')
            
            ax_ln.plot(ln_sdf_time-t_on[0], ln_sdf[:,glo_id*num_lns_glo:((glo_id+1)*num_lns_glo)], '--',color=fig_color[ll], 
                                  linewidth=lw, label='LN')
            glo_id = glo_id+1
            
        ax_conc.set_xlim(t2plot)
        ax_orn.set_xlim(t2plot)
        ax_pn.set_xlim(t2plot)
        ax_ln.set_xlim(t2plot)
        
        ax_orn.set_ylim((0, 150))
        ax_pn.set_ylim((0, 180))
        ax_ln.set_ylim((0, 230))

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

        # # tmp
        # if not(stim_type == 'pl'):
        #     title_fs = 30
        #     if (params2an[1] ==0) & (params2an[1] ==0):
        #         ax_conc.set_title('a. Independent', fontsize=title_fs)
        #     elif (params2an[1] >0):
        #         ax_conc.set_title('b. LN inhib.', fontsize=title_fs)
        #     else:
        #          ax_conc.set_title('c. NSI', fontsize=title_fs)   
             
        ax_conc.spines['right'].set_color('none')
        ax_conc.spines['top'].set_color('none')
        ax_orn.spines['right'].set_color('none')
        ax_orn.spines['top'].set_color('none')
        ax_pn.spines['right'].set_color('none')
        ax_pn.spines['top'].set_color('none')
        ax_ln.spines['right'].set_color('none')
        ax_ln.spines['top'].set_color('none')
        
        
        # Manual positioning of subplot
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
    
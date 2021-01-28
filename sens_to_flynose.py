#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:33:34 2020

sens_to_flynose.py

This script is the raw model for flynose2.0. It runs NSI_ORN_LIF.py to 
generate ORN activity and then run the AL dynamics.

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
#    ln_params = np.array([tau_s, tau_v, a_s_ln, vrev_ln, vrest_ln, vln_noise])
    tau_v = ln_params[1]
    
    vrev = ln_params[3]
    vrest = ln_params[4]
    vln_noise = ln_params[5]*1*(-.5+np.random.uniform(0, 1, size=np.shape(x0)))
    
    # PN -> LN equations:
    # ORN -> PN equations:
    dt = t[1]-t[0]
    b = -(1 + s)/tau_v
    a = (vrest + s*vrev + vln_noise)/tau_v
    y = (x0 + a/b)*np.exp(b*dt)-a/b
    #dvdt = ((vrest-v) + s*(vrev-v) + v_bckgnd)/tau_v
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
#    pn_params  = np.array([tau_s, tau_v, a_s_pn, vrev_pn, vrest_pn, vpn_noise])
    tau_v = pn_params[1]
    
    vrev = pn_params[3]
    vrest = pn_params[4]
    vpn_noise = pn_params[5]*(-.5+np.random.uniform(0, 1, size=np.shape(x0)))
    
    # ORN -> PN equations:
    dt = t[1]-t[0]
    b = -(1 + s)/tau_v
    a = (vrest + s*vrev + vpn_noise)/tau_v
    y = (x0 + a/b)*np.exp(b*dt)-a/b
#    dvdt = (vrest + s*vrev + v_bckgnd)/tau_v  - v*(1 + g*s)/tau_v
    return y

def x_adapt_ex(x0,t,u_orn, tau, a_ad,):
    b = (-a_ad*u_orn-1)/tau
    a = a_ad*u_orn/tau
    dt = t[1]-t[0]
    y = (x0 + a/b)*np.exp(b*dt)-a/b
    return y

# Simulation INI

# # Save params 
# stim_data_fld = ''
# fld_analysis    = 'NSI_analysis/trials'
   
# Stimulus params
stim_params     = dict([
                    ('stim_type' , 'rs'),   # 'ts'  # 'ss' # 'pl'
                    ('pts_ms' , 5),         # simulated pts per ms 
                    ('n_od', 2),            # number of odours
                    ('t_tot', 4000),        # ms 
                    ('conc0', [2.85e-04]),    # 2.854e-04
                    ('od_noise', 00), 
                    ('r_noise', 2.0), 
                    ('filter_frq', 0.006),#0.001
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
tau_sdf             = 41
dt_sdf              = 5
sdf_params          = [tau_sdf, dt_sdf, ]

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
                    ('w_nsi', .00000002), 
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
# orn_layer.append(sens_params2)
# orn_layer.append(sens_params1)

n_sens_type       = orn_layer.__len__()  # number of type of sensilla
n_recep_list      = np.zeros(n_sens_type, dtype=int)
for st in range(n_sens_type):
    n_recep_list[st]      = orn_layer[st]['n_neu'] #[n_neu, n_neu]    # number of ORNs per sensilla

n_recep_tot       = sum(n_recep_list) # number of receptors in total

# AL + ORN layer network parameters
n_orns_pn         = n_orns_recep    # number of ORNs per each PN in each glomerulus
n_orns_tot        = n_orns_recep*n_recep_tot  # total number of ORNs 

n_pns_recep       = 5     # number of PNs per each glomerulus
n_lns_recep       = 3     # number of LNs per each glomerulus
n_pns_tot         = n_pns_recep*n_recep_tot # number of total PNs
n_lns_tot         = n_lns_recep*n_recep_tot # number of total LNs 

pts_ms              = stim_params['pts_ms']
t_tot               = stim_params['t_tot']
n2sim               = pts_ms*t_tot       # number of time points
u_orn               = np.zeros((n2sim, n_pns_tot))
sdf_size            = int(t_tot/dt_sdf)

# ORN, PN and LN PARAMETERS
spike_length        = int(4*pts_ms)     # [ms]
t_ref               = 2*pts_ms          # ms; refractory period 
theta               = 1                 # [mV] firing threshold

orn_spike_height    = .3
pn_spike_height     = .3
ln_spike_height     = .3

# Each PN belongs to ONLY one of the glomeruli
ids_recep     = np.arange(n_recep_tot)
ids_pn_recep  = np.repeat(ids_recep, n_pns_recep)

# Connectivity matrices between ORNs and PNs 
orn_pn_mat          = np.zeros((n_orns_tot, n_pns_tot))  
for pp in range(n_pns_tot):
    rnd_ids             = np.random.permutation(n_orns_recep) 
    tmp_ids             = rnd_ids[:n_orns_pn] + \
        n_orns_recep*ids_pn_recep[pp]
    # ids_orn_pn[pp,:]    = tmp_ids
    orn_pn_mat[tmp_ids, pp] = orn_spike_height

# Connectivity matrices between PNs and LNs
pn_ln_mat           = np.zeros((n_pns_tot, n_lns_tot))
for pp in range(n_recep_tot):
    pn_ln_mat[pp*n_pns_recep:(pp+1)*n_pns_recep,
              pp*n_lns_recep:(pp+1)*n_lns_recep] = pn_spike_height

recep_id = 0        
ln_pn_mat           = np.zeros((n_lns_tot,n_pns_tot))
for pp in range(n_sens_type):
    num_recep = n_recep_list[pp]
    # Inhibitory LN connectivity within receptors cluster
    ln_pn_mat[(recep_id*n_lns_recep):((recep_id+num_recep)*n_lns_recep),
              (recep_id*n_pns_recep):((recep_id+num_recep)*n_pns_recep)] = ln_spike_height
    for qq in range(num_recep):
        # PN innervating LN are not inhibited
        ln_pn_mat[((recep_id+qq)*n_lns_recep):((recep_id+qq+1)*n_lns_recep),
                  ((recep_id+qq)*n_pns_recep):((recep_id+qq+1)*n_pns_recep)] = 0
    recep_id = recep_id + num_recep


# flynose verbose description 
print('flynose Simulation ')    
print('')
print('In the ORNs layer there are %d types of sensilla' %(n_sens_type, ))
print('and %d identical sensilla of each type' %(n_orns_recep, ))

for st in range(n_sens_type):
    print('   Sensillum %d has %d ORNs of different type' %(st, n_recep_list[st]))
print('In total, there are %d ORNs of %d different types' %(n_orns_tot, n_recep_tot))
print('')

print('In the AL there are %d glomeruli. One per each receptor type.' %n_recep_tot)
print('Each glomerulus has %d PNs and %d LNs' %(n_pns_recep, n_lns_recep))
print('In total, AL is compound of %d PNs and %d LNs' %(n_pns_tot, n_lns_tot))
print('Each PNs receives input from %d random ORNs of the same type' %(n_orns_pn))
print('')

print('flynose is presented with an odour mixtures containing %d odorants' %n_od)


#%% ***************************************************************************
# SIMULATIONS

# ORN LIF SIMULATION
orn_spikes_t = np.zeros((n2sim, n_orns_tot))
orn_sdf = np.zeros((sdf_size, n_orns_tot))
tic = tictoc()
id_orn0 = 0 

for id_sens, n_neu in enumerate(n_recep_list):
    orn_lif_out   = NSI_ORN_LIF.main(orn_params, stim_params, 
                                 sdf_params, orn_layer[id_sens])
    [t, u_od, r_orn, v_orn, y_orn, 
     n_spikes_orn_tmp, spike_matrix, orn_sdf_tmp, orn_sdf_time] = orn_lif_out 
    
    ids_orn = np.arange(n_neu*n_orns_recep) + id_orn0 
    
    orn_spikes_t[:, ids_orn] = n_spikes_orn_tmp
    orn_sdf[:, ids_orn] = orn_sdf_tmp
    
    id_orn0 = ids_orn[-1]+1


# Generate input to PNs
orn_spikes_all = orn_spikes_t.dot(orn_pn_mat) 
u_orn =  orn_spikes_all
t_zeros = np.zeros((1, n_pns_tot))
for tt in range(spike_length-1):
    orn_spikes_all = np.concatenate((t_zeros, orn_spikes_all[:-1,:]))
    u_orn = u_orn + orn_spikes_all

orn_spikes_all = None
t_zeros = None 

toc = tictoc()
# print('ORNs sim time: %.2f s' %(toc-tic,))


# %%  AL DYNAMICS SIMULATION 

tic = tictoc()

al_dyn = 1
  
alpha_ln = 0.0#3                           

# PN and LN PARAMETERS and OUTPUT VECTORS
tau_v               = .5        # [ms]
tau_s               = 10        # [ms]

# PN PARAMETERS
a_s_pn              = 2.5       #     
vrest_pn            = -6.5      # [mV] resting potential
vrev_pn             = 15.0      # [mV] reversal potential
vpn_noise           = 6         # extra noise input to PNs

alpha_x             = 2.         # ORN input coeff for adaptation variable x_pn
tau_x               = 600    # [ms] time scale for dynamics of adaptation variable x_pn
x_pn0               = 0.48*np.ones(n_pns_tot)     # 0.27

pn_params  = np.array([tau_s, tau_v, a_s_pn, vrev_pn, vrest_pn, vpn_noise])

# LN PARAMETERS
a_s_ln              = 2.5       #     
vrest_ln            = -3.0      # -1.5 [mV] resting potential
vrev_ln             = 15.0      # [mV] reversal potential
vln_noise           = 1         # extra noise input to LNs

tau_y               = 600    # [ms] time scale for dynamics of adaptation variable y_ln
y_ln0               = 0.025*np.ones(n_pns_tot) # 0.2
ln_params = np.array([tau_s, tau_v, a_s_ln, vrev_ln, vrest_ln, vln_noise])

# Initialize LN to PN output vectors
x_pn            = np.zeros((n2sim, n_pns_tot))
u_pn            = np.zeros((n2sim, n_lns_tot))
s_pn            = np.zeros((n2sim, n_pns_tot))
v_pn            = np.ones((n2sim, n_pns_tot))*vrest_pn

u_ln            = np.zeros((n2sim, n_pns_tot))
y_ln            = np.zeros((n2sim, n_pns_tot))

# Initialize PN output vectors
num_spike_pn    = np.zeros((n2sim, n_pns_tot))

pn_ref_cnt      = np.zeros(n_pns_tot) # Refractory period counter starts from 0

# Initialize LN output vectors
s_ln            = np.zeros((n2sim, n_lns_tot))
v_ln            = np.zeros((n2sim, n_lns_tot))
num_spike_ln    = np.zeros((n2sim, n_lns_tot))  

# PN and LN params initial conditions
s_pn[0, :]      = 0.2*(1 + np.random.standard_normal((1, n_pns_tot)))
x_pn[0, :]      = x_pn0*(1 + np.random.standard_normal((1, n_pns_tot)))
v_pn[0,:]      = .5*np.ones((1, n_pns_tot)) \
    + np.random.standard_normal((1, n_pns_tot)) 

s_ln[0, :]      = 0.2*(1 + np.random.standard_normal((1, n_lns_tot)))
y_ln[0, :]      = y_ln0*(1 + np.random.standard_normal((1, n_pns_tot)))
v_ln[0,:]      = .5*np.ones((1, n_lns_tot)) \
    + np.random.standard_normal((1, n_lns_tot)) 

ln_ref_cnt      = np.zeros(n_lns_tot) # initially the ref period cnter is equal to 0

if al_dyn:
    # *****************************************************************
    # solve ODE for PN and LN
    for tt in range(1, n2sim-t_ref-1):
        # span for next time step
        tspan = [t[tt-1],t[tt]]
        
        pp_rnd = np.arange(n_pns_tot) # np.random.permutation(n_pns_tot)
        
        # Adaptation variable of PN neuron
        x_pn[tt, pp_rnd] = x_adapt_ex(x_pn[tt-1,pp_rnd],tspan, 
                u_orn[tt, pp_rnd], tau_x, alpha_x, )        
    
        # Inhibitory input to PNs
        y_ln[tt, pp_rnd] = y_ln_fun_ex(y_ln[tt-1, pp_rnd],tspan, 
                u_ln[tt, pp_rnd], tau_y, alpha_ln, )
    
        # ORN -> PN synapses
        
        # PNs whose ref_cnt is equal to zero:
        pn_ref_0 = pn_ref_cnt==0
        s_pn[tt, pn_ref_0] = orn2pn_s_ex(s_pn[tt-1, pn_ref_0],tspan, 
            u_orn[tt, pn_ref_0], x_pn[tt-1, pn_ref_0], y_ln[tt-1, pn_ref_0], pn_params, )
        v_pn[tt, pn_ref_0] = orn2pn_v_ex(v_pn[tt-1, pn_ref_0],tspan, 
                s_pn[tt-1, pn_ref_0], pn_params, )
        
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
        
        # PN -> LN synapses        
            
        # LNs whose ref_cnt is equal to zero:
        ln_ref_0 = ln_ref_cnt==0
        s_ln[tt, ln_ref_0] = pn2ln_s_ex(s_ln[tt-1, ln_ref_0], tspan, 
                    u_pn[tt, ln_ref_0], ln_params, )
        v_ln[tt, ln_ref_0] = pn2ln_v_ex(v_ln[tt-1, ln_ref_0], tspan, 
                    s_ln[tt-1, ln_ref_0], ln_params, )
        
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
        
    # Calculate the spike matrix of PNs and LNs
    pn_spike_matrix = np.asarray(np.where(num_spike_pn))
    pn_spike_matrix[0,:] = pn_spike_matrix[0,:]/pts_ms
    pn_spike_matrix = np.transpose(pn_spike_matrix)
    
    ln_spike_matrix = np.asarray(np.where(num_spike_ln))
    ln_spike_matrix[0,:] = ln_spike_matrix[0,:]/pts_ms
    ln_spike_matrix = np.transpose(ln_spike_matrix)
    toc = tictoc()

    # print('AL sim time: %.2f s' %(toc-tic,))

else:
    # output values w/o simulations
    [pn_spike_matrix, ln_spike_matrix, ] = [np.nan, np.nan]
    print('No AL dynamics')
    
#%% ORN correlation analysis
corr_orn = np.zeros((n_orns_tot, n_orns_tot))
corr_vorn = np.zeros((n_orns_tot, n_orns_tot))
for nn1 in range(n_orns_tot):
    for nn2 in range(n_orns_tot):
        if nn2>nn1:
            pip1 = v_orn[::5, nn1]
            pip2 = v_orn[::5, nn2]
            corr_vorn[nn1, nn2] = np.corrcoef((pip1,pip2))[0,1]
            corr_vorn[nn2, nn1] = corr_vorn[nn1, nn2]
            
            pip1 = np.zeros(t_tot)
            pip2 = np.zeros(t_tot)
            pip1[spike_matrix[spike_matrix[:,1] == nn1, 0]] = 1
            pip2[spike_matrix[spike_matrix[:,1] == nn2, 0]] = 1
            corr_orn[nn1, nn2] = np.corrcoef((pip1,pip2))[0,1]
            corr_orn[nn2, nn1] = corr_orn[nn1, nn2]
            
tmp_corr = corr_vorn[:n_orns_recep, :n_orns_recep]
tmp_corr[tmp_corr!=0]
corr_orn_hom = np.mean(tmp_corr[tmp_corr!=0])
corr_orn_het = np.mean(corr_vorn[:n_orns_recep, n_orns_recep:]) # corr_pn[0,-1]
print('ORNs, Hom and Het Potent corr: %.3f and %.3f' 
      %(corr_orn_hom, corr_orn_het))

tmp_corr = corr_orn[:n_orns_recep, :n_orns_recep]
tmp_corr[tmp_corr!=0]
corr_orn_hom = np.mean(tmp_corr[tmp_corr!=0])
corr_orn_het = np.mean(corr_orn[:n_orns_recep, n_orns_recep:]) # corr_pn[0,-1]
print('ORNs, Hom and Het spk cnt corr: %.3f and %.3f' 
      %(corr_orn_hom, corr_orn_het))

orn_avg = np.mean(orn_sdf)
print('ORNs, FR avg: %.2f Hz' %orn_avg)
print('')

if al_dyn:   
    #%% PNs correlation analysis
    corr_pn = np.zeros((n_pns_tot, n_pns_tot))
    corr_vpn = np.zeros((n_pns_tot, n_pns_tot))
    for nn1 in range(n_pns_tot):
        for nn2 in range(n_pns_tot):
            if nn2>nn1:
                pip1 = np.zeros(t_tot)
                pip2 = np.zeros(t_tot)
                pip1[pn_spike_matrix[pn_spike_matrix[:,1] == nn1, 0]] = 1
                pip2[pn_spike_matrix[pn_spike_matrix[:,1] == nn2, 0]] = 1
    
                corr_pn[nn1, nn2] = np.corrcoef((pip1,pip2))[0,1]
                corr_pn[nn2, nn1] = corr_pn[nn1, nn2]
                
                pip1 = v_pn[::pts_ms, nn1]
                pip2 = v_pn[::pts_ms, nn2]
                corr_vpn[nn1, nn2] = np.corrcoef((pip1, pip2))[0,1]
                corr_vpn[nn2, nn1] = corr_vpn[nn1, nn2]
                
    tmp_corr = corr_vpn[:n_pns_recep, :n_pns_recep]
    tmp_corr[tmp_corr!=0]
    corr_pn_hom = np.mean(tmp_corr[tmp_corr!=0])
    corr_pn_het = np.mean(corr_vpn[:n_pns_recep, n_pns_recep:]) # corr_pn[0,-1]
    print('PNs, Hom and Het Potent corr: %.3f and %.3f' 
          %(corr_pn_hom, corr_pn_het))
    
    tmp_corr = corr_pn[:n_pns_recep, :n_pns_recep]
    tmp_corr[tmp_corr!=0]
    corr_pn_hom = np.mean(tmp_corr[tmp_corr!=0])
    corr_pn_het = np.mean(corr_pn[:n_pns_recep, n_pns_recep:]) # corr_pn[0,-1]
    print('PNs, Hom and Het spk cnt corr: %.3f and %.3f' 
          %(corr_pn_hom, corr_pn_het))
    
    print('')
    #%% LNs correlation analysis
    corr_ln = np.zeros((n_lns_tot, n_lns_tot))
    corr_vln = np.zeros((n_lns_tot, n_lns_tot))
    for nn1 in range(n_lns_tot):
        for nn2 in range(n_lns_tot):
            if nn2>nn1:
                pip1 = np.zeros(t_tot)
                pip2 = np.zeros(t_tot)
                pip1[pn_spike_matrix[pn_spike_matrix[:,1] == nn1, 0]] = 1
                pip2[pn_spike_matrix[pn_spike_matrix[:,1] == nn2, 0]] = 1
    
                corr_ln[nn1, nn2] = np.corrcoef((pip1,pip2))[0,1]
                corr_ln[nn2, nn1] = corr_ln[nn1, nn2]
                
                pip1 = v_ln[::pts_ms, nn1]
                pip2 = v_ln[::pts_ms, nn2]
                corr_vln[nn1, nn2] = np.corrcoef((pip1, pip2))[0,1]
                corr_vln[nn2, nn1] = corr_vln[nn1, nn2]
                
    tmp_corr = corr_vln[:n_lns_recep, :n_lns_recep]
    tmp_corr[tmp_corr!=0]
    corr_ln_hom = np.mean(tmp_corr[tmp_corr!=0])
    corr_ln_het = np.mean(corr_vln[:n_lns_recep, n_lns_recep:]) # corr_ln[0,-1]
    print('LNs, Hom and Het Potent corr: %.3f and %.3f' 
          %(corr_ln_hom, corr_ln_het))
    
    tmp_corr = corr_ln[:n_lns_recep, :n_lns_recep]
    tmp_corr[tmp_corr!=0]
    corr_ln_hom = np.mean(tmp_corr[tmp_corr!=0])
    corr_ln_het = np.mean(corr_ln[:n_lns_recep, n_lns_recep:]) # corr_ln[0,-1]
    print('LNs, Hom and Het spk cnt corr: %.3f and %.3f' 
          %(corr_ln_hom, corr_ln_het))
    print('')
    
    
    
# %% FIGURE ORN, PN, LN
# Generate a figure per each sensillum type

al_fig = 1
stim_type = stim_params['stim_type']

panels_id = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
rs = 4 # number of rows
cs = 1 # number of cols
fig_size = [7, 8] 
recep_clrs = ['purple','green','cyan','red']
trsp = .3
    
t_on    = np.min(stim_params['t_on'])
t2plot = -t_on, t_tot-t_on, 

if al_dyn & al_fig:
    # Calculate the SDF for PNs and LNs
    pn_sdf, pn_sdf_time = sdf_krofczik.main(pn_spike_matrix, sdf_size,
                                                 tau_sdf, dt_sdf)  # (Hz, ms)
    pn_sdf= pn_sdf*1e3

    ln_sdf, ln_sdf_time = sdf_krofczik.main(ln_spike_matrix, sdf_size,
                                                 tau_sdf, dt_sdf)  # (Hz, ms)
    ln_sdf= ln_sdf*1e3
    
    # Calculate the mean for PNs and LNs                
    pn_avg = np.mean(pn_sdf)
    ln_avg = np.mean(ln_sdf)
    print('FR avg PNs: %.2f Hz' %pn_avg)
    print('FR avg LNs: %.2f Hz' %ln_avg)
    
    
    recep_id = 0
    for qq in range(n_sens_type):
        num_recep = n_recep_list[qq]
        # for id_sens, n_neu in enumerate(n_recep_list):
        fig_al = plt.figure(figsize=fig_size)
        ax_conc = plt.subplot(rs, cs, 1)
        ax_orn = plt.subplot(rs, cs, 2)
        ax_pn = plt.subplot(rs, cs, 3)
        ax_ln = plt.subplot(rs, cs, 4)
        # ax_conc.plot(t-t_on, 100*u_od[:,0], color=green, linewidth=lw+2, 
        #                   label='glom : '+'%d'%(1))
        # ax_conc.plot(t-t_on, 100*u_od[:,1], '--',color=purple, linewidth=lw+1, 
        #                   label='glom : '+'%d'%(2))
        
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
            # for nn1 in range(n_pns_recep):
            #     pn_t_spikes = pn_spike_matrix[pn_spike_matrix[:,1] == rr*n_pns_recep+nn1, 0]
            #     ax_pn.scatter(pn_t_spikes-t_on, np.ones_like(pn_t_spikes)*(rr*n_pns_recep+nn1),
            #                     color=recep_clrs[rr], s=10)
            # ax_ln.plot(t-t_on, 
            #             v_pn[:, recep_id*n_pns_recep:((recep_id+1)*n_pns_recep)], '--', #pn_sdf
            #             color=recep_clrs[rr], linewidth=lw,)
                
            # ax_pn.plot(pn_sdf_time-t_on, 
                    # pn_sdf[:, recep_id*n_pns_recep:((recep_id+1)*n_pns_recep)], 
                    # '--', color=recep_clrs[rr], linewidth=lw,)
            
            ax_ln.plot(ln_sdf_time-t_on, 
                        ln_sdf[:,recep_id*n_lns_recep:((recep_id+1)*n_lns_recep)], 
                        '--', color=recep_clrs[rr], linewidth=lw, )
            
            for nn1 in range(n_lns_recep):
                ln_t_spikes = ln_spike_matrix[ln_spike_matrix[:,1] == rr*n_lns_recep+nn1, 0]
                ax_pn.scatter(ln_t_spikes-t_on, np.ones_like(ln_t_spikes)*(rr*n_lns_recep+nn1),
                                color=recep_clrs[rr], s=10)
            
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
            
        plt.show()
        
#%% ORN FIGURE

orn_fig = 0
pts_ms  = stim_params['pts_ms']
vrest   = orn_params['vrest']
vrev    = orn_params['vrev']

trsp    = .1

if orn_fig:          
    # One figure per each sensilla type
    for st in range(n_sens_type):
        n_neu   = orn_layer[st]['n_neu'] # TEMP: THIS SHOULD CHANGE PER EACH TYPE OF SENSILLA
        
        rs = 5      # number of rows
        cs = n_neu  #  number of cols
                        
        fig_orn, ax_orn = plt.subplots(rs, cs, figsize=[8.5, 9])
        fig_orn.tight_layout()
        
        
        # Create Transduction Matrix to plot odour 
        transd_mat = np.zeros((n_neu, n_od))
        for pp in range(n_neu):
            transd_mat[pp,:] = orn_layer[st]['od_pref'][pp,:]
        
        if n_neu == 1:
            weight_od = u_od*transd_mat[0,:]
            
            # PLOT
            ax_orn[0].plot(t-t_on, weight_od, linewidth=lw+1, )
            for rr in range(1, rs):
                X0 = t-t_on
                if rr == 1:
                    X1 = r_orn
                elif rr == 2:
                    X1 = y_orn
                elif rr == 3:
                    X1 = v_orn
                    ax_orn[3].plot([t[0]-t_on, t[-1]-t_on], [vrest, vrest], 
                                   '--', linewidth=lw, color=red,)
                    # ax_orn[3].plot([t[0]-t_on, t[-1]-t_on], [vrev, vrev], 
                    #                '-.', linewidth=lw, color=red,)
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
            for rr in range(rs):
                ax_orn[rr].tick_params(axis='both', which='major', labelsize=ticks_fs)
                ax_orn[rr].text(-.15, 1.25, panels_id[rr], transform=ax_orn[0].transAxes, 
                             fontsize=panel_fs, fontweight='bold', va='top', ha='right')
                ax_orn[rr].spines['right'].set_color('none')
                ax_orn[rr].spines['top'].set_color('none')
                ax_orn[rr].set_xlim((t2plot))
                
            for rr in range(rs-1):
                ax_orn[rr].set_xticklabels('')
                     
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
            
            plt.show()
            
        else:
            trsp = .3
            for id_neu in range(n_neu):
                
                # PLOT
                weight_od = u_od*transd_mat[id_neu,:]
                ax_orn[0, id_neu].plot(t-t_on, weight_od, linewidth=lw+1, 
                                       color=black,) 
                
                for rr in range(1, rs):
                    X0 = t-t_on
                    if rr == 1:
                        X1 = r_orn[:, id_neu*n_orns_recep:((id_neu+1)*n_orns_recep)]
                    elif rr == 2:
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
                    
                    ax_orn[rr, id_neu].plot(X0, mu1,  
                                  linewidth=lw+1, color=recep_clrs[id_neu],)
                    for nn in range(n_orns_recep):
                        ax_orn[rr, id_neu].plot(X0, X1[:, nn], 
                                  linewidth=lw-1, color=recep_clrs[id_neu], alpha=trsp)
            
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
                    
                      
        
            plt.show()
     

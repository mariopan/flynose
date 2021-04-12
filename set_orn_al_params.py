#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 16:51:37 2021

set_orn_al_params.py

This function is to set the params of flynose in a easy way.

@author: mario
"""
import numpy as np

def main(n_orn=2):
    
    #%% Stimulus params
    stim_params     = dict([
                        ('stim_type' , 'ss'),   # 'ts'  # 'ss' # 'pl'
                        ('pts_ms' , 10),         # simulated pts per ms 
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
    
    # ornXXC_params = dict([
    #                     ('n', .822066870*transd_vect_3A), 
    #                     ('alpha_r', 12.6228808*transd_vect_3A), 
    #                     ('beta_r', 7.6758436748e-02*transd_vect_3A),
    #                     ])
    
    # sensillum 0
    if n_orn ==1:
        transd_params0 = (ab3A_params)
    elif n_orn == 2:
        transd_params0 = (ab3A_params, ab3B_params)
        
    sens_params0     = dict([
                        ('n_neu', transd_params0.__len__()), # number of ORN cohoused in the sensillum
                        ('n_orns_recep', n_orns_recep),
                        ('od_pref' , od_pref),
        # NSI params
                        ('w_nsi', 1e-10), 
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
                        ('theta', -35),                     # 1
                        ('t_ref', orn_params['t_ref']),
                        ('n_recep_list', n_recep_list),
                        ('n_sens_type', n_sens_type),                    
                        ('n_orns_recep', n_orns_recep),                    
                            ])
    
    pn_ln_params = dict([
                        # PNs and LNs potential params
                        ('vrev_ex',       0),         # 15 [mV] excitatory reversal potential
                        ('vrest_pn',    -65),         # -6.5 [mV] resting potential
                        ('vrest_ln',    -65),       # -3[mV] resting potential
                        ('vrev_inh',    -80),       # 15  [mV] inhibitory reversal potential
    
                        ('c_pn_ln',     10),      # .5 [--]
                        
                        # leakage conductance
                        ('g_l_pn',      6.2),      #  leak conductance for PN
                        ('g_l_ln',      10),      #  0.5 leak conductance for LN                  
                                            
                        # PNs adaptation params
                        ('tau_ad',      258),       # 600 [ms] time scale for dynamics of adaptation
                        ('alpha_ad',    .02),      # 2.4 [unitless] ORN input coeff for adaptation variable x_pn
                        ('g_ad',        12.2),      # [muS] maximal conductance of spike rate adaptation current
                                                    # variable x_pn
                        # ORN on PN
                        ('tau_orn',     26.8),        # 10 [ms]
                        ('alpha_orn',   .5),         # 43 [unitless] coeff for the ORN input to PNs                    
                        ('g_orn',       .6),      # [muS] maximal conductance of ORN to PN synapses
                        
                        # PNs on LNs
                        ('tau_pn',      19),        # 10 [ms]                    
                        ('alpha_pn',    .25),       # [unitless] 3  # coeff for the PN input to LNs
                        ('g_pn',        2.1),       # [muS] maximal conductance of PN to LN synapses
                        
                        # LNs on PNs
                        ('tau_ln',      25),
                        ('alpha_ln',    0),       # [unitless]
                        ('g_ln',        1),      # [muS] LN to PN "maximal" conductance
                        
                        # background noise to PN and LN
                        ('vpn_noise',   11),       # NEW # extra noise input to PNs
                        ('vln_noise',   12),       #2.5 NEW
                    ])
    
    for sst in range(n_sens_type):
        orn_layer_params[sst]['w_nsi']  = 0.0
        
    
    # ORNS layer SIMULATION
    params_al_orn = dict([
                ('stim_params', stim_params),
                ('orn_layer_params', orn_layer_params),
                ('orn_params', orn_params),
                ('sdf_params', sdf_params),
                ('al_params', al_params),
                ('pn_ln_params',pn_ln_params),
                ])


    return params_al_orn
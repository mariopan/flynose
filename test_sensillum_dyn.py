#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 07:58:06 2021


test_sensillum_dyn.py

@author: mario
"""



import numpy as np
import matplotlib.pyplot as plt
import timeit
import pickle


import plot_orn
import set_orn_al_params
import plot_hist_isi
import sensillum_dyn
import stim_fcn_oop



# import sensillum_dyn_old


# DEFINE FUNCTIONS

# tic toc
def tictoc():
    return timeit.default_timer()



fld_analysis = 'NSI_analysis/ORN_LIF_dynamics/'
name_data = 'tmp_od_orn.pickle'
fig_name= "ORN_lif_dyn_500ms_300dpi"

# params_al_orn = set_orn_al_params.main(1)
params_al_orn = set_orn_al_params.main(n_orn=1, n_od=1)

stim_params         = params_al_orn['stim_params']
orn_layer_params    = params_al_orn['orn_layer_params']
sens_params         = orn_layer_params[0]
orn_params          = params_al_orn['orn_params']
sdf_params          = params_al_orn['sdf_params']
# output_params       = params_al_orn['output_params']

# al_params           = params_al_orn['al_params']
# pn_ln_params        = params_al_orn['pn_ln_params']

stim_params['stim_type']    = 'ss'      
stim_params['conc0']        = 1.85e-10
stim_params['t_tot']        = 1800            # [ms]
stim_params['concs']        = np.array([1e-3,])

# ##############################################################
# # real plume example
       
# # stim params
# delay                       = 0    
# stim_params['stim_type']    = 'pl'      
# rd_seed                     = np.random.randint(0, 1000)
# stim_params['stim_seed']    = 299#rd_seed
# print(stim_params['stim_seed'])
# stim_params['t_on']         = np.array([1000, 1000])
# stim_params['concs']        = np.array([10e-3,10e-3, ])


# # real plumes params
# plume_params        = stim_params['plume_params']
# plume_params['whiff_max']   = 3
# plume_params['rho_t_exp']   = 0   #[0, 1, 3, 5]

# sdf_params['tau_sdf']       = 30
# sdf_params['dt_sdf']        = 5

# stim_params['plume_params'] = plume_params
# ##############################################################
    
params_1sens   = dict([
                    ('stim_params', stim_params),
                    ('sens_params', sens_params),
                    ('orn_params', orn_params),
                    ('sdf_params', sdf_params),
                    ])

run_sim = 1


if run_sim:
    print('Run sim')
    # ORN LIF SIMULATION
      
    
    tic = timeit.default_timer()
    
    # GENERATE ODOUR STIMULUS/I and UPDATE STIM PARAMS
    plume           = stim_fcn_oop.main(stim_params, verbose=True)
    u_od            = plume.u_od
    
    
    output_orn      = sensillum_dyn.main(params_1sens, u_od)
     # output_orn      = sensillum_dyn_old.main(params_1sens, u_od)
    toc = timeit.default_timer()
    print('plume sim and ORNs sim time: %.2f s'%(toc-tic))
    
    data_od_orn = dict([('u_od', u_od), ('output_orn', output_orn)])
    with open(fld_analysis + name_data, 'wb') as f:
        pickle.dump(data_od_orn, f)
                
else:
    print('LOAD DATA')
    # ORN load data
    data_od_orn = pickle.load(open(fld_analysis+ name_data,  "rb" ))
    
    u_od = data_od_orn['u_od']
    output_orn = data_od_orn['output_orn']



fig = plot_orn.main(params_1sens, output_orn, )
    
fig_save=1
    
plt.show()              

if fig_save:
    fig.savefig(fld_analysis + fig_name, dpi=300)
    

# # fig, axs = plot_hist_isi.main(params_1sens, output_orn)
# plt.show()

# # fig.savefig(fld_analysis + hist_fig_name)
    


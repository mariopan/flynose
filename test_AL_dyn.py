#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 17:11:52 2021

test_AL_dyn.py

@author: mario
"""

import numpy as np
import matplotlib.pyplot as plt
import timeit
import pickle

import stim_fcn
import plot_orn
import plot_al_orn
import set_orn_al_params

import sensillum_dyn
import AL_dyn

import AL_dyn_old

# DEFINE FUNCTIONS

# tic toc
def tictoc():
    return timeit.default_timer()



fld_analysis = 'NSI_analysis/trials/'
name_data = 'tmp_od_orn.pickle'


# params_al_orn = set_orn_al_params.main(1)
params_al_orn = set_orn_al_params.main(2)

stim_params         = params_al_orn['stim_params']
orn_layer_params    = params_al_orn['orn_layer_params']
sens_params         = orn_layer_params[0]
orn_params          = params_al_orn['orn_params']
sdf_params          = params_al_orn['sdf_params']
# output_params       = params_al_orn['output_params']

# al_params           = params_al_orn['al_params']
# pn_ln_params        = params_al_orn['pn_ln_params']

stim_params['stim_type']    = 'ss'      
stim_params['conc0']        = 1.85e-4
stim_params['t_tot']        = 100000            # [ms]
stim_params['concs']        = np.array([10e-3,])

##############################################################
# real plume example
       
# stim params
delay                       = 0    
stim_params['stim_type']    = 'pl'      
rd_seed                     = np.random.randint(0, 1000)
stim_params['stim_seed']    = rd_seed
print('seed ORN sim: %d'%stim_params['stim_seed'])
stim_params['t_on']  = np.array([1000, 1000])
stim_params['concs']        = np.array([10e-3,10e-3, ])


# real plumes params
plume_params        = stim_params['plume_params']
plume_params['whiff_max']   = 3
plume_params['rho_t_exp']   = 0   #[0, 1, 3, 5]

sdf_params['tau_sdf']       = 30
sdf_params['dt_sdf']        = 5

stim_params['plume_params'] = plume_params
    
params_1sens   = dict([
                    ('stim_params', stim_params),
                    ('sens_params', sens_params),
                    ('orn_params', orn_params),
                    ('sdf_params', sdf_params),
                    ])

run_sim = True

if run_sim:
    # ORN LIF SIMULATION
    
    # GENERATE ODOUR STIMULUS/I and UPDATE STIM PARAMS
    tic = timeit.default_timer()
    u_od            = stim_fcn.main(stim_params, verbose=False)
    
    output_orn      = sensillum_dyn.main(params_1sens, u_od)
    toc = timeit.default_timer()
    
    print('elapsed time new ORN sim: %.2f'%(toc-tic))
    
    data_od_orn = dict([('u_od', u_od), ('output_orn', output_orn)])
    with open(fld_analysis + name_data, 'wb') as f:
        pickle.dump(data_od_orn, f)
                
else:
    # ORN load data
    data_od_orn = pickle.load(open(fld_analysis+ name_data,  "rb" ))
    
    u_od = data_od_orn['u_od']
    output_orn = data_od_orn['output_orn']

fig = plot_orn.main(params_1sens, output_orn, )
# fig.savefig(fld_analysis + timecourse_fig_name)    
    
plt.show()                


#%% NEW AL SIM
tic = timeit.default_timer()
[t, u_od, r_orn, v_orn, y_orn, spike_orn, spike_matrix, orn_sdf,
     orn_sdf_time,]  = output_orn
output_al = AL_dyn.main(params_al_orn, spike_orn, )
toc = timeit.default_timer()

print('elapsed time new AL sim: %.2f'%(toc-tic))

output_orn_lyr = [t, u_od, spike_orn, orn_sdf, orn_sdf_time,]

fig_al = plot_al_orn.main(params_al_orn, output_orn_lyr, output_al)

plt.show()                
#%% OLD AL SIM
tic = timeit.default_timer()
output_al = AL_dyn_old.main(params_al_orn, spike_orn, )
toc = timeit.default_timer()

print('elapsed time old AL sim: %.2f'%(toc-tic))

fig_al = plot_al_orn.main(params_al_orn, output_orn_lyr, output_al)

plt.show()                






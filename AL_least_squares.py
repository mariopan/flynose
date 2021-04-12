#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 10:05:09 2021

AL_least_squares.py

This function fits several average values of PNs activity during the stimulation 
of 500ms as a function of the ORNs

It runs the dynamics of ORNs once and then look for the correct parameters over
many loops

NOTE: scipy.optimize.least_squares simply requires a function that returns 
whatever value you'd like to be minized, in this case the difference between 
your actual y data and the fitted function data. The actual important variables 
in least_squares are the parameters you want to fit for, not the x and y data. 
The latter are passed as extra arguments.

Concatenating vectors yields to awkward function to fit (probably not a smooth 
derivative, for example), therefore it is quite essential to
   1. provide good starting values (params0, so all the ...0 values).
   2. don't have data or parameters which span orders of magnitude. The 
   closer everything is around 1 (a few orders of magnitude is certainly ok), 
   the better.
   
This function repeats the fit until a good enough result is achieved. 
The flag bad_fit indicates when stop the loop, that could happen for low 
cost_goal achieved or for enough number of loops achieved.

@author: mario
"""

# %% IMPORT MODULES
import numpy as np
import timeit
from scipy.optimize import least_squares
import pickle

# import the two models of ORN
import AL_dyn
import ORNs_layer_dyn

# DEFINE FUNCTIONS

def tictoc():
    return timeit.default_timer()

def fitpar_pn_ln_params(params2fit):
    # FITTING PARAMETERS 
    [g_l_pn, tau_ad, alpha_ad, g_ad, tau_orn, alpha_orn, g_orn, vpn_noise] = params2fit
    
    # AL params
    pn_ln_params['g_l_pn']        = g_l_pn
    pn_ln_params['tau_ad']        = tau_ad
    pn_ln_params['alpha_ad']      = alpha_ad
    pn_ln_params['g_ad']          = g_ad
    pn_ln_params['tau_orn']       = tau_orn
    pn_ln_params['alpha_orn']     = alpha_orn
    pn_ln_params['g_orn']         = g_orn
    pn_ln_params['vpn_noise']     = vpn_noise
    

def al_dyn0(params2fit, orn_spikes_t):    
    
    # # FITTING PARAMETERS 
    fitpar_pn_ln_params(params2fit)


    # AL simulation
    output_al = AL_dyn.main(params_al_orn, orn_spikes_t)
    [t, pn_spike_matrix, pn_sdf, pn_sdf_time,
                  ln_spike_matrix, ln_sdf, ln_sdf_time,] = output_al
    
    return [pn_sdf_time, pn_sdf]

    
def leastsq_al(params_tmp, *args):
    # Calculate all the single curves from the params and the observed x;
    # It returns the difference between the objective y and the calculated yfit.
    
    t_on    = args[0]
    t_off   = args[1]
    n_pns_recep     = int(args[2])
    
    
    nu_obs  = args[3:7]
    concs   = args[7:] #[8, 75, 130, 150]
    nu_pn_s     = np.zeros(len(concs))

    
    for id_c, cc in enumerate(concs):
        stim_params['concs'] = np.array([cc, cc])
        
        # tic= tictoc()
        # output_orn = ORNs_layer_dyn.main(params_al_orn)
        output_orn = pickle.load(open(fld_analysis + 'ORN2fitAL_' + str(id_c) + '.pickle',  "rb" ))

        [t, u_od,  orn_spikes_t, orn_sdf,orn_sdf_time] = output_orn 

        
        #AL dynamics
        [pn_sdf_time, pn_sdf] = al_dyn0(params_tmp, orn_spikes_t)
        
        
        pn_id_stim = np.flatnonzero((pn_sdf_time>t_on) & (pn_sdf_time<t_off))
        
        # nu_pn_w = np.mean(pn_sdf[pn_id_stim, :n_pns_recep])
        nu_pn_s[id_c] = np.mean(pn_sdf[pn_id_stim, n_pns_recep:])
        
        # toc= tictoc()
        # print('AL sim: %.2f'%(toc-tic))
        
    dy = (nu_pn_s-nu_obs)
    # print(dy)
    return dy 







# INITIALIZE PARAMETERS

# figure and data params
fld_analysis    = 'fit_AL/' 
fig_name        = 'AL_fit_'
data_name       = 'AL_params_fit_'

# LOAD PARAMS FROM A FILE

fld_params = 'NSI_analysis/trials/' #Olsen2010
name_data = 'params_al_orn.ini'
params_al_orn = pickle.load(open(fld_params+ name_data,  "rb" ))

stim_params     = params_al_orn['stim_params']
orn_layer_params= params_al_orn['orn_layer_params']
orn_params      = params_al_orn['orn_params']
sdf_params      = params_al_orn['sdf_params']
al_params       = params_al_orn['al_params']
pn_ln_params    = params_al_orn['pn_ln_params']

n_pns_recep     = params_al_orn['al_params']['n_pns_recep']
n_orns_recep     = params_al_orn['al_params']['n_orns_recep']

# stim params
delay                       = 0
t0                          = 1000
stim_name                   = ''

stim_params['stim_type']    = 'ss' # 'ss'  # 'ts' # 'rs' # 'pl'
stim_dur                    = 500        # 10, 50, 200

stim_params['stim_dur']     = np.array([stim_dur, stim_dur])
stim_params['t_tot']        = t0+delay+stim_dur+300
stim_params['t_on']         = np.array([t0, t0+delay])

stim_params['conc0']        = 1.85e-4    # 1.85e-4  # fitting value: 2.85e-4
peak_ratio                  = 1
concs                       = [1.85e-4, 5e-4, 1.5e-3, 2e-2,]

# PNs average activity during 500ms stimulation (see Olsen et al. 2010)
nu_obs                      = [8, 75, 130, 150]
    
stim_params['pts_ms']       = 10
n_sens_type       = orn_layer_params.__len__()  # number of type of sensilla
for sst in range(n_sens_type):
    orn_layer_params[sst]['w_nsi']  = 0   

# Run the fit until cost is very low or for a max of max_nn times
max_nn          = 250
bad_fit         = True
nn              = 0
diff_step       = 1e-4

# routine to check the single functions
t_on            = t0
t_off           = t0+stim_dur
args_fit        = [t_on, t_off, n_pns_recep, ]
args_fit        = np.concatenate((args_fit, nu_obs))
args_fit        = np.concatenate((args_fit, concs))

# Run ORNs dynamics and save output before running AL dynamics
for id_c, cc in enumerate(concs):
        stim_params['concs'] = np.array([cc, cc])
        
        tic= tictoc()
        output_orn = ORNs_layer_dyn.main(params_al_orn)
        [t, u_od,  orn_spikes_t, orn_sdf,orn_sdf_time] = output_orn 
        
        
        with open(fld_analysis + 'ORN2fitAL_' + str(id_c) + '.pickle', 'wb') as f:
            pickle.dump(output_orn, f)
        
        orn_id_stim = np.flatnonzero((orn_sdf_time>t_on) & (orn_sdf_time<t_off))
        
        nu_orn_w = np.mean(orn_sdf[orn_id_stim, :n_orns_recep])
        
        nu_orn_s = np.mean(orn_sdf[orn_id_stim, n_orns_recep:])
        print(nu_orn_s)
        

#%%np.random.uniform(0, 100, size=len(params2fit))


# s, x, y dynamics without (1-s)
# par2test  = [10.9, 28.2, 7.57, 3.8, 31.3, 3.43, 0.61, 8.33] #params2fit*(1+.5*np.random.randn(len(params2fit)))
# par2test = [13.85421082, 12.82625519,  8.51026598, 10.68393157, 61.7348968 ,        3.01304013,  0.35764161,  9.65594671]
par2test = [6.2, 208, 0.02, 12.2, 26.8, .5, .6, 11]

# fitting params: min_params = [6.71481343e+00, 2.58521904e+02, 5.56803509e-02, 1.22318181e+01,
       # 2.68234116e+01, 4.95968803e-01, 6.28658533e-01, 1.73528442e+01]
#[5, 200, .1, 10, 15,.5, 1, 12] # mario by eye
# Thomas: [10,200,.1,10,15,.5,2,9]
print(par2test)
dnu = leastsq_al(par2test, *args_fit)


print(dnu+nu_obs)
cost_est = 0.5 * np.sum(dnu**2)
print('estimated cost: %.2f'%cost_est)


#%% FIT OBSERVED DATA TO SIMULATIONS
cost_goal = 500

tic_tot     = tictoc()
data_save   = 1
min_cost    = cost_est

# FITTING PARAMS
params2fit = par2test
   
fitpar_pn_ln_params(params2fit)

# LOWER BOUNDS
# lb          = np.zeros_like(params2fit)
lb = np.array([0,100,.05,5,0,.05,0,0])
# UPPER BOUNDS
ub          = np.ones_like(params2fit)*300
ub  = np.array([100, 300,.9,50,30,1,50,100])


while bad_fit: 
    params_0 = params2fit*(1+.5*np.random.randn(len(params2fit)))

    # check that the starting params are all w/in the bounds
    new_try = 0
    while ~np.all((params_0>lb) & (params_0<ub)):
        new_try += 1
        params_0 = params2fit*(1+.5*np.random.randn(len(params2fit)))
        
    print('trial #: %d, params 0:'%nn)
    print(params_0)
        
    # linear diff fit
    tic = tictoc()
    res_lsq = least_squares(leastsq_al, params_0, #method='lm',
                            bounds=(lb, ub), 
                            #loss='soft_l1', f_scale=0.1, 
                            diff_step=diff_step,# loss='cauchy', # tr_solver='lsmr', 
                            gtol=None, #xtol=None, ftol=None, 
                            args=args_fit)
    
    toc = tictoc()
    
    
    print('linear diff fit time: %.1f s'%(toc-tic))
    params_lsq = res_lsq.x
    cost = res_lsq.cost
    print('exit status: %d, cost: %f'%(res_lsq.status, cost))
    print('params fit:')
    print(params_lsq)
    dnu = leastsq_al(params_lsq, *args_fit)
    print('nu out: ')
    print(dnu+nu_obs)
    print('')
    if cost<min_cost:
        min_cost = cost
        min_params = params_lsq
        print('NEW params fit wtih min cost!')
        #print(params_lsq)
    
    #########################################
    # FIGURE AND DATA SAVING
    # if fig_fit:
        # fig = plt.figure()
        # ax = fig.subplots(1, 1,)
        # figure_fit(params_lsq, ax)
        # ax.set_title('fit linear diff')
    
        # # SAVE FIGURE AND DATA
        # fig.savefig(fld_analysis + fig_name + str(nn) +'.png')
        # plt.close()
        # plt.show()
    
    
    if data_save:
        with open(fld_analysis+ data_name + str(nn) + '.pickle', 'wb') as f:
            saved_pars = ['min_cost', 'min_params', 'params_0']
            pickle.dump([], f)
    if (cost < cost_goal) or (nn == max_nn):
        bad_fit = False
    nn += 1
    
toc_tot = tictoc()    
print('tot time: %.2f min'%((toc_tot-tic_tot)/60))




 
# #%%
# # LOAD EXPERIMENT PARAMETERS AND ANALYSIS OF THE RESULTS
# # figure and data params
# import pickle
# import numpy as np

# fld_analysis    = 'fit_ORN/'#'LIF_ORN_fit_50tpts/' 
# fig_name        = 'ORN_fit_'
# data_name       = 'ORN_params_fit_'


# cost =  np.empty(250)
# for nn in range(len(cost)):
    
#     all_params    = pickle.load(open(fld_analysis+data_name + str(nn) + '.pickle', "rb" ))
#     [t_tot, t_on, t_tot_fit, t_on_fit, delay_obs, 
#         pts_ms, n_tpts, t_fit, t_fit_obs, params2fit,
#         nu_obs_all, concs_obs, concs_sim, file_names, 
#         params_0, args_fit, lb, ub, 
#         diff_step, res_lsq, saved_pars] = all_params
#     cost[nn] =  res_lsq.cost
    
#     if cost[nn] < 21000:
#         print('cost(%d): %.2f' %(nn, cost[nn]))
#         print('Fit params:')
#         print(str(res_lsq.x))
#         print('distance from start:')
#         print(str(np.round(1000*params_0/res_lsq.x)/10))
#         print('distance from best fit: ')
#         print(str(np.round(1000*res_lsq.x/params2fit)/10))
#         print()
        

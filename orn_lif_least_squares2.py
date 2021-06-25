#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 17:24:23 2020

orn_lif_least_squares.py

This function fits multiple curves coming from ORN_depalo or study_ORN_LIF
with the function scipy.optimize.least_squares. 

It loads the data from Martelli2013/ethyl_acetate for multiple concentrations
and run multiple simulations of the chose ORN dynamics.

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
import matplotlib.pyplot as plt
import pickle

from scipy.interpolate import interp1d
import pandas
    

# to obtain lines with a color of different shades
import matplotlib as mpl

# import the two models of ORN
import ORN_LIF 
import ORN_depalo

# DEFINE FUNCTIONS
def tictoc(*args):
    toc = timeit.default_timer()
    if len(args)>0:
        toc = toc - args[0] 
        print(toc)        
    return toc

def orn_depalo0(params2fit, *args):    
    t_tot   = args[0] # 1000 # approximately t[-1]-t[0]
    t_on    = args[1] #  170   # approximately -t[0]+50 # added 50ms delay to reach the antenna
    conc    = args[2]
    
    # FITTING PARAMETERS 
    [n, alpha_r, beta_r, 
     c_rect, a_rect, nu_max_rect, 
     a_y, b_y, b_x, c_x, d_x, a_r, conc0] = np.array([.5, .9, .09, 
                       1, 3.3, 250,
                       .25, .002, .2, .0028, 1, 1, 1e-5])
    
    [n, alpha_r, beta_r, 
     c_rect, a_rect, nu_max_rect, 
     a_y, b_y, b_x, c_x, d_x, a_r, conc0] = params2fit
    
    # stimulus params
    stim_params         = dict([
                    ('stim_dur' , 500),
                    ('stim_type' , 'ss'),   # 'ts'  # 'ss' # 'pl'
                    ('pts_ms' , 1),         # simulated pts per ms 
                    ('t_tot', t_tot),        # ms 
                    ('t_on', t_on),          # ms
                    ('concs', conc),
                    ('conc0', conc0),
                    ])
    
    # ORN PARAMS
    orn_params = dict([
        # Transduction params
                        ('n', n),
                        ('alpha_r', alpha_r), 
                        ('beta_r', beta_r),
        # rectification params        
                        ('c_rect', c_rect),
                        ('a_rect', a_rect), 
                        ('nu_max_rect', nu_max_rect),
        # Spiking machine params
                        ('t_ref', 2*stim_params['pts_ms']), # ms; refractory period 
                        ('a_y', a_y),  
                        ('b_y', b_y),  
                        ('b_x', b_x),       # 0.12
                        ('c_x', c_x),     # 0.004
                        ('d_x', d_x),           
                        ('a_r', a_r),
                        ])                    
    
    # *****************************************************************
    # ORN simulation
    orn_out = ORN_depalo.main(orn_params, stim_params)
    [t, u_od, r_orn, x_orn, y_orn, nu_orn, ]  = orn_out
        
    return [t-stim_params['t_on'], nu_orn]


def orn_lif0(params2fit, *args):    
    t_tot   = args[0] # 1000 # approximately t[-1]-t[0]
    t_on    = args[1] #  170   # approximately -t[0]+50 # added 50ms delay to reach the antenna
    conc    = args[2]
    
    
    # FITTING PARAMS
    [n, alpha_r, beta_r, 
     tau_v, vrest,  
     g_y, g_r, alpha_y, beta_y, 
     conc0, tau_sdf,] =  params2fit
    
    # analysis params
    # tau_sdf = 50
    dt_sdf  = 5
    sdf_params = [tau_sdf, dt_sdf]
    
    # Stimulus PARAMETERS 
    stim_params         = dict([
                    ('stim_dur' , 500),
                    ('stim_type' , 'ss'),   # 'ts'  # 'ss' # 'pl'
                    ('pts_ms' , 1),         # simulated pts per ms 
                    ('t_tot', t_tot),        # ms 
                    ('t_on', t_on),          # ms
                    ('concs', conc),
                    ('conc0', conc0),
                    ])
    
    # ORN PARAMETERS 
    orn_params = dict([
        # Transduction params
                        ('n', n),
                        ('alpha_r', alpha_r), 
                        ('beta_r', beta_r),
        # LIF params
                        ('t_ref', 2*stim_params['pts_ms']), # ms; refractory period 
                        ('theta', -30),                 # [mV] firing threshold
                        ('tau_v', tau_v),        # [ms]
                        ('vrest', -vrest),      # [mV] resting potential
                        ('vrev', 0),  # [mV] reversal potential
                        # ('v_k', vrest),
                        ('g_y', g_y),       
                        ('g_r', g_r),       
        # Adaptation params
                        ('alpha_y', alpha_y), 
                        ('beta_y', beta_y), ])
                  
    # ORN simulation
    orn_out = ORN_LIF.main(orn_params, stim_params, sdf_params)
    [t, u_od, r_orn, v_orn, y_orn, num_spikes, spike_matrix, orn_sdf,
     t_sdf]  = orn_out
    
    if np.any(orn_sdf)==np.nan:
        print('error')
    return [t_sdf-stim_params['t_on'], orn_sdf]


def ethyl_data(conc2fit, ):
    # load observe ORN data and return into t and freq
    data_fld = 'ab3A_ethyl_acetate/'
    
    # file_names = ['ethyl_ab3A_10.csv', 'ethyl_ab3A_17.csv', 'ethyl_ab3A_20.csv', 
    #               'ethyl_ab3A_22.csv', 'ethyl_ab3A_27.csv', 'ethyl_ab3A_30.csv',
    #               'ethyl_ab3A_35.csv', 'ethyl_ab3A_40.csv', 'ethyl_ab3A_80.csv',]
    
    file_name = ['ethyl_ab3A_%d'%int(conc2fit)+'.csv']

    df = pandas.read_csv(data_fld  + file_name[0], names = ['time', 'freq'])
    
    t = np.array(df.time)
    freq = np.array(df.freq)
    return [t, freq] 


def leastsq_orn(params, *args):
    # Calculate all the single curves from the params and the observed x;
    # It returns the difference between the objective y and the calculated yfit.
    n_tpts  = args[0]   # number of time points (redundant)
    nu_obs  = args[1]   # frequency goal series
    t_fit   = args[2]
    t_tot   = args[3] # 1000 # approximately t[-1]-t[0]
    t_on    = args[4] #  170   # approximately -t[0]+50 # added 50ms delay to reach the antenna
    concs   = args[5:]

    nu_sim  = np.empty_like(nu_obs)
    args0 = [t_tot, t_on, np.nan]
    for id_c, cc in enumerate(concs):
        args0[2] = cc
        [t_orn, nu_orn] = orn_func(params, *args0)
        
        nu_sim_fcn = interp1d(t_orn, nu_orn)
        nu_sim_spl = nu_sim_fcn(t_fit)
        
        ss = id_c*n_tpts
        ee = (id_c+1)*n_tpts
        nu_sim[ss:ee] = nu_sim_spl
    
    if any(np.isinf(nu_sim)):
        print('NAN OR INF IN THE SIMULATION!')
        nu_sim  = np.zeros_like(nu_obs)
    dy = nu_obs-nu_sim
    return dy 



def select_nu_obs():
    
    nu_obs_all = np.empty(n_cs*n_tpts,)
    
    # select oserved activity curves:
    for cc_ob, idc in zip(concs_obs, range(n_cs)):
        
        # Load the observed ORN activity (from Martelli 2013, ethyl acetate, ab3)
        [t_obs, nu_obs] = ethyl_data(cc_ob, )
        
        # interpolate observed activity 
        nu_obs_fcn = interp1d(t_obs, nu_obs)
        y_true = nu_obs_fcn(t_fit_obs)
        
        ss = idc*n_tpts
        ee = (1+idc)*n_tpts
        nu_obs_all[ss:ee] = y_true
    
    return nu_obs_all


def plot_obs_data(ax, nu_obs_all):      
    n_clr = n_cs+2
    c = np.arange(1, n_clr )
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    greenmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Greens)
   
    # generate and plot 'real' and fit curves:
    for cc_ob, cc_sim, idc in zip(concs_obs, concs_sim, range(n_cs)):
        
        ss = idc*n_tpts
        ee = (1+idc)*n_tpts
        y_true = nu_obs_all[ss:ee]
        
        
        # PLOT
        clr = greenmap.to_rgba(n_clr-idc)
        ax.plot(t_fit_obs, y_true, '-.', color=clr, label='observ %.5f'%cc_sim)
        
    ax.plot([0, 0], [0, 300], 'k--', linewidth=2)
    ax.plot([500, 500], [0, 300], 'k--', linewidth=2)
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.legend()

    return ax


def figure_fit(params2an, ax, nu_obs_all):
    args0 = [t_tot, t_on, np.nan]
    n_clr = n_cs+2
    c = np.arange(1, n_clr )
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    greenmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Greens)
   
    # generate and plot 'real' and fit curves:
    for cc_ob, cc_sim, idc in zip(concs_obs, concs_sim, range(n_cs)):
        
        ss = idc*n_tpts
        ee = (1+idc)*n_tpts
        y_true = nu_obs_all[ss:ee]
        
        # Simulate ORN with fitted params
        args0[2] = cc_sim
        [t_orn, nu_orn] = orn_func(params2an, *args0)
        
        # interpolate simulated activity values
        nu_sim_fcn = interp1d(t_orn, nu_orn)
        y_fit = nu_sim_fcn(t_fit)
        
        # PLOT
        clr = greenmap.to_rgba(n_clr-idc)
        ax.plot(t_fit_obs, y_true, '-.', color=clr, label='observ %.5f'%cc_sim)
        ax.plot(t_fit_obs, y_fit, '-*', color=clr, label='fit %.5f'%cc_sim)
        
    ax.plot([0, 0], [0, 300], 'k--', linewidth=2)
    ax.plot([500, 500], [0, 300], 'k--', linewidth=2)
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.legend()

    return ax

# INITIALIZE PARAMETERS

# figure and data params
plt.ioff()      # ioff() # to avoid showing the plot every time   

fld_analysis    = 'fit_ORN/new_fit/' 
fig_name        = 'ORN_fit_'
data_name       = 'ORN_params_fit_'

# simulation params
t_tot           = 1200  # ms
t_on            = 300   # ms, stimulus onset

t_0_obs         = -110
t_1_obs         = 790

# fitting parameters
t_tot_fit       = 900
delay_obs       = 0    # ms, delay between open valve and antenna

t_on_fit        = t_1_obs-delay_obs-t_tot_fit # ms, stimulus onset

pts_ms          = 1     # number of pts per ms for the simulations

n_tpts          = 200    # number of datapoints in each curve

t_fit           = np.linspace(t_on_fit, t_tot_fit+t_on_fit, n_tpts)
t_fit_obs       = t_fit+delay_obs

# choose the ORN model: LIF or De Palo
orn_model       = 'lif' # 'dp' 'lif'


if orn_model == 'dp':
    orn_func        = orn_depalo0  
    # [n, alpha_r, beta_r, 
    # c_rect, a_rect, nu_max_rect, 
    # a_y, b_y, b_x, c_x, d_x, a_r] = params2fit
    params2fit      = np.array([.5, .9, .09, 
                        1, 3.3, 250,
                        .25, .002, .2, .0028, 1, 1, 1e-5])

elif orn_model == 'lif':
    orn_func        = orn_lif0 
    
    
    # [n, alpha_r, beta_r, 
    #  tau_v, -vrest, 
    #  g_y, g_r, alpha_y, beta_y, 
    #  conc0, tau_sdf, ] =  params2fit
    
    params2fit = [.82, 12.62, 0.076,  
                  2.26, 33, 
                  5.8e-1, 0.86, 0.45, 3.4e-3,  
                  1.85e-4, 41,]    


file_names = ['ethyl_ab3A_10.csv', 'ethyl_ab3A_17.csv', 'ethyl_ab3A_20.csv', 
                  'ethyl_ab3A_22.csv', 'ethyl_ab3A_27.csv', 'ethyl_ab3A_30.csv',
                  'ethyl_ab3A_35.csv', 'ethyl_ab3A_40.csv', 'ethyl_ab3A_80.csv',]
    
concs_obs       = np.array([10, 20,30, ]) # 40,80list of concentrations to list
concs_sim       =  10**-np.array(concs_obs/10) # list of concentrations to list
n_cs            = len(concs_obs)



nu_obs_all      = select_nu_obs()
fig, axs = plt.subplots(1,1)
plot_obs_data(axs, nu_obs_all)
plt.show()

#%% extract observed ORN and simulate with standard params.01*
# fig = plt.figure()
# ax = fig.subplots(1, 1,)
# figure_fit(params2fit, ax, nu_obs_all)
# plt.show()


# routine to check the single functions
args_fit        = list([n_tpts, nu_obs_all, t_fit, t_tot, t_on, ])
args_fit        = np.concatenate((args_fit, concs_sim))

dnu = leastsq_orn(params2fit, *args_fit)
cost_est = 0.5 * np.sum(dnu**2)
print('estimated cost: %.2f'%cost_est)




#%% **************************************************************************
# FIT OBSERVED DATA TO SIMULATIONS
tic_tot = tictoc()

diff_step       = 1e-4

# LOWER BOUNDS
lb              = np.zeros_like(params2fit)
lb[-1]          = 20

# UPPER BOUNDS
if orn_model=='lif':
    ub          = np.ones_like(params2fit)*50
elif orn_model=='dp':
    ub          = np.ones_like(params2fit)*300
ub[0]           = 1       # upper bounds for n, transduction params
ub[-1]          = 150

# Run the fit until cost is very low or for a max of max_nn times
max_nn = 250
bad_fit = True
nn = 0
cost_goal = cost_est/500
print('cost goal: %d'%cost_goal)


while bad_fit: 

    params_0 = ub
    # check that the starting params are all w/in the bounds
    new_try = 0
    while ~np.all((params_0>lb) & (params_0<ub)):
        new_try += 1
        params_0 = params2fit*(1+.2*np.random.randn(len(params2fit)))
        
    # print('trial #: %d, params 0:'%nn)
    # print(params_0)
    # print('Perc Diff params 0-params_good:')
    # print(100*(params_0-params2fit)/params2fit)
        
    # linear diff fit
    tic = tictoc()
    res_lsq = least_squares(leastsq_orn, params_0, #method='lm',
                            bounds=(lb, ub), 
                            loss='soft_l1', f_scale=0.1, 
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
    print('')
    
    fig = plt.figure()
    ax = fig.subplots(1, 1,)
    figure_fit(params_lsq, ax, nu_obs_all)
    ax.set_title('fit linear diff')

    # SAVE FIGURE AND DATA
    fig.savefig(fld_analysis + fig_name + str(nn) +'.png')
    plt.close()
    
    fit_data = dict([
        ('t_tot', t_tot), 
        ('t_on',  t_on), 
        ('t_tot_fit',  t_tot_fit ), 
        ('t_on_fit', t_on_fit  ), 
        ('delay_obs', delay_obs), 
        ('pts_ms',  pts_ms ), 
        ('n_tpts',  n_tpts ), 
        ('t_fit',   t_fit), 
        ('t_fit_obs', t_fit_obs  ), 
        ('params2fit',   params2fit), 
        ('nu_obs_all',   nu_obs_all), 
        ('concs_obs',   concs_obs), 
        ('concs_sim',  concs_sim ), 
        ('file_names',    file_names), 
        ('params_0',  params_0 ), 
        ('args_fit',   args_fit), 
        ('lb',  lb ), 
        ('ub',   ub), 
        ('diff_step',   diff_step), 
        ('res_lsq', res_lsq ), 
        ])
        
    with open(fld_analysis+ data_name + str(nn) + '.pickle', 'wb') as f:
        pickle.dump(fit_data, f)
        
    if (cost < cost_goal) or (nn == max_nn):
        bad_fit = False
    nn += 1
    print('attempt number: %d'%nn)

toc_tot = tictoc()    
print('tot time: %.2f min'%((toc_tot-tic_tot)/60))
 
#%%
# LOAD EXPERIMENT PARAMETERS
# figure and data params
import pickle
import numpy as np

fld_analysis    = 'fit_ORN/new_fit/'#'LIF_ORN_fit_50tpts/' 
fig_name        = 'ORN_fit_'
data_name       = 'ORN_params_fit_'

pippo = 0

good_params = np.zeros((40,12))
good_params_dist= np.zeros((40,))

cost =  np.empty(175)
for nn in range(len(cost)):
    
    all_params    = pickle.load(open(fld_analysis+data_name + str(nn) + '.pickle', "rb" ))
    res_lsq         = all_params['res_lsq']
    cost[nn]        = res_lsq.cost

    params_0        = all_params['params_0']
    
    if cost[nn] < 600:
        
        print('cost(%d): %.2f' %(nn, cost[nn]))
        print('Fit params:')
        print(str(res_lsq.x))
        good_params [pippo, :11] = np.round(1000*res_lsq.x/params2fit)/10
        
        
        good_params_dist[pippo] = np.mean(good_params [pippo, :11])
        
        dnu = leastsq_orn(res_lsq.x, *args_fit)
        cost_est = np.mean(np.abs(dnu)/nu_obs_all)
        good_params [pippo, 11] = cost_est#cost[nn]/5
        
        pippo = pippo + 1
        print('distance from start:')
        print(str(np.round(1000*params_0/res_lsq.x)/10))
        
        print('distance from best fit: ')
        print(str(good_params_dist))
        print()
        


nn_min = np.argmin(cost)
best_prms = pickle.load(open(fld_analysis+data_name + str(nn_min) + '.pickle', "rb" ))
res_lsq = best_prms['res_lsq']
params_bestfit = res_lsq.x

nu_obs_all      = select_nu_obs()

# extract observed ORN and simulate with standard params
fig, axs = plt.subplots(1, 2,)
figure_fit(params_bestfit, axs[0], nu_obs_all)

axs[1].imshow(good_params)
plt.show()



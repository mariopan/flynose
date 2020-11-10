#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 17:24:23 2020

study_least_squares.py

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

@author: mario
"""

# %% IMPORT MODULES
import numpy as np
import timeit
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
import pandas
    
import os

# to obtain lines with a color of different shades
import matplotlib as mpl

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
    
    #***********************************************
    # stimulus params
    stim_params         = dict([
                    ('stim_dur' , 500),
                    ('stim_type' , 'ss'),   # 'ts'  # 'ss' # 'pl'
                    ('pts_ms' , 1),         # simulated pts per ms 
                    ('t_tot', t_tot),        # ms 
                    ('t_on', t_on),          # ms
                    ('concs', conc),
                    ])
    
    # ORN PARAMETERS 
    [n, alpha_r, beta_r, 
     c_rect, a_rect, nu_max_rect, 
     a_y, b_y, b_x, c_x, d_x, a_r] = params2fit
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
    [t, u_od, r_orn, x_orn, y_orn, 
     nu_orn, ]  = orn_out
        
    return [t-stim_params['t_on'], nu_orn]


def ethyl_data(conc2fit, ):
    # load observe ORN data and return into t and freq
    my_home = os.path.expanduser("~/")    
    data_fld = my_home + 'MEGA/WORK/OdorObjects/Data/Martelli2013/ab3A_ethyl_acetate'
    
    # file_names = ['ethyl_ab3A_10.csv', 'ethyl_ab3A_17.csv', 'ethyl_ab3A_20.csv', 
    #               'ethyl_ab3A_22.csv', 'ethyl_ab3A_27.csv', 'ethyl_ab3A_30.csv',
    #               'ethyl_ab3A_35.csv', 'ethyl_ab3A_40.csv', 'ethyl_ab3A_80.csv',]
    
    file_name = ['ethyl_ab3A_%d'%int(conc2fit)+'.csv']

    df = pandas.read_csv(data_fld + '/' + file_name[0], names = ['time', 'freq'])
    
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
        [t_orn, nu_orn] = orn_depalo0(params, *args0)
        nu_sim_fcn = interp1d(t_orn, nu_orn)
        nu_sim_spl = nu_sim_fcn(t_fit)
        ss = id_c*n_tpts
        ee = (id_c+1)*n_tpts
        nu_sim[ss:ee] = nu_sim_spl
        
    return nu_obs-nu_sim

#  LOAD OBSERVED ORN ACTIVITY and FIGURE

def figure_fit(params2an, ax):
    args0 = [t_tot, t_on, np.nan]
    
    nu_obs_all = np.empty(n_cs*n_tpts,)
    
    tic = tictoc()
    # *********************
    n_clr = n_cs+2
    c = np.arange(1, n_clr )
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    greenmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Greens)
    
    # generate and plot 'real' and fit curves:
    for cc_ob, cc_sim, idc in zip(concs_obs, concs_sim, range(n_cs)):
        
        # Load the observed ORN activity (from Martelli 2013, ethyl acetate, ab3)
        [t_obs, nu_obs] = ethyl_data(cc_ob, )
        
        # interpolate observed activity 
        nu_obs_fcn = interp1d(t_obs, nu_obs)
        y_true = nu_obs_fcn(t_fit_obs)
        
        ss = idc*n_tpts
        ee = (1+idc)*n_tpts
        nu_obs_all[ss:ee] = y_true
        
        # Simulate ORN with fitted params
        args0[2] = cc_sim
        [t_orn, nu_orn] = orn_depalo0(params2an, *args0)
        
        # interpolate simulated activity values
        nu_sim_fcn = interp1d(t_orn, nu_orn)
        y_fit = nu_sim_fcn(t_fit)
        
        # PLOT
        # y_soft_l1 = funct0(x0, Ts_soft_l1[id_t], pars_soft_l1)
        clr = greenmap.to_rgba(n_clr-idc)
        ax.plot(t_fit, y_true, '-.', color=clr, label='observ %.5f'%cc_sim)
        ax.plot(t_fit, y_fit, '-*', color=clr, label='fit %.5f'%cc_sim)
        # ax.plot(x0, y_soft_l1, '--', label='soft l1 loss')
    
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.legend()
    #plt.show()
    
    toc = tictoc()
    print(toc-tic)

    return nu_obs_all

#%% INITIALIZE PARAMETERS

# fitting parameters
t_tot           = 1000  # ms
t_on            = 160   # ms, stimulus onset
t_tot_fit       = 900
delay_obs       = 50    # ms, delay between open valve and antenna

pts_ms          = 1     # number of pts per ms for the simulations

n_tpts          = 300    # number of datapoints in each curve
t_fit           = np.linspace(-t_on, t_tot_fit-t_on, n_tpts)
t_fit_obs       = t_fit+delay_obs
pts_ms_obs      = t_tot_fit/n_tpts  # number of pts per ms for the observed data



file_names = ['ethyl_ab3A_10.csv', 'ethyl_ab3A_17.csv', 'ethyl_ab3A_20.csv', 
                  'ethyl_ab3A_22.csv', 'ethyl_ab3A_27.csv', 'ethyl_ab3A_30.csv',
                  'ethyl_ab3A_35.csv', 'ethyl_ab3A_40.csv', 'ethyl_ab3A_80.csv',]
    
concs_obs       = np.array([10, 20, 30,40]) # list of concentrations to list
concs_sim       = 10**-np.array(concs_obs/10) # list of concentrations to list
n_cs            = len(concs_obs)

# [n, alpha_r, beta_r, 
# c_rect, a_rect, nu_max_rect, 
# a_y, b_y, b_x, c_x, d_x, a_r] = params2fit
params2fit      = np.array([.5, .9, .09, 
                   1, 3.3, 250,
                   .25, .002, .2, .0028, 1, 1])

fig = plt.figure()
ax = fig.subplots(1, 1,)
nu_obs_all = figure_fit(params2fit, ax)
ax.set_title('default params')


# %% FIT OBSERVED DATA TO SIMULATIONS
args_fit = [n_tpts, nu_obs_all, t_fit, t_tot, t_on, ]
args_fit = np.concatenate((args_fit, concs_sim))

tic = tictoc()

# lower and upper bounds
lb = np.zeros_like(params2fit)
lb[1:3] = -np.inf
ub = np.ones_like(params2fit)*np.inf
ub[0] = 3
ub[5] = 1000
params_0 = 3*np.ones_like(params2fit)

# check that the starting params are all w/in the bounds
while ~np.all((params_0>lb) & (params_0<ub)):
    params_0 = params2fit*(1+1*np.random.randn())

fig = plt.figure()
ax = fig.subplots(1, 1,)
figure_fit(params_0, ax)
ax.set_title('starting params')


#%% linear diff fit
tic = tictoc()
res_lsq = least_squares(leastsq_orn, params_0, 
                        bounds=(lb, ub), ftol=None, xtol=None,
                        args=args_fit)
params_lsq = res_lsq.x
print('exit status: %d'%res_lsq.status)
fig = plt.figure()
ax = fig.subplots(1, 1,)
figure_fit(params_lsq, ax)
ax.set_title('fit linear diff')

toc = tictoc()
print('lindiff fit time: %.1f s'%(toc-tic))

#%%  soft diff fit
tic = tictoc()
res_soft_l1 = least_squares(leastsq_orn, params_0,
                        loss='soft_l1', f_scale=0.1, 
                        bounds=(lb, ub), #ftol=None, xtol=None,
                        args=args_fit)
pars_soft = res_soft_l1.x
print('exit status: %d'%res_soft_l1.status)
fig = plt.figure()
ax = fig.subplots(1, 1,)
figure_fit(pars_soft, ax)
ax.set_title('fit soft diff')


toc = tictoc()
print('soft fit time: %.1f s'%(toc-tic))

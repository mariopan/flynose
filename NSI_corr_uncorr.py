#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 17:17:34 2018
NSI_corr_uncorr.py

Plot for the explanatory figure of the main hypothesis. It uses the model 
developed by De Palo et al.

@author: mp525
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint



# STANDARD FIGURE PARAMS
lw = 4
fs = 13
plt.rc('text', usetex=True)  # laTex in the polot
#plt.rc('font', family='serif')
fig_position = 1300,10
title_fs = 20 # font size of ticks
label_fs = 20 # font size of labels
panel_fs = 30
legend_fs = 12

black   = 'xkcd:black'
blue    = 'xkcd:blue'
red     = 'xkcd:red'
green   = 'xkcd:green'
purple  = 'xkcd:purple'
orange  = 'xkcd:orange'
#cmap    = plt.get_cmap('rainbow')

def depalo_eq(z,t,u,u2,ax, cx, bx, by,dy,):
    x = z[0]
    y = z[1]
    w = z[2]
    q = z[3]
    dydt = u - cx*x*(1+dy*y) - by*y - .05*w 
    dxdt = ax*y - bx*x
    dwdt = u2 - cx*q*(1+dy*w) - by*w - .05*y
    dqdt = ax*w - bx*q
    dzdt = [dxdt,dydt,dwdt,dqdt]
    return dzdt

def rect_func(b, x):
    ot = b[0]/(1 + np.exp(-b[1]*(x-b[2])))
    return ot

def rise_func(t, last_stim, c, tau_on):
    rf = last_stim+(c-last_stim)*(1.0-np.exp(-t/tau_on))
    return rf

def decay_func(t, last_stim, c, tau_off):
    df = c - (c-last_stim)* np.exp(-t/tau_off)
    return df



# c =  0.7 for seed = 3200 #
# c =  1 for seed = 2679 # boring
# c = -1    for seed = 3457 #
# c =  1    for seed = 4645 #  1221 nice
# c = -0.93 for seed = 7873 #
# c = -0.61 for seed = 2197 #
# c =  0.17 for seed = 4576 #
# c =  0.3  for seed = 6489  #strange one
# c =  0.6  for seed = 1888  #

seed = 4645 # round(np.random.uniform()*10000)
print(seed)
np.random.seed(seed)

fld_analysis = 'hypotheses/'
fig_save = 0
if seed == 7873:
    fig_name = 'NSI_negative_corr'
elif seed == 4645:
    fig_name = 'NSI_high_corr'
else:
    fig_save = False
    


corr_wanted = .95

stimuli_length  = 2500 +1# ms, approximate total length of the stimuli
t_open_valve    = 100  # ms, 1/lambda, pulse mean interval 
n_switch_ip     = int(stimuli_length/t_open_valve) # approximate number of valvees

tau_on          = 20     # rising/onset timescale 
tau_off         = 60     # decay/offset timescale

conc_y          = .2 # max value of conc. for stimuli to ORN_y
conc_w          = .2  # max value of conc. for stimuli to ORN_w


    
# initial condition adaptation variables
z0 = [0,0,0,0]      
# receptor binding params
#b       = 0.25
#d       = 0.22
# rectification params
c = 1;
a = 3.3;
nu_max = 250;
B0 = [nu_max, a, c]

#adaptation params
ax      = 0.25
bx      = 0.002
cx      = 0.004
by      = 0.15
dy      = 0.7


# old parameters
#ax = .1;
#bx = .01;
#cx = .001;   % or g or cx
#by = .001;   % or b or by

# nagel times:
switch_time_all = np.linspace(t_open_valve, int(t_open_valve*n_switch_ip),n_switch_ip).astype(int)   # valve in time [integers]
rnd_flip_y = np.random.randint(2, size=n_switch_ip)     # random flip of 0s and 1s
rnd_flip_y = ((np.round(rnd_flip_y)-.5)*2).astype(int)  # random flip of 1s and -1s
switch_time_y = switch_time_all[rnd_flip_y == 1]
n_switch_y = np.size(switch_time_y)  # number of times the valve change state        
rand_uni = np.random.uniform(0,1,n_switch_ip)           # random numers between 0 and 1
rand_shf = np.round(rand_uni + corr_wanted/2)           # random flip of 0s and 1s, prob.(1) = 0.5+crr/2
rand_shf = ((np.round(rand_shf)-.5)*2).astype(int)      # random flip of 1s and -1s, prob.(1) = 0.5+crr/2
rnd_flip_w = rand_shf*rnd_flip_y                        # rnd flip of 1s and -1s, slightly different

switch_time_w = switch_time_all[rnd_flip_w == 1]

corr_flip_tmp = np.corrcoef(rnd_flip_w, rnd_flip_y)
corr_flip = corr_flip_tmp[1,0]


n_switch_w = np.size(switch_time_w)  # number of times the valve change state
max_switch_time = np.max([switch_time_y[-1], switch_time_w[-1]])

time_tot = max_switch_time+1     # [ms] tot time of simulation
n_time_pts = time_tot+1           # number of time points

# define vectors for the simulation
t = np.linspace(0,time_tot, n_time_pts)   # time vector# store solution
stim_y = np.zeros_like(t)       # stimulus vector  
stim_w = np.zeros_like(t)       # stimulus vector  
valve_y = np.zeros_like(t)       # valve vector
valve_w = np.zeros_like(t)       # valve vector


# ------------------------------------------------------------
# STIMULATION FOR Y
# record initial conditions
switcher = 1                   # variable to change valve's value
stim_y[0] = 0.0 
valve_y[0:switch_time_y[0]] = -1  
last_stim = stim_y[0]
for ii in range(1,n_switch_y):
    t_start = switch_time_y[ii-1]
    t_stop = switch_time_y[ii]
    valve_y[t_start:t_stop] = switcher
    
    if switcher == 1:
        stim_y[t_start:t_stop] = rise_func(t[t_start:t_stop]-t[t_start], last_stim, conc_y*(1+0.0*np.random.normal()), tau_on)
    elif switcher == -1:
        stim_y[t_start:t_stop] = decay_func(t[t_start:t_stop]-t[t_start], last_stim, 0.0, tau_off)
    last_stim = stim_y[t_stop-1]
    switcher = -1 * switcher


# ------------------------------------------------------------
# STIMULATION FOR W
# record initial conditions
switcher = 1#np.sign(corr_wanted)                   # variable to change valve's value
stim_w[0] = 0.0
valve_w[0:switch_time_w[0]] = -1  
# STIMULATION FOR W
last_stim = stim_w[0]
for ii in range(1,n_switch_w):
    t_start = switch_time_w[ii-1]
    t_stop = switch_time_w[ii]
    valve_w[t_start:t_stop] = switcher
    
    if switcher == 1:
        stim_w[t_start:t_stop] = rise_func(t[t_start:t_stop]-t[t_start], last_stim, conc_w*(1+0.0*np.random.normal()), tau_on)
    elif switcher == -1:
        stim_w[t_start:t_stop] = decay_func(t[t_start:t_stop]-t[t_start], last_stim, 0.0, tau_off)
    last_stim = stim_w[t_stop-1]
    switcher = -1 * switcher

corr_stim_tmp = np.corrcoef(stim_y[1000:-1], stim_w[1000:-1])
corr_stim = corr_stim_tmp[1,0]
corr_valve_tmp = np.corrcoef(valve_y[1000:-1], valve_w[1000:-1])
corr_valve = corr_valve_tmp[1,0]

print('corr stims:' + '%0.2f'%(corr_stim))
# %% ------------------------------------------------------------
# SOLVE ODE for ORNs model
        
# store solution
x = np.empty_like(t)
y = np.empty_like(t)
w = np.empty_like(t)
q = np.empty_like(t)
x[0] = z0[0]
y[0] = z0[1]
w[0] = z0[2]
q[0] = z0[3]

# solve ODE
for i in range(1,n_time_pts):
    # span for next time step
    tspan = [t[i-1],t[i]]
    # solve for next step
    z = odeint(depalo_eq,z0,tspan,args=(stim_y[i],stim_w[i],ax,cx, bx,by,dy,))
    # store solution for plotting
    x[i] = z[1][0]
    y[i] = z[1][1]
    w[i] = z[1][2]
    q[i] = z[1][3]
    # next initial condition
    z0 = z[1]
# --------------------------------------------------------
    
         
# rectification 
y_rect = rect_func(B0, y);
w_rect = rect_func(B0, w);

# %% ------------------------------------------------------------
# FIGURE
# Figure parameters
fig_size = [13, 5]
fig_position = 1000,50
ticks_fs = 20 # font size of ticks
label_fs = 30 # font size of ticks
lw = 5 # linewidth

rs = 1 # number of rows
cs = 3 # number of cols
black = 'xkcd:black'
blue = 'xkcd:blue'
red = 'xkcd:red'
green = 'xkcd:green'
purple = 'xkcd:purple'
orange = 'xkcd:orange'

fig = plt.figure(figsize=fig_size)
#fig.canvas.manager.window.wm_geometry("+%d+%d" % fig_position )
#fig.tight_layout()

ax1 = plt.subplot(rs,cs, 1)
ax2 = plt.subplot(rs,cs, 3)


ax1.plot(stim_y, color=green, linewidth=lw+1, label='stim 1')
ax1.plot(stim_w, '--', color=purple, linewidth=lw-1, label='stim 2')

ax2.plot(y_rect, color=green, linewidth=lw+1, label='out 1')
ax2.plot(w_rect, '--', color=purple, linewidth=lw-1, label='out 2')

ax1.set_ylabel('Odourant conc. (a.u.)', size=label_fs,fontname='Sans')
ax1.set_xlabel('Time (s)', size=label_fs,fontname='Sans')
ax1.set_xlim(1000, 2000)
ax1.set_xticks((1500, 2000))
ax1.set_xticklabels((0.5, 1))
ax1.set_yticks([]) 
ax1.tick_params(labelsize=ticks_fs)

ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')

ll, bb, ww, hh = ax1.get_position().bounds
ax1.set_position([ll, bb+.05, ww, hh])
    
ax2.set_ylabel('ORN response (Hz)', size=label_fs,fontname='Sans')
ax2.set_xlabel('Time (s)', size=label_fs,fontname='Sans')
ax2.set_ylim(0, 150)
ax2.set_xlim(1000, 2000)
ax2.set_xticks((1500, 2000))
ax2.set_xticklabels((0.5, 1))
ax2.set_yticks((0, 100)) 
ax2.tick_params(labelsize=ticks_fs)

ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')

ll, bb, ww, hh = ax2.get_position().bounds
ax2.set_position([ll, bb+.05, ww, hh])

if fig_save:
    plt.savefig(fld_analysis+ fig_name)
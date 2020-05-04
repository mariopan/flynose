#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:06:08 2020
plot_hyp1.py

@author: mario
"""


import numpy as np
import matplotlib.pyplot as plt


# *****************************************************************
# STANDARD FIGURE PARAMS
lw = 4
fs = 13
plt.rc('text', usetex=True)  # laTex in the polot
#plt.rc('font', family='serif')
fig_position = 1300,10
title_fs = 20 # font size of ticks
label_fs = 20 # font size of labels
panel_fs = 30 # font size of panel' letters
legend_fs = 12

black   = 'xkcd:black'
blue    = 'xkcd:blue'
red     = 'xkcd:red'
green   = 'xkcd:green'
purple  = 'xkcd:purple'
orange  = 'xkcd:orange'
#cmap    = plt.get_cmap('rainbow')



def olsen_orn_pn(nu_orn, sigma, nu_max):
    nu_pn = nu_max * np.power(nu_orn, 1.5)/(np.power(nu_orn, 1.5) 
        + np.power(sigma,1.5))
    return nu_pn

fig_save = 1
fld_analysis = '../hypotheses/'
fig_hyp1_name = 'hypothesis1'

sigma = 15
nu_max = 250 # Hz

sigma2 = 40

nu_orn = np.linspace(5, 140, 200)
nu_pn = olsen_orn_pn(nu_orn, sigma, nu_max)
nu_pn2 = olsen_orn_pn(nu_orn, sigma2, nu_max)

x2plot_a = [15, 30,]
y2plot_a = olsen_orn_pn(x2plot_a, sigma, nu_max)
y2plot_a2 = olsen_orn_pn(x2plot_a, sigma2, nu_max)

x2plot_b = [60, 120]
y2plot_b = olsen_orn_pn(x2plot_b, sigma, nu_max)
y2plot_b2 = olsen_orn_pn(x2plot_b, sigma2, nu_max)

#**********************************************************
# FIGURE
#**********************************************************

rs = 1
cs = 1
fig, axs = plt.subplots(rs,cs, figsize=(7,4), )

plt.rc('text', usetex=True)

axs.plot(nu_orn, nu_pn, label='w/o NSI', color= 'black',lw=lw)
axs.plot(nu_orn, nu_pn2, '--', color= 'red', label='with NSI', lw=lw)

axs.errorbar(x2plot_a, y2plot_a,  fmt='o', color='blue')
axs.errorbar(x2plot_b, y2plot_b,  fmt='o', color='blue')
axs.errorbar(x2plot_a, y2plot_a2,  fmt='d', color='blue')
axs.errorbar(x2plot_b, y2plot_b2,  fmt='d', color='blue')

#axs.errorbar(x2plot_a, np.zeros_like(y2plot_a),  fmt='-', color='blue', lw=lw-1)
#axs.errorbar(x2plot_b, np.zeros_like(y2plot_b),  fmt='-', color='blue', lw=lw-1)

axs.errorbar(np.zeros_like(x2plot_a), y2plot_a,  fmt='-', color='black', lw=lw)
axs.errorbar(np.zeros_like(x2plot_b), y2plot_b,  fmt='-', color='black', lw=lw)

axs.errorbar(np.ones_like(x2plot_a)*2, y2plot_a2,  fmt='-', color='red', lw=lw)
axs.errorbar(np.ones_like(x2plot_b)*2, y2plot_b2,  fmt='-', color='red', lw=lw)



axs.tick_params(axis='both', labelsize=label_fs)
axs.set_xticks(np.concatenate((x2plot_a,x2plot_b)))
axs.set_xticklabels(['A1','B1','A2','B2',] )

#axs.set_xticklabels('')
axs.set_yticklabels('')
axs.spines['right'].set_color('none')
axs.spines['top'].set_color('none')

axs.set_ylabel(r'ORN Firing rates (Hz)', fontsize=label_fs)
axs.set_xlabel(r'Odor concentration (au)', fontsize=label_fs)
            
axs.legend(loc='lower right', fontsize=label_fs, frameon=False)
            
dy = 0.07
dx = 0.05
ll, bb, ww, hh = axs.get_position().bounds
axs.set_position([ll+dx, bb+dy, ww, hh])

axs.text(-.1, 1.05, 'b.', transform=axs.transAxes, color=blue,
   fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            

if fig_save:
    fig.savefig(fld_analysis+  fig_hyp1_name+'.png')
       
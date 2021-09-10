#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:06:08 2020
plot_hyp1.py

Plot for the explanatory figure of the secondary hypothesis

@author: mario
"""


import numpy as np
import matplotlib.pyplot as plt


# *****************************************************************
# STANDARD FIGURE PARAMS
thin = 1.5
fs = 20
lw = 3
plt.rc('text', usetex=True)  # laTex in the plot
# plt.rc('font', family='serif')

fig_size = [10, 6]
fig_position = 1300,10
title_fs = 20 # font size of ticks
label_fs = 20 # font size of labels
panel_fs = 30 # font size of panel letters
ticks_fs = label_fs - 3

black   = 'xkcd:black'
blue    = 'xkcd:blue'
red     = 'xkcd:red'
green   = 'xkcd:green'
purple  = 'xkcd:purple'
orange  = 'xkcd:orange'
magenta = 'xkcd:magenta'
cyan    = 'xkcd:cyan'
pink    = 'xkcd:pink'
cmap = 'inferno'
colors = plt.cm.winter_r
clr_fct = 30        # color factor
    
np.set_printoptions(precision=5)




def orn_nu(conc):
    c0      = 1e-5
    k       = 1e4
    nu_max  = 210
    nu_min  = -180
    
    return nu_min + (nu_max- nu_min)/(1 + (np.exp(-(conc-c0)*k)))





def olsen_orn_pn(nu_orn, sigma, nu_max):
    nu_pn = nu_max * np.power(nu_orn, 1.5)/(np.power(nu_orn, 1.5) 
        + np.power(sigma,1.5))
    return nu_pn




#############################################################################
# NEW HYPOTHESIS 
fig_save = 0
fig_name = 'new_hyp1.png'
fld_output = 'images/hypotheses/'

nsi = .07
label_fs = 40
lw = 5

c2plot = np.logspace(-6, -3, 100)
nu = orn_nu(c2plot)


nu1 = 30 # Hz
c_nu1 = next(x[0] for x in enumerate(nu) if x[1] > nu1)

c_ratio = c2plot[c_nu1:]/c2plot[c_nu1]
nu_ratio = nu[c_nu1:]/nu1

nu1_nsi = nu1-nu[c_nu1:]*nsi
nu_nsi_ratio = nu[c_nu1:]/nu1_nsi

fig, axs  = plt.subplots(1, 1, figsize=[10, 8])
left, bottom, width, height = [0.32, 0.6, 0.22, 0.28]
ax2 = fig.add_axes([left, bottom, width, height])


# PLOT 
axs.plot(c2plot, nu, color=green, linewidth=lw)
axs.plot(c2plot, np.ones_like(c2plot)*nu1, '--',color=purple,  linewidth=lw)
axs.plot(c2plot[c_nu1:], nu1_nsi, ':', color=purple, linewidth=lw)
# PLOT inset
ax2.plot(c_ratio, nu_ratio, color=pink, linewidth=lw)
ax2.plot(c_ratio, c_ratio, 'k--', linewidth=lw)
ax2.plot(c_ratio, nu_nsi_ratio, color=cyan, linewidth=lw)


# SETTINGS
axs.set_xscale('log')
axs.spines['right'].set_color('none')
axs.spines['top'].set_color('none')
axs.set_xlabel('odor concentration (a.u.)', fontsize=label_fs)
axs.set_ylabel('ORN rates (Hz)', fontsize=label_fs)
axs.tick_params(axis='both', labelsize=label_fs)
axs.text(3e-4, 150, 'ORN \n strong', fontsize= label_fs, color=green)
axs.text(1e-4, 35, 'ORN weak ctrl', fontsize= label_fs, color=purple)
axs.text(2e-5, 0, 'ORN weak NSI', fontsize= label_fs, color=purple)

# SETTINGS inset
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_xlabel('conc. ratio', fontsize=label_fs-1)
ax2.set_ylabel('ORN ratio', fontsize=label_fs-1)
ax2.tick_params(axis='both', labelsize=label_fs-1)
ax2.text(15, 4,'ctrl', fontsize= label_fs, color=pink)
ax2.text(3, 10,'NSI', fontsize= label_fs, color=cyan)
ax2.set_yticks([1, 10])
ax2.set_yticklabels([1, 10])
ax2.set_xticks([1, 10])
ax2.set_xticklabels([1, 10])

ax2.set_xlim((0.9, 20))
ax2.set_ylim((9e-1, 20))
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')


dy = 0.07
dx = 0.05
ll, bb, ww, hh = axs.get_position().bounds
axs.set_position([ll+dx, bb+dy, ww, hh])


plt.show()

if fig_save:
    fig.savefig(fld_output +  fig_name,dpi=300)

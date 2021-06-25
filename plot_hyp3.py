#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 14:27:21 2021

@author: mario
"""

import numpy as np

import matplotlib.pyplot as plt

# *****************************************************************
# STANDARD FIGURE PARAMS
thin = 1.5
fs = 20
lw = 2
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




def olsen_orn_pn(nu_orn, sigma, nu_max):
    nu_pn = nu_max * np.power(nu_orn, 1.5)/(np.power(nu_orn, 1.5) 
        + np.power(sigma,1.5))
    return nu_pn


label_fs = 40
lw = 5
fig_save = 0
fld_analysis = 'images/hypotheses/'
fig_hyp3_name = 'hyp_dynrange'

sigma   = 15
nu_max  = 200 # Hz
xmin    = 0
xmax    = 250
sigma2  = 40
msz     = 15

nu_orn = np.linspace(xmin, xmax, 1000)
nu_pn = olsen_orn_pn(nu_orn, sigma, nu_max)
nu_pn2 = olsen_orn_pn(nu_orn, sigma2, nu_max)

thr_10_ctrl = nu_pn[next(x[0] for x in enumerate(nu_pn) if x[1] > np.max(nu_pn)*.1)]
thr_10_nsi = nu_pn2[next(x[0] for x in enumerate(nu_pn2) if x[1] > np.max(nu_pn)*.1)]

thr_90_ctrl = nu_pn[next(x[0] for x in enumerate(nu_pn) if x[1] > np.max(nu_pn)*.9)]
thr_90_nsi = nu_pn2[next(x[0] for x in enumerate(nu_pn2) if x[1] > np.max(nu_pn)*.9)]

x_thr_10_ctrl = nu_orn[next(x[0] for x in enumerate(nu_pn) if x[1] > np.max(nu_pn)*.1)]
x_thr_10_nsi = nu_orn[next(x[0] for x in enumerate(nu_pn2) if x[1] > np.max(nu_pn)*.1)]

x_thr_90_ctrl = nu_orn[next(x[0] for x in enumerate(nu_pn) if x[1] > np.max(nu_pn)*.9)]
x_thr_90_nsi = nu_orn[next(x[0] for x in enumerate(nu_pn2) if x[1] > np.max(nu_pn)*.9)]

# FIGURE
rs = 1
cs = 1
fig, axs = plt.subplots(rs,cs, figsize=(10,8), )
y_max = 200
arrow_size = 7
plt.rc('text', usetex=True)
axs.plot([x_thr_10_ctrl, x_thr_10_ctrl], [0, y_max], '--', lw=lw-3, color='grey', )
axs.plot([x_thr_90_ctrl, x_thr_90_ctrl], [0, y_max], '--', lw=lw-3, color='grey', )
axs.plot([x_thr_10_nsi, x_thr_10_nsi], [0, y_max], '--', lw=lw-3, color='grey', )
axs.plot([x_thr_90_nsi, x_thr_90_nsi], [0, y_max], '--', lw=lw-3, color='grey', )


axs.plot(nu_orn, nu_pn, label='w/o NSI', color= 'pink',lw=lw)
axs.plot(nu_orn, nu_pn2, color= 'cyan', label='with NSI', lw=lw)

axs.errorbar(x_thr_10_ctrl, thr_10_ctrl,  fmt='o', markersize= msz, color='pink', mfc= 'white', mew= thin)
axs.errorbar(x_thr_90_ctrl, thr_90_ctrl,  fmt='o', markersize= msz, color='pink', mfc= 'white', mew= thin)

axs.plot([x_thr_10_ctrl+2, x_thr_90_ctrl-2], [70, 70], lw=lw, color='pink', )
axs.arrow(x_thr_90_ctrl, 70, 2, 0, shape='full', lw=0, 
          length_includes_head=True, head_width=arrow_size, color='pink')
axs.arrow(x_thr_10_ctrl, 70, -2, 0, shape='full', lw=0, 
          length_includes_head=True, head_width=arrow_size, color='pink')


axs.errorbar(x_thr_10_nsi, thr_10_nsi,  fmt='o', markersize= msz, color='cyan', mfc= 'white', mew= thin)
axs.errorbar(x_thr_90_nsi, thr_90_nsi,  fmt='o', markersize= msz, color='cyan', mfc= 'white', mew= thin)

axs.plot([x_thr_10_nsi+2, x_thr_90_nsi-2], [50, 50], lw=lw, color='cyan', )
axs.arrow(x_thr_90_nsi, 50, 2, 0, shape='full', lw=0, 
          length_includes_head=True, head_width=arrow_size, color='cyan')
axs.arrow(x_thr_10_nsi*.98, 50, -2, 0, shape='full', lw=0, 
          length_includes_head=True, head_width=arrow_size, color='cyan')

# axs.text(x_thr_10_ctrl*5, thr_10_ctrl, '10\% ', fontsize=label_fs)
# axs.text(x_thr_90_ctrl*1.2, thr_90_ctrl*.9, '90\% ', fontsize=label_fs)

axs.tick_params(axis='both', labelsize=label_fs, )
# axs.ticklabel_format(axis='both', fontweight='bold')
axs.set_yticklabels('')
axs.set_xticklabels('')
axs.spines['right'].set_color('none')
axs.spines['top'].set_color('none')

axs.text(25, 180,'ctrl', fontsize= label_fs, color=pink)
axs.text(100,130,'NSI', fontsize= label_fs, color=cyan)


axs.set_ylabel(r'ORN Firing rates (au)', fontweight='bold',fontsize=label_fs)
axs.set_xlabel(r'Odor concentration (au)', fontweight='bold',fontsize=label_fs)
        

axs.set_ylim((0, y_max))
dy = 0.07
dx = 0.05
ll, bb, ww, hh = axs.get_position().bounds
axs.set_position([ll+dx, bb+dy, ww, hh])
   
axs.text(-.1, 1.05, 'e.', transform=axs.transAxes, #color=blue,
   fontsize=label_fs, fontweight='bold', va='top', ha='right')      
plt.show()



if fig_save:
    fig.savefig(fld_analysis+fig_hyp3_name+'.png', dpi=300)

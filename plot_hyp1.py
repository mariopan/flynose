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

fig_save = 0
fld_analysis = '../hypotheses/'
fig_hyp1_name = 'hypothesis1'

sigma = 15
nu_max = 250 # Hz
xmin= 5
xmax= 140
sigma2 = 40
msz= 8

nu_orn = np.linspace(xmin, xmax, 200)
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
fig, axs = plt.subplots(rs,cs, figsize=(5,4), )

plt.rc('text', usetex=True)

axs.plot(nu_orn, nu_pn, label='w/o NSI', color= 'black',lw=lw)
axs.plot(nu_orn, nu_pn2, '--', color= 'red', label='with NSI', lw=lw)

axs.errorbar(x2plot_a, y2plot_a,  fmt='o', markersize= msz, color='black',mfc= 'white', mew= thin)
axs.errorbar(x2plot_b, y2plot_b,  fmt='o', markersize= msz, color='black', mfc= 'white', mew= thin)
axs.errorbar(x2plot_a, y2plot_a2,  fmt='d', markersize= msz, color='red', mfc= 'white', mew= thin)
axs.errorbar(x2plot_b, y2plot_b2,  fmt='d', markersize= msz, color='red', mfc= 'white', mew= thin)

#axs.errorbar(x2plot_a, np.zeros_like(y2plot_a),  fmt='-', color='blue', lw=lw-1)
#axs.errorbar(x2plot_b, np.zeros_like(y2plot_b),  fmt='-', color='blue', lw=lw-1)

axs.errorbar(np.zeros_like(x2plot_a), y2plot_a,  fmt='-', color='black', lw=lw)
axs.errorbar(np.zeros_like(x2plot_b), y2plot_b,  fmt='-', color='black', lw=lw)

axs.errorbar(np.ones_like(x2plot_a)*xmax+5, y2plot_a2,  fmt='-', capsize= 100, capthick= 2, color='red', lw=lw)
axs.errorbar(np.ones_like(x2plot_b)*xmax+5, y2plot_b2,  fmt='-', capsize= 100, capthick= 2, color='red', lw=lw)

for i in [0, 1]:
    axs.plot([x2plot_a[i], xmax+5], [y2plot_a2[i], y2plot_a2[i]], linestyle= '--', color='red', lw= thin)
    axs.plot([x2plot_b[i], xmax+5], [y2plot_b2[i], y2plot_b2[i]], linestyle= '--', color='red', lw= thin)
    
    axs.plot([0, x2plot_a[i]], [y2plot_a[i], y2plot_a[i]], linestyle= '--', color='black', lw= thin)
    axs.plot([0, x2plot_b[i]], [y2plot_b[i], y2plot_b[i]], linestyle= '--', color='black', lw= thin)


axs.tick_params(axis='both', labelsize=label_fs)
axs.set_xticks(np.concatenate((x2plot_a,x2plot_b)))
axs.set_xticklabels(['A1','B1','A2','B2',] )

#axs.set_xticklabels('')
axs.set_yticklabels('')
axs.spines['right'].set_color('none')
axs.spines['top'].set_color('none')

axs.set_ylabel(r'ORN Firing rates (Hz)', fontsize=label_fs)
axs.set_xlabel(r'Odor concentration (au)', fontsize=label_fs)
            
#axs.legend(loc='lower right', fontsize=label_fs, frameon=False)
            
dy = 0.07
dx = 0.05
ll, bb, ww, hh = axs.get_position().bounds
print(ww)
print(hh)
axs.set_position([ll+dx, bb+dy, ww, hh])


axs.text(-.1, 1.05, 'b.', transform=axs.transAxes, #color=blue,
   fontsize=panel_fs, fontweight='bold', va='top', ha='right')
            

if fig_save:
    fig.savefig(fig_hyp1_name+'.png')
       
plt.show()

#%%############################################################################
# NEW HYPOTHESIS 


fig_save = 0
fig_name = 'new_hyp1.png'
fld_output = 'images/'

nsi = .07

c2plot = np.logspace(-6, -3, 100)
nu = orn_nu(c2plot)


nu1 = 30 # Hz
c_nu1 = next(x[0] for x in enumerate(nu) if x[1] > nu1)

c_ratio = c2plot[c_nu1:]/c2plot[c_nu1]
nu_ratio = nu[c_nu1:]/nu1

nu1_nsi = nu1-nu[c_nu1:]*nsi
nu_nsi_ratio = nu[c_nu1:]/nu1_nsi

fig, axs  = plt.subplots(1, 1, figsize=[10, 8])
left, bottom, width, height = [0.25, 0.6, 0.22, 0.28]
ax2 = fig.add_axes([left, bottom, width, height])


# PLOT 
axs.plot(c2plot, nu, color=green, linewidth=lw)
axs.plot(c2plot, np.ones_like(c2plot)*nu1, '--',color=purple,  linewidth=lw)
axs.plot(c2plot[c_nu1:], nu1_nsi, '--', color=cyan, linewidth=lw)
# PLOT inset
ax2.plot(c_ratio, nu_ratio, color=pink, linewidth=lw)
ax2.plot(c_ratio, c_ratio, 'k--', linewidth=lw)
ax2.plot(c_ratio, nu_nsi_ratio, color=cyan, linewidth=lw)

label_fs = 30

# SETTINGS
axs.set_xscale('log')
axs.spines['right'].set_color('none')
axs.spines['top'].set_color('none')
axs.set_xlabel('odor conc (au)', fontsize=label_fs)
axs.set_ylabel('ORN rates (Hz)', fontsize=label_fs)
axs.tick_params(axis='both', labelsize=label_fs)
axs.text(2.5e-4, 150, 'ORN strong', fontsize= label_fs, color=green)
axs.text(1e-4, 35, 'ORN weak', fontsize= label_fs, color=purple)
axs.text(1e-4, 6, 'ORN weak NSI', fontsize= label_fs, color=cyan)
# axs.text(-.1, 1.1, 'b', transform=axs.transAxes,
#     fontsize=panel_fs, color=black, weight='bold', va='top', ha='right')

# SETTINGS inset
ax2.set_yscale('log')
ax2.set_xlabel('conc ratio', fontsize=label_fs-1)
ax2.set_ylabel('FR ratio', fontsize=label_fs-1)
ax2.tick_params(axis='both', labelsize=label_fs-1)
ax2.text(15, 4,'ctrl', fontsize= label_fs, color=pink)
ax2.text(3, 10,'NSI', fontsize= label_fs, color=cyan)
ax2.set_xlim((0, 20))
ax2.set_ylim((9e-1, 20))
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')

plt.show()

if fig_save:
    fig.savefig(fld_output +  fig_name,dpi=300)


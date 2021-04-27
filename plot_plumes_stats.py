#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:36:50 2019
plot_plumes_stats.py


This script makes three plot in a single figure:
a) Probability distribution of the whiff durations for odorants emitted at 
distances larger than 60 m (See Yee 1995). 

b) Probability distribution of the blank durations for odorants emitted at 
    distances larger than 60 m (See Yee 1995). 

c) Probability distribution of the normalized concentration for odorants 
    emitted at 75 m distance from the source (See Mylne 1991).




@author: mp525
"""

import matplotlib.pyplot as plt
from scipy.integrate import quad
import numpy as np

import stats_for_plumes as stats

# *****************************************************************
# STANDARD FIGURE PARAMS
lw = 4
fs = 13
plt.rc('text', usetex=True)  # laTex in the polot
#plt.rc('font', family='serif')
fig_size = [12, 12]
fig_position = 1300,10
title_fs = 25 # font size of ticks
label_fs = 20 # font size of labels
panel_fs = 30  # font size of panel's letter
tick_fs = label_fs-3
black   = 'xkcd:black'
blue    = 'xkcd:blue'
red     = 'xkcd:red'
green   = 'xkcd:green'
purple  = 'xkcd:purple'
orange  = 'xkcd:orange'
cmap    = plt.get_cmap('rainbow')



# *****************************************************************
# Fig. PlumesStats
fld_stimuli = 'open_field_stimuli/images'
fig_save    = 0
fig_name    = '/plumes_stats'
panels_id   = ['a', 'b', 'c', ]




# *******************************************************************
#  PARAMS FOR WHIFF AND BLANK DISTRIOBUTIONS

distance_yee2 = np.array([20, 40, 60, 100, 180, 220, 330])
dur_up = np.array([.87, .5,.37, .45, .8,1.1,1.95])/2
dur_wh =  np.array([.36,.2,.20,.26,.53,.73,1.1])/2
dur_bl = dur_up - dur_wh # [0.51, 0.3 , 0.09, 0.27, 0.37, 0.85]/2
interm_yee2 = dur_wh/dur_up
print(interm_yee2)
print('duration whiff:' + ' '.join(map(str, dur_wh)))
print('duration blank: ' + ' '.join(map(str, dur_bl)))

distance_yee_obs  = np.array([25,  50,  100,  25,  50,  50,  50,  50, 230, 185, 60, 330, 391, 638])
interm_yee_obs  = np.array([.32, .19, .31, .11, .26, .25, .08, .23, .94, .66, .27, .90, .54, .75])
wind_yee_obs =  np.array([1.3, 1.4, 2.5, 2.6, 3.3, 1.0, 1.4, 1.4, 3.3, 5.3, 5.6, 4.7, 2.2, 1.8])

g = -1/2        # for a power law of -3/2
whiff_min = 3e-3      # [s]
whiff_maxs = np.array([3, 50, 150])# [s]
bl_min = 3e-3      # [s]
bl_maxs = np.array([25, 25, 25])# [s]



# *******************************************************************
# Mylne Mason 1991 params

# PARAMS CONCENTRATION 
numlinbins  = 100
c_0         = 0
c_inf       = 15
linbins     = np.linspace(c_0, c_inf, numlinbins)
w_linbins   = np.diff(linbins)[0]

num_samples_c = 50*numlinbins  
# fit of average concentration at 75 m, Mylne and Mason 1991, Fig.11 a
b1 = -(np.log10(1-.997) + np.log10(1-.5))/10.7
a1 = -0.3*b1 - np.log10(1-.5)

pdf_th_mm75 = stats.pdf_mylne_75m(linbins, a1, b1)
cdf_th_mm75 = stats.cdf_mylne_75m(linbins, a1, b1)
        
    
print(quad(lambda x: stats.pdf_mylne_75m(x, a1, b1), c_0, c_inf)) # integrate the pdf to check it sum to 1

unif = np.random.random(size=num_samples_c)
conc_mm75 = stats.rnd_mylne_75m(a1, b1, unif)


#%% *****************************************************
# FIGURE
wh_mean = np.zeros(3)
bl_mean = np.zeros(3)

rs = 1 # number of rows
cs = 3 # number of columns
fig = plt.figure(figsize=[13, 4.6])    
# fig.canvas.manager.window.wm_geometry("+%d+%d" % fig_position )
# fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

# whiff, blank and concentration panels
ax_wh = plt.subplot(rs,cs, 1)
ax_bl = plt.subplot(rs,cs, 2)
ax_conc = plt.subplot(rs,cs, 3)
        


pdf_th_wh, logbins_wh, wh_mean[0] = stats.whiffs_blanks_pdf(whiff_min, whiff_maxs[0], g)
ax_wh.plot(logbins_wh, pdf_th_wh, color='blue', linewidth=5, label='source dist. 60m')

pdf_th_wh, logbins_wh, wh_mean[1]  = stats.whiffs_blanks_pdf(whiff_min, whiff_maxs[1], g)
ax_wh.plot(logbins_wh, pdf_th_wh, color='green', linewidth=3, label='source dist. 220m')

pdf_th_wh, logbins_wh, wh_mean[2]  = stats.whiffs_blanks_pdf(whiff_min, whiff_maxs[2], g)
ax_wh.plot(logbins_wh, pdf_th_wh, '--', color='red', linewidth=1, label='source dist. 330m')

#
pdf_th_bl, logbins_bl, bl_mean[0] = stats.whiffs_blanks_pdf(bl_min, bl_maxs[0], g)
ax_bl.plot(logbins_bl, pdf_th_bl, color='blue', linewidth=5, label='Theor blanks 60m')

pdf_th_bl, logbins_bl, bl_mean[1] = stats.whiffs_blanks_pdf(bl_min, bl_maxs[1], g)
ax_bl.plot(logbins_bl, pdf_th_bl, color='green', linewidth=3, label='Theor blanks 220m')

pdf_th_bl, logbins_bl, bl_mean[2] = stats.whiffs_blanks_pdf(bl_min, bl_maxs[2], g)
ax_bl.plot(logbins_bl, pdf_th_bl, '--', color='red', linewidth=1, label='Theor blanks 330m')

# 
ax_conc.plot(linbins, -np.log10(1-cdf_th_mm75), linewidth=4, label='Theor Mylne91, 75m')


# FIGURE SETTINGS
ax_wh.legend(fontsize=label_fs-5, frameon=False)
ax_wh.set_xscale('log')
ax_wh.set_yscale('log')
ax_wh.set_ylabel('pdf', fontsize=label_fs)
ax_wh.set_xlabel('duration (s)', fontsize=label_fs)
ax_wh.set_title('Whiff durations', fontsize=title_fs)
ax_wh.tick_params(axis='both', which='major', labelsize=label_fs-5)

#
ax_bl.set_xscale('log')
ax_bl.set_yscale('log')
ax_bl.set_xlabel('duration (s)', fontsize=label_fs)
ax_bl.set_title('Clean air durations', fontsize=title_fs)
ax_bl.tick_params(axis='both', which='major', labelsize=label_fs-5)


ax_wh.spines['right'].set_color('none')
ax_wh.spines['top'].set_color('none')
ax_bl.spines['right'].set_color('none')
ax_bl.spines['top'].set_color('none')
ax_conc.spines['right'].set_color('none')
ax_conc.spines['top'].set_color('none')

# letters on panels
ax_wh.text(-.1, 1.1, panels_id[0], transform=ax_wh.transAxes, 
           fontsize=panel_fs, fontweight='bold', va='top', ha='right')
ax_bl.text(-.1, 1.1, panels_id[1], transform=ax_bl.transAxes,
           fontsize=panel_fs, fontweight='bold', va='top', ha='right')
ax_conc.text(-.1, 1.1, panels_id[2], transform=ax_conc.transAxes,
           fontsize=panel_fs, fontweight='bold', va='top', ha='right')


# Concentration panel d.
y_label_conc_cdf_eff = np.array([0, 0.3, .5, .9, .99, .999, .9999])
y_label_conc_cdf = -np.log10(1-y_label_conc_cdf_eff)

ax_conc.set_yticks(y_label_conc_cdf)
ax_conc.set_yticklabels(y_label_conc_cdf_eff)
ax_conc.set_ylabel('cdf ' , fontsize=label_fs)
ax_conc.set_xlabel(r'$C/<C>$', fontsize=label_fs)
ax_conc.tick_params(axis='both', which='major', labelsize=tick_fs)
ax_conc.set_ylim((0, 3.1))
ax_conc.set_xlim((0, 11.5))

# reset panels positions
ll, bb, ww, hh = ax_wh.get_position().bounds
ax_wh.set_position([ll-.04, bb+.05, ww*1.1, hh])
ll, bb, ww, hh = ax_bl.get_position().bounds
ax_bl.set_position([ll+0.01, bb+.05, ww*1.1, hh])
ll, bb, ww, hh = ax_conc.get_position().bounds
ax_conc.set_position([ll+.06, bb+.05, ww*1.1, hh])

plt.show()

if fig_save:
    fig.savefig(fld_stimuli + fig_name + '.png')
    
    
#%% *****************************************************
# OLD FIGUREs
wh_mean = np.zeros(3)
bl_mean = np.zeros(3)

rs = 2 # number of rows
cs = 3 # number of columns
fig = plt.figure(figsize=[14, 8])    
#fig.canvas.manager.window.wm_geometry("+%d+%d" % fig_position )
#fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

ax = plt.subplot(rs, cs, 1)
ax.plot(distance_yee_obs, interm_yee_obs, 'o', color=blue, label='Obs Yee 1993')
ax.set_ylabel('Interm. wh/(wh+bl) (s/s)', color=blue, fontsize=fs)
ax.set_xlim((0, 650))
ax.set_ylim((0, 1))


ax3 = plt.subplot(rs, cs, 4)
ax3.plot(distance_yee2, interm_yee2, 'o', color=blue)
ax3.set_ylabel('Interm. wh/(wh+bl) (s/s)', color=blue, fontsize=fs)
ax3.set_xlabel('distance  (m)', color=blue, fontsize=fs)
ax3.set_xlim((0, 650))
ax3.set_ylim((0, 1))

y_label_conc_cdf_eff = np.array([0, 0.3, .5, .9, .99, .999, .9999])
y_label_conc_cdf = -np.log10(1-y_label_conc_cdf_eff)


ax_wh = plt.subplot(rs,cs, 2)
ax_wh.set_title('Mylne 1991', fontsize=title_fs)


pdf_th_wh, logbins_wh, wh_mean[0] = stats.whiffs_blanks_pdf(whiff_min, whiff_maxs[0], g)
ax_wh.plot(logbins_wh, pdf_th_wh, color='blue', linewidth=5, label='Theor whiffs 60m')

pdf_th_wh, logbins_wh, wh_mean[1]  = stats.whiffs_blanks_pdf(whiff_min, whiff_maxs[1], g)
ax_wh.plot(logbins_wh, pdf_th_wh, color='green', linewidth=2, label='Theor whiffs 220m')

pdf_th_wh, logbins_wh, wh_mean[2]  = stats.whiffs_blanks_pdf(whiff_min, whiff_maxs[2], g)
ax_wh.plot(logbins_wh, pdf_th_wh, color='red', linewidth=.5, label='Theor whiffs 330m')
ax_wh.yaxis.set_label_coords(-0.1,0.5)

ax_wh.set_xscale('log')
ax_wh.set_yscale('log')
ax_wh.set_ylabel('pdf whiff durations', fontsize=fs)
print(wh_mean)


ax_bl = plt.subplot(rs,cs, 5)

#ax_bl.plot(logbins_bl, pdf_th_bl, color='green', linewidth=5, label='Theor blanks')
pdf_th_bl, logbins_bl, bl_mean[0] = stats.whiffs_blanks_pdf(bl_min, bl_maxs[0], g)
ax_bl.plot(logbins_bl, pdf_th_bl, color='blue', linewidth=5, label='Theor blanks 60m')

pdf_th_bl, logbins_bl, bl_mean[1] = stats.whiffs_blanks_pdf(bl_min, bl_maxs[1], g)
ax_bl.plot(logbins_bl, pdf_th_bl, color='green', linewidth=2, label='Theor blanks 220m')

pdf_th_bl, logbins_bl, bl_mean[2] = stats.whiffs_blanks_pdf(bl_min, bl_maxs[2], g)
ax_bl.plot(logbins_bl, pdf_th_bl, color='red', linewidth=.5, label='Theor blanks 330m')

ax_bl.set_xscale('log')
ax_bl.set_yscale('log')
ax_bl.set_ylabel('pdf blank durations', fontsize=fs)
ax_bl.set_xlabel('duration (s)', fontsize=fs)

print('mean blanks: ')
print(bl_mean)

interm3 = wh_mean/(wh_mean+bl_mean)
ax.plot(distance_yee2[[2, 5,6]], interm3, label='simul')
ax.legend(fontsize=fs)



#***********************************************

ax_wh_cdf = plt.subplot(rs,cs, 3)
ax_wh_cdf.set_title('Mylne 1991, cdf', fontsize=title_fs)

cdf_th_wh, logbins_wh = stats.whiffs_blanks_cdf(whiff_min, whiff_maxs[0], g)
ax_wh_cdf.plot(logbins_wh, cdf_th_wh, color='blue', linewidth=5, label='Theor whiffs 60m')

cdf_th_wh, logbins_wh = stats.whiffs_blanks_cdf(whiff_min, whiff_maxs[1], g)
ax_wh_cdf.plot(logbins_wh, cdf_th_wh, color='green', linewidth=2, label='Theor whiffs 220m')

cdf_th_wh, logbins_wh = stats.whiffs_blanks_cdf(whiff_min, whiff_maxs[2], g)
ax_wh_cdf.plot(logbins_wh, cdf_th_wh, color='red', linewidth=.5, label='Theor whiffs 330m')

#ax_wh_cdf.set_xscale('log')
#ax_wh_cdf.set_yscale('log')
ax_wh_cdf.set_ylabel('cdf Whiff durations', fontsize=fs)
ax_wh_cdf.set_xlim((0, 1))
ax_wh_cdf.yaxis.set_label_coords(-0.1,0.5)



ax_bl_cdf = plt.subplot(rs,cs, 6)

cdf_th_bl, logbins_bl = stats.whiffs_blanks_cdf(bl_min, bl_maxs[0], g)
ax_bl_cdf.plot(logbins_bl, cdf_th_bl, color='blue', linewidth=5, label='Theor blanks 60m')

cdf_th_bl, logbins_bl = stats.whiffs_blanks_cdf(bl_min, bl_maxs[1], g)
ax_bl_cdf.plot(logbins_bl, cdf_th_bl, color='green', linewidth=2, label='Theor blanks 220m')

cdf_th_bl, logbins_bl = stats.whiffs_blanks_cdf(bl_min, bl_maxs[2], g)
ax_bl_cdf.plot(logbins_bl, cdf_th_bl, color='red', linewidth=.5, label='Theor blanks 330m')

#ax_bl_cdf.set_xscale('log')
#ax_bl_cdf.set_yscale('log')
ax_bl_cdf.set_ylabel('cdf blank durations', fontsize=fs)
ax_bl_cdf.set_xlabel('duration (s)', fontsize=fs)
ax_bl_cdf.legend(fontsize=fs)


#if fig_save:
#    fig.savefig(fld_stimuli + '/whiff_blanks_obs_theor.png')
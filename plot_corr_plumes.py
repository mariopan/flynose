#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:14:13 2020
plot_corr_plumes.py

@author: mario
"""


import timeit
import numpy as np
import matplotlib.pyplot as plt

import corr_plumes 
import stats_for_plumes as stats

#import flynose.corr_plumes as corr_plumes
#import flynose.stats_for_plumes as stats

def overlap(a,b):
    a = (a>0)*1.0
    b = (b>0)*1.0
    return np.sum(a*b)*2.0/(np.sum(a)+np.sum(b))

import pandas as pd 
import seaborn as sns
import matplotlib
# *****************************************************************
# STANDARD FIGURE PARAMS
fs = 20
lw = 2
# plt.rc('text', usetex=True)  # laTex in the polot
# plt.rc('font', family='serif')
matplotlib.rcParams['text.usetex'] = True
fig_size = [20, 10]
fig_position = 700,10
title_fs = 20 # font size of ticks
label_fs = 20 # font size of labels
panel_fs = 30 # font of the letters on each panel
tick_fs = label_fs-3
black   = 'xkcd:black'
blue    = 'xkcd:blue'
red     = 'xkcd:red'
green   = 'xkcd:green'
purple  = 'xkcd:purple'
orange  = 'xkcd:orange'
# *****************************************************************


quenched        = True      # if True Tbl and Twh are chosen to compensate the distance between stimuli

n_loop          = 30
n_repet_loop    = 1         # repeat n_loop n_repet_loop times
fig_plumes_tc   = 1         # time course figure
fig_plumes_dist = 1     
fig_save        = 0

fld_output    = 'open_field_stimuli/images/'
rhos            = [0, 1, 3, 5]

# *******************************************************************
# PARAMS FOR STIMULUS GENERATION
t2sim           = 199.7              # [s] time duration of whole simulated stimulus
pts_ms          = 5
sample_rate     = pts_ms*1000       # [Hz] num of samples per each sec
n_sample2       = 5                 # [ms] num of samples with constant concentration

tot_n_samples   = int(t2sim*sample_rate) # [] duration of whole simulated stimulus in number of samples

# *******************************************************************
#  PARAMS FOR WHIFF AND BLANK DISTRIOBUTIONS
g               = -1/2  # -1/2 for a power law of -3/2, 1 for uniform distribution
whiff_min       = 3e-3  # [s]
whiff_max       = 3     # [s] 3, 50,150

blank_min       = 3e-3   # [s]
blank_max       = 25     # [s]  25

# *******************************************************************
# PARAMS FOR CONCENTRATION DISTRIBUTIONS
# fit of average concentration at 75 m, Mylne and Mason 1991, Fig.10 
b_conc = -(np.log10(1-.997) + np.log10(1-.5))/10.7
a_conc = -0.3*b_conc - np.log10(1-.5)

rho_c       = 1     # corr. between normal distribution to generate concentration        

# CALCULATE THE THEORETICAL MEAN WHIFF, MEAN BLANK DURATIONS AND INTERMITTENCY
pdf_wh, logbins, wh_mean = stats.whiffs_blanks_pdf(whiff_min, whiff_max, g)
pdf_bl, logbins, bl_mean = stats.whiffs_blanks_pdf(blank_min, blank_max, g)

print('main Stim. params:')
print('durs:%.2fs'%t2sim)
# print('rho_{wh,bl}:1-10^{%d}' %round(np.log10(1-rho_t)))
print('rho_{conc}: %.3f' %rho_c)

# still not set params
rho_t       = np.nan
seed_num    = np.nan
# *******************************************************************
# arguments for the generation of stimuli function
stim_params = [t2sim, sample_rate, n_sample2, g, whiff_min, whiff_max, 
       blank_min, blank_max, a_conc, b_conc,rho_c, rho_t, quenched, seed_num, ]   


# *******************************************************************
# REPEATED STIMULI GENERATION
n_rhos = len(rhos)
# n_loop = 30
n_obs = n_rhos*n_loop

cor_stim        = -np.ones((n_obs,1))*2
overlap_stim    = -np.ones((n_obs,1))*2
cor_whiff       = -np.ones((n_obs,1))*2
interm_est      = -np.ones((n_obs,1))*2
interm_est2     = -np.ones((n_obs,1))*2
conc_est        = -np.ones((n_obs,1))*2
conc_est2       = -np.ones((n_obs,1))*2
perc_dif        = -np.ones(n_repet_loop)
seeds           = -np.ones((n_obs,1))*2
th_rhos         = -np.ones((n_obs,1))*2

th_rhos_4 = [0, .9, .999, .99999]
start_seed      = np.random.randint(1, 1000, n_repet_loop)
print('seed: %d'%start_seed)
for pp in range(n_repet_loop):
    
    rr =-1 
    for id_rho, rho in enumerate(rhos):
        rho_t       = 1-10**-rho # correlation between normal distribution to generate whiffs and blanks
        print('rho_{wh,bl}:1-10^{%d}' %round(np.log10(1-rho_t)))
        stim_params[-3] = rho_t

        tic = timeit.default_timer()    
        for ll in range(n_loop):
            rr = rr + 1
            # set a new seed at each rr loop:
            stim_params[-1] = ll + start_seed[pp]
            out_y, out_w, t_dyn, t_dyn_cor, = corr_plumes.main(*stim_params)
            
            th_rhos[rr, 0] = th_rhos_4[id_rho]
           # np.round(100*rho_t)/100
            seeds[rr, 0] = stim_params[-1]
            
            conc_est[rr, 0] = 1.5*np.mean(out_y)
            conc_est2[rr, 0] = 1.5*np.mean(out_w)
            interm_est[rr, 0] = np.sum(out_y>0)/(t2sim*sample_rate)
            interm_est2[rr, 0] = np.sum(out_w>0)/(t2sim*sample_rate)
    
            if (np.sum(out_y)!=0) & (np.sum(out_w)!=0):
                cor_stim[rr,0] = np.corrcoef(out_y, out_w)[1,0]
                overlap_stim[rr, 0] = overlap(out_y, out_w)
                nonzero_concs1  = out_y[(out_y>0) & (out_w>0)]
                nonzero_concs2  = out_w[(out_y>0) & (out_w>0)]
                if np.size(nonzero_concs1)>0 & np.size(nonzero_concs2)>0:
                    cor_whiff[rr, 0]   = np.corrcoef(nonzero_concs1, nonzero_concs2)[0, 1] 
    
        perc_dif[pp] = (np.mean(conc_est) - np.mean(conc_est2))/np.mean(conc_est)
        #cor_stim = cor_stim[cor_stim>=-1.0]
        #overlap_stim = overlap_stim[overlap_stim>=-1.0]
    
        interm_th = wh_mean/(wh_mean+bl_mean)
        print('Theor. Interm.: %.2g'%interm_th)
        print('')
        print('Estimated values:')
        print('stim., Mean: %.2f'%np.mean(conc_est) +', std: %.2g' %(np.std(conc_est)))
        print('stim.2, Mean: %.2f'%np.mean(conc_est2) +', std: %.2g' %(np.std(conc_est2)))
        print('percent difference: %.2g'%(perc_dif[pp]))
        print('')
        print('Interm., Mean: %.2g'%np.mean(interm_est) +', std: %.2g' %(np.std(interm_est)))
        print('Interm. 2, Mean: %.2g'%np.mean(interm_est2)+', std: %.2g' %(np.std(interm_est2)))
        print('Corr blank+whiff, Mean: %.2g' %(np.mean(cor_stim))+
                     ', std: %.2g' %(np.std(cor_stim)))
        print('Overlap, Mean: %.2g' %(np.mean(overlap_stim))+
                     ', std: %.2g' %(np.std(overlap_stim)))
        print('Corr whiff, Mean: %.2g' %(np.mean(cor_whiff))+
                     ', std: %.2g' %(np.std(cor_whiff)))
        
    #    print(perc_dif)
    #    print('Mean perc_dif: %.2f'%np.mean(perc_dif))
    
        toc = timeit.default_timer()
        print('Tot time, %d repetition: %.3fs' %(n_loop, (toc-tic)))
        print('Time single repetition: %.3fs' %((toc-tic)/n_loop))

#*********************************************************
# SAVE DATA INTO PANDAS DATAFRAME

# create a DataFrame
data2fr = np.concatenate((seeds, th_rhos, overlap_stim, cor_stim), axis = 1)

df = pd.DataFrame(data=data2fr, columns=['seeds', 'rho_ts', 'overlap_stim', 'cor_stim'])

def repeat(arr):
    return  np.concatenate((arr, arr), axis=0)

# second dataframe
seeds2 = repeat(seeds)
th_rhos2 = repeat(th_rhos)
intmt = np.concatenate((interm_est,interm_est2), axis=0)
avg_conc = np.concatenate((conc_est, conc_est2), axis=0)
glo = np.ones_like(seeds2)
glo[n_obs:] = 2

data2fr_2 = np.concatenate((seeds2, th_rhos2, glo, intmt, avg_conc), axis = 1)

df_2 = pd.DataFrame(data=data2fr_2, columns=['seeds', 'rho_ts', 'glo', 
                                             'intmt', 'avg_conc', ])


#%% *********************************************************
# FIGURE VIOLIN PLOT

# flatui = ["#9b59b6",  "#2ecc71"]
flatui = ["xkcd:purple",  "xkcd:green"]
# sns.palplot(sns.color_palette(flatui))
th_corr = [0, 0.9, 0.999, 1]#np.round(100*(1-np.exp(-np.array([0,1,3,5]))))/100

rs = 1
cs = 4

fig, axs = plt.subplots(rs, cs, figsize=(9, 3.2))

axs[0] = sns.violinplot(x="rho_ts", y="intmt", hue="glo", split=True, 
                        data=df_2,palette=flatui, ax=axs[0])
axs[1] = sns.violinplot(x="rho_ts", y="overlap_stim", data=df, palette="Blues", ax=axs[1])
axs[2] = sns.violinplot(x="rho_ts", y="cor_stim", data=df, palette="Blues", ax=axs[2])
axs[3] = sns.violinplot(x="rho_ts", y="avg_conc", hue="glo", split=True, 
                        data=df_2,palette=flatui, ax=axs[3])


# FIGURE SETTINGS
axs[0].set_title('Intermittency', fontsize=title_fs)
axs[1].set_title('Overlap ', fontsize=title_fs)
axs[2].set_title('Correlation', fontsize=title_fs)
axs[3].set_title('Avg Input', fontsize=title_fs)

axs[0].set_ylabel('\\textrm{Observed values} \n \\textrm{(unitless)}', fontsize=label_fs)
axs[0].set_xlabel('', fontsize=label_fs)

axs[1].set_ylabel('', fontsize=label_fs)
axs[1].set_xlabel('', fontsize=label_fs)

axs[2].set_xlabel('\\textrm{Theor. correlation (unitless)}', fontsize=label_fs)
axs[2].set_ylabel('', fontsize=label_fs)
axs[2].xaxis.set_label_coords(-0.1, -0.2)

axs[3].set_xlabel('', fontsize=label_fs)
axs[3].set_ylabel('', fontsize=label_fs)

label_ids = ['a','b','c','d']

for cc in range(4):
    # axs[cc].text(-.05, 1.1, label_ids[cc], transform=axs[cc].transAxes, 
    #               fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    axs[cc].set_ylim((-.02, 1.02))
    axs[cc].set_xticks(range(4))
    axs[cc].set_xticklabels(th_corr)
    if cc>0:
        axs[cc].spines['left'].set_color('none')
        axs[cc].set_yticks([])
        axs[cc].set_yticklabels('')
        
sns.despine(offset=10, trim=True)

   
axs[0].legend().set_visible(False)
axs[3].legend().set_visible(False)

for cc in range(4):
    ll, bb, ww, hh = axs[cc].get_position().bounds
    axs[cc].set_position([ll+.01, bb+.15, ww*1.2, hh*0.8])


if fig_save:
    fig.savefig(fld_output + '/plumes_stats_violins.png')

#%%
#********************************************************************
# FIGURE STIMULI TIME COURSE
    
if fig_plumes_tc:
    t2plot = np.linspace(0, t2sim, np.size(out_y))
    t2plot_lim = 0, t2sim
    t2plot_lim2 = 0, .5
    
    rs = 2
    cs = 1
    fig = plt.figure(figsize=(10, 5), )    
    
    ax_st = plt.subplot(rs,cs,1)
    ax_st.plot(t2plot, out_y, color=green, label='Stimulus 1')  
    ax_st.plot(t2plot, out_w, color=purple, label='Stimulus 2')  
    
    ax_st.set_xlim(t2plot_lim)
    ax_st.set_ylabel('Concentration', fontsize=label_fs)
    ll, bb, ww, hh = ax_st.get_position().bounds
    ax_st.set_position([ll, bb+.04, ww, hh])
    
    ax_st2 = plt.subplot(rs,cs, 2)
    ax_st2.plot(t2plot, out_y, color=green, label='Stimulus 1')  
    ax_st2.plot(t2plot, out_w, color=purple, label='Stimulus 2')  
    
    ax_st2.set_xlim(t2plot_lim2)
    ax_st2.set_xlabel('Time   (s)', fontsize=label_fs)
    ax_st2.set_ylabel('Concentration', fontsize=label_fs)
    ll, bb, ww, hh = ax_st2.get_position().bounds
    ax_st2.set_position([ll, bb+.04, ww, hh])
    if fig_save:
        fig.savefig(fld_output + 
                            '/corr_plumes_timecourse_dur%.1fs_rhoT%d_rhoC%d.png'%(t2sim, rho,100*rho_c))
        
#%%********************************************************************
# FIGURE STIMULI DISTRIBUTION OF CORRELATION, AVERAGE ETC ...
if fig_plumes_dist:
    
    n_bins  = 20
    
    rs      = 1
    cs      = 3
    fig2, (ax_int, ax_ov, ax_cor) = plt.subplots(rs, cs, figsize=(13, 5), )    

    n_tmp, _, _= ax_int.hist(interm_est, bins=n_bins, label='Stim 1', color=green, alpha=.5, density=True,)  
    n_tmp2, _, _= ax_int.hist(interm_est2, bins=n_bins, label='Stim 2', color=purple, alpha=.5, density=True,)  
    ax_int.plot([interm_th, interm_th], [0, np.max([n_tmp,n_tmp2])], '--', label='theor.', color=black,)
    
    ax_cor.hist(cor_stim, bins=n_bins, label='corr blank+whiff', color=blue, alpha=.5, density=True,)  
    
    ax_ov.hist(overlap_stim, bins=n_bins, label='overlap', color=red, alpha=.5, density=True,)  
#    ax_cor.hist(cor_whiff, bins=n_bins, label='corr whiff', color=green, alpha=.5, density=True,)  
    
    # FIGURE SETTINGS
    ax_cor.set_xlabel('Correlation', fontsize=label_fs)
    
    ax_ov.set_xlabel('Overlap', fontsize=label_fs)
    
    ax_int.set_xlabel('Intermittency', fontsize=label_fs)
    ax_int.set_ylabel('probab distr funct', fontsize=label_fs)
    ax_int.legend(fontsize=label_fs-5, frameon=False)
    
    ax_int.text(-.2, 1.2, 'a', transform=ax_int.transAxes, 
                  fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    ax_ov.text(-.2, 1.2, 'b', transform=ax_ov.transAxes, 
                  fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    ax_cor.text(-.2, 1.2, 'c', transform=ax_cor.transAxes, 
                  fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    
    ax_int.spines['top'].set_color('none')
    ax_int.spines['right'].set_color('none')
    ax_ov.spines['top'].set_color('none')
    ax_ov.spines['right'].set_color('none')
    ax_cor.spines['top'].set_color('none')
    ax_cor.spines['right'].set_color('none')

    ll, bb, ww, hh = ax_int.get_position().bounds
    ax_int.set_position([ll, bb+.05, ww, hh])
    ll, bb, ww, hh = ax_ov.get_position().bounds
    ax_ov.set_position([ll, bb+.05, ww, hh])
    ll, bb, ww, hh = ax_cor.get_position().bounds
    ax_cor.set_position([ll, bb+.05, ww, hh])
    
    
    if fig_save:
        fig2.savefig(fld_output + '/corr_plumes_distr_dur%.1fs_rhoT%d_rhoC%d.png'%(t2sim, rho,100*rho_c))

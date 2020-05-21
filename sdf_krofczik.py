#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:54:28 2020
sdf_krofczik.py

@author: mario
"""

#function sdfGlo = OO_st2sdf(tau, dt, iFile, t_max, numN)


#% sdfGlo = OO_st2sdf( tau, dt, iFile,t_max, numN)
#% This function makes an asymmetric Spike Density Function (SDF) as used by 
#% Krofczik et al. 2009 this is done with a t*exp(-t/tau) kernel shifted by 
#% tshift= centre of mass of the kernel.


import numpy as np
import matplotlib.pyplot as plt
import timeit

# *****************************************************************
# STANDARD FIGURE PARAMS
lw = 2
fs = 13
plt.rc('text', usetex=True)  # laTex in the polot
#plt.rc('font', family='serif')
fig_size = [12, 12]
fig_position = 1300,10
title_fs = 20 # font size of ticks
label_fs = 20 # font size of labels
black   = 'xkcd:black'
blue    = 'xkcd:blue'
red     = 'xkcd:red'
green   = 'xkcd:green'
purple  = 'xkcd:purple'
orange  = 'xkcd:orange'
cmap    = plt.get_cmap('rainbow')
save_fig = 1
# *****************************************************************

def poisson_isi(rate, n_spikes):
    # return a vector of n_spikes elements of inter-spike intervals
    isi = -np.log(1.0 - np.random.random_sample(n_spikes)) / rate
    return isi


def add_to_sdf(sdf, tme, krnl_bins, tau_sdf, dt_sdf, sdf_size):

    for j in range(1,krnl_bins): # to check, substitute 1 with 0
        t  = tme + j*dt_sdf - tau_sdf
        dt = j*dt_sdf
        s  = dt*np.exp(-dt/tau_sdf)
        iTime = int(t/dt_sdf)
        if ((iTime > 0) and (iTime < sdf_size)): 
            sdf[iTime] = sdf[iTime] + s
            
    return sdf 


def kernel_prep(eps_KRNL,tau_sdf, dt_sdf):

    krnl_bins = (tau_sdf/dt_sdf) # number of bins, it should be an integer
    
    sum_tmp     = 0.0
    krnl_tmp    = 0.0
    i           = 0    
    
    while ((i < krnl_bins) or (krnl_tmp > eps_KRNL)): 
        t           = i*dt_sdf
        krnl_tmp    = t*np.exp(-t/tau_sdf)
        sum_tmp     = sum_tmp + krnl_tmp
        i           = i + 1
    
    krnl_bins = i
    krnl_intg = sum_tmp*dt_sdf; # this is the integral

    return [krnl_bins, krnl_intg]

# **************************************************************

def main(spike_mat, tau_sdf = 100, dt_sdf = 20, ):

    eps_KRNL = 0.01
    
    t_max = np.max(spike_mat[:, 0])  # max time (ms)
    n_neu  = int(1+np.max(spike_mat[:, 1]))  # number of neurons
    
    sdf_size = int(t_max/dt_sdf)
    # inizialize sdf variable
    sdf_matrix = np.zeros((sdf_size, n_neu))
    
    [krnl_bins, krnl_intg] = kernel_prep(eps_KRNL,tau_sdf, dt_sdf)
    
    for idS in range(np.size(spike_mat, 0)):
        tSpike = spike_mat[idS,0]
        id_neu = int(spike_mat[idS,1])
        
        sdf_tmp = sdf_matrix[:, id_neu]
        # add to sdf the spike
        #   sdf = add_to_SDF(sdf, tme, krnl_bins, tau_sdf, dt_sdf, sdf_size)
        sdf_tmp = add_to_sdf(sdf_tmp, tSpike, krnl_bins, 
           tau_sdf, dt_sdf, sdf_size)
        
        sdf_matrix[:, id_neu] = sdf_tmp
    sdf_norm = sdf_matrix / krnl_intg
    
    time_sdf = np.linspace(0, dt_sdf*sdf_size,sdf_size)

    return sdf_norm,time_sdf

if __name__ == '__main__':
    #% Input
    #%       tau_sdf: the value of integration time of the kernel
    #%       dt_sdf:  interval of subdivision
    #%       spike_mat: matrix Nspikes x 2, [spike-time NeuId]
    #% Output
    #%       sdf_norm:
    
    # **************************************************************
    # PARAMETERS
    n_neurons   = 2
    n_spikes    = 1000
    rate        = 200 # Hz
    tau_sdf     = 20 # ms
    dt_sdf      = 5  # ms    
    stimulus    = 'poi' # 'poi' or 'det'
    
    print('tau sdf: %d ms'%tau_sdf)
    print('dt sdf: %d ms'%dt_sdf)
    print('rate : %d Hz'%rate)
        
    if stimulus == 'poi':
        # Poissonian spikes:
        isi = 1e3*poisson_isi(rate = n_neurons*rate, n_spikes=n_spikes)
    elif stimulus == 'det':
        # Deterministic spikes:
        isi = 1e3/(n_neurons*rate)*np.ones(n_spikes)

    isi_mean = np.mean(isi)
    
    t_spikes = 2000+np.cumsum(isi)

    spike_mat = np.zeros((np.size(t_spikes), 2))
    spike_mat[:, 0] = t_spikes
    spike_mat[:, 1] = np.random.randint(n_neurons, size=n_spikes)
    
    print('tot time: %.1f s'%(t_spikes[-1]/1e3))
    
    
    # ******************************************
    # SDF calculus
    tic = timeit.default_timer()
    sdf_norm, time_sdf = main(spike_mat = spike_mat, tau_sdf=tau_sdf, dt_sdf = dt_sdf)  # (Hz, ms)
    toc = timeit.default_timer()
    print('mean rate sdf: %.1f'%np.mean(sdf_norm))
    print('time to caclulate sdf: %f'%(toc-tic))
    # ******************************************    
    
    # ******************************************
    # FIGURE
    fig_sdf = 1
    if fig_sdf:
        rs = 2
        cs = 1
        
        fig, axs = plt.subplots(rs, cs, figsize=[8, 8])
        axs[0].plot(time_sdf/1e3, sdf_norm*1e3, '.', label='sdf')
    #    axs[0].plot(t_spikes/1e3, np.ones_like(t_spikes), 'r.', label='spikes')
        axs[0].plot(time_sdf/1e3, np.ones_like(time_sdf)*rate, 'r.', label='th. rate')
        axs[0].set_ylabel('sdf (Hz)', fontsize=label_fs)
        axs[0].set_xlabel('time (s)', fontsize=label_fs)
        axs[0].set_title('rate:%d Hz, tau:%d ms, dt:%d ms'%(rate,tau_sdf,dt_sdf), fontsize=title_fs)
        axs[0].legend(fontsize=label_fs-3)
        
        axs[1].hist(isi, bins=30, label='spikes')
        axs[1].set_xlabel('isi  (ms)', fontsize=label_fs)
        axs[1].set_ylabel('pdf ()', fontsize=label_fs)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 09:51:12 2021
Generate odorant stimuli of several kinds	

see also corr_plumes.py

@author: mario
"""
import numpy as np
from scipy import signal
    
import corr_plumes
    
class Plume:
    # default constructor
    def __init__(self, stim_params):
        
        self.t_tot      = stim_params['t_tot']
        self.pts_ms     = stim_params['pts_ms']
        self.conc0      = stim_params['conc0']
        


class SimPlume(Plume):
    def __init__(self, stim_params):
        super().__init__(stim_params)
        self.od_noise   = stim_params['od_noise']
        self.od_filter_frq = stim_params['od_filter_frq']
        self.concs      = stim_params['concs']
        self.t_on       = stim_params['t_on']
        self.stim_dur   = stim_params['stim_dur']
        self.t_off      = self.t_on+self.stim_dur
        
        self.n2sim      = int(self.pts_ms*self.t_tot)        
        self.n_od       = len(self.concs)
        
        # Create an order 3 lowpass butterworth filter:
        self.stim_seed       = stim_params['stim_seed']
        
        if not(np.isnan(self.stim_seed)):
            np.random.seed(seed=self.stim_seed)
            
            
        extra_t         = self.pts_ms*200                
        rand_ts         =  np.random.randn(self.n2sim+extra_t, self.n_od)*self.od_noise
        filter_ord      = 3
        b, a            = signal.butter(filter_ord, self.od_filter_frq)    
        filt_ts         = np.zeros_like(rand_ts)
        filt_ts         = signal.filtfilt(b, a, rand_ts.T).T    
    
        self.u_od       = (np.ones((self.n2sim+extra_t, self.n_od)) * 
                           self.conc0*(1 + filt_ts*np.sqrt(1/self.pts_ms)))
        
        self.u_od       = self.u_od[extra_t:, :]
        
    
class PulseTriangular(SimPlume):
    def __init__(self, stim_params):
        super().__init__(stim_params)
        
        for nn in range(self.n_od):
            t_peak          = self.t_on[nn] + self.stim_dur[nn]/2     # [ms]
            stim_peak       = int(t_peak*self.pts_ms)
            
            stim_on         = self.t_on[nn]*self.pts_ms   # [num. of samples]
            stim_off        = self.t_off[nn]*self.pts_ms 
            
            self.u_od[stim_on:stim_peak, nn] = (np.linspace(self.conc0, 
                                                self.concs[nn], stim_peak-stim_on))
            self.u_od[stim_peak:stim_off, nn] = np.linspace(self.concs[nn], 
                                                self.conc0, stim_off-stim_peak)
        
        
class PulseStep(SimPlume):
    def __init__(self, stim_params):
        super().__init__(stim_params)
        tau_on          = 50
        for nn in range(self.n_od):
            stim_on         = self.t_on[nn]*self.pts_ms   # [num. of samples]
            stim_off        = self.t_off[nn]*self.pts_ms    
            
            # stimulus onset
            t_tmp           = \
                np.linspace(0, self.t_off[nn]-self.t_on[nn], stim_off-stim_on)
            
            self.u_od[stim_on:stim_off, nn] += \
                + (self.concs[nn] - self.conc0)*(1 - np.exp(-t_tmp/tau_on)) # conc0
            
            # stimulus offset
            t_tmp           = \
                np.linspace(0, self.t_tot-self.t_off[nn], self.n2sim-stim_off)    
            
            self.u_od[stim_off:, nn]  += \
                (self.u_od[stim_off-1, nn]- self.conc0)*np.exp(-t_tmp/tau_on)
                
        self.u_od[self.u_od<0] = 0
        
        
        
class RealPlume(SimPlume):
    def __init__(self, stim_params):
        super().__init__(stim_params)
        # PARAMS FOR GENERATION OF PLUMES
        
        plume_params    = stim_params['plume_params']        
        
        quenched        = True          # if True Tbl and Twh are chosen to compensate the distance between stimuli
        t2sim_s         = (self.t_tot-self.t_on[0])/1000  # [s]
        sample_rate     = 1000*self.pts_ms   # [Hz] num of samples per each sec
        n_sample2       = 5             # [ms] num of samples with constant concentration

        #  PARAMS FOR WHIFF AND BLANK DISTRIOBUTIONS
        g               = plume_params['g']            # -1/2 for a power law of -3/2, 1 for uniform distribution
        whiff_min       = plume_params['whiff_min']    # 3e-3 [s]
        whiff_max       = plume_params['whiff_max']    # [s] 3, 50,150
        
        blank_min       = plume_params['blank_min']      # 3e-3 [s]
        blank_max       = plume_params['blank_max']       # [s]  25, 35
        
        # PARAMS FOR CONCENTRATION DISTRIBUTIONS
        # fit of average concentration at 75 m, Mylne and Mason 1991, Fig.10 
        b_conc          = -(np.log10(1-.997) + np.log10(1-.5))/10.7
        a_conc          = -0.3*b_conc - np.log10(1-.5)
        
        rho_c           = plume_params['rho_c']      # correlation between normal distribution to generate whiffs and blanks
        rho_t_exp       = plume_params['rho_t_exp']     # correlation between normal distribution to generate concentration        
        rho_t           = 1-10**-rho_t_exp
        
        # arguments for the generation of stimuli function
        corr_plumes_in = [t2sim_s, sample_rate, n_sample2, g, whiff_min, whiff_max, 
               blank_min, blank_max, a_conc, b_conc,rho_c, rho_t, quenched, self.stim_seed]
        
        # PLUME GENERATION
        out_corr_plumes = corr_plumes.main(*corr_plumes_in)
        
        for nn in range(self.n_od):
            stim_on         = self.t_on[nn]*self.pts_ms   # [num. of samples]
            self.u_od[stim_on:, nn] = out_corr_plumes[nn]*self.concs[nn] + .9*self.u_od[stim_on:, nn]
        
        
        
class ExtPlume(Plume):
    def __init__(self, stim_params):
        super().__init__(stim_params)
        stim_data_name = stim_params['stim_data_name'] 
        ex_stim = np.loadtxt(stim_data_name+'.dat')
     
        # Sims params
        self.t_tot           = ex_stim[-1,0]*1e3 # [ms] t_tot depends on data
        self.n2sim           = int(self.t_tot*self.pts_ms)+1 
        # n_ex_stim       = np.size(ex_stim, axis=0)#pts_ms*t_tot + 1      # number of time points
        stim_params['t_tot'] = self.t_tot
        
        self.u_od            = np.zeros((self.n2sim, 2))
        self.u_od[:, 0]      = .00004*ex_stim[:,1]
        self.u_od[:, 1]      = .00004*(ex_stim[0,1]+ex_stim[-1,1])/2
        
        

# stimulus
def main(stim_params, verbose=False):   
    
    verbose_dict = {
        'ss' : 'u_od is single step',
        'ts' :  'u_od is triangular pulse', 
        'pl' :  'u_od is extracted from real plumes',
        'ext' : 'ext stimuli, from Kim et al. 2011',
        'rs' : 'u_od is constant',
        }
    
    plume_dict = {
        'ss' : PulseStep,
        'ts' :  PulseTriangular, 
        'pl' :  RealPlume,
        'ext' : ExtPlume,
        'rs' : SimPlume,
        } 
    
    plume_type = stim_params['stim_type']
    plume = plume_dict[plume_type](stim_params)
    
    if verbose: 
        print(verbose_dict[plume_type])
    
    
    return plume
    
    
    
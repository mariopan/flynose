#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:04:01 2020

This script runs the simulations for the ORNs with a LIF dynamics:
We describe ORN activity in terms of an odorant transduction process combined 
with a biophysical spike generator \cite{lazar2020molecular}. During 
transduction, odorants bind and unbind at olfactory receptors according to 
simple rate equations. As we are not interested in the competition of 
different odorants for the same receptors, we simplify the customary 
two-stages binding and activation model \cite{rospars2008competitive, 
                                              nowotny2013data,chan2018odorant} 
to a single binding rate equation for the fraction r of receptors bound to an 
odorant, 

\begin{eqnarray}
\label{eq:transduction}
\dot{r} &=& b_r C^n (1-r) - d_r r 
\end{eqnarray}

We described the spike generator by a leaky integrate-and-fire neuron with 
adaptation,

\begin{eqnarray}
  \tau_{\text{V}} \dot{V} &=& (1+y)(V_{\text{rest}}^{\text{ORN}} - V) 
  + r\, (V_{\text{rev}}^{\text{ORN}} - V) \\
  \dot{y} &=& - b_y y + a_y  \sum_{t_S \in S} \delta(t-t_s) 
\end{eqnarray}

where, $V_{\text{rest}}$ is the resting potential, $V_{\text{rev}}$ the 
reversal potential of the ion channels opened due to receptor activation,  
$\zeta_x$ is a normally distributed white noise process representing receptor 
noise, and $y$ is a spike rate adaptation variable with decay time constant 
$b_y$. 


See also: 
    Treves, A (1993). Mean-field analysis of neuronal spike dynamics Network: 
        Computation in Neural Systems 4: 259-284.
    http://www.scholarpedia.org/article/Spike_frequency_adaptation

@author: mario
"""
import numpy as np
import matplotlib.pyplot as plt
import timeit
from scipy import signal

import sdf_krofczik

# %% STANDARD FIGURE PARAMS
lw = 2
fs = 13
plt.rc('text', usetex=True)  # laTex in the polot
#plt.rc('font', family='serif')
fig_size = [12, 12]
fig_position = 1300,10
title_fs = 20 # font size of ticks
label_fs = 20 # font size of labels
ticks_fs = label_fs - 3
panel_fs = 30 # font size of panel's letter
black   = 'xkcd:black'
blue    = 'xkcd:blue'
red     = 'xkcd:red'
green   = 'xkcd:green'
purple  = 'xkcd:purple'
orange  = 'xkcd:orange'
cmap    = plt.get_cmap('rainbow')
recep_clrs = ['green','purple','cyan','red']


#%% DEFINE FUNCTIONS

# tic toc
def tictoc():
    return timeit.default_timer()

# stimulus
def stim_fcn(stim_params):
    
    tmp_ks = \
        ['stim_type', 'stim_dur', 'pts_ms', 't_tot', 
         't_on', 'concs', 'conc0', 'od_noise', 'od_filter_frq']    
    [stim_type, stim_dur, pts_ms, t_tot, t_on, concs, conc0, od_noise, od_filter_frq] = [
        stim_params[x] for x in tmp_ks]  
    
    # Stimulus params    
    n_od            = len(concs)
    t_off           = t_on+stim_dur
    
    n2sim           = int(pts_ms*t_tot) + 1       # number of time points
    
    rand_ts =  np.random.randn(n2sim, n_od)*od_noise
    # Create an order 3 lowpass butterworth filter:
    filter_ord = 3
    b, a = signal.butter(filter_ord, od_filter_frq)
    
    filt_ts = np.zeros_like(rand_ts)
    filt_ts = signal.filtfilt(b, a, rand_ts.T).T    

    u_od = np.ones((n2sim, n_od)) * conc0*(1 + filt_ts)
    
    
    if stim_type == 'ext':
        stim_data_name = stim_params['stim_data_name'] 
        ex_stim = np.loadtxt(stim_data_name+'.dat')
     
        # Sims params
        t_tot           = ex_stim[-1,0]*1e3 # [ms] t_tot depends on data
        n2sim           = int(t_tot*pts_ms)+1 
        # n_ex_stim       = np.size(ex_stim, axis=0)#pts_ms*t_tot + 1      # number of time points
        stim_params['t_tot'] = t_tot
        
        u_od            = np.zeros((n2sim, 2))
        u_od[:, 0]      = .00004*ex_stim[:,1]
        u_od[:, 1]      = .00004*(ex_stim[0,1]+ex_stim[-1,1])/2
        
        
    elif (stim_type == 'rs'):
        # baseline stimuli
        print('u_od is constant')
        
    elif (stim_type == 'ss'):
        # Single Step Stimuli
        # print('u_od is single step')
        
        n2sim           = int(t_tot*pts_ms)+1 
        tau_on          = 50
        for nn in range(n_od):
            stim_on         = t_on[nn]*pts_ms   # [num. of samples]
            stim_off        = t_off[nn]*pts_ms    
            
            # stimulus onset
            t_tmp           = \
                np.linspace(0, t_off[nn]-t_on[nn], stim_off-stim_on)
            
            u_od[stim_on:stim_off, nn] += \
                + concs[nn]*(1 - np.exp(-t_tmp/tau_on)) # conc0
            
            # stimulus offset
            t_tmp           = \
                np.linspace(0, t_tot-t_off[nn], n2sim-stim_off)    
            
            u_od[stim_off:, nn]  += \
                (u_od[stim_off-1, nn]-conc0)*np.exp(-t_tmp/tau_on)
 
    u_od[u_od<0] = 0
    return u_od


# transduction function
def transd(r_0,t,u, transd_params,):

    alpha_r     = transd_params['alpha_r']
    alpha_r[alpha_r==0] = 1e-16
    beta_r      = transd_params['beta_r']
    beta_r[beta_r==0] = 1e16
    n           = transd_params['n']
    
    dt = t[1]-t[0]
    b = -alpha_r * u**n - beta_r
    a = alpha_r * u**n 
    r = (r_0 + a/b)*np.exp(b*dt)-a/b
    # drdt = alpha_r*u**n*(1-r) - beta_r*r
    return r    

# LIF 1
def v_orn_ode(v0, t, r, y, vrev, orn_params, ):
    tau_v = orn_params['tau_v']
    # vrev = orn_params['vrev']
    vrest = orn_params['vrest']
    r = r*orn_params['g_r']
    y = y*orn_params['g_y']
    
    dt = t[1]-t[0]
    b = -(1 + y + r)/tau_v
    a = (vrest*(1+y) + r*vrev)/tau_v
    v = (v0 + a/b)*np.exp(b*dt)-a/b
    #dvdt = ((1+g_y*y)*(vrest-v) + g_r*r*(vrev-v) )/tau_v
    return v

# adaptation
def y_adapt(y0, t, orn_params):
#   orn_params = np.array([b_r, d_r, n, tau_v, vrev, vrest, g, t_ref, theta, ay, tau_y])
    beta_y = orn_params['beta_y']
    
    dt = t[1] - t[0]
    y = y0 * np.exp(-dt*beta_y)
    #dydt = -y/tau_y + ay * sum(delta(t-t_spike))
    return y

# 1 Co-housed ORN
def solo_ORN(w_nsi, v_orn, nsi_vect, vrest, vrev, t, ):
    vrev_t = vrev
    return vrev_t

# 2 Co-housed ORNs
def duo_ORN(w_nsi, r_orn, nsi_vect, vrest, vrev, t, ):
    vect_a = nsi_vect[:, 1]
    vrev_t = vrev + w_nsi*r_orn[t, vect_a]*(vrest-vrev)
    return vrev_t

# 3 Co-housed ORNs
def tri_ORN(w_nsi, v_orn, nsi_vect, vrest, vrev, t, ):
    vect_a = [nsi_vect[x, 1] for x in range(0, len(nsi_vect[:, 0]), 2)]
    vect_b = [nsi_vect[x, 1] for x in range(1, len(nsi_vect[:, 0]), 2)]
     
    vrev_t = vrev + w_nsi*r_orn[t, vect_a]*(vrest-vrev) \
                    + w_nsi*r_orn[t, vect_b]*(vrest-vrev) 

    vrev_t = (w_nsi*(v_orn[t, vect_a] - vrest)+
              w_nsi*(v_orn[t, vect_b] - vrest))
    return vrev_t

# 4 Co-housed ORNs
def quad_ORN(w_nsi, v_orn, nsi_vect, vrest, vrev, t, ):
    vect_a = [nsi_vect[x, 1] for x in range(0, len(nsi_vect[:, 0]), 3)]
    vect_b = [nsi_vect[x, 1] for x in range(1, len(nsi_vect[:, 0]), 3)]
    vect_c = [nsi_vect[x, 1] for x in range(2, len(nsi_vect[:, 0]), 3)]
    
    vrev_t = vrev + w_nsi*r_orn[t, vect_a]*(vrest-vrev) \
                + w_nsi*r_orn[t, vect_b]*(vrest-vrev) \
              + w_nsi*r_orn[t, vect_c]*(vrest-vrev) 
                    
    # vrev_t = (w_nsi*(v_orn[t, vect_a] - vrest)+
               # w_nsi*(v_orn[t, vect_b] - vrest)+
              # w_nsi*(v_orn[t, vect_c] - vrest))
    return vrev_t

              

# ************************************************************************
# main function of the LIF ORN 
def main(params2an):
    
    stim_params = params2an['stim_params']
    sens_params = params2an['sens_params']
    orn_params = params2an['orn_params']
    sdf_params = params2an['sdf_params']
    
    # GENERATE ODOUR STIMULUS/I and UPDATE STIM PARAMS
    u_od            = stim_fcn(stim_params)
    
    # SDF PARAMETERS 
    tau_sdf = sdf_params['tau_sdf']
    dt_sdf  = sdf_params['dt_sdf']
    
    # STIMULI PARAMETERS 
    tmp_ks = ['pts_ms', 't_tot', 'n_od', 'r_noise', 'r_filter_frq']    
    [pts_ms, t_tot, n_od, r_noise, r_filter_frq] = [
        stim_params[x] for x in tmp_ks]    
    
    # SENSILLUM PARAMETERS
    n_neu           = sens_params['n_neu']
    w_nsi           = sens_params['w_nsi']
    n_orns_recep    = sens_params['n_orns_recep']
    
    # Connectivity matrix for ORNs
    nsi_mat = np.zeros((n_neu*n_orns_recep, n_neu*n_orns_recep))
    
    for pp in range(n_orns_recep*n_neu):
        nn = np.arange(np.mod(pp,n_orns_recep), n_neu*n_orns_recep, 
                       n_orns_recep,dtype='int')
        nsi_mat[pp, nn] = 1
    np.fill_diagonal(nsi_mat, 0)
    
    # Convert connectivity matrix to vector
    nsi_vect = np.transpose(np.asarray(np.where(nsi_mat == 1)))
    
    # Run correct ORN number
    rev_dict = {
        1 : solo_ORN,
        2 :  duo_ORN, 
        3 :  tri_ORN,
        4 : quad_ORN,
        } 
    
    # ORN PARAMETERS 
    t_ref           = orn_params['t_ref']
    theta           = orn_params['theta']
    alpha_y         = orn_params['alpha_y']
    vrest           = orn_params['vrest']
    vrev            = orn_params['vrev']
    y0              = orn_params['y0']
    r0              = orn_params['r0']
    
    # INITIALIZE OUTPUT VECTORS
    n2sim           = int(pts_ms*t_tot)   + 1   # number of time points
    t               = np.linspace(0, t_tot, n2sim) # time points
    n_neu_tot       = n_neu*n_orns_recep

    # u_od            = np.zeros((n2sim, n_od))
    
    r_orn_od        = np.zeros((n2sim, n_neu, n_od)) 
    v_orn           = np.ones((n2sim, n_neu_tot)) *vrest
    r_orn_od[0,:,:] = r0*np.ones((1, n_neu, n_od)) #+ np.random.standard_normal((1, n_neu, n_od)) 
    v_orn[0,:]      = .5*(np.ones((1, n_neu_tot)) + .01*np.random.standard_normal((1, n_neu_tot))) 
    y_orn           = np.zeros((n2sim, n_neu_tot))
    y_orn[0,:]      = y0*(np.ones((1, n_neu_tot)) +.01*np.random.standard_normal((1, n_neu_tot))) 
    
    vrev_t          = np.ones(n_neu_tot)*vrev
    num_spikes      = np.zeros((n2sim, n_neu_tot))
    orn_ref         = np.zeros(n_neu_tot)
    
    
    
    
    # Transduction for different ORNs and odours
    for tt in range(1, n2sim):
        # span for next time step
        tspan = [t[tt-1],t[tt]] 
        for id_neu in range(n_neu):
            transd_params = sens_params['transd_params'][id_neu]
            r_orn_od[tt, id_neu, :] = transd(
                r_orn_od[tt-1, id_neu, :], tspan, 
                              u_od[tt, :], transd_params)   

    # Create an order 3 lowpass butterworth filter:
    filter_ord = 3
    b, a = signal.butter(filter_ord, r_filter_frq)
    
    # Replicate to all sensilla and add noise    
    r_tmp = np.sum(r_orn_od, axis=2)
    r_orn = np.zeros((n2sim, n_neu*n_orns_recep))
    for nn in range(n_neu):
        for ss in range(n_orns_recep):
            rand_ts = r_noise*np.random.standard_normal((int(n2sim*1.3)))
            filt_ts = signal.filtfilt(b, a, rand_ts)
            filt_ts = filt_ts[-n2sim:]
            r_orn[:, ss+nn*n_orns_recep] = r_tmp[:, nn] + filt_ts
            #r_orn[:, ss+nn*n_orns_recep] = r_tmp[:, nn] * (1+ filt_ts)
    r_orn[r_orn<0] = 0
    
    # ********************************************************
    # LIF ORN DYNAMICS
    for tt in range(1, n2sim-t_ref-1):
        # span for next time step
        tspan = [t[tt-1],t[tt]]
        
        # adaptation variable
        y_orn[tt, :] = y_adapt(y_orn[tt-1, :], tspan, orn_params)
        
        # NSI effect on reversal potential 
        vrev_t = rev_dict[n_neu](w_nsi, r_orn, nsi_vect, vrest, vrev, tt-1, )
        
        # ORNs whose ref_cnt is equal to zero:
        orn_ref0 = (orn_ref==0)
        if n_neu == 1:
            v_orn[tt, orn_ref0] = v_orn_ode(v_orn[tt-1, orn_ref0], tspan, 
                                        r_orn[tt, orn_ref0], y_orn[tt, orn_ref0], 
                                        vrev_t, orn_params)
        else:
            v_orn[tt, orn_ref0] = v_orn_ode(v_orn[tt-1, orn_ref0], tspan, 
                                        r_orn[tt, orn_ref0], y_orn[tt, orn_ref0], 
                                        vrev_t[orn_ref0], orn_params)
        
        # ORNs whose Voltage is above threshold AND whose ref_cnt is equal to zero:
        orn_above_thr = (v_orn[tt, :] >= theta) & (orn_ref==0)
        num_spikes[tt, orn_above_thr] = num_spikes[tt, orn_above_thr] + 1
        orn_ref[orn_above_thr] = t_ref
        y_orn[tt:(tt+t_ref), orn_above_thr] = y_orn[tt, orn_above_thr]+alpha_y
           
        # ORNs whose ref_cnt is different from zero:
        orn_ref_no0 = (orn_ref!=0)
        # Refractory period count down
        orn_ref[orn_ref_no0] = orn_ref[orn_ref_no0] - 1 
    
    # Calculate the spike matrix 
    spike_matrix = np.asarray(np.where(num_spikes))
    spike_matrix[0,:] = spike_matrix[0,:]/pts_ms
    spike_matrix = np.transpose(spike_matrix)
    
    # SDF extraction from the spike matrix
    sdf_size    = int(stim_params['t_tot']/dt_sdf)
    t_sdf = np.linspace(0, dt_sdf*sdf_size, sdf_size)
    orn_sdf = np.zeros((sdf_size, n_neu_tot))
    
    if ~(np.sum(spike_matrix) == 0):
        orn_sdf, t_sdf = sdf_krofczik.main(spike_matrix, sdf_size,
                                                tau_sdf, dt_sdf)  # (Hz, ms)
        # orn_sdf = orn_sdf[:,0]*1e3    
        orn_sdf = orn_sdf*1e3 
    orn_lif_out = [t, u_od, r_orn, v_orn, y_orn, 
                   num_spikes, spike_matrix, orn_sdf, t_sdf,]
    
    return  orn_lif_out 


# ************************************************************
# Launching script and Figure
if __name__ == '__main__':
    print('run directly')
    
    # stimulus params
    stim_params     = dict([
                        ('stim_type' , 'ss'),   # 'rs' # 'ts'  # 'ss' # 'pl' # 'ext'
                        ('pts_ms' , 5),         # simulated pts per ms 
                        ('n_od', 2), 
                        ('t_tot', 2000),        # ms  
                        ('conc0', [1.9e-04]),    # 2.854e-04
                        ('od_noise', 5), #3.5
                        ('od_filter_frq', 0.002), #.002
                        ('r_noise', .50), #6.0
                        ('r_filter_frq', 0.002), # 0.002
                        ])
    
    
    n_od = stim_params['n_od']
    if n_od == 1:
        concs_params    = dict([
                        ('stim_dur' , np.array([500])),     # ms
                        ('t_on', np.array([300])),          # ms
                        ('concs', np.array([0.003])),
                        ])
    elif n_od == 2:
        concs_params    = dict([
                        ('stim_dur' , np.array([500, 500])),  # ms
                        ('t_on', np.array([1000, 1000])), # ms
                        ('concs', np.array([.50, .00])),
                        ])
    
    stim_params.update(concs_params)
        
    
    # Transduction parameters
    od_pref = np.array([[1,0], [0,1],]) # ORNs' sensibilities to each odours
         
    transd_vect_3A = od_pref[0,:]
    transd_vect_3B = od_pref[1,:]
    
    # TEMP: Each ORN will have its transduction properties based on DoOR
    ab3A_params = dict([
                        ('n', .822066870*transd_vect_3A), 
                        ('alpha_r', 12.6228808*transd_vect_3A), 
                        ('beta_r', 7.6758436748e-02*transd_vect_3A),
                        ])
    
    ab3B_params = dict([
                        ('n', .822066870*transd_vect_3B), 
                        ('alpha_r', 12.6228808*transd_vect_3B), 
                        ('beta_r', 7.6758436748e-02*transd_vect_3B),
                        ])
    
    # Sensilla/network parameters
    transd_params       = (ab3A_params, )#ab3B_params)
    
    n_orns_recep        = 20         # number of ORNs per each receptor
    n_neu               = transd_params.__len__()         # number of ORN cohoused in the sensillum
    
    
    
    # TEMP: Each sensillum will have its properties based on DoOR
    sens_params     = dict([
                        ('n_neu', n_neu),
                        ('n_orns_recep', n_orns_recep),
                        ('od_pref' , od_pref),
        # NSI params
                        ('w_nsi', 0.),  # 0.3
                        ('transd_params', transd_params),
                        ])
        
    # ORN Parameters 
    orn_params  = dict([
        # LIF params
                        ('t_ref', 2*stim_params['pts_ms']), # ms; refractory period 
                        ('theta', 1),                   # [mV] firing threshold
                        # fitted values
                        ('tau_v', 2.26183540),          # [ms]
                        ('vrest', -0.969461053),        # [mV] resting potential
                        ('vrev', 21),  #25wnsi.2 30wnsi.5         # 21.1784081 [mV] reversal potential
                        # ('v_k', vrest),
                        ('g_y', .5853575783),       
                        ('g_r', .864162073), 
                        # initial values of y anr r
                        ('r0', 0.15), 
                        ('y0', 1), 
        # Adaptation params
                        ('alpha_y', .45310619), 
                        ('beta_y', 3.467184e-03), 
                        ])

    
    # analysis params
    sdf_params      = dict([
                        ('tau_sdf', 41),
                        ('dt_sdf', 5),
                        ])
     # [tau_sdf, dt_sdf]
    
    params2an   = dict([
                        ('stim_params', stim_params),
                        ('sens_params', sens_params),
                        ('orn_params', orn_params),
                        ('sdf_params', sdf_params),
                        ])
    #*********************************************************************
    # ORN LIF SIMULATION
    tic = timeit.default_timer()
    orn_lif_out = main(params2an)
    toc = timeit.default_timer()
    
    print('sim run time: %.2f s' %(toc-tic))
    
    [t, u_od, r_orn, v_orn, y_orn, num_spikes, spike_matrix, orn_sdf,
     orn_sdf_time,]  = orn_lif_out
    
    
    #%% FIGURE, time course and histograom of ISI and POTENTIAL of ORNs
    
    t_on    = np.min(stim_params['t_on'])
    stim_dur = stim_params['stim_dur'][0]
    t_tot   = stim_params['t_tot']
    pts_ms  = stim_params['pts_ms']
    vrest   = orn_params['vrest']
    vrev    = orn_params['vrev']
    n_neu   = sens_params['n_neu']
    
    n_neu_tot       = n_neu*n_orns_recep
    n_isi = np.zeros((n_neu_tot,))
    rs = 2
    cs = 2
    
    fig, axs = plt.subplots(rs, cs, figsize=(7,7))    
    
    for nn1 in range(n_neu):
        isi = []
        for nn2 in range(n_orns_recep):
            nn = nn2+n_orns_recep*nn1     
            min_isi = 10
            spks_tmp = spike_matrix[spike_matrix[:,1]==nn][:,0]
            spks_tmp = spks_tmp[spks_tmp>10]
            if stim_params['stim_type'] != 'rs':
                spks_tmp = spks_tmp[spks_tmp<t_on]
            n_isi[nn] = len(spks_tmp)-1
            isi = np.append(isi, np.diff(spks_tmp))
            if np.shape(isi)[0]>0:
                min_isi = np.min((np.min(isi), min_isi))
                
            axs[0,0].plot(np.diff(spks_tmp), '.-', color=recep_clrs[nn1], alpha=.25)
        
        if len(isi)>3:
            axs[0, 1].hist(isi, bins=int(len(isi)/3), color=recep_clrs[nn1], alpha=.25, 
                    orientation='horizontal')
    
    fr_mean_rs = 1000/np.mean(isi)
    print('ORNs, FR avg no stimulus: %.2f Hz' %fr_mean_rs)
    
    fr_peak = np.max(np.mean(orn_sdf[:, :n_orns_recep], axis=1)) 
    print('ORNs, FR peak: %.2f Hz' %fr_peak)
    
    # Comparison with Poissonian hypothesis
    # t_tmp = np.linspace(0, np.max(isi),100)
    # isi_pois = fr_mean_rs*np.exp(-fr_mean_rs*t_tmp*1e-3) # poisson    
    # axs[1].plot(isi_pois, t_tmp, 'k.-')
    # SETTINGS
    axs[0, 0].set_xlabel('id spikes', fontsize=label_fs)
    axs[0, 0].set_ylabel('ISI spikes (ms)', fontsize=label_fs)
    
    dbb = 1.5
    ll, bb, ww, hh = axs[0,0].get_position().bounds
    axs[0,0].set_position([ll, bb, ww*dbb , hh])
    
    ll, bb, ww, hh = axs[0,1].get_position().bounds
    axs[0, 1].set_position(
        [ll+(dbb - 1)*ww, bb, ww*(2-dbb), hh])
    
    # V ORNs
    
    X0 = t-t_on
    trsp = .3
    if n_neu == 1:
        X1 = v_orn
        axs[1, 0].plot([t[0]-t_on, t[-1]-t_on], [vrest, vrest], 
             '--', linewidth=lw, color=black,)
        mu1 = X1.mean(axis=1)
        sigma1 = X1.std(axis=1)
        
        axs[1, 0].plot(X0, mu1, linewidth=lw+1, 
                color=recep_clrs[0], )
        for nn in range(n_orns_recep):
            axs[1, 0].plot(X0, X1[:, nn], '.', linewidth= lw-1, 
                color=recep_clrs[0], alpha=trsp)
            
        axs[1, 1].hist(X1[(t_on*pts_ms):(t_on+250)*pts_ms, nn], 
            bins=50, color=recep_clrs[0], alpha=.25, 
                    orientation='horizontal')
    
    
    else:
        for id_neu in range(n_neu):
            X1 = v_orn[:, id_neu*n_orns_recep:((id_neu+1)*n_orns_recep)]
            axs[1, 0].plot([t[0]-t_on, t[-1]-t_on], [vrest, vrest], 
                         '--', linewidth=lw, color=red,)
            mu1 = X1.mean(axis=1)
            sigma1 = X1.std(axis=1)
            
            # axs[1, 0].fill_between(X0, mu1+sigma1, mu1-sigma1, 
                        # facecolor=recep_clrs[id_neu], alpha=trsp)
            
            axs[1, 0].plot(X0, mu1,  
                linewidth=lw+1, color=recep_clrs[id_neu],)
            
            for nn in range(n_orns_recep):
                axs[1, 0].plot(X0, X1[:, nn], '.', linewidth= lw-1, 
                    color=recep_clrs[id_neu], alpha=trsp)
            
            axs[1, 1].hist(X1[(t_on*pts_ms):(t_on+250)*pts_ms, nn], bins=50, 
                    alpha=.25, color=recep_clrs[id_neu], 
                    orientation='horizontal')

    axs[1, 0].set_xlabel('time (ms)', fontsize=label_fs)
    axs[1, 0].set_ylabel('V (mV)', fontsize=label_fs)
    axs[1, 1].set_ylabel('pdf', fontsize=label_fs)
    
    dbb = 1.5
    ll, bb, ww, hh = axs[1,0].get_position().bounds
    axs[1, 0].set_position([ll, bb, ww*dbb , hh])
    
    ll, bb, ww, hh = axs[1,1].get_position().bounds
    axs[1, 1].set_position(
        [ll+(dbb - 1)*ww, bb, ww*(2-dbb), hh])

                        
    plt.show()
    
    fld_analysis = 'NSI_analysis/trials'
    hist_fig_name = '/ORN_lif_dyn_hist' + \
                            '.png'
    fig.savefig(fld_analysis + hist_fig_name)
        
    #%% correlation analysis
    tic = tictoc()
    corr_orn = np.zeros((n_neu_tot,n_neu_tot))
    corr_vorn = np.zeros((n_neu_tot,n_neu_tot))
    for nn1 in range(n_neu_tot):
        for nn2 in range(n_neu_tot):
            if nn2>nn1:
                pip1 = v_orn[::5, nn1]
                pip2 = v_orn[::5, nn2]
                corr_vorn[nn1, nn2] = np.corrcoef((pip1,pip2))[0,1]
                corr_vorn[nn2, nn1] = corr_vorn[nn1, nn2]
                
                pip1 = np.zeros(int(t_tot))
                pip2 = np.zeros(int(t_tot))
                pip1[spike_matrix[spike_matrix[:,1] == nn1, 0]] = 1
                pip2[spike_matrix[spike_matrix[:,1] == nn2, 0]] = 1
                corr_orn[nn1, nn2] = np.corrcoef((pip1,pip2))[0,1]
                corr_orn[nn2, nn1] = corr_orn[nn1, nn2]
                
    tmp_corr = corr_vorn[:n_orns_recep, :n_orns_recep]
    tmp_corr[tmp_corr!=0]
    corr_orn_hom = np.mean(tmp_corr[tmp_corr!=0])
    corr_orn_het = np.mean(corr_vorn[:n_orns_recep, n_orns_recep:]) # corr_pn[0,-1]
    print('ORNs, Hom and Het Potent corr: %.3f and %.3f' 
          %(corr_orn_hom, corr_orn_het))
    
    tmp_corr = corr_orn[:n_orns_recep, :n_orns_recep]
    tmp_corr[tmp_corr!=0]
    corr_orn_hom = np.mean(tmp_corr[tmp_corr!=0])
    corr_orn_het = np.mean(corr_orn[:n_orns_recep, n_orns_recep:]) # corr_pn[0,-1]
    print('ORNs, Hom and Het spk cnt corr: %.3f and %.3f' 
          %(corr_orn_hom, corr_orn_het))
    
    
    
    #%% FIGURE ORN dynamics
    # Create Transduction Matrix to plot odour 
    orn_fig = 1
    if orn_fig:
        # output params 
        fld_analysis = 'NSI_analysis/trials'
        orn_fig_name = '/ORN_lif_dyn' + \
                            '.png'
                            
        transd_mat = np.zeros((n_neu, n_od))
        for pp in range(n_neu):
            transd_mat[pp,:] = sens_params['od_pref'][pp,:]
        
    
        t2plot = -t_on, t_tot-t_on#np.min([1000-t_on, t_tot-t_on])
        
        panels_id = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
        
        rs = 5      # number of rows
        cs = n_neu  #  number of cols
                        
        fig_orn, ax_orn = plt.subplots(rs, cs, figsize=[8.5, 9])
        fig_orn.tight_layout()
            
        
            
        if n_neu == 1:
            weight_od = u_od*transd_mat[0,:]
            
            # PLOT
            ax_orn[0].plot(t-t_on, weight_od, linewidth=lw+1, )
            for rr in range(1, rs):
                X0 = t-t_on
                trsp = .3
                if rr == 1:
                    X1 = r_orn
                    trsp = .3            
                elif rr == 2:
                    X1 = y_orn
                elif rr == 3:
                    X1 = v_orn
                    ax_orn[3].plot([t[0]-t_on, t[-1]-t_on], [vrest, vrest], 
                                   '--', linewidth=lw, color=black,)
                    # ax_orn[3].plot([t[0]-t_on, t[-1]-t_on], [vrev, vrev], 
                    #                '-.', linewidth=lw, color=black,)
                elif rr == 4:
                    X1 = orn_sdf
                    X0 = orn_sdf_time-t_on
                mu1 = X1.mean(axis=1)
                sigma1 = X1.std(axis=1)
                
                ax_orn[rr].plot(X0, mu1,  
                              linewidth=lw+1, color=recep_clrs[0],)
                for nn in range(n_orns_recep):
                    ax_orn[rr].plot(X0, X1[:, nn], 
                                    linewidth=lw-1, color=recep_clrs[0], alpha=trsp)
            
            # SETTINGS
            # ax_orn[4].set_ylim(0, 30)
            for rr in range(rs):
                ax_orn[rr].tick_params(axis='both', which='major', labelsize=ticks_fs)
                ax_orn[rr].text(-.15, 1.25, panels_id[rr], transform=ax_orn[0].transAxes, 
                             fontsize=panel_fs, fontweight='bold', va='top', ha='right')
                ax_orn[rr].spines['right'].set_color('none')
                ax_orn[rr].spines['top'].set_color('none')
                ax_orn[rr].set_xlim((t2plot))
                
            # for rr in range(rs-1):
            #     ax_orn[rr].set_xticklabels('')
                     
            ax_orn[0].set_ylabel('Input (a.u.)', fontsize=label_fs)
            ax_orn[3].set_ylabel(r'V (a.u.)', fontsize=label_fs)
            ax_orn[4].set_ylabel('firing rates (Hz)', fontsize=label_fs)   
            ax_orn[1].set_ylabel(r'r (a.u.) ', fontsize=label_fs, )
            ax_orn[2].set_ylabel(r'y adapt (a.u.)', fontsize=label_fs, )
            ax_orn[4].set_xlabel('Time  (ms)', fontsize=label_fs) 
            
            ll, bb, ww, hh = ax_orn[0].get_position().bounds
            ww_new = ww - 0.08
            bb_plus = 0.015
            ll_new = ll + 0.075
            hh_new = hh - 0.05
            ax_orn[0].set_position([ll_new, bb+2.1*bb_plus, ww_new, hh_new])
            
            ll, bb, ww, hh = ax_orn[1].get_position().bounds
            ax_orn[1].set_position([ll_new, bb+2.0*bb_plus, ww_new, hh])
            
            ll, bb, ww, hh = ax_orn[2].get_position().bounds
            ax_orn[2].set_position([ll_new, bb+1.9*bb_plus, ww_new, hh])
            
            ll, bb, ww, hh = ax_orn[3].get_position().bounds
            ax_orn[3].set_position([ll_new, bb+1.8*bb_plus, ww_new, hh])
            
            ll, bb, ww, hh = ax_orn[4].get_position().bounds
            ax_orn[4].set_position([ll_new, bb+1.7*bb_plus, ww_new, hh])
            
            
        else:
            for id_neu in range(n_neu):
                          
                # PLOT    
                weight_od = u_od*transd_mat[id_neu,:]
                ax_orn[0, id_neu].plot(t-t_on, weight_od, linewidth=lw+1, 
                                       color=black,) 
                
                for rr in range(1, rs):
                    X0 = t-t_on
                    trsp = .1
                    if rr == 1:
                        X1 = r_orn[:, id_neu*n_orns_recep:((id_neu+1)*n_orns_recep)]
                    elif rr == 2:
                        trsp = .1
                        X1 = y_orn[:, id_neu*n_orns_recep:((id_neu+1)*n_orns_recep)]
                    elif rr == 3:
                        X1 = v_orn[:, id_neu*n_orns_recep:((id_neu+1)*n_orns_recep)]
                        ax_orn[3, id_neu].plot([t[0]-t_on, t[-1]-t_on], [vrest, vrest], 
                                       '--', linewidth=lw, color=red,)
                        # ax_orn[3, id_neu].plot([t[0]-t_on, t[-1]-t_on], [vrev, vrev], 
                        #                '-.', linewidth=lw, color=red,)
                    elif rr == 4:
                        X1 = orn_sdf[:, id_neu*n_orns_recep:((id_neu+1)*n_orns_recep)] 
                        X0 = orn_sdf_time-t_on
                    mu1 = X1.mean(axis=1)
                    sigma1 = X1.std(axis=1)
                    
                    ax_orn[rr, id_neu].fill_between(X0, mu1+sigma1, mu1-sigma1, 
                                facecolor=recep_clrs[id_neu], alpha=trsp)
                    
                    ax_orn[rr, id_neu].plot(X0, mu1,  
                                   linewidth=lw+1, color=recep_clrs[id_neu],)
                    # for nn in range(n_orns_recep):
                        # ax_orn[rr, id_neu].plot(X0, X1[:, nn], 
                                  # linewidth=lw-1, color=recep_clrs[id_neu], alpha=trsp)
                    
            
                # FIGURE SETTINGS
                for rr in range(rs):
                    ax_orn[rr, id_neu].tick_params(axis='both', which='major', labelsize=ticks_fs)
                    ax_orn[rr, id_neu].set_xlim((t2plot))      
                    ax_orn[rr, id_neu].spines['top'].set_color('none')
                    ax_orn[rr, id_neu].spines['right'].set_color('none')
                                
                ax_orn[4, id_neu].set_xlabel('Time  (ms)', fontsize=label_fs) 
            
                # ax_orn[4, id_neu].set_ylim(0, 30)
                # LABELING THE PANELS
                # ax_orn[0, id_neu].text(-.15, 1.25, panels_id[0+id_neu], 
                #                        transform=ax_orn[0, id_neu].transAxes, 
                #                   fontsize=panel_fs, fontweight='bold', va='top', ha='right')
                # ax_orn[1, id_neu].text(-.15, 1.25, panels_id[0+id_neu], transform=ax_orn[0, id_neu].transAxes, 
                #                   fontsize=panel_fs, fontweight='bold', va='top', ha='right')
                # ax_orn[2, id_neu].text(-.15, 1.25, panels_id[0+id_neu], transform=ax_orn[0, id_neu].transAxes, 
                #                   fontsize=panel_fs, fontweight='bold', va='top', ha='right')
                # ax_orn[3, id_neu].text(-.15, 1.25, panels_id[n_neu+id_neu], transform=ax_orn[3, id_neu].transAxes, 
                #                   fontsize=panel_fs, fontweight='bold', va='top', ha='right')
                # ax_orn[4, id_neu].text(-.15, 1.25, panels_id[(n_neu*2)+id_neu], transform=ax_orn[4, id_neu].transAxes, 
                #                   fontsize=panel_fs, fontweight='bold', va='top', ha='right')
                
                for rr in range(rs-1):
                    ax_orn[rr, id_neu].set_xticklabels('')
        
                if id_neu == 0:
                    ax_orn[0, id_neu].set_ylabel('Input (a.u.)', fontsize=label_fs)
                    ax_orn[1, id_neu].set_ylabel(r'r (a.u.) ', fontsize=label_fs, )
                    ax_orn[2, id_neu].set_ylabel(r'y adapt (a.u.)', fontsize=label_fs)
                    ax_orn[3, id_neu].set_ylabel(r'V (a.u.)', fontsize=label_fs)
                    ax_orn[4, id_neu].set_ylabel('firing rates (Hz)', fontsize=label_fs)        
                                             
                    ll, bb, ww, hh = ax_orn[0, id_neu].get_position().bounds
                    ww_new = ww - 0.08
                    bb_plus = 0.015
                    ll_new = ll + 0.075
                    hh_new = hh - 0.05
                    ax_orn[0, id_neu].set_position([ll_new, bb+2.1*bb_plus, ww_new, hh_new])
                    
                    ll, bb, ww, hh = ax_orn[1, id_neu].get_position().bounds
                    ax_orn[1, id_neu].set_position([ll_new, bb+2.0*bb_plus, ww_new, hh])
                    
                    ll, bb, ww, hh = ax_orn[2, id_neu].get_position().bounds
                    ax_orn[2, id_neu].set_position([ll_new, bb+1.9*bb_plus, ww_new, hh])
                    
                    ll, bb, ww, hh = ax_orn[3, id_neu].get_position().bounds
                    ax_orn[3, id_neu].set_position([ll_new, bb+1.8*bb_plus, ww_new, hh])
                    
                    ll, bb, ww, hh = ax_orn[4, id_neu].get_position().bounds
                    ax_orn[4, id_neu].set_position([ll_new, bb+1.7*bb_plus, ww_new, hh])
                    
                else:
                    ll, bb, ww, hh = ax_orn[0, id_neu].get_position().bounds
                    ww_new = ww - 0.08
                    bb_plus = 0.015
                    ll_new = ll + (0.075-(0.03*id_neu))
                    hh_new = hh - 0.05
                    ax_orn[0, id_neu].set_position([ll_new, bb+2.1*bb_plus, ww_new, hh_new])
                    
                    ll, bb, ww, hh = ax_orn[1, id_neu].get_position().bounds
                    ax_orn[1, id_neu].set_position([ll_new, bb+2.0*bb_plus, ww_new, hh])
                    
                    ll, bb, ww, hh = ax_orn[2, id_neu].get_position().bounds
                    ax_orn[2, id_neu].set_position([ll_new, bb+1.9*bb_plus, ww_new, hh])
                    
                    ll, bb, ww, hh = ax_orn[3, id_neu].get_position().bounds
                    ax_orn[3, id_neu].set_position([ll_new, bb+1.8*bb_plus, ww_new, hh])
                    
                    ll, bb, ww, hh = ax_orn[4, id_neu].get_position().bounds
                    ax_orn[4, id_neu].set_position([ll_new, bb+1.7*bb_plus, ww_new, hh])
                    
                      
        
        fig_orn.align_labels() 
        plt.show()
        fig_orn.savefig(fld_analysis + orn_fig_name)
         
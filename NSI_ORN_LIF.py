#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:04:01 2020

It simulates single sensensillum dynamics with LIF interacting neurons.

For the ORNs with a LIF dynamics:
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
import stim_fcn
import plot_orn
import set_orn_al_params
import plot_hist_isi

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
def tri_ORN(w_nsi, r_orn, nsi_vect, vrest, vrev, t, ):
    vect_a = [nsi_vect[x, 1] for x in range(0, len(nsi_vect[:, 0]), 2)]
    vect_b = [nsi_vect[x, 1] for x in range(1, len(nsi_vect[:, 0]), 2)]
     
    vrev_t = vrev + w_nsi*r_orn[t, vect_a]*(vrest-vrev) \
                    + w_nsi*r_orn[t, vect_b]*(vrest-vrev) 
    return vrev_t

# 4 Co-housed ORNs
def quad_ORN(w_nsi, r_orn, nsi_vect, vrest, vrev, t, ):
    vect_a = [nsi_vect[x, 1] for x in range(0, len(nsi_vect[:, 0]), 3)]
    vect_b = [nsi_vect[x, 1] for x in range(1, len(nsi_vect[:, 0]), 3)]
    vect_c = [nsi_vect[x, 1] for x in range(2, len(nsi_vect[:, 0]), 3)]
    
    vrev_t = vrev + w_nsi*r_orn[t, vect_a]*(vrest-vrev) \
                + w_nsi*r_orn[t, vect_b]*(vrest-vrev) \
              + w_nsi*r_orn[t, vect_c]*(vrest-vrev) 
    return vrev_t

                
# ************************************************************************
# main function of the LIF ORN 
def main(params_1sens, verbose=False):
    
    stim_params = params_1sens['stim_params']
    sens_params = params_1sens['sens_params']
    orn_params = params_1sens['orn_params']
    sdf_params = params_1sens['sdf_params']
    
    # GENERATE ODOUR STIMULUS/I and UPDATE STIM PARAMS
    u_od            = stim_fcn.main(stim_params, verbose=verbose)

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
   
    r_orn_od        = np.zeros((n2sim, n_neu, n_od)) 
    v_orn           = np.ones((n2sim, n_neu_tot)) *vrest
    y_orn           = np.zeros((n2sim, n_neu_tot))
    
    r_orn_od[0,:,:] = r0/n_od*(np.ones((1, n_neu, n_od)) +.01*np.random.standard_normal((1, n_neu, n_od))) 
    v_orn[0,:]      = vrest*(np.ones((1, n_neu_tot)) + .01*np.random.standard_normal((1, n_neu_tot))) 
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
            r_orn[:, ss+nn*n_orns_recep] = r_tmp[:, nn] + filt_ts*np.sqrt(1/pts_ms)
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
        orn_sdf_tmp, t_sdf = sdf_krofczik.main(spike_matrix, sdf_size,
                                                tau_sdf, dt_sdf)  # (Hz, ms)
        for nn in range(np.size(orn_sdf_tmp,1)):
            orn_sdf[:, nn] = orn_sdf_tmp[:, nn]*1e3    
        # orn_sdf = orn_sdf*1e3 
        
        
    orn_lif_out = [t, u_od, r_orn, v_orn, y_orn, 
                   num_spikes, spike_matrix, orn_sdf, t_sdf,]    
   
    return  orn_lif_out 

# ************************************************************
# Launching script and Figure
if __name__ == '__main__':
    print('run directly')
    
    fld_analysis = 'NSI_analysis/trials/'
    timecourse_fig_name = 'ORN_lif_timecourse.png'
    hist_fig_name = 'ORN_lif_dyn_hist.png'
    
    
    params_al_orn = set_orn_al_params.main(1)

    stim_params         = params_al_orn['stim_params']
    orn_layer_params    = params_al_orn['orn_layer_params']
    sens_params         = orn_layer_params[0]
    orn_params          = params_al_orn['orn_params']
    sdf_params          = params_al_orn['sdf_params']
    # al_params           = params_al_orn['al_params']
    # pn_ln_params        = params_al_orn['pn_ln_params']
    
    stim_params['conc0'] = 1.85e-4
    stim_params['t_tot']  = 2000
    stim_params['t_on']  = np.array([1000, 1000])
    
    params_1sens   = dict([
                        ('stim_params', stim_params),
                        ('sens_params', sens_params),
                        ('orn_params', orn_params),
                        ('sdf_params', sdf_params),
                        ])

    # ORN LIF SIMULATION
    tic = timeit.default_timer()
    output_orn = main(params_1sens)
    toc = timeit.default_timer()
    
    print('sim run time: %.2f s' %(toc-tic))
    
    [t, u_od, r_orn, v_orn, y_orn, num_spikes, spike_matrix, orn_sdf,
     orn_sdf_time,]  = output_orn
    
    fig = plot_orn.main(params_1sens, output_orn, )
    fig.savefig(fld_analysis + timecourse_fig_name)
    
    
    fig, axs = plot_hist_isi.main(params_1sens, output_orn)
    plt.show()
    
    fig.savefig(fld_analysis + hist_fig_name)
        
    
    
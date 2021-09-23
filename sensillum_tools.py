#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 10:36:50 2021

A collection of classes and function to run simulation of the sensillum

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

sensillum_tools.py

@author: mario
"""

import numpy as np
from scipy import signal

def y_adapt(y0, t, orn_params):
    """ adaptation variable dynamics"""
    #   orn_params = np.array([b_r, d_r, n, tau_v, vrev, vrest, g, t_ref, theta, ay, tau_y])
    beta_y = orn_params['beta_y']
    
    dt = t[1] - t[0]
    y = y0 * np.exp(-dt*beta_y)
    #dydt = -y/tau_y + ay * sum(delta(t-t_spike))
    return y



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
    v = (v0 + a/b)*np.exp(b*dt) - a/b
    #dvdt = ((1+g_y*y)*(vrest-v) + g_r*r*(vrev-v) )/tau_v
    return v



def transd(r_0,t,u, transd_params,):

    """transduction function with 2 states"""
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
    return [r, r]

def transd2(r01_0, t, u, transd_params,):
    
    """transduction function with 3 states"""
    [r0_0, r1_0] = r01_0
    r0 = r0_dyn(r0_0, t, u, r1_0, transd_params,)
    r1 = r1_dyn(r1_0, t, u, r0, transd_params,)
    
    rstar = 1 - r0 - r1
    return [[r0, r1], rstar]

def r0_dyn(r0_0, t, u, r1, transd_params):
    """r0 dynamics"""
    k_u     = transd_params['k_u']
    k_b     = transd_params['k_b']
    n           = transd_params['n']
    
    dt = t[1]-t[0]
    a = k_u*r1
    b = - k_b*u**n
    r0 = (r0_0 + a/b)*np.exp(b*dt)-a/b
    
    return r0


def r1_dyn(r1_0, t, u, r0, transd_params):
    """r1 dynamics"""
    k_u     = transd_params['k_u']
    k_a     = transd_params['k_a']
    k_b     = transd_params['k_b']
    k_d     = transd_params['k_d']
    n           = transd_params['n']
    
    dt = t[1]-t[0]
    a = k_b*u**n*r0 + k_d*(1-r0)
    b = -(k_u+k_d+k_a)
    r1 = (r1_0 + a/b)*np.exp(b*dt)-a/b
    
    return r1    


def transd_1sens(params_1sens, u_od, t_part):
    """  RECEPTORS DYNAMICS for single sensillum and many odours """   
    
    # STIMULI and SENSILLUM PARAMETERS 
    stim_params = params_1sens['stim_params']
    sens_params = params_1sens['sens_params']
    orn_params = params_1sens['orn_params']
    
    tmp_ks = ['pts_ms', 't_tot', 'n_od', 'r_noise', 'r_filter_frq']    
    [pts_ms, t_tot, n_od, r_noise, r_filter_frq] = [
        stim_params[x] for x in tmp_ks]    
    
    n_neu           = sens_params['n_neu']
    r0              = orn_params['r0']
    
    # INITIALIZE OUTPUT VECTORS
    # t_part      = 2000      # [ms] repetition time window 
    
    if t_tot >= t_part:
        n_rep       = int(np.ceil(t_tot / t_part))
        extra_time  = int(t_tot% t_part)
    else:
        n_rep       = 1
        extra_time  = t_tot
        
    n2sim_tot       = int(pts_ms*t_tot)    # number of time points
    
    r_orn_od        = np.zeros((n2sim_tot, n_neu, n_od)) 
    r_orn_od_last   = []
    
    tt_rep          = 0
    for id_rep in range(n_rep):
        
        if (extra_time>0) &  (id_rep == (n_rep-1)):
            n2sim = int(pts_ms*extra_time)      # number of time points
            t     = np.linspace(0, extra_time, n2sim) # time points
        else:
            n2sim = int(pts_ms*t_part)          # number of time points
            t     = np.linspace(0, t_part, n2sim) # time points
        
        
        r_orn_od_rep    = np.zeros((n2sim, n_neu, n_od)) 
        
        if id_rep == 0:
            # typical starting values 
            r_orn_od_rep[0,:,:] = r0/n_od*(np.ones((1, n_neu, n_od)) 
                                           +.01*np.random.standard_normal((1, n_neu, n_od))) 
        else:
            # starting values are the last of the previous iteration
            r_orn_od_rep[0,:,:] = r_orn_od_last
            
  
        for id_neu in range(n_neu):
            transd_params = sens_params['transd_params'][id_neu]
            if 'alpha_r' in transd_params:
                transd_fcn = transd
                r01_0 = np.zeros((1, n_od))                
            else:
                transd_fcn = transd2
                r01_0 = np.zeros((2, n_od))
                r01_0 [0,:] = np.ones((1, n_od))
                r01_0 [1,:] = 1 - r01_0 [0,:]
            for tt in range(1, n2sim):
                # span for next time step
                tspan = [t[tt-1],t[tt]] 
                [r01_0, r_orn_od_rep[tt, id_neu, :]] = transd_fcn(r01_0, tspan, 
                                  u_od[int(tt+tt_rep), :], transd_params)   
                # r_orn_od_rep[tt, id_neu, :] = transd(
                #     r_orn_od_rep[tt-1, id_neu, :], tspan, 
                #                   u_od[int(tt+tt_rep), :], transd_params)   
        # save temporary values
        # starting values are the last of the previous iteration
        r_orn_od_last = r_orn_od_rep[tt, :, :] 
        
        r_orn_od[tt_rep:(tt_rep+tt), :, :] = r_orn_od_rep[:tt, :, :]
        tt_rep  += tt
        
        
    return r_orn_od

def noise_in_transd(params_1sens, r_orn_od, t_part):
    """ Add noise to each sensilla """
    stim_params = params_1sens['stim_params']
    sens_params = params_1sens['sens_params']
    
    # STIMULI PARAMETERS 
    tmp_ks = ['pts_ms', 't_tot', 'r_noise', 'r_filter_frq']    
    [pts_ms, t_tot, r_noise, r_filter_frq] = [
        stim_params[x] for x in tmp_ks]    
    
    # SENSILLUM PARAMETERS
    n_neu           = sens_params['n_neu']
    n_orns_recep    = sens_params['n_orns_recep']
    
    n2sim_tot   = int(pts_ms*t_tot)   # number of time points
    
    if t_tot >= t_part:
        n_rep       = int(np.ceil(t_tot / t_part))
        extra_time  = int(t_tot% t_part)
    else:
        n_rep       = 1
        extra_time  = t_tot
        
    # Create an order 3 lowpass butterworth filter:
    filter_ord = 3
    b, a = signal.butter(filter_ord, r_filter_frq)
    
    r_orn           = np.zeros((n2sim_tot, n_neu*n_orns_recep))
    
    tt_rep          = 0
    for id_rep in range(n_rep):
        
        if (extra_time>0) &  (id_rep == (n_rep-1)):
            n2sim = int(pts_ms*extra_time)   # number of time points
        else:
            n2sim = int(pts_ms*t_part)   # number of time points

        r_orn_rep       = np.zeros((n2sim, n_neu*n_orns_recep))
        r_tmp           = np.sum(r_orn_od[tt_rep:(tt_rep+n2sim), :, :], axis=2)
        for nn in range(n_neu):
            for ss in range(n_orns_recep):
                rand_ts = r_noise*np.random.standard_normal((int(n2sim*1.3)))
                filt_ts = signal.filtfilt(b, a, rand_ts)
                filt_ts = filt_ts[-n2sim:]
                
                r_orn_rep[:, ss+nn*n_orns_recep] = r_tmp[:, nn] + filt_ts*np.sqrt(1/pts_ms)
        r_orn_rep[r_orn_rep<0] = 0

        r_orn[tt_rep:(tt_rep+n2sim), :] = r_orn_rep[:n2sim, :]
        tt_rep  += n2sim
    
    return r_orn




def solo_ORN(w_nsi, r_orn, nsi_vect, vrest, vrev, t, ):
    """ 1 Co-housed ORNs """
    vrev_t = vrev*np.ones_like(r_orn[0,:])
    return vrev_t

def duo_ORN(w_nsi, r_orn, nsi_vect, vrest, vrev, t, ):
    """ 2 Co-housed ORNs """
    vect_a = nsi_vect[:, 1]
    vrev_t = vrev + w_nsi*r_orn[t, vect_a]*(vrest-vrev)
    return vrev_t

def tri_ORN(w_nsi, r_orn, nsi_vect, vrest, vrev, t, ):
    """ 3 Co-housed ORNs"""
    vect_a = [nsi_vect[x, 1] for x in range(0, len(nsi_vect[:, 0]), 2)]
    vect_b = [nsi_vect[x, 1] for x in range(1, len(nsi_vect[:, 0]), 2)]
     
    vrev_t = vrev + w_nsi*r_orn[t, vect_a]*(vrest-vrev) \
                    + w_nsi*r_orn[t, vect_b]*(vrest-vrev) 
    return vrev_t

def quad_ORN(w_nsi, r_orn, nsi_vect, vrest, vrev, t, ):
    """ 4 Co-housed ORNs"""
    vect_a = [nsi_vect[x, 1] for x in range(0, len(nsi_vect[:, 0]), 3)]
    vect_b = [nsi_vect[x, 1] for x in range(1, len(nsi_vect[:, 0]), 3)]
    vect_c = [nsi_vect[x, 1] for x in range(2, len(nsi_vect[:, 0]), 3)]
    
    vrev_t = vrev + w_nsi*r_orn[t, vect_a]*(vrest-vrev) \
                + w_nsi*r_orn[t, vect_b]*(vrest-vrev) \
              + w_nsi*r_orn[t, vect_c]*(vrest-vrev) 
    return vrev_t


# Vreversal depends on the number of co-housed neurons
vrev_dict = {
    1 : solo_ORN,
    2 :  duo_ORN, 
    3 :  tri_ORN,
    4 : quad_ORN,
    } 

class SensillumORNS:
    
    # default constructor
    def __init__(self, params_1sens, n2sim):
    
        self.params = params_1sens
        sens_params = self.params['sens_params']
        orn_params = self.params['orn_params']
    
        self.n_neu           = sens_params['n_neu']
        n_orns_recep    = sens_params['n_orns_recep']
        
        # ORN PARAMETERS 
        self.w_nsi           = sens_params['w_nsi']
        self.t_ref           = orn_params['t_ref']
        self.theta           = orn_params['theta']
        self.alpha_y         = orn_params['alpha_y']
        self.vrest           = orn_params['vrest']
        self.vrev            = orn_params['vrev']
        self.y0              = orn_params['y0']
        
        # STIMULI PARAMETERS 
        self.n_neu_tot   = self.n_neu*n_orns_recep
        
        # Connectivity matrix for ORNs
        nsi_mat = np.zeros((self.n_neu_tot, self.n_neu_tot))
        
        for pp in range(self.n_neu_tot):
            nn = np.arange(np.mod(pp,n_orns_recep), self.n_neu_tot, 
                           n_orns_recep,dtype='int')
            nsi_mat[pp, nn] = 1
        np.fill_diagonal(nsi_mat, 0)
        
        # Convert connectivity matrix to vector
        self.nsi_vect = np.transpose(np.asarray(np.where(nsi_mat == 1)))
        
        # INITIALIZE OUTPUT VECTORS
        self.v_orn           = np.ones((n2sim, self.n_neu_tot)) *self.vrest
        self.y_orn           = np.zeros((n2sim, self.n_neu_tot))
        
        self.vrev_t          = np.ones(self.n_neu_tot)*self.vrev
        self.spikes_orn      = np.zeros((n2sim, self.n_neu_tot)) 

        # initialize the values for y and v at t0
        self.v_orn[0,:]      = self.vrest*(np.ones((1, self.n_neu_tot)) + .01*np.random.standard_normal((1, self.n_neu_tot))) 
        self.y_orn[0,:]      = self.y0*(np.ones((1, self.n_neu_tot)) +.01*np.random.standard_normal((1, self.n_neu_tot))) 
        
        self.orn_ref         = np.zeros(self.n_neu_tot)
        
    def update_t0values(self, y_orn_last, v_orn_last):
        # Initialize output values at intermediate repetitions 
        self.v_orn           = np.ones_like(self.v_orn) *self.vrest
        self.y_orn           = np.zeros_like(self.y_orn)
        
        self.vrev_t          = np.ones_like(self.vrev_t)*self.vrev
        self.spikes_orn      = np.zeros_like(self.spikes_orn)
        
        # starting values are the last of the previous iteration
        self.v_orn[0,:]      = v_orn_last
        self.y_orn[0,:]      = y_orn_last
        
    def run_1step(self, r_orn_rep, tt, t_rep):
        # Simulate a single step of the ORNs of the sensillum
        
        tspan = [t_rep[tt-1], t_rep[tt]]
        
        orn_params = self.params['orn_params']
        
        # update adaptation variable
        self.y_orn[tt, :] = y_adapt(self.y_orn[tt-1, :], tspan, orn_params)
        
        # NSI effect on reversal potential 
        self.vrev_t = vrev_dict[self.n_neu](self.w_nsi, r_orn_rep, self.nsi_vect, self.vrest, self.vrev, tt-1, )
        
        # ORNs whose ref_cnt is equal to zero:
        orn_ref0 = (self.orn_ref==0)
        self.v_orn[tt, orn_ref0] = v_orn_ode(self.v_orn[tt-1, orn_ref0], tspan, 
                                    r_orn_rep[tt, orn_ref0], self.y_orn[tt, orn_ref0], 
                                    self.vrev_t[orn_ref0], orn_params)
        
        # ORNs whose Voltage is above threshold AND whose ref_cnt is equal to zero:
        orn_above_thr = (self.v_orn[tt, :] >= self.theta) & (self.orn_ref==0)
        self.spikes_orn[tt, orn_above_thr] = self.spikes_orn[tt, orn_above_thr] + 1
        self.orn_ref[orn_above_thr] = self.t_ref
        self.y_orn[tt:(tt+self.t_ref), orn_above_thr] = self.y_orn[tt, orn_above_thr]+self.alpha_y
           
        # ORNs whose ref_cnt is different from zero:
        orn_ref_no0 = (self.orn_ref!=0)
        # Refractory period count down
        self.orn_ref[orn_ref_no0] = self.orn_ref[orn_ref_no0] - 1 
            
            
    def last_values(self, tt):
        # return the last values of y and v 
        v_orn_last = self.v_orn[tt, :]
        y_orn_last = self.y_orn[tt, :]        
        return [v_orn_last,y_orn_last] 
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:49:50 2020

@author: mario
"""

import numpy as np
from scipy.integrate import quad

def rnd_pow_law(a, b, g, r):
    """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b"""
    ag, bg = a**g, b**g
    return (ag + (bg - ag)*r)**(1./g)

def pdf_pow_law(x, a, b, g):
    ag, bg = a**g, b**g
    return g * x**(g-1) / (bg - ag)

def cdf_pow_law(x, a, b, g):
    ag, bg = a**g, b**g
    return (x**g - ag) / (bg - ag)

def whiffs_blanks_pdf(dur_min, dur_max,g):
    logbins  = np.logspace(np.log10(dur_min),np.log10(dur_max))
    pdf_th = pdf_pow_law(logbins, dur_min, dur_max, g)      # theoretical values of the pdf
#    tot_prob_blank = quad(lambda x: pdf_pow_law(x, blank_min, blank_max, g), blank_min, blank_max) # integrate the pdf to check it sum to 1
    dur_mean = quad(lambda x: x*pdf_pow_law(x, dur_min, dur_max, g), dur_min, dur_max) # integrate the pdf to check it sum to 1
    return pdf_th, logbins, dur_mean[0]

def whiffs_blanks_cdf(dur_min, dur_max, g):
    logbins  = np.logspace(np.log10(dur_min),np.log10(dur_max))
    cdf_th = cdf_pow_law(logbins, dur_min, dur_max, g)      # theoretical values of the pdf
    return cdf_th, logbins 

# concentration values drawn from fit of Mylne and Mason 1991 fit of fig.10 
def rnd_mylne_75m(a1, b1, r):
    """Mylne and Mason 1991 fit of fig.11 """
    y = ((1-np.heaviside(r-.5, 0.5)) * .3*r/.5 + 
         np.heaviside(r-.5, 0.5)* (-(a1 + np.log10(1-r))/b1))
    return y

def pdf_mylne_75m(x, a1, b1):
    y = ((1-np.heaviside(x-.3, 0.5)) * .5/.3 + 
         np.heaviside(x-.3, 0.5)* (b1*np.log(10)*10**(-a1-b1*x)))
    return y

def cdf_mylne_75m(x, a1, b1):
    y = ((1-np.heaviside(x-.3, 0.5)) * .5*x/.3 + 
         np.heaviside(x-.3, 0.5)* (1-10**(-(a1+b1*x))))
    return y

def overlap(a,b):
    a = (a>0)*1.0
    b = (b>0)*1.0
    return np.sum(a*b)*2.0/(np.sum(a)+np.sum(b))


# ******************************************************************* 
def main():
    # dummy function to make a method
    y = 20
    return y


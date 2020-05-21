# flynose
The goal of this project is to develop a model of the early olfactory system of *Drosophila* using Python as programming language. 
For now the model consists of a subset of the early olfactory system of insects from ORNs to the AL using only two groups of ORNs (ORN$_a$ and ORN$_{b}$) and their respective PNs and LNs. Each ORN type, a and b, is tuned to a specific set of odorants (e.g. individual pheromone component) and converges onto its corresponding PNs. PNs impinge into their respective LNs, but receive inhibitory input from LNs of the other type.

|      |      |      |
| ---- | ---- | ---- |
|      |      |      |
|      |      |      |
|      |      |      |



The core script is **flynose.py**. 

There are three script to lauch the core flynose: **flynose_examples.py**, **batch_flynose_ratio.py**, **batch_flynose_real_plumes.py**.



**flynose.py** receives as input a set of parameters:

> params2an = [nsi_str, alpha_ln, stim_dur, delays2an, peak, 
>                  peak_ratio, rho, stim_type,w_max,b_max, spont_dur]

flynose_example.py is a collection of 7 different examples of how to use flynose.py (6 of them were used in the publication *Pannunzi et al.*: 

*  Real plumes, example figure
  * Run **flynose** for a single run with a real plume 
* Fig.ImpulseResponse
  * Run a single triangular stimulation (100ms) for the three different networks (*indep*,*LN-inhib*, *NSI*)
* Olsen-Wilson 2010 figure
  * Run multiple times a step stimulation (constant stimuli lasting 500ms) for the three different networks (*indep*,*LN-inhib*, *NSI*)
* Lazar and Kim data reproduction
* FIG. ORN_response
* (Trials and errors)










# flynose

The goal of this project is to develop a model of the early olfactory system of *Drosophila* using Python as programming language. 
For now the model consists of a subset of the early olfactory system of insects from ORNs to the AL using only two groups of ORNs (ORN$_a$ and ORN$_{b}$) and their respective PNs and LNs. Each ORN type, a and b, is though to be tuned to a specific set of odorants (e.g. individual pheromone component) and converges onto its corresponding PNs. PNs impinge into their respective LNs, but receive inhibitory input from LNs of the other type.

The core script is **flynose.py**. 

## Structure

**flynose.py** consists of three separated parts:

1. stimulus generation: four kinds of stimuli can be generated (each have different id): triangular (*'ts'*), step (*'ss'*), real plumes (*'pl'*) and loaded from an external file (undefined)
2. ORNs simulation
3. AL simulation

### Syntax

>  [t, u_d, orn_sp_mat, pn_sp_mat, ln_sp_mat] = flynose(params2an, flig_opts)

### Parameters

​     *params2an* is a list containing the main parameters that are used for the most common simulations: the strength of the NSI, *nsi_str* and the strength of the AL lateral inhibition, *alpha_ln*. The third element is another list with the parameters for the stimulus: *stim_params*.

   *fig_opts* is a list containing several flags to decide from outside **flynose.py** whether for exanple if we want the figure of the ORNs dynamics, *orn_fig*, or to save the figure, *fig_save*, or to simulate the AL, *al_dyn*.

### Return

> [t, u_d, orn_sp_mat, pn_sp_mat, ln_sp_mat] : list of five ndarray

The output is a single variable that consists of 5 vectors:

* t: 1d np array, the time series of the time used for the simulation (ms)
* u_od: 2d np array of the odorant concentration input to the ORNs
* orn_sp_mat: 2d np array of the spikes trains of the ORNs. The first element is spike time and the second element is the ORN id
* pn_sp_mat:2d np array of the spikes trains of the PNs. The first element is spike time and the second element is the PN id
* ln_sp_mat: 2d np array of the spikes trains of the LNs. The first element is       spike time and the second element is the LN id

### Stimuli

There are four different  types of stimuli: 

1. single step - constant concentration to both ORNs, with different onsets and offsets, different peaks
2. triangle - linear increase and decrease, with different onsets and offsets, different peaks
3. real plumes - two correlated realistic plumes, with a given distribution of whiffs, clean air and concentration
4. external - input given from an external file

There are three script to lauch the core flynose: **flynose_examples.py**, **batch_flynose_ratio.py**, **batch_flynose_real_plumes.py**. 

To analyse the output of the batch files, use **analysis_flynose_ratio.py** and **analysis_flynose_real_plumes.py**.

**plot_flynose.py** plots part of the output of **flynose_examples.py**

### Complementary functions

**plot_flynose.py** produces plots for

1. the simulations with external stimuli (*Lazar et al. 2019*). It loads data from the folder *lazar_sim* that are produced with **flynose_example.py**.
2. the simulations showing the Intensity invariance of the ORNs response for several concentration values (*Martelli et al. 2013*). It loads data from the folder *Olsen2010_Martelli2013/data* that were produced with  **flynose_example.py**.
3. the simulations showing the sigmoidal curve of the relation between ORNs responses and PNs responses (Olsen and Wilson 2010). It loads data from the folder *Olsen2010_Martelli2013/data* that were produced with  **flynose_example.py**.

**batch_flynose_ratio.py** 

**flynose_example.py** is a collection of different examples of how to use flynose.py (6 of them were used in the publication *Pannunzi et al.*: 

* ORN_response
  * run **flynose** a single time the 3 networks (*indep*, *LN-inhib*, *NSI*) with step stimulus for 500ms and make the figure for ORN_response
* Olsen-Wilson 2010
  * Run **flynose** multiple times with a step stimulation (constant stimuli lasting 500ms) for the three different networks (*indep*,*LN-inhib*, *NSI*)
* Lazar and Kim data reproduction
  * Run **flynose** with an external input. The input are extracted from the fig.4 of Lazar et al. (2019).
* Real plumes, example figure
  * Run **flynose** for a single run with a real plume 
* ImpulseResponse
  * Run **flynose** with two triangular syncrhonous stimuli (50ms durations), for the three different networks 
* DelayResponse
  * Run **flynose** with two triangular syncrhonous stimuli (50ms durations, 100ms delay), for the three different networks 
* (Trials and errors)


# flynose

In this project we develop a model of the early olfactory system of *Drosophila*. 

The model consists of a subset of the early olfactory system of insects from ORNs to the AL using only two groups of ORNs (ORN_a and ORN_b) and their respective PNs and LNs. Each ORN type, *a* and *b*, is though to be tuned to a specific set of odorants (e.g. individual pheromone component) and converges onto its corresponding PNs. 

<img src="images/Topology_NSI.png" title="Model topology of the early olfactory areas of a *Drosophila*" style="zoom:60%;" />  

Each pair of ORNs is co-housed in a same sensillum and therefore they interact via Non-synaptic mechanism (see figure below). PNs impinge into their respective LNs, but receive inhibitory input from LNs of the other type.

<img src="images/NSI_figure.png" title="Model of the Non-synaptic interaction between ORNs" style="zoom:60%;" />





## Program structure

Python is the present programming language. It consists of three separated parts:

1. stimulus generation: four kinds of stimuli can be generated (each have different id): triangular (*'ts'*), step (*'ss'*), real plumes (*'pl'*) and loaded from an external file (undefined). The function managing this part is **stim_fcn.py**. To generate the *real plumes* we used the scripts **stats_for_plumes.py** and **corr_plumes.py**.
2. ORNs simulation. The function managing this part is **NSI_ORN_LIF.py** for a single sensillum and **ORNs_layer_dyn.py** for the dynamics of all the ORN's types. NSI_ORN_LIF receives the input from stim_fcn, and it is repeated in ORNs_layer_dyn for a number of times equal to the implemented number of receptor types. Both functions return a spike matrix and spike density function (calculated with **sdf_krofczik.py**) of the ORNs.
3. Antennal lobe (AL) simulation. The function managing this part is **AL_dyn.py**. It receives the output from both NSI_ORN_LIF or ORNs_layer_dyn as they are homogeneous.  It returns spike matrix and spike density function (calculated with sdf_krofczik) of the PNs and LNs.



### Parameters

All the parameters of the model are collected into the dictionary **params_al_orn**, that is formed by several dictionaries.



### Stimuli

There are four different  types of stimuli: 

1. single step - ss - constant concentration to both ORNs, with different onsets and offsets, different peaks
2. triangle - ts - linear increase and decrease, with different onsets and offsets, different peaks
3. real plumes - pl - two correlated realistic plumes, with a given distribution of whiffs, clean air and concentration. To simulate the correlated naturalistic plumes use corr_plumes.
4. external - ext - input given from an external file. For example, we used the data observed by Kim, Lazar and colleagues and saved in the folder *lazar_data_hr*.



### Analysis and visualization

There are several scripts to launch the core **flynose.py**. For example, to explore the parameters using different kind of stimuli: **flynose_examples.py**, **batch_ratio.py**, **batch_delays.py**, **batch_real_plumes.py**.

To analyse the output of the batch files, use **analysis_ratio_delays.py** and **analysis_real_plumes.py**.



**plot_flynose.py** produces plots for:

1. the simulations with external stimuli (*Lazar et al. 2019*). It loads data from the folder *NSI_analysis/lazar_sim* that are produced with **flynose_example.py**.
2. the simulations showing the Intensity invariance of the ORNs response for several concentration values (*Martelli et al. 2013*). It loads data from the folder *Olsen2010_Martelli2013/data* that were produced with  **flynose_example.py**.
3. the simulations showing the sigmoidal curve of the relation between ORNs responses and PNs responses (Olsen and Wilson 2010). It loads data from the folder *Olsen2010_Martelli2013/data* that were produced with  **flynose_example.py**.



**plot_plumes_stats.py** helps visualize the temporal and statistical properties of the naturalistic plumes reproducing the data furnished in literature (i.e. *Yee et al. 1995, Murlis et al. 1991*).



**plot_corr_plumes.py** helps visualize the temporal and statistical properties of the simulated naturalistic plumes. It runs launching multiple times the function **corr_plumes.py**.



**plot_diag_shades.py** plots the network response  for multiple runs for three different models (*NSI-model, LN-model and ctrl-model*) to three kind of stimuli:

* Real plumes (fig_is='pl') runs **flynose.py** with a real plume 
* Impulse response (fig_is='pl') runs **flynose.py** with two triangular syncrhonous stimuli (50ms durations)
* Delayed impulse response (fig_is='pl') runs **flynose.py** with two triangular syncrhonous stimuli (50ms durations, 100ms delay)



**flynose_example.py** is a collection of different examples of how to use flynose.py (6 of them were used in the publication *Pannunzi et al.*: 

* ORN_response
  * run **flynose.py** a single time the 3 networks (*indep*, *LN-inhib*, *NSI*) with step stimulus for 500ms and make the figure for ORN_response
* Olsen-Wilson 2010
  * Run **flynose.py** multiple times with a step stimulation (constant stimuli lasting 500ms) for the three different networks (*indep*,*LN-inhib*, *NSI*)
* Lazar and Kim data reproduction
  * Run **flynose.py** with an external input. The input are extracted from the fig.4 of Lazar et al. (2019).
* Real plumes, example figure
  * Run **flynose.py** for a single run with a real plume 
* ImpulseResponse
  * Run **flynose** with two triangular syncrhonous stimuli (50ms durations), for the three different networks 
* DelayResponse
  * Run **flynose** with two triangular syncrhonous stimuli (50ms durations, 100ms delay), for the three different networks 
* (Trials and errors)



#### ... in the same folder

**NSI_analysis.zip** contains figures and data of the results of Pannunzi and Nowotny (*writing*).

**Olsen2010_Martelli2013.zip** contains figures and data to show the model reproduction of the results from *Olsen et al. 2010* and *Martelli et al. 2013*.

**plot_hyp1.py** and **NSI_corr_uncorr.py** to plot the first and the second hypotheses of the possible role of the NSI in the insects' olfaction illustrated in Pannunzi and Nowotny (*writing*).



#### ... for the paper

see the resuming table to generate the figures 'figures.md'



### Related publication: 

[Olsen, S. R., Bhandawat, V., & Wilson, R. I. (2010). Divisive Normalization in Olfactory Population Codes. In *Neuron* (Vol. 66, Issue 2, pp. 287–299).](http://dx.doi.org/10.1016/j.neuron.2010.04.009)

[Pannunzi, M., & Nowotny, T. (2019). Odor Stimuli: Not Just Chemical Identity. *Frontiers in Physiology*, *10*, 1428](https://www.frontiersin.org/articles/10.3389/fphys.2019.01428/full)

[Yee, E., Wilson, D. J., & Zelt, B. W. (1993). Probability distributions of concentration fluctuations of a weakly diffusive passive plume in a turbulent boundary layer. In Boundary-Layer Meteorology (Vol. 64, Issue 4, pp. 321–354)](https://link.springer.com/article/10.1007/BF00708930)
[Su, C.-Y., Menuz, K., Reisert, J., & Carlson, J. R. (2012). Non-synaptic inhibition between grouped neurons in an olfactory circuit. Nature](https://pubmed.ncbi.nlm.nih.gov/23172146/)
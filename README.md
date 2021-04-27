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

There are several scripts to analyse the model. For example, to explore the parameters using different kind of stimuli: **ORN_dyn_plot.py**, **AL_dyn_batch.py**, **batch_ratio2.py**, and **batch_real_plumes2.py**. To analyse the output of the batch files, use **analysis_ratio_delays2.py** and **analysis_real_plumes2.py**.



**ORN_dyn_plot.py** produces plots for ORN layer (ORNs_layer_dyn.py) multiple times with a single pulse for several peak values (a la Martelli et al. 2013) and with different shapes (step, ramp and parabola a la Kim et al. 2011). It can even loads old data and plots them.

**AL_dyn_batch.py** runs AL and ORNs layer dynamics for multiple values of peaks and inhibitory conditions. Then it loads and plots dynamics to do a plot like Olsen et al. 2010.

**figure_orn.py** plots the dynamics of a single simulation from the output of NSI_ORN_LIF.py  

**plot_al_orn.py** plots the dynamics of a single simulation from the output of **ORNs_layer_dyn.py** and **AL_dyn.py**.

**plot_plumes_stats.py** helps visualize the temporal and statistical properties of the naturalistic plumes reproducing the data furnished in literature (i.e. *Yee et al. 1995, Murlis et al. 1991*). This script compares experimental probability distribution as observed by Mylne et al. 1991, with a truncated power law, for whiff and blank durations. Moreover, it plots the intermittency values obtained by Yee et al. 1993 for several downwind distances going from 20 to 330 m.

**plot_corr_plumes.py** helps visualize the temporal and statistical properties of the simulated naturalistic plumes. It runs launching multiple times the function **corr_plumes.py**. It runs statistics and makes the plot of the output of corr_plumes.py. The plot shows in a violin plot the distribution of the intermittency, the correlation, the overlap and the average value of the two output.

**flynose_examples.py** plots the network response  for multiple runs for three different models (*NSI-model, LN-model and ctrl-model*) to three kind of stimuli:

* Real plumes (fig_is='ts_s') runs flynose for the three inhibitory conditions with a real plume 
* Impulse response (fig_is='ts_a') runs flynose for the three inhibitory conditions with two triangular asynchronous stimuli (50ms duration)
* Delayed impulse response (fig_is='pl') runs flynose for the three inhibitory conditions with two triangular synchronous stimuli (50ms durations, 100ms delay)

**plot_hyp1.py**  and **NSI_corr_uncorr.py**  plot the first and the second hypotheses of the possible role of the NSI in the insects' olfaction illustrated in Pannunzi and Nowotny (*writing*).



#### ... in the same folder

**flynose0_backup_10112020.zip** contains the source code of the first version of the model described in Pannunzi and Nowotny (*bioRxiv* 2020).

**NSI_analysis.zip** contains figures and data of the results of the first version of the model described in Pannunzi and Nowotny (*bioRxiv* 2020).

**Olsen2010_Martelli2013.zip** contains figures and data to show the model reproduction of the results from *Olsen et al. 2010* and *Martelli et al. 2013* of the first version of the model described in Pannunzi and Nowotny (*bioRxiv* 2020).



### Related publication: 

[Olsen, S. R., Bhandawat, V., & Wilson, R. I. (2010). Divisive Normalization in Olfactory Population Codes. In *Neuron* (Vol. 66, Issue 2, pp. 287–299).](http://dx.doi.org/10.1016/j.neuron.2010.04.009)

[Pannunzi, M., & Nowotny, T. (2019). Odor Stimuli: Not Just Chemical Identity. *Frontiers in Physiology*, *10*, 1428](https://www.frontiersin.org/articles/10.3389/fphys.2019.01428/full)

[Yee, E., Wilson, D. J., & Zelt, B. W. (1993). Probability distributions of concentration fluctuations of a weakly diffusive passive plume in a turbulent boundary layer. In Boundary-Layer Meteorology (Vol. 64, Issue 4, pp. 321–354)](https://link.springer.com/article/10.1007/BF00708930)
[Su, C.-Y., Menuz, K., Reisert, J., & Carlson, J. R. (2012). Non-synaptic inhibition between grouped neurons in an olfactory circuit. Nature](https://pubmed.ncbi.nlm.nih.gov/23172146/)
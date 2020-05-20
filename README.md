# flynose
The goal of this project is to develop a model of the early olfactory system of *Drosophila* using Python as programming language. 
For now the model consists of a subset of the early olfactory system of insects from ORNs to the AL using only two groups of ORNs (ORN$_a$ and ORN$_{b}$) and their respective PNs and LNs. Each ORN type, a and b, is tuned to a specific set of odorants (e.g. individual pheromone component) and converges onto its corresponding PNs. PNs impinge into their respective LNs, but receive inhibitory input from LNs of the other type.

<img src="/home/mario/MEGA/WORK/OdorObjects/Docs/images/NSI_model_topology_antennae_ink.png" alt="NSI_model_topology_antennae_ink" style="zoom:50%;" />

The core script is **flynose.py**. 

This script is called by **flynose_example.py**, **batch_flynose_ratio.py**, **batch_flynose_real_plumes.py**.

flynose_example.py is a collection of examples of how to use flynose.py.






# flynose
The goal of this project is to develop a model of the early olfactory system of *Drosophila* using Python as programming language. 
For now the model consists of a subset of the early olfactory system of insects from ORNs to the AL using only two groups of ORNs (ORN$_{a}$ and ORN$_{b}$) and their respective PNs and LNs. Each ORN type, a and b, is tuned to a specific set of odorants (e.g. individual pheromone component) and converges onto its corresponding PNs. PNs impinge into their respective LNs, but receive inhibitory input from LNs of the other type.

The core script is flynose.py

Nested Graph Neural Networks
============================

About
-----
Nested Graph Neural Network (NGNN) is a general framework to improve a base GNN's expressive power and performance. It consists of a base GNN (usually a weak message-passing GNN) and an outer GNN. In NGNN, we extract a rooted subgraph around each node, and let the base GNN to learn a subgraph representation from the rooted subgraph, which is used as the root node's representation. Then, the outer GNN further learns a graph representation from these root node representations returned from the base GNN (in this paper, we simply let the outer GNN be a global pooling layer without graph convolution). NGNN is proved to be more powerful than 1-WL, being able to discriminate almost all r-regular graphs where 1-WL always fails. In contrast to other high-order GNNs, NGNN only incurs a constant time higher time complexity than its base GNN (given the rooted subgraph size is bounded). NGNN often shows immediate performance gains in real-world datasets when applying it to a weak base GNN.

Requirements
------------
Stable: Python 3.8 + PyTorch 1.8.1 + PyTorch\_Geometric 1.7.0 + OGB 1.3.1
Latest: Python 3.8 + PyTorch 1.9.0 + PyTorch\_Geometric 1.7.2 + OGB 1.3.1 (GINE+ and its nested version not trainable with this configuration)

Install [PyTorch](https://pytorch.org/)

Install [PyTorch\_Geometric](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html)

Install [OGB](https://ogb.stanford.edu/docs/home/)

Install rdkit by 

    conda install -c conda-forge rdkit

To run 1-GNN, 1-2-GNN, 1-3-GNN, 1-2-3-GNN and their nested versions on QM9, install k-gnn by executing

    python setup.py install

under "software/k-gnn-master/".

Other required python libraries include: numpy, scipy, tqdm etc.

Usages
------

### TU dataset

To run Nested GCN on MUTAG (with subgraph height=3 and base GCN #layers=4), type:

    python run_tu.py --model NestedGCN --h 3 --layers 4 --node_label spd --use_rd --data MUTAG

To compare with a base GCN model only, type:

    python run_tu.py --model GCN --layers 4 --data MUTAG

Use "--data all" and "--model all" to run all models (NestedGCN, NestedGraphSAGE, NestedGIN, NestedGAT) on all datasets.

### QM9

We include the commands for reproducing the QM9 experiments in "run_all_targets_qm9.sh". Uncomment the corresponding command in this file, and then run

    ./run_all_targets_qm9.sh 0 11

to execute this command repeatedly for all 12 targets.

### OGB molecular datasets

To reproduce the ogb-molhiv experiment, run

    python run_ogb_mol.py --h 4 --num_layer 6 --save_appendix _h4_l6_spd_rd --dataset ogbg-molhiv --node_label spd --use_rd --drop_ratio 0.65 --runs 10 

When finished, to get the ensemble test result, run

    python run_ogb_mol.py --h 4 --num_layer 6 --save_appendix _h4_l6_spd_rd --dataset ogbg-molhiv --node_label spd --use_rd --drop_ratio 0.65 --runs 10 --continue_from 100 --ensemble

To reproduce the ogb-molpcba experiment, run

    python run_ogb_mol.py --h 3 --num_layer 4 --save_appendix _h3_l4_spd_rd --dataset ogbg-molpcba --subgraph_pooling center --node_label spd --use_rd --drop_ratio 0.35 --epochs 150 --runs 10

When finished, to get the ensemble test result, run
    
    python run_ogb_mol.py --h 3 --num_layer 4 --save_appendix _h3_l4_spd_rd --dataset ogbg-molpcba --subgraph_pooling center --node_label spd --use_rd --drop_ratio 0.35 --epochs 150 --runs 10 --continue_from 150 --ensemble --ensemble_lookback 140

### Simulation on r-regular graphs

To reproduce Appendix C Figure 3, run the following commands:
    
    python run_simulation.py --n 10 20 40 80 160 320 640 1280 --save_appendix _node --N 10 --h 10

    python run_simulation.py --n 10 20 40 80 160 320 640 1280 --save_appendix _graph --N 100 --h 10 --graph


Miscellaneous
-------------

We have tried our best to clean the code. We will keep polishing it after the author response. If you encounter any errors or bugs, please let us know in OpenReview. Hope you enjoy NGNN and the code!

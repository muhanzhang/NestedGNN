#! /bin/bash

# Run 10 times on OGB mol datasets. Usage:
# ./run_ten_times.sh DATANAME
# Replace DATANAME with flixster, douban or yahoo_music.
# After running, type python summarize_fdy.py to summarize the results.


data=${1}
for i in $(seq 1 5)  # to run with different seeds
do
  #python run_ogb_mol.py --subgraph --h 4 --num_layer 6 --save_appendix _hoplabel_h4_l6_mean_mean_s${i} --seed ${i} --use_hop_label --dataset ${data}
  #python run_ogb_mol.py --subgraph --h 4 --num_layer 6 --save_appendix _hoplabel_h4_l6_cpv_center_mean_s${i} --center_pool_virtual --subgraph_pooling center --seed ${i} --use_hop_label --dataset ${data}
  #python run_ogb_mol.py --subgraph --h 3 --num_layer 4 --save_appendix _hoplabel_h3_l4_cpv_center_mean_s${i} --center_pool_virtual --subgraph_pooling center --seed ${i} --use_hop_label --dataset ${data}
  #python run_ogb_mol.py --subgraph --h 3 --num_layer 4 --save_appendix _rd_h3_l4_cpv_center_mean_s${i} --center_pool_virtual --subgraph_pooling center --seed ${i} --use_resistance_distance --dataset ${data}
  #python run_ogb_mol.py --subgraph --h 4 --num_layer 6 --save_appendix _rd_h4_l6_cpv_center_mean_s${i} --center_pool_virtual --subgraph_pooling center --seed ${i} --use_resistance_distance --dataset ${data}
  python run_ogb_mol.py --subgraph --h 4 --num_layer 6 --save_appendix _hoplabel_jt_h4_l6_mean_mean_s${i} --seed 1 --use_hop_label --use_junction_tree --dataset ${data}
done

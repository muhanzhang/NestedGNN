#! /bin/bash

# Run 10 times on OGB mol datasets. Usage:
# ./run_ten_times.sh DATANAME
# Replace DATANAME with flixster, douban or yahoo_music.
# After running, type python summarize_fdy.py to summarize the results.


#data=${1}
#for data in ogbg-molbace ogbg-molbbbp ogbg-molclintox ogbg-molsider ogbg-moltox21 ogbg-moltoxcast ogbg-molesol ogbg-molfreesolv ogbg-mollipo  # smaller datasets
for data in ogbg-molmuv ogbg-moltox21 ogbg-moltoxcast  # smaller datasets
do
  python run_ogb_mol.py --h 4 --num_layer 6 --save_appendix _h4_l6_spd_rd_d065 --dataset ${data} --node_label spd --drop_ratio 0.65 --runs 5 --use_rd
  #python run_ogb_mol.py --subgraph --h 5 --num_layer 7 --save_appendix _rd_h5_l7_mean_mean_s${i} --seed ${i} --use_resistance_distance --dataset ${data}
  #python run_ogb_mol.py --subgraph --h 1 --num_layer 2 --save_appendix _rd_h1_l2_mean_mean_s${i} --seed ${i} --use_resistance_distance --dataset ${data}
  #python run_ogb_mol.py --subgraph --h 4 --num_layer 6 --save_appendix _rd_h4_l6_mean_mean_s${i} --seed ${i} --use_resistance_distance --dataset ${data}
  #python run_ogb_mol.py --subgraph --h 3 --num_layer 5 --save_appendix _rd_h3_l5_mean_mean_s${i} --seed ${i} --use_resistance_distance --dataset ${data}
  #python run_ogb_mol.py --subgraph --h 2 --num_layer 4 --save_appendix _rd_h2_l4_mean_mean_s${i} --seed ${i} --use_resistance_distance --dataset ${data}
  #python run_ogb_mol.py --save_appendix _h0_l4_s${i} --num_layer 4 --seed ${i} --dataset ${data}
  #python run_ogb_mol.py --save_appendix _h0_s${i} --seed ${i} --dataset ${data}
  #python run_ogb_mol.py --save_appendix _h0_scheduler_s${i} --scheduler --seed ${i} --dataset ${data}
  #python run_ogb_mol.py --subgraph --h 4 --num_layer 6 --save_appendix _hoplabel_h4_l6_mean_mean_s${i} --use_hop_label --seed ${i} --dataset ${data}
  #python run_ogb_mol.py --subgraph --h 4 --num_layer 6 --save_appendix _hoplabel_h4_l6_mean_mean_scheduler_s${i} --scheduler --use_hop_label --seed ${i} --dataset ${data}
  #python run_ogb_mol.py --subgraph --h 5 --num_layer 7 --save_appendix _hoplabel_h5_l7_mean_mean_scheduler_s${i} --scheduler --use_hop_label --seed ${i} --dataset ${data}
  #python run_ogb_mol.py --subgraph --h 3 --num_layer 5 --save_appendix _hoplabel_h3_l5_mean_mean_scheduler_s${i} --scheduler --use_hop_label --seed ${i} --dataset ${data}
  #python run_ogb_mol.py --subgraph --h 2 --num_layer 4 --save_appendix _hoplabel_h2_l4_mean_mean_scheduler_s${i} --scheduler --use_hop_label --seed ${i} --dataset ${data}

  #python run_ogb_mol.py --subgraph --multiple_h '2,3,4' --num_more_layer 2  --save_appendix _hoplabel_mh234_ml2_mean_mean_s${i} --use_hop_label --seed ${i} --dataset ${data}

  #python run_ogb_mol.py --subgraph --h 4 --num_layer 6 --save_appendix _hoplabel_h4_l6_mean_mean_scheduler_s${i} --use_hop_label --scheduler --seed ${i} --dataset ${data}
  #python run_ogb_mol.py --subgraph --h 5 --num_layer 7 --save_appendix _hoplabel_h5_l7_mean_mean_scheduler_s${i} --use_hop_label --scheduler --seed ${i} --dataset ${data}
  #python run_ogb_mol.py --subgraph --h 3 --num_layer 5 --save_appendix _hoplabel_h3_l5_mean_mean_scheduler_s${i} --use_hop_label --scheduler --seed ${i} --dataset ${data}
  #python run_ogb_mol.py --subgraph --multiple_h '3,4,5' --num_layer 2 --save_appendix _hoplabel_mh345_ml2_mean_mean_scheduler_s${i} --use_hop_label --scheduler --seed ${i} --dataset ${data}
done

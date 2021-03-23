#! /bin/bash

# Run all targets on QM9. Usage:
# ./run_all_targets_qm9.sh 2 10
# run qm9 from targets 2 to 10


T0=${1}
T1=${2}
for target in $(seq ${T0} ${T1})
do
  '''
  python run_qm9.py --h 3 --model k1_GNN_sub --save_appendix _k1_h3_spd_rd --use_rd --target ${target}
  python run_qm9.py --h 3 --model k12_GNN_sub --save_appendix _k12_h3_spd_rd --use_rd --target ${target}
  python run_qm9.py --h 3 --model k13_GNN_sub --save_appendix _k13_h3_spd_rd --use_rd --target ${target}
  python run_qm9.py --h 3 --model k123_GNN_sub --save_appendix _k123_h3_spd_rd --use_rd --target ${target}

  python run_qm9.py --h 3 --model k1_GNN_sub --save_appendix _k1_h3_no --node_label no --target ${target}
  python run_qm9.py --h 3 --model k12_GNN_sub --save_appendix _k12_h3_no --node_label no --target ${target}
  python run_qm9.py --h 3 --model k13_GNN_sub --save_appendix _k13_h3_no --node_label no --target ${target}
  python run_qm9.py --h 3 --model k123_GNN_sub --save_appendix _k123_h3_no --node_label no --target ${target}
  '''
  
  python run_qm9.py --model k1_GNN --save_appendix _k1_RNI --RNI --target ${target}
  #python run_qm9.py --h 3 --model k1_GNN_sub --save_appendix _k1_h3_spd_rd_RNI --RNI --use_rd --target ${target}
done

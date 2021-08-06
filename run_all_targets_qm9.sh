#! /bin/bash

# Run all targets on QM9. Usage:
#   ./run_all_targets_qm9.sh 2 10
# to run qm9 from targets 2 to 10
# When you finish all the runs, type
#   for i in `seq 0 11`; do tail -1 QM9_${i}_k1_h3_spd_rd/log.txt; done
# for example to summarize the results


T0=${1}
T1=${2}
for target in $(seq ${T0} ${T1})
do
  # The following 4 commands reproduce the NGNN results in Table 4.
  python run_qm9.py --h 3 --model Nested_k1_GNN --save_appendix _k1_h3_spd_rd --use_rd --target ${target}
  #python run_qm9.py --h 3 --model Nested_k12_GNN --save_appendix _k12_h3_spd_rd --use_rd --target ${target}
  #python run_qm9.py --h 3 --model Nested_k13_GNN --save_appendix _k13_h3_spd_rd --use_rd --target ${target}
  #python run_qm9.py --h 3 --model Nested_k123_GNN --save_appendix _k123_h3_spd_rd --use_rd --target ${target}

  # The following 4 commands reproduce the NGNN (no DE features) in Table 5 of Appendix E.
  #python run_qm9.py --h 3 --model Nested_k1_GNN --save_appendix _k1_h3_no --node_label no --target ${target}
  #python run_qm9.py --h 3 --model Nested_k12_GNN --save_appendix _k12_h3_no --node_label no --target ${target}
  #python run_qm9.py --h 3 --model Nested_k13_GNN --save_appendix _k13_h3_no --node_label no --target ${target}
  #python run_qm9.py --h 3 --model Nested_k123_GNN --save_appendix _k123_h3_no --node_label no --target ${target}
done

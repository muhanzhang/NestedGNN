#! /bin/bash

# Run 10 times on OGB mol datasets. Usage:
# ./run_ten_times.sh DATANAME
# Replace DATANAME with flixster, douban or yahoo_music.
# After running, type python summarize_fdy.py to summarize the results.


for data in ogbg-molbace ogbg-molbbbp ogbg-molclintox ogbg-molsider ogbg-moltox21 ogbg-moltoxcast ogbg-molesol ogbg-molfreesolv ogbg-mollipo  # smaller datasets
# for data in ogbg-molhiv ogbg-molpcba ogbg-molmuv  # larger datasets
do
  ./run_ten_times.sh ${data}
done


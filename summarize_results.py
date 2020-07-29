from __future__ import print_function
import os
import re
import numpy as np
import torch
import pdb
from scipy import stats


seed_range = range(1, 11)


datasets = [
'ogbg-molbace', 
'ogbg-molbbbp', 
'ogbg-molclintox', 
'ogbg-molsider', 
'ogbg-moltox21', 
'ogbg-moltoxcast', 
'ogbg-molesol', 
'ogbg-molfreesolv', 
'ogbg-mollipo'
]
'ogbg-molmuv', 
prefixs = [
'_h0_s', 
'_h0_l2_s', 
'_h0_l3_s', 
'_h0_l4_s', 
#'_h0_scheduler_s', 
#'_hoplabel_h4_l6_mean_mean_s', 
#'_hoplabel_h4_l6_mean_mean_scheduler_s', 
#'_hoplabel_mh234_ml2_mean_mean_s'
'_rd_h1_l2_mean_mean_s', 
'_rd_h2_l4_mean_mean_s', 
'_rd_h3_l5_mean_mean_s', 
'_rd_h4_l6_mean_mean_s',
'_rd_h5_l7_mean_mean_s', 
#'_hoplabel_h5_l7_mean_mean_scheduler_s', 
#'_hoplabel_h3_l5_mean_mean_scheduler_s', 
#'_hoplabel_h2_l4_mean_mean_scheduler_s', 
]


'''
datasets = [
    'ogbg-molclintox', 
    'ogbg-molsider', 
    'ogbg-moltoxcast', 
]
prefixs = [
    '_h0_s', 
    '_hoplabel_h4_l6_mean_mean_scheduler_s',
    '_hoplabel_h5_l7_mean_mean_scheduler_s', 
    '_hoplabel_h3_l5_mean_mean_scheduler_s', 
]

datasets = [
    'ogbg-molpcba', 
]
prefixs = [
    '_h0_s', 
    '_hoplabel_h4_l6_mean_mean_s', 
]

'''


print()
for dataset in datasets:
    print('Results of ' + dataset)
    for prefix in prefixs:
        res_base = 'results/' + dataset + prefix
        train, val, test = [], [], []
        for seed in seed_range:
            res_dir = res_base + str(seed)
            res = torch.load(res_dir)
            train.append(res['Train'])
            val.append(res['Val'])
            test.append(res['Test'])
        train = np.array(train)
        val = np.array(val)
        test = np.array(test)
        print('\033[91m Results of ' + prefix + '\033[00m')
        print(test)
        print('Mean and std of train, val, test:')
        print('%.4f$\pm$%.4f'%(np.around(np.mean(train), 4), np.around(np.std(train), 4)))
        print('%.4f$\pm$%.4f'%(np.around(np.mean(val), 4), np.around(np.std(val), 4)))
        print('%.4f$\pm$%.4f'%(np.around(np.mean(test), 4), np.around(np.std(test), 4)))
    print()


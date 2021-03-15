"""
training configures
"""
version = '5.9' # 需要测试的模型版本号
num_iter = 60000 # 模型的迭代次数，加载时根据该次数进行加载对应模型

import importlib
conf = importlib.import_module('configures.transformer.train_tfm_v{}'.format(version.replace('.', '_'))).conf

debug = False

solver_conf = conf['solver_conf']
solver_conf['solver_name'] = 'tfm_chip_test_v{}'.format(version.replace('.', '_'))
solver_conf['gpu_ids'] = '0,1,2,3,4,5,6,7' # set '-1' to disable gpu
solver_conf['metric_names'] = ['acc_ctc', 'acc_reg', 'acc', 'acc_ord', 'acc_map', 'acc_tfm']
solver_conf['checkpoints_dir'] = './checkpoints/' + solver_conf['solver_name']
solver_conf['log_dir'] = './checkpoints/' + solver_conf['solver_name'] + '/logs'
solver_conf['load_state'] = True
solver_conf['solver_state_path'] = './checkpoints/tfm_chip_v{}/network_{:0>6d}.pkl'.format(version, num_iter)
solver_conf['phase'] = 'test'
solver_conf['use_dist'] = False
solver_conf['load_epoch'] = False
solver_conf['save_freq'] = 40000000
solver_conf['print_freq'] = 1 if debug else 100

solver_conf['prob_conf'] = {}
# solver_conf['prob_conf']['foreign_method'] = 'openmax'
solver_conf['prob_conf']['foreign_method'] = 'normal'
solver_conf['prob_conf']['weibull_feats_type'] = 'all_chips'

solver_conf['foreign_prob_type'] = 'normal'


loader_conf = conf['loader_conf']
loader_conf['dataset_name'] = 'chip'
loader_conf['dataset_dir'] = 's3://tianmaoqing.data/chips'

# loader_conf['file_path_str'] = 'inputs/list_v2/foreign_list/noflip_faban_20200707+modeldealrealdata+foreign12+SZ+100K_{}.txt'
# loader_conf['file_name_list'] = ['']

loader_conf['file_path_str'] = 'inputs/list_v3/foreign_list/test/unknow/{}.txt'
loader_conf['file_name_list'] = ['mix+foreign_unknow_gt',
                                 'real+foreign_unknow_gt', 'unknow_1_gt',
                                 'unknow_2_gt', 'unknow_3_gt', 'unknow_4_gt'][0:]

# 如果需在正常筹码测试集上测试精度，使用下面的路径即可
# loader_conf['file_path_str'] = 'inputs/list_v3/stand_chip/test/{}.txt'
# loader_conf['file_name_list'] = ['real_test_sl_usetype', 'mix_sl_test']

loader_conf['solver_name'] = solver_conf['solver_name']
loader_conf['batch_size'] = 32
loader_conf['use_augment'] = False
loader_conf['num_workers'] = 16
loader_conf['tfm_label_type'] = 'individual'

## Network Options
net_conf = conf['net_conf']
net_conf['temperature'] = 1
# net_conf['transformer']['linear_type'] = 'cosine_face'

conf = {'net_conf':net_conf, 'solver_conf':solver_conf, 'loader_conf':loader_conf}

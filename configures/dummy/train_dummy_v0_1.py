"""
training configures
"""
debug = False

solver_conf = {}
solver_conf['gpu_ids'] = '0' # set '-1' to disable gpu
solver_conf['solver_name'] = 'dummy_v0.1'
solver_conf['solver_path'] = 'models.dummy_model'
solver_conf['net_path'] = 'models.dummy_model'
solver_conf['metric_names'] = ['acc']
solver_conf['checkpoints_dir'] = './checkpoints/' + solver_conf['solver_name']
solver_conf['log_dir'] = './checkpoints/' + solver_conf['solver_name'] + '/logs'
solver_conf['load_state'] = False
solver_conf['solver_state_path'] = './checkpoints/dummy_net/network_008000.pkl'
solver_conf['phase'] = 'train'
solver_conf['use_dist'] = False
solver_conf['max_iter'] = 80000
solver_conf['max_epoch'] = 100
solver_conf['load_epoch'] = True
solver_conf['phase'] = 'train'
solver_conf['save_freq'] = 4000
solver_conf['print_freq'] = 1 if debug else 400
solver_conf['record_fail'] = True

solver_conf['optimizer_name'] = 'Adam'
solver_conf['lr_conf'] = {}
solver_conf['lr_conf']['decay_type'] = 'LUT'
solver_conf['lr_conf']['warm_up'] = [0.00001, 0.0001, 5000]
solver_conf['lr_conf']['decay_steps'] = [5000, 20000, 40000]
solver_conf['lr_conf']['decay_base'] = 0.1
solver_conf['lr_conf']['init_lr'] = solver_conf['lr_conf']['warm_up'][1]


loader_conf = {}
loader_conf['solver_name'] = solver_conf['solver_name']
loader_conf['batch_size'] = 2 if solver_conf['use_dist'] else 8
loader_conf['use_augment'] = True
loader_conf['resize_type'] = 'no_pad'
loader_conf['num_classes'] = 10 # (the extra 2 is for transformer's starting and ending symbol)
loader_conf['num_workers'] = 8
loader_conf['use_ceph'] = True
loader_conf['dataset_name'] = 'chip'
loader_conf['dataset_dir'] = 's3://tianmaoqing.data/chips'
loader_conf['use_validate'] = False
loader_conf['file_path'] = 'inputs/list_v3/stand_chip/train/faban_20200707+modeldealrealdata.txt'
loader_conf['img_H'] = 384
loader_conf['img_W'] = 128
loader_conf['hw_ratio'] = 2.0
loader_conf['random_ratio'] = 0.5
loader_conf['label_len'] = 25
loader_conf['label_pad'] = 0 # use which number to pad the label
loader_conf['num_used_data'] = -1
loader_conf['color'] = True


## Network Options
net_conf = {}

net_conf['enc_net'] = {}
net_conf['enc_net']['net_name'] = 'unet_resnet_50'
net_conf['enc_net']['encoder_depth'] = 5
net_conf['enc_net']['decoder_depth'] = 3
net_conf['enc_net']['num_in_channels'] = 3
net_conf['enc_net']['num_out_channels'] = 256
net_conf['enc_net']['decoder_channels'] = [1024, 512, 256]
net_conf['enc_net']['use_y_avg_pool'] = False
net_conf['enc_net']['use_x_avg_pool'] = False

channels = [256, 128]

net_conf['loss_weight'] = [1]
net_conf['num_classes'] = loader_conf['num_classes']

conf = {'net_conf':net_conf, 'solver_conf':solver_conf, 'loader_conf':loader_conf}

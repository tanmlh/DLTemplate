def get_net_conf(num_classes, label_len):

    net_conf = {}

    net_conf['enc_net'] = {}
    net_conf['enc_net']['net_name'] = 'pspnet_resnet_50'
    net_conf['enc_net']['depth'] = 4
    net_conf['enc_net']['num_in_channels'] = 3
    net_conf['enc_net']['num_out_channels'] = 128
    net_conf['enc_net']['upsampling'] = 16

    temp_channels = [128, 64]

    net_conf['seg_net'] = {}
    net_conf['seg_net']['down_sample'] = True
    net_conf['seg_net']['type'] = 'conv'
    # net_conf['seg_net']['channels'] = [temp_channels[0], temp_channels[1], num_classes]
    net_conf['seg_net']['channels'] = [temp_channels[0], num_classes]
    # net_conf['seg_net']['freeze'] = True

    net_conf['loc_net'] = {}
    # net_conf['loc_net']['channels'] = [temp_channels[0], temp_channels[1], 1]
    net_conf['loc_net']['channels'] = [temp_channels[0], 1]
    # net_conf['loc_net']['freeze'] = True

    net_conf['ord_net'] = {}
    net_conf['ord_net']['down_sample'] = True
    # net_conf['ord_net']['channels'] = [temp_channels[0], temp_channels[1], label_len]
    net_conf['ord_net']['channels'] = [temp_channels[0], label_len]
    net_conf['ord_net']['type'] = 'conv'
    # net_conf['ord_net']['freeze'] = True

    net_conf['len_net'] = {}
    # net_conf['len_net']['channels'] = [temp_channels[0], temp_channels[1]]
    net_conf['len_net']['channels'] = [temp_channels[0]]

    return net_conf

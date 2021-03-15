def get_net_conf(num_classes, label_len):

    net_conf = {}

    net_conf['enc_net'] = {}
    net_conf['enc_net']['net_name'] = 'resnet_50_4x'
    net_conf['enc_net']['num_out_channels'] = 2048

    temp_channels = [512, 256]

    net_conf['seg_net'] = {}
    net_conf['seg_net']['down_sample'] = True
    net_conf['seg_net']['type'] = 'conv'
    net_conf['seg_net']['channels'] = [temp_channels[0], temp_channels[1], num_classes]
    # net_conf['seg_net']['freeze'] = True

    net_conf['loc_net'] = {}
    net_conf['loc_net']['channels'] = [temp_channels[0], temp_channels[1], 1]
    # net_conf['loc_net']['freeze'] = True

    net_conf['ord_net'] = {}
    net_conf['ord_net']['down_sample'] = True
    net_conf['ord_net']['channels'] = [temp_channels[0], temp_channels[1], label_len]
    net_conf['ord_net']['type'] = 'conv'
    # net_conf['ord_net']['freeze'] = True

    net_conf['len_net'] = {}
    net_conf['len_net']['channels'] = [temp_channels[0], temp_channels[1]]

    net_conf['attention'] = {}
    net_conf['attention']['dim'] = 1024
    net_conf['attention']['num_heads'] = 8
    net_conf['attention']['num_hiddens'] = 512

    return net_conf

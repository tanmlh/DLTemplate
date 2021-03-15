import pdb
import os
import sys
sys.path.append(os.getcwd())
import importlib
import argparse
import torch
from torch.utils.data import DataLoader
# import warnings
# warnings.filterwarnings('ignore')

from datasets.dummy_dataset import DummyDataset
from models import base_solver
# import utils.distributed as dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, default='configures.xxx')
    parser.add_argument('--state_path', type=str, default=None)

    args = parser.parse_args()

    if args.state_path is not None:
        solver = base_solver.get_solver_from_solver_state(args.state_path)
        solver_conf = solver.conf['solver_conf']
        loader_conf = solver.conf['loader_conf']

    else:
        ## Load configurations ##
        conf = importlib.import_module(args.conf_path).conf
        solver_conf = conf['solver_conf']
        loader_conf = conf['loader_conf']
        solver_path = solver_conf['solver_path']
        solver = importlib.import_module(solver_path).get_solver(conf)
        if solver_conf['load_state']:
            solver.load_solver_state(torch.load(solver_conf['solver_state_path']))

    batch_size = loader_conf['batch_size']
    num_workers = loader_conf['num_workers']

    if solver_conf['phase'] == 'train':

        max_iter = solver_conf['max_iter']

        train_dataset = DummyDataset(loader_conf, phase='train')
        train_sampler = None

        if solver_conf['use_dist']:
            train_sampler = dist.DistributedGivenIterationSampler(train_dataset,
                                                                  max_iter,
                                                                  batch_size,
                                                                  last_iter=solver.global_step-1)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True if train_sampler is None else False,
                                  num_workers=num_workers, pin_memory=False,
                                  sampler=train_sampler)

        solver.train(train_loader, None)

    elif solver_conf['phase'] == 'test':

        for file_name in loader_conf['file_name_list']:

            loader_conf['file_path'] = loader_conf['file_path_str'].format(file_name)
            test_dataset = ChipDataset(loader_conf, phase='test')
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                     num_workers=num_workers, pin_memory=False)
            state = solver.test(test_loader)

    elif solver_conf['phase'] == 'val':

        test_dataset = ChipDataset(loader_conf, phase='test')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=False)
        state = solver.validate(test_loader)

        print(state)





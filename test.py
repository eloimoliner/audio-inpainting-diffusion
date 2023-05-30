import os
import re
import json
import hydra
#import click
import torch
import utils.dnnlib as dnnlib
from utils.torch_utils import distributed as dist
import utils.setup as setup

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.


def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------


def _main(args):


    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    dirname = os.path.dirname(__file__)
    args.model_dir = os.path.join(dirname, str(args.model_dir))
    if dist.get_rank() == 0:
        print(args.model_dir)
        if not os.path.exists(args.model_dir):
            raise Exception(f"Model directory {args.model_dir} does not exist")

    args.exp.model_dir=args.model_dir

    torch.multiprocessing.set_start_method('spawn')

    diff_params=setup.setup_diff_parameters(args)
    network=setup.setup_network(args, device)

    test_set=setup.setup_dataset_test(args)

    tester=setup.setup_tester(args, network=network, diff_params=diff_params, test_set=test_set, device=device) #this will be used for making demos during training
    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0()
    dist.print0(f'Output directory:        {args.model_dir}')
    dist.print0(f'Network architecture:    {args.network.callable}')
    dist.print0(f'Diffusion parameterization:  {args.diff_params.callable}')
    dist.print0(f'Tester:                  {args.tester.callable}')
    dist.print0(f'Experiment:                  {args.exp.exp_name}')
    dist.print0()

    # Train.
    print("loading checkpoint path:", args.tester.checkpoint)
    if args.tester.checkpoint != 'None':

        tester.load_checkpoint(os.path.join(args.model_dir,args.tester.checkpoint))
    else:
        print("trying to load latest checkpoint")
        tester.load_latest_checkpoint()
    
    tester.dodajob()


@hydra.main(config_path="conf", config_name="conf")
def main(args):
    _main(args)

if __name__ == "__main__":
    main()


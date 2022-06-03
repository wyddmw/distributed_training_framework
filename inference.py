from random import sample
from model import get_model
from model.model_utils import get_loss
from dataset.kitti_dataset import KittiDataset
from dataset.data_list import TEST_INFO_PATH, TRAIN_INFO_PATH
import argparse
from utils.train_utils import *
import datetime
import tqdm
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, dataloader


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--output_dir', type=str, default='./output', help='output saving path')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=2, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--output_dir', type=str, default='./output', help='output saving path')
    parser.add_argument('--batch_size_per_gpu', type=int, default=8, help='batch size for distributed training')
    parser.add_argument('--model_tag', type=str, default='point2_rot_dir', help='inidicate the training model name')
    parser.add_argument('--device_ids', type=str, default='0', help='indicates CUDA devices, e.g. 0,1,2')
    parser.add_argument('--pretrained_model', type=str, default='None', help='pretrained model')
    opt = parser.parse_args()
    return opt

def main():
    opt = parse_config()
    if opt.launcher == 'slurm':
        total_gpus, opt.local_rank = init_dist_slurm(opt.tcp_port, opt.local_rank, backend='nccl')

    opt.output_dir = opt.output_dir + '/' + opt.model_tag
    ckpt_dir = opt.output_dir + '/ckpt'
    if not os.path.exists(ckpt_dir):
        os.system('mkdir -p %s' % (ckpt_dir))

    log_file = opt.output_dir + '/log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logger = create_logger(log_file, rank=opt.local_rank)

    logger.info('***************Start Initializing model*****************')
    # initialize model
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    for key, val in vars(opt).items():
        logger.info('{:16} {}'.format(key, val))

    # Initialize model
    model = get_model(opt.model_name).cuda()

    start_epoch = 0
    # TODO: Implement continue training
    logger.info('Loading pretrained model from %s' % (opt.pretrained_model))
    state_params = torch.load(opt.pretrained_model, map_location=torch.device('cpu'))
    model_state = state_params['model_state']
    model.load_state_dict(model_state)
    logger.info('succeed in loading model')
    logger.info(model)
    # launch trainer
    model.eval()
    # Data distributed training
    if opt.launcher == 'slurm':
        logger.info('total batch size: %d ' % (total_gpus * opt.batch_size_per_gpu))
        model = nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank % torch.cuda.device_count()])

    # Dataparallel training
    elif len(opt.device_ids) > 1:
        logger.info('Dataparallel training')
        device_ids = [int(item) for item in opt.device_ids.split(',')]
        model = nn.parallel.DataParallel(model, device_ids=device_ids)

    # loading dataset
    train_set = KittiDataset(info_path=TRAIN_INFO_PATH, logger=logger, opt=opt, training=True)
    test_set = KittiDataset(info_path=TEST_INFO_PATH, logger=logger, opt=opt)
    logger.info('testing file length is %d' % (len(test_set)))

    if opt.launcher == 'slurm':
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=opt.batch_size_per_gpu, shuffle=False, drop_last=False, \
                                    pin_memory=True, num_workers=opt.workers, sampler=test_sampler)
    else:
        test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, \
                                    pin_memory=True)

    logger.info('********Start Evaluating***********')
    pbar = tqdm.tqdm(total=len(test_loader), leave=False, desc='eval')
    for i, data in enumerate(test_loader):
        # needs specific modification
        input_data = data['input_data'].cuda()
        pred = model(input_data)

        if opt.local_rank == 0:
            pbar.update()

    logger.info('Eval info')

if __name__ == '__main__':
    main()

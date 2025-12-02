import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import data
from experiment import Experiment
from timeit import default_timer as timer
import utils
import faulthandler
faulthandler.enable()
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


"""
nohup python run.py --lr 1e-3 --num_workers 2 --batch_size 4 --epochs 60 --cuda --ngpu 1 --refs 2 --patch_size 35 --patch_stride 30 --test_patch 75 --pretrained encoder.pth --save_dir out --train_dir data/train --val_dir data/val --test_dir data/val &> out.log &
"""

# 获取模型运行时必须的一些参数，
parser = argparse.ArgumentParser(description='Acquire some parameters for fusion restore')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='the initial learning rate')
parser.add_argument('--batch_size', type=int, default=4,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=30,
                    help='number of epochs to train')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--num_workers', type=int, default=2, help='number of threads to load data')
parser.add_argument('--save_dir', type=Path, default=Path('./out'),     #保存一些参数或者输出结果到项目根目录。
                    help='the output directory')
parser.add_argument('--pretrained', type=Path,default=(data.pretrained), help='the path of the pretained encoder')

# 获取对输入数据进行预处理时的一些参数
parser.add_argument('--refs', type=int, default=2,help='the reference data counts for fusion')  #需要几对参考图像
parser.add_argument('--train_dir', type=Path, default=(data.data_dir / 'train'),
                    help='the training data directory')
parser.add_argument('--val_dir', type=Path, default=(data.data_dir / 'val'),
                    help='the validation data directory')
parser.add_argument('--test_dir', type=Path, default=(data.data_dir / 'test'),
                    help='the test data directory')
parser.add_argument('--image_size', type=int, nargs='+', default=[1640, 1640],    #高宽(y x)
                    help='the size of the coarse image (width, height)')
parser.add_argument('--patch_size', type=int, nargs='+', default=80,
                    help='the coarse image patch size for training restore')    #用于训练恢复的粗图像块大小
parser.add_argument('--patch_padding', type=int, nargs='+', default=20,
                    help='the coarse patch padding for image division')

parser.add_argument('--patch_stride', type=int, nargs='+', default=80,
                    help='the coarse patch stride for image division')     #图像分割中的粗块步长
parser.add_argument('--test_patch', type=int, nargs='+', default=80,
                    help='the coarse image patch size for fusion test')

parser.add_argument('--num_res_blocks', type=str, default='16+16+8+4',
                    help='The number of residual blocks in each stage')
parser.add_argument('--n_feats', type=int, default=64,
                    help='The number of channels in network')
parser.add_argument('--res_scale', type=float, default=1.,
                    help='Residual scale')
parser.add_argument('--num_steps', type=int, nargs='+', default=2,
                    help='the FB number for model')

parser.add_argument('--eval', type=str2bool, default=False,
                    help='Evaluation mode')
parser.add_argument('--test', type=str2bool, default=False,
                    help='Test mode')



opt = parser.parse_args()

torch.manual_seed(2019)
if not torch.cuda.is_available():
    opt.cuda = False
if opt.cuda:
    torch.cuda.manual_seed_all(2019)
    cudnn.benchmark = True
    cudnn.deterministic = True

if __name__ == '__main__':
    project_start = timer()
    experiment = Experiment(opt)

    logger = utils.get_logger("train")
    logger.info(f'option：{opt}')

    if opt.epochs > 0:
        experiment.train(opt.train_dir, opt.val_dir,
                         opt.patch_size, opt.patch_stride, opt.batch_size,
                         opt.refs, patch_padding = opt.patch_padding,num_workers=opt.num_workers, epochs=opt.epochs)
    experiment.test(opt.test_dir, opt.test_patch, opt.refs,patch_padding = opt.patch_padding,
                    num_workers=opt.num_workers)
    project_end = timer()
    logger.info(f'Project Cost[{project_end-project_start}S')





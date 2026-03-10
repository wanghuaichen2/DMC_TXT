import os
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type = str,
        default='cfgs/Tooth_models/PoinTr.yaml',
        help = 'yaml config file')
    parser.add_argument(
        '--config_SAP',
        type=str,
        default='SAP/configs/learning_based/noise_small/ours.yaml',
        help='yaml config file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')
    # 记录数据集文件的命名关键字【’基牙‘，‘对牙’，’牙冠‘】
    parser.add_argument(
        '--file_key_words',
        type=str,
        nargs='+',
        default=['Preparation', 'Antagonist', 'Crown'],
        help='file name keywords (multiple allowed)'
    )
    parser.add_argument(
        '--psr_npz',
        action='store_true',
        default=False,
        help='Is there a psr.npz file')

    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--shell_center', type=int, nargs='+', default=[-25, 5, -4],help='shell_center')
    parser.add_argument('--use_crown', action='store_true', default=False, help='Whether to use the crown data (requires --test)')
    # seed
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    # bn
    parser.add_argument(
        '--sync_bn',
        action='store_true',
        default=False,
        help='whether to use sync bn')
    # some args
    parser.add_argument('--exp_name', type = str, default='default', help = 'experiment name')
    parser.add_argument('--start_ckpts', type = str, default=None, help = 'reload used ckpt path')
    parser.add_argument('--ckpts', type = str, default=None, help = 'test used ckpt path')
    parser.add_argument('--val_freq', type = int, default=1, help = 'test freq')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help = 'autoresume training (interrupted by accident)')
    parser.add_argument(
        '--test',
        action='store_true',
        default=False,
        help = 'test mode for certain ckpt')
    parser.add_argument(
        '--mode',
        choices=['easy', 'median', 'hard', None],
        default=None,
        help = 'difficulty mode for shapenet')
    args = parser.parse_args()

    if not args.test and not args.use_crown:
        args.use_crown = True # 训练模式下必须使用真实冠数据
        #parser.error("When --test is not enabled, --no_use_crown must be True (crown data required)")

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.test:
        args.exp_name = 'test_' + args.exp_name
    if args.mode is not None:
        args.exp_name = args.exp_name + '_' +args.mode
    args.experiment_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem, args.exp_name)
    args.tfboard_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem,'TFBoard' ,args.exp_name)
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)


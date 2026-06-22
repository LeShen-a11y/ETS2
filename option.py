import argparse

parser = argparse.ArgumentParser(description='SNN-M')
parser.add_argument('--feat-extractor', default='i3d', choices=['i3d', 'c3d'])
parser.add_argument('--feature-size', type=int, default=256, help='size of feature (default: 2048)')
parser.add_argument('--gt', default='list/gt-ucf.npy', help='file of ground truth ')
parser.add_argument('--gpus', default=1, type=int, choices=[0], help='gpus')
parser.add_argument('--lr', type=str, default='[0.001]*15000', help='learning rates for steps(list form)')
parser.add_argument('--batch-size', type=int, default=32, help='number of instances in a batch of data (default: 16)')
parser.add_argument('--workers', default=4, help='number of workers in dataloader')
parser.add_argument('--model-name', default='snn-mix', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--num-classes', type=int, default=1, help='number of class')
parser.add_argument('--dataset', default='ucf', help='dataset to train on (default: )')
parser.add_argument('--plot-freq', type=int, default=10, help='frequency of plotting (default: 10)')
parser.add_argument('--max-epoch', type=int, default=300, help='maximum iteration to train (default: 15000)')
# 1000
# Early-stopping 超参
parser.add_argument('--early-stop-patience', type=int, default=15,
                    help='max epochs to wait for metric improvement (default: 10)')
parser.add_argument('--early-stop-delta', type=float, default=1e-4,
                    help='minimum delta to qualify as improvement (default: 1e-4)')
parser.add_argument('--save-best-only', action='store_true',default=True,
                    help='only save the best ckpt instead of every 5 epoch')


parser.add_argument(
    '--modality',
    type=str,
    default='mix',
    choices=['mix', 'rgb', 'dvs'],
    help='Input modality: mix, rgb, or dvs.'
)

parser.add_argument(
    '--F-rgb',
    dest='F_rgb',
    type=int,
    default=768
)

parser.add_argument(
    '--F-dvs',
    dest='F_dvs',
    type=int,
    default=256
)

parser.add_argument(
    '--d-model',
    dest='d_model',
    type=int,
    default=256
)
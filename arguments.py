#encoding=utf-8

import argparse
parser = argparse.ArgumentParser("QSV")

parser.add_argument('--config', type=str, default='exp/clip/noisy/q4.yaml')

parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--phase', type=str, default='train')
parser.add_argument('--retrain', type=bool, default=False)

parser.add_argument('--log_dir', type=str, default='logs')

args = parser.parse_args()
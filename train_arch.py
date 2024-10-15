#encoding=utf-8

import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import json

from model.arch_net import ArchNet
from multimodal_data import CircIdealDataset

from arguments import args
from util.logger import setup_logger
from util.config import configs

logger = setup_logger(args.log_dir, name='arch')

def train(model, loader, loader_test, args, board):
    model.train()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.train.cuda and torch.cuda.is_available():
        model.cuda()

    loss_fn = nn.MSELoss()

    params = [
        {'params': model.parameters(), 'initial_lr': args.train.lr, 'lr': args.train.lr},
    ]

    optimizer = torch.optim.Adam(params, betas=(0.5, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.train.decay_step, gamma=0.5, last_epoch=-1)
    for i in range(args.train.n_epoch):
        if device == 'cuda':
            model.cuda()

        for cir1, cir2, label, _ in loader:
            feat1 = model(cir1.to(device))
            feat2 = model(cir2.to(device))
            sim = (torch.sum(torch.mul(feat1, feat2), dim=1) + 1)/2

            loss = loss_fn(sim, label.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch={i}, loss={loss.item()}, lr={lr}')
        board.add_scalar('loss', loss.item(), i+1)

        scheduler.step()
        if (i+1) % args.train.save_freq == 0:
            preds, labels, keys = [], [], []
            with torch.no_grad():
                model.eval()
                for cir1, cir2, label, key in loader_test:
                    feat1 = model(cir1.to(device))
                    feat2 = model(cir2.to(device))
                    sim = (torch.sum(torch.mul(feat1, feat2), dim=1)+1)/2
                    preds.append(sim.cpu().detach().numpy())
                    labels.append(label.cpu().detach().numpy())
                preds = np.concatenate(preds, axis=0)
                labels = np.concatenate(labels, axis=0)
                mse = np.mean((preds-labels)**2)
                mae = np.mean(np.abs(preds-labels))
                logger.info(f'Test mse = {mse}, mae={mae}')
                model.train()

            torch.save(model.cpu().state_dict(), os.path.join(args.log_dir, f'arch_c{args.dataset.n_circuit}_e{i+1}.pth'))

def test(model, loader):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.cuda and torch.cuda.is_available():
        model.cuda()

    preds, labels, keys = [], [], []
    feat_query, feat_ref = [], []
    with torch.no_grad():
        for cir1, cir2, label, key in loader:
            feat1 = model(cir1.to(device))
            feat2 = model(cir2.to(device))
            sim = (torch.sum(torch.mul(feat1, feat2), dim=1)+1)/2
            preds.append(sim.cpu().detach().numpy())
            labels.append(label.cpu().detach().numpy())
            feat_query.append(feat1.cpu().detach().numpy())
            feat_ref.append(feat2.cpu().detach().numpy())
            keys = keys + key
        return np.concatenate(preds, axis=0), np.concatenate(labels, axis=0), keys, np.concatenate(feat_query, axis=0), np.concatenate(feat_ref, axis=0)

if __name__ == '__main__':
    torch.manual_seed(0)
    if torch.cuda.is_available():
        if not args.cuda:
            logger.info("WARNING: You have a CUDA device, so you should probably run with --cuda")

    configs.load(args.config, recursive=True)

    logger.info('begin')
    logger.info(f'arguments = {configs}')
    args.log_dir = os.path.join(args.log_dir, '/'.join(args.config.split('/')[1:])[:-5])
    configs.log_dir = args.log_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    model = ArchNet(configs.dataset.circ_feat_dim, configs.model.dims)
    if args.phase == 'train':
        if args.retrain:
            model.load_state_dict(torch.load(os.path.join(args.log_dir, f'arch_c{configs.dataset.n_circuit}_e{configs.train.n_epoch}.pth')))
        names = os.listdir(args.log_dir)
        for name in names:
            if 'event' in name:
                os.remove(os.path.join(args.log_dir, name))
        board = SummaryWriter(log_dir=args.log_dir)

        if configs.model.loss_type == 'rl':
            train_dataset = CircIdealDataset(configs.dataset, test=False)
            test_mea_name = configs.dataset.mea_data.replace('train', 'test')
            test_mea_name = test_mea_name.replace('p1000', 'p100')
            test_arch_name = configs.dataset.arch_data.replace('train', 'test')
            test_arch_name = test_arch_name.replace('p1000', 'p100')
            configs.dataset.mea_data = test_mea_name
            configs.dataset.arch_data = test_arch_name
            test_dataset = CircIdealDataset(configs.dataset, test=True)
            logger.info(f'Train={len(train_dataset)}, test={len(test_dataset)}')
        logger.info('----------build dataloader------------')
        train_loader =  DataLoader(train_dataset, batch_size=configs.train.bs, shuffle=True, num_workers=configs.train.n_worker, pin_memory=True)
        test_loader =  DataLoader(test_dataset, batch_size=configs.train.bs, shuffle=True, num_workers=configs.train.n_worker, pin_memory=True)

        logger.info('----------begin training------------')
        train(model, train_loader, test_loader, configs, board)
    elif args.phase == 'test':
        model.load_state_dict(torch.load(os.path.join(args.log_dir, f'arch_c{configs.dataset.n_circuit}_e{configs.train.n_epoch}.pth')))

        test_mea_name = configs.dataset.mea_data.replace('train', 'test')
        test_mea_name = test_mea_name.replace('p1000', 'p100')
        test_arch_name = configs.dataset.arch_data.replace('train', 'test')
        test_arch_name = test_arch_name.replace('p1000', 'p100')
        configs.dataset.mea_data = test_mea_name
        configs.dataset.arch_data = test_arch_name

        test_dataset = CircIdealDataset(configs.dataset, test=True)
        test_loader =  DataLoader(test_dataset, batch_size=configs.train.bs, shuffle=False, num_workers=configs.train.n_worker, pin_memory=True)
        preds, labels, keys, feat_query, feat_ref = test(model, test_loader)
        mse = np.mean((preds-labels)**2)
        logger.info(f'{len(preds)} samples, MSE={mse}')

        res = {}
        for i, key in enumerate(keys):
            res[key] = {'pred': preds[i].tolist(), 'label': labels[i].tolist()}

        with open(os.path.join(args.log_dir, f'test_arch_c{configs.dataset.n_circuit}.json'), 'w') as f:
            json.dump(res, f)

        with open(os.path.join(args.log_dir, f'test_arch_feat_c{configs.dataset.n_circuit}.json'), 'w') as f:
            res = {}
            for i, key in enumerate(keys):
                res[key] = {'query': feat_query[i].tolist(), 'ref': feat_ref[i].tolist(), 'fi': labels[i].tolist()}
            json.dump(res, f)
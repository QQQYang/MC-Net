#encoding=utf-8

import torch
from torch import nn
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import shutil
from sklearn.metrics import r2_score

from model.combine_net import CombineNet
from multimodal_data import MeaCircIdealDataset, MeaCircIdealLargeDataset

from arguments import args
from util.logger import setup_logger
from util.config import configs

logger = setup_logger(args.log_dir, name='measurement+arch')

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
    res_test = {}
    res_train = {}
    for i in range(args.train.n_epoch_combine):
        if device == 'cuda':
            model.cuda()

        for mea_vec1, mea_vec2, cir1, cir2, label, _ in loader:
            feat1 = model(mea_vec1.to(device), cir1.to(device))
            feat2 = model(mea_vec2.to(device), cir2.to(device))
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
            preds, labels = [], []
            with torch.no_grad():
                model.eval()
                for mea_vec1, mea_vec2, cir1, cir2, label, _ in loader_test:
                    feat1 = model(mea_vec1.to(device), cir1.to(device))
                    feat2 = model(mea_vec2.to(device), cir2.to(device))
                    sim = (torch.sum(torch.mul(feat1, feat2), dim=1)+1)/2
                    preds.append(sim.cpu().detach().numpy())
                    labels.append(label.cpu().detach().numpy())
                preds = np.concatenate(preds, axis=0)
                labels = np.concatenate(labels, axis=0)
                mse = np.mean((preds-labels)**2)
                mae = np.mean(np.abs(preds-labels))
                logger.info(f'Test mse = {mse}, mae={mae}, r2={r2_score(labels, preds)}')
                res_test[str(i+1)] = mse.tolist()

                preds, labels = [], []
                for mea_vec1, mea_vec2, cir1, cir2, label, _ in loader:
                    feat1 = model(mea_vec1.to(device), cir1.to(device))
                    feat2 = model(mea_vec2.to(device), cir2.to(device))
                    sim = (torch.sum(torch.mul(feat1, feat2), dim=1)+1)/2
                    preds.append(sim.cpu().detach().numpy())
                    labels.append(label.cpu().detach().numpy())
                preds = np.concatenate(preds, axis=0)
                labels = np.concatenate(labels, axis=0)
                mse = np.mean((preds-labels)**2)
                mae = np.mean(np.abs(preds-labels))
                res_train[str(i+1)] = mse.tolist()

                model.train()

    torch.save(model.cpu().state_dict(), os.path.join(args.log_dir, f'combine_c{args.dataset.n_circuit}_s{args.dataset.shadow_size}_e{args.train.n_epoch_combine}.pth'))
    return res_test, res_train

def test(model, args, loader):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.train.cuda and torch.cuda.is_available():
        model.cuda()

    preds, labels, keys = [], [], []
    feat_query, feat_ref = [], []
    with torch.no_grad():
        for mea_vec1, mea_vec2, cir1, cir2, label, key in loader:
            feat1 = model(mea_vec1.to(device), cir1.to(device))
            feat2 = model(mea_vec2.to(device), cir2.to(device))
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
    # args.log_dir = os.path.join(args.log_dir, f'{args.cir_type}-combine-{args.loss_type}-{args.label_type}_d{args.obs_feat_dim}_lr{round(args.lr, 5)}_s{args.shadow_size}_p{args.n_phase}')
    args.log_dir = os.path.join(args.log_dir, '/'.join(args.config.split('/')[1:])[:-5])
    configs.log_dir = args.log_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    shutil.copy(args.config, args.log_dir)
    model = CombineNet(configs.dataset.obs_feat_dim, configs.dataset.circ_feat_dim, configs.model.dims, configs.dataset.average, fuse=configs.model.fuse)
    # model.model_obs.load_state_dict(torch.load(os.path.join(f'/home/yqia7342/Documents/quantum/quantum state verification/logs/HEA-obsnet-rl-{args.label_type}_d{args.obs_feat_dim}_lr0.0001_s{args.shadow_size}_p{args.n_phase}', f'obsnet_e500.pth')), strict=False)
    model.model_obs.load_state_dict(torch.load(os.path.join(args.log_dir, f'{configs.model.net_type}_c{configs.dataset.n_circuit}_s{configs.dataset.shadow_size}_e{configs.train.n_epoch}.pth')))
    model.model_arch.load_state_dict(torch.load(os.path.join(args.log_dir, f'arch_c{configs.dataset.n_circuit}_e{configs.train.n_epoch}.pth')))#, strict=False)
    # model.model_arch.load_state_dict(torch.load(os.path.join(f'/home/yqia7342/Documents/quantum/quantum state verification/logs/HEA-arch-rl-{args.label_type}_d{args.obs_feat_dim}_lr0.0001_p{args.n_phase}_bs128_head', f'arch_e250.pth')), strict=False)

    if args.pretrain:
        model.load_state_dict(torch.load(os.path.join(args.log_dir, f'clip_c1000_s500_e250.pth')), strict=False)

    if args.phase == 'train':
        if args.retrain:
            model.load_state_dict(torch.load(os.path.join(args.log_dir, f'combine_c{configs.dataset.n_circuit}_s{configs.dataset.shadow_size}_e{configs.train.n_epoch_combine}.pth')))
        names = os.listdir(args.log_dir)
        for name in names:
            if 'event' in name:
                os.remove(os.path.join(args.log_dir, name))
        board = SummaryWriter(log_dir=args.log_dir)

        if configs.model.loss_type == 'rl':
            if configs.dataset.split:
                train_dataset = MeaCircIdealLargeDataset(configs.dataset, test=False)
            else:
                train_dataset = MeaCircIdealDataset(configs.dataset, test=False)
            test_mea_name = configs.dataset.mea_data.replace('train', 'test')
            test_mea_name = test_mea_name.replace('p1000', 'p100')
            test_arch_name = configs.dataset.arch_data.replace('train', 'test')
            test_arch_name = test_arch_name.replace('p1000', 'p100')
            configs.dataset.mea_data = test_mea_name
            configs.dataset.arch_data = test_arch_name

            if configs.dataset.split:
                test_dataset = MeaCircIdealLargeDataset(configs.dataset, test=True)
            else:
                test_dataset = MeaCircIdealDataset(configs.dataset, test=True)
            logger.info(f'Train={len(train_dataset)}, test={len(test_dataset)}')
        train_loader =  DataLoader(train_dataset, batch_size=configs.train.bs, shuffle=True, num_workers=configs.train.n_worker, pin_memory=True)
        test_loader =  DataLoader(test_dataset, batch_size=configs.train.bs, shuffle=True, num_workers=configs.train.n_worker, pin_memory=True)

        res_test, res_train = train(model, train_loader, test_loader, configs, board)
        with open(os.path.join(args.log_dir, f'test_combine_epoch.json'), 'w') as f:
            json.dump(res_test, f)
        with open(os.path.join(args.log_dir, f'train_combine_epoch.json'), 'w') as f:
            json.dump(res_train, f)
    elif args.phase == 'test':
        model.load_state_dict(torch.load(os.path.join(args.log_dir, f'combine_c{configs.dataset.n_circuit}_s{configs.dataset.shadow_size}_e{configs.train.n_epoch_combine}.pth')))

        test_mea_name = configs.dataset.mea_data.replace('train', 'test')
        test_mea_name = test_mea_name.replace('p1000', 'p100')
        test_arch_name = configs.dataset.arch_data.replace('train', 'test')
        test_arch_name = test_arch_name.replace('p1000', 'p100')
        configs.dataset.mea_data = test_mea_name
        configs.dataset.arch_data = test_arch_name
        # configs.dataset.n_circuit = 100
        if configs.dataset.split:
            test_dataset = MeaCircIdealLargeDataset(configs.dataset, test=True)
        else:
            test_dataset = MeaCircIdealDataset(configs.dataset, test=True)
        test_loader =  DataLoader(test_dataset, batch_size=configs.train.bs, shuffle=False, num_workers=configs.train.n_worker, pin_memory=True)
        preds, labels, keys, feat_query, feat_ref = test(model, configs, test_loader)
        mse = np.mean((preds-labels)**2)
        mae = np.mean(np.abs(preds-labels))
        logger.info(f'{len(preds)} samples, MSE={mse}, MAE={mae}, r2={r2_score(labels, preds)}')

        res = {}
        for i, key in enumerate(keys):
            res[key] = {'pred': preds[i].tolist(), 'label': labels[i].tolist()}
        with open(os.path.join(args.log_dir, f'test_combine_c{configs.dataset.n_circuit}_s{configs.dataset.shadow_size}.json'), 'w') as f:
            json.dump(res, f)

        with open(os.path.join(args.log_dir, f'test_combine_feat_c{configs.dataset.n_circuit}_s{configs.dataset.shadow_size}.json'), 'w') as f:
            res = {}
            for i, key in enumerate(keys):
                res[key] = {'query': feat_query[i].tolist(), 'ref': feat_ref[i].tolist(), 'fi': labels[i].tolist()}
            json.dump(res, f)
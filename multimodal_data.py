#encoding=utf-8
import torch
import torch.utils.data as torch_data

import json
import numpy as np
import h5py
import pickle

import sys
sys.path.append('.')

def load_data_from_pyg(file_name):
    file = open(file_name + '.data', "rb")
    data = pickle.load(file)
    file.close()
    print("Size of the data: ", len(data))
    return data

class MeaIdealDataset(torch_data.Dataset):
    def __init__(self, args, test=False):
        super().__init__()

        print('---load data-----')
        self.f = h5py.File(args.mea_data)
        if test:
            self.n_sample = len(self.f['label'][:])
        else:
            self.n_sample = len(self.f['label'][:args.n_circuit*args.n_noise_pair])
        self.shadow_size = args.shadow_size
        if args.normalize:
            with open(args.mean_std_data, 'r') as f:
                mean_std = json.load(f)
                self.mea_mean = np.array(mean_std['mean'][:args.obs_feat_dim], dtype=np.float32)
                self.mea_std = np.array(mean_std['std'][:args.obs_feat_dim], dtype=np.float32)
        else:
            self.mea_mean = 0
            self.mea_std = 1
        self.average = args.average
        self.obs_feat_dim = args.obs_feat_dim

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        if self.average:
            return torch.from_numpy((np.mean(self.f['mea1'][index][:, :self.obs_feat_dim], axis=0)-self.mea_mean)/(1e-8+self.mea_std))[:self.shadow_size], torch.from_numpy((np.mean(self.f['mea2'][index][:, :self.obs_feat_dim], axis=0)-self.mea_mean)/(1e-8+self.mea_std))[:self.shadow_size], self.f['label'][index], self.f['key'][index].decode('UTF-8')
        else:
            return torch.from_numpy((self.f['mea1'][index][:, :self.obs_feat_dim]-self.mea_mean)/(1e-8+self.mea_std))[:self.shadow_size], torch.from_numpy((self.f['mea2'][index][:, :self.obs_feat_dim]-self.mea_mean)/(1e-8+self.mea_std))[:self.shadow_size], self.f['label'][index], self.f['key'][index].decode('UTF-8')

class MeaIdealLargeDataset(torch_data.Dataset):
    def __init__(self, args, test=False) -> None:
        super().__init__()

        self.f = h5py.File(args.mea_data)
        self.f_ref = h5py.File(args.mea_data.replace('query', 'ref'))

        self.n_sample = len(self.f['label'][:])
        query2ref = self.f['query2ref'][:]
        uniq_idx = np.unique(query2ref)
        ref_idx = {}
        for i in range(len(uniq_idx)):
            ref_idx[uniq_idx[i]] = i
        self.query2ref = []
        for i in range(len(query2ref)):
            self.query2ref.append(ref_idx[query2ref[i]])
        self.shadow_size = args.shadow_size
        if args.normalize:
            with open(args.mean_std_data, 'r') as f:
                mean_std = json.load(f)
                self.mea_mean = np.array(mean_std['mean'][:args.obs_feat_dim], dtype=np.float32)
                self.mea_std = np.array(mean_std['std'][:args.obs_feat_dim], dtype=np.float32)
        else:
            self.mea_mean = 0
            self.mea_std = 1
        self.average = args.average

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        if self.average:
            return torch.from_numpy((np.mean(self.f['mea'][index], axis=0)-self.mea_mean)/(1e-8+self.mea_std))[:self.shadow_size], torch.from_numpy((np.mean(self.f_ref['mea'][self.query2ref[index]], axis=0)-self.mea_mean)/(1e-8+self.mea_std))[:self.shadow_size], self.f['label'][index], self.f['key'][index].decode('UTF-8')+'-1'
        else:
            return torch.from_numpy((self.f['mea'][index]-self.mea_mean)/(1e-8+self.mea_std))[:self.shadow_size], torch.from_numpy((self.f_ref['mea'][self.query2ref[index]]-self.mea_mean)/(1e-8+self.mea_std))[:self.shadow_size], self.f['label'][index], self.f['key'][index].decode('UTF-8')+'-1'

class MeaCircIdealDataset(torch_data.Dataset):
    def __init__(self, args, test=False) -> None:
        super().__init__()

        np.random.seed(0)
        self.f = h5py.File(args.mea_data)
        if test:
            self.n_sample = len(self.f['label'][:])
        else:
            self.n_sample = len(self.f['label'][:args.n_circuit*args.n_noise_pair])
        self.shadow_size = args.shadow_size
        circuit_graph = load_data_from_pyg(args.arch_data)
        self.cir1 = circuit_graph['cir1']
        self.cir2 = circuit_graph['cir2']
        self.labels = circuit_graph['label']
        keys = circuit_graph['key']
        self.key2index = {}
        for i in range(len(keys)):
            self.key2index[keys[i]] = i

        if args.normalize:
            with open(args.mean_std_data, 'r') as f:
                mean_std = json.load(f)
                self.mea_mean = np.array(mean_std['mean'][:args.obs_feat_dim], dtype=np.float32)
                self.mea_std = np.array(mean_std['std'][:args.obs_feat_dim], dtype=np.float32)
        else:
            self.mea_mean = 0
            self.mea_std = 0
        self.average = args.average

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        if self.average:
            return torch.from_numpy((np.mean(self.f['mea1'][index], axis=0)-self.mea_mean)/(1e-8+self.mea_std))[:self.shadow_size], torch.from_numpy((np.mean(self.f['mea2'][index], axis=0)-self.mea_mean)/(1e-8+self.mea_std))[:self.shadow_size], self.cir1[self.key2index[self.f['key'][index].decode('UTF-8')]], self.cir2[self.key2index[self.f['key'][index].decode('UTF-8')]], self.labels[self.key2index[self.f['key'][index].decode('UTF-8')]], self.f['key'][index].decode('UTF-8')
        else:
            return torch.from_numpy((self.f['mea1'][index]-self.mea_mean)/(1e-8+self.mea_std))[:self.shadow_size], torch.from_numpy((self.f['mea2'][index]-self.mea_mean)/(1e-8+self.mea_std))[:self.shadow_size], self.cir1[self.key2index[self.f['key'][index].decode('UTF-8')]], self.cir2[self.key2index[self.f['key'][index].decode('UTF-8')]], self.labels[self.key2index[self.f['key'][index].decode('UTF-8')]], self.f['key'][index].decode('UTF-8')

class CircIdealDataset(torch_data.Dataset):
    def __init__(self, args, test=False) -> None:
        super().__init__()

        np.random.seed(0)
        self.f = h5py.File(args.mea_data)
        if test:
            self.n_sample = len(self.f['label'][:])
        else:
            self.n_sample = len(self.f['label'][:args.n_circuit*args.n_noise_pair])
        self.key_plus_1 = ''
        if 'query' in args.mea_data:
            self.key_plus_1 = '-1'

        circuit_graph = load_data_from_pyg(args.arch_data)
        self.cir1 = circuit_graph['cir1']
        self.cir2 = circuit_graph['cir2']
        self.labels = circuit_graph['label']
        keys = circuit_graph['key']
        self.key2index = {}
        for i in range(len(keys)):
            self.key2index[keys[i]] = i

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        return self.cir1[self.key2index[self.f['key'][index].decode('UTF-8')+self.key_plus_1]], self.cir2[self.key2index[self.f['key'][index].decode('UTF-8')+self.key_plus_1]], self.labels[self.key2index[self.f['key'][index].decode('UTF-8')+self.key_plus_1]], self.f['key'][index].decode('UTF-8')+self.key_plus_1

class MeaCircIdealLargeDataset(torch_data.Dataset):
    def __init__(self, args, test=False) -> None:
        super().__init__()

        np.random.seed(0)
        self.f = h5py.File(args.mea_data)
        if test:
            self.n_sample = len(self.f['label'][:])
        else:
            self.n_sample = len(self.f['label'][:args.n_circuit*args.n_noise_pair])

        self.f_ref = h5py.File(args.mea_data.replace('query', 'ref'))

        query2ref = self.f['query2ref'][:]
        uniq_idx = np.unique(query2ref)
        ref_idx = {}
        for i in range(len(uniq_idx)):
            ref_idx[uniq_idx[i]] = i
        self.query2ref = []
        for i in range(len(query2ref)):
            self.query2ref.append(ref_idx[query2ref[i]])

        circuit_graph = load_data_from_pyg(args.arch_data)
        self.cir1 = circuit_graph['cir1']
        self.cir2 = circuit_graph['cir2']
        self.labels = circuit_graph['label']
        keys = circuit_graph['key']
        self.key2index = {}
        for i in range(len(keys)):
            self.key2index[keys[i]] = i

        if args.normalize:
            with open(args.mean_std_data, 'r') as f:
                mean_std = json.load(f)
                self.mea_mean = np.array(mean_std['mean'][:args.obs_feat_dim], dtype=np.float32)
                self.mea_std = np.array(mean_std['std'][:args.obs_feat_dim], dtype=np.float32)
        else:
            self.mea_mean = 0
            self.mea_std = 0

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        return torch.from_numpy((self.f['mea'][index]-self.mea_mean)/(1e-8+self.mea_std)), torch.from_numpy((self.f_ref['mea'][self.query2ref[index]]-self.mea_mean)/(1e-8+self.mea_std)), self.cir1[self.key2index[self.f['key'][index].decode('UTF-8')+'-1']], self.cir2[self.key2index[self.f['key'][index].decode('UTF-8')+'-1']], self.labels[self.key2index[self.f['key'][index].decode('UTF-8')+'-1']], self.f['key'][index].decode('UTF-8')+'-1'


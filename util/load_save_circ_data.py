import pickle
import sys
from tqdm import tqdm
import numpy as np
import json
import copy

sys.path.append('.')

import torch
from util.circ_dag_converter import raw_pyg_converter
from arguments import args


def normalize_data(file_name):
    file = open(file_name+'.data', "rb")
    data = pickle.load(file)
    file.close()

    all_features = None
    for k, dag in enumerate(data['cir1']):
        if not k:
            all_features = dag.x
        else:
            all_features = torch.cat([all_features, dag.x])

    means = all_features.mean(0)
    stds = all_features.std(0)
    for k, dag in enumerate(data['cir1']):
        data['cir1'][k].x = (dag.x - means) / (1e-8 + stds)

    for k, dag in enumerate(data['cir2']):
        data['cir2'][k].x = (dag.x - means) / (1e-8 + stds)
    file = open(file_name + "-normalized.datameta", "wb")
    pickle.dump([means, stds], file)
    file.close()

    file = open(file_name + '-normalized.data', "wb")
    pickle.dump(data, file)
    file.close()


def load_data_and_save():
    with open('-'.join(args.mea_data.split('-')[:-1])+'-qasm.json', 'r') as f:
        circuit_graph = json.load(f)

    if args.label_type == 'fi':
        with open('-'.join(args.mea_data.split('-')[:-1])+'-label.json', 'r') as f:
            label = json.load(f)
    else:
        with open('-'.join(args.mea_data.split('-')[:-1])+'-label-shadow2.json', 'r') as f:
            label = json.load(f)
    cir1 = []
    cir2 = []
    labels = []
    keys = []

    cir_dict = {}
    cnt = 0
    noise_strength = [0.0001, 0.0003, 0.002, 0.004, 0.007, 0.01, 0.013, 0.018, 0.023, 0.03, 0.04] # #qubit=30
    noise_strength = [0.0001, 0.0003, 0.0012, 0.0028, 0.004, 0.006, 0.008, 0.01, 0.014, 0.0188, 0.027] # qubit=50
    for arch in tqdm(label, desc='arch'):
        if cnt == args.n_phase:
            break
        cnt += 1
        for key in label[arch]:
            labels.append(float(label[arch][key][args.label_type]))
            noise2_idx = len(key.split('-'))//2
            keys.append(arch+'-'+key)

            if '-'.join([arch, key.split('-')[0]]) not in cir_dict:
                cir1.append(raw_pyg_converter(circuit_graph[arch], (int(key.split('-')[0]))/args.n_depolar*0.1))
                # cir1.append(raw_pyg_converter(circuit_graph[arch], noise_strength[int(key.split('-')[0])-1]))
                cir_dict['-'.join([arch, key.split('-')[0]])] = cir1[-1]
            else:
                cir1.append(copy.deepcopy(cir_dict['-'.join([arch, key.split('-')[0]]) ]))
            if '-'.join([arch, key.split('-')[noise2_idx]]) not in cir_dict:
                cir2.append(raw_pyg_converter(circuit_graph[arch], (int(key.split('-')[2]))/args.n_depolar*0.1))
                # cir2.append(raw_pyg_converter(circuit_graph[arch], noise_strength[int(key.split('-')[1])-1]))
                cir_dict['-'.join([arch, key.split('-')[noise2_idx]])] = cir2[-1]
            else:
                cir2.append(copy.deepcopy(cir_dict['-'.join([arch, key.split('-')[noise2_idx]]) ]))
    labels = torch.from_numpy(np.array(labels, dtype=np.float32))


    pyg_data = {
        'cir1': cir1,
        'cir2': cir2,
        'label': labels,
        'key': keys
    }
    file = open(args.mea_data.split('.')[0] + f'-p{args.n_phase}' + '-pyg-hpc.data', "wb")
    pickle.dump(pyg_data, file)
    file.close()


if __name__ == "__main__":
    load_data_and_save()
    filename = args.mea_data.split('.')[0] + f'-p{args.n_phase}-pyg-hpc'
    normalize_data(filename)

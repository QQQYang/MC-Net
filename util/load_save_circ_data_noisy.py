import pickle
import sys
from tqdm import tqdm
import numpy as np
import json
import copy

sys.path.append('.')

import torch
from util.circ_dag_converter_noisy import raw_pyg_converter
from util.config import configs
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
    with open('-'.join(args.mea_data.split('-')[:-1])+'-qasm-transpile.json', 'r') as f:
        circuit_graph = json.load(f)

    with open('data/ibmq_device_info', 'rb') as f:
        device_info = pickle.load(f)

    if args.label_type == 'fi':
        with open('-'.join(args.mea_data.split('-')[:-1])+'-label.json', 'r') as f:
            label = json.load(f)
    else:
        with open('-'.join(args.mea_data.split('-')[:-1])+'-label-shadow2.json', 'r') as f:
            label = json.load(f)

    configs.load('exp/noisy/q4.yaml', recursive=True)

    cir1 = []
    cir2 = []
    labels = []
    keys = []

    cir_dict = {}
    cnt = 0
    for arch in tqdm(label, desc='arch'):
        if cnt == args.n_phase:
            break
        cnt += 1
        for key in label[arch]:
            labels.append(float(label[arch][key][args.label_type]))

            if '-'.join([arch, key.split('-')[0]]) not in cir_dict:
                backend_name = configs.dataset.backends[int(key.split('-')[0])]
                cir1.append(raw_pyg_converter(circuit_graph[arch][backend_name], device_info[backend_name]['noise']))
                cir_dict['-'.join([arch, key.split('-')[0]])] = cir1[-1]
            else:
                cir1.append(copy.deepcopy(cir_dict['-'.join([arch, key.split('-')[0]]) ]))
            if '-'.join([arch, key.split('-')[1]]) not in cir_dict:
                backend_name = configs.dataset.backends[int(key.split('-')[1])]
                cir2.append(raw_pyg_converter(circuit_graph[arch][backend_name], device_info[backend_name]['noise']))
                cir_dict['-'.join([arch, key.split('-')[1]])] = cir2[-1]
            else:
                cir2.append(copy.deepcopy(cir_dict['-'.join([arch, key.split('-')[1]]) ]))
            keys.append(arch+'-'+key)
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

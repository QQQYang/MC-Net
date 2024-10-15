#encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GraphConv, global_mean_pool

class ArchNet(nn.Module):
    def __init__(self, channel=1, dims=[128, 256, 512]):
        super(ArchNet, self).__init__()
        self.conv1 = GraphConv(channel, dims[0])
        self.conv2 = GraphConv(dims[0], dims[1])
        self.conv3 = GraphConv(dims[1], dims[2])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))

        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, data.batch)
        return F.normalize(x)
    
class ArchBranNet(nn.Module):
    def __init__(self, channel=1, dims=[128, 256, 512]):
        super(ArchBranNet, self).__init__()
        self.conv1 = GraphConv(channel, dims[0])
        self.conv2 = GraphConv(dims[0], dims[1])
        self.conv3 = GraphConv(dims[1], dims[2])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))

        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, data.batch)
        return x
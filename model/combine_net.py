#encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('.')

from model.arch_net import ArchBranNet
from model.obs_net import ObsBranNet, ObsAverageBranNet

class LowRankBilinearPooling(nn.Module):
	def __init__(self, in_channels1, in_channels2, hidden_dim, out_channels, sum_pool = True):
		super().__init__()
		self.sum_pool = sum_pool
		self.proj1 = nn.Linear(in_channels1, hidden_dim, bias = False)
		self.proj2 = nn.Linear(in_channels2, hidden_dim, bias = False)
		self.proj = nn.Linear(hidden_dim, out_channels)
		
	def forward(self, x1, x2):
		x1_ = torch.tanh(self.proj1(x1))
		x2_ = torch.tanh(self.proj2(x2))
		lrbp = self.proj(x1_ * x2_)
		return lrbp

class CombineNet(nn.Module):
    def __init__(self, obs_feat_dim, circ_feat_dim, dims=[128, 256, 512], average=False, fuse='lrbp'):
        super(CombineNet, self).__init__()
        self.model_arch = ArchBranNet(circ_feat_dim, dims)
        if average:
            self.model_obs = ObsAverageBranNet(obs_feat_dim, dims)
        else:
            self.model_obs = ObsBranNet(obs_feat_dim, dims)

        self.fuse = fuse
        if fuse == 'lrbp':
            self.lrbp = LowRankBilinearPooling(dims[-1], dims[-1], dims[-1], dims[-1])
        elif fuse == 'concat':
            self.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(dims[-1]*2, dims[-1]*2),
            )
        else:
            self.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(dims[-1], dims[-1]),
            )

    def forward(self, obs_feat, circ_feat):
        obs_feat = self.model_obs(obs_feat)
        circ_feat = self.model_arch(circ_feat)
        if self.fuse == 'lrbp':
            return F.normalize(self.lrbp(obs_feat, circ_feat))
        elif self.fuse == 'concat':
            feat = torch.cat((circ_feat, obs_feat), dim=1)
            return F.normalize(self.fc(feat))
        else:
            return F.normalize(self.fc(circ_feat+obs_feat))

#encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class ObsNet(nn.Module):
    def __init__(self, channel=1, dims=[256, 512, 512]):
        super(ObsNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, dims[0], 1)
        self.conv2 = torch.nn.Conv1d(dims[0], dims[1], 1)
        self.conv3 = torch.nn.Conv1d(dims[1], dims[2], 1)
        self.bn1 = nn.BatchNorm1d(dims[0])
        self.bn2 = nn.BatchNorm1d(dims[1])
        self.bn3 = nn.BatchNorm1d(dims[2])
        self.fc = nn.Sequential(
            nn.Linear(dims[2], dims[2]),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dims[2], dims[2]),
        )

    def forward(self, x):
        B, N, D = x.size()
        x = x.transpose(2, 1)
        x = self.bn1(F.relu(self.conv1(x)))

        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.sum(x, 2, keepdim=True)
        x = x.view(B, -1)
        return F.normalize(self.fc(x))


class ObsBranNet(nn.Module):
    def __init__(self, channel=1, dims=[256, 512, 512]):
        super(ObsBranNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, dims[0], 1)
        self.conv2 = torch.nn.Conv1d(dims[0], dims[1], 1)
        self.conv3 = torch.nn.Conv1d(dims[1], dims[2], 1)
        self.bn1 = nn.BatchNorm1d(dims[0])
        self.bn2 = nn.BatchNorm1d(dims[1])
        self.bn3 = nn.BatchNorm1d(dims[2])
        self.fc = nn.Sequential(
            nn.Linear(dims[2], dims[2]),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dims[2], dims[2]),
        )

    def forward(self, x):
        B, N, D = x.size()
        x = x.transpose(2, 1)
        x = self.bn1(F.relu(self.conv1(x)))

        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.sum(x, 2, keepdim=True)
        x = x.view(B, -1)
        return self.fc(x)
    
class ObsAverageNet(nn.Module):
    def __init__(self, channel=1, dims=[256, 512, 512]):
        super(ObsAverageNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, dims[0]),
            nn.ReLU(),
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dims[2], dims[2]),
        )

    def forward(self, x):
        return F.normalize(self.fc(x))

class ObsAverageBranNet(nn.Module):
    def __init__(self, channel=1, dims=[256, 512, 512]):
        super(ObsAverageBranNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, dims[0]),
            nn.ReLU(),
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dims[2], dims[2]),
        )

    def forward(self, x):
        return self.fc(x)
import torch
import torch.nn as nn

class AvoidNet(nn.Module):
    def __init__(self):
        super(AvoidNet, self).__init__()
        self.conv1 = nn.Conv2d(1,16,3,2,1)
        self.conv2 = nn.Conv2d(16,32,3,2,1)
        self.conv3 = nn.Conv2d(32,64,3,2,1)
        self.pool = nn.AdaptiveAvgPool2d((2,2))
        self.fc = nn.Linear(64*4,3) # 输出：前进，左转，右转
    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.flatten(1)
        out = self.fc(x)
        return out
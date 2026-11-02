import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from model import AvoidNet

class DepthDataset(Dataset):
    def __init__(self):
        self.data = []
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]

if __name__ == "__main__":
    model = AvoidNet()
    opt = torch.optim.Adam(model.parameters(),lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    ds = DepthDataset()
    dl = DataLoader(ds,batch_size=8,shuffle=True)
    for epoch in range(30):
        for x,y in dl:
            pred = model(x)
            loss = loss_fn(pred,y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    torch.save(model.state_dict(),"avoid_ckpt.pth")
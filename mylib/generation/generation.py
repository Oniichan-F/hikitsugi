from .models.my_ae import calc_loss

import numpy as np
from torch.autograd import Variable


def train_ae(device, model, loader, optimizer, criterion):
    losses = []
    for x, _, _ in loader:
        x = Variable(x).to(device)
        loss, _, _ = calc_loss(model, x, criterion)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().numpy())
        
    ep_loss = np.sum(losses) / len(loader.dataset)
    return ep_loss


def train_vae(device, model, loader, optimizer):
    losses = []
    for x, _, _ in loader:
        x = Variable(x).to(device)
        loss, _, _, _, _ = model.loss(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().numpy())
    
    ep_loss = np.sum(losses) / len(loader.dataset)
    return ep_loss
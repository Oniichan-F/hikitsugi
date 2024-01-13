import torch

from torch import nn
from torch.autograd import Variable


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
        
    def forward(self,x):
        return x.view(self.shape)

def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

class ConvAE(nn.Module):
    def __init__(self, latent_dim):
        super(ConvAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0, True),
            nn.MaxPool2d(4),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0,True),
            nn.MaxPool2d(4),
            
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0,True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0,True),
            
            nn.Flatten(),
            nn.Linear(8*8*256, latent_dim), nn.LeakyReLU(0,True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8*8*256), nn.LeakyReLU(0,True),
            Reshape(-1, 256, 8, 8),
            nn.ConvTranspose2d(256, 128, 1, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0,True),
            
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0,True),
            
            nn.ConvTranspose2d(64, 32, 4, stride=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0,True),
        
            nn.ConvTranspose2d(32, 3, 4, stride=4),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_pred = self.decoder(z)
        return x_pred, z

def calc_loss(model, x, loss_func):
    x = get_torch_vars(x)
    x_pred, z = model(x)
    loss = loss_func(x, x_pred)
    return loss, x_pred, z
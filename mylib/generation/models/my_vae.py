import torch

from torch import nn
from torch.autograd import Variable


class ConvVAE(nn.Module):
    def __init__(self, latent_dim):
        super(ConvVAE, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0, True),
            nn.MaxPool2d(4)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0,True),
            nn.MaxPool2d(4)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0,True),
            nn.MaxPool2d(2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0,True),
        )
            
        self.mean = nn.Sequential(
            nn.Linear(256*8*8, latent_dim),
        )
        
        self.var = nn.Sequential(
            nn.Linear(256*8*8, latent_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256*8*8),
            nn.BatchNorm1d(256*8*8),
            nn.LeakyReLU(0,True),
        )
        
        self.convTrans1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 1, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0,True),
        )
        
        self.convTrans2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0,True),
        )
        
        self.convTrans3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0,True),
        )
        
        self.convTrans4 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 4, stride=4),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )
        

    def _sample_z(self, mean, var):
        std = var.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mean)
    
    
    def _encoder(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 256*8*8)
        mean = self.mean(x)
        var = self.var(x)
        return mean, var
    
    def _decoder(self, z):
        z = self.decoder(z)
        z = z.view(-1, 256, 8, 8)
        x = self.convTrans1(z)
        x = self.convTrans2(x)
        x = self.convTrans3(x)
        x = self.convTrans4(x)
        return x
  
    def forward(self, x):
        mean,var = self._encoder(x)
        z = self._sample_z(mean, var)
        x = self._decoder(z)
        return x, mean, var, z
    
    def loss(self, x):
        mean, var = self._encoder(x)
        KL = -0.5 * torch.mean(torch.sum(1 + var - mean**2 - var.exp()))
        z = self._sample_z(mean, var)
        y = self._decoder(z)
        delta = 1e-7
        reconstruction = torch.mean(torch.sum(x * torch.log(y+delta) + (1 - x) * torch.log(1 - y + delta)))
        lower_bound = [-KL, reconstruction]
        print(f"\rKLdivergence:{KL} / Reconstruction:{reconstruction}", end='')
        return -sum(lower_bound), y, mean, var, z
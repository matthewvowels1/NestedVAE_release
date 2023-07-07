# adapted from https://github.com/IouJenLiu/Variational-Autoencoder-Pytorch/blob/master/vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, in_size, h_size=32, n_layers=4, dropout_rate=0.3, latent_size=8):
        super(Encoder, self).__init__()

        self.latent_size = latent_size
        self.fc1 = nn.Linear(in_size, h_size)

        layers = []
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(h_size, h_size))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.LeakyReLU())

        self.enc_layers = nn.Sequential(*layers)

        self.fc_last = nn.Linear(h_size, latent_size*2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.enc_layers(x)
        mu_sig = self.fc_last(x)
        mu = mu_sig[:, :self.latent_size]
        log_sigma = mu_sig[:, self.latent_size:]
        return mu, log_sigma


class Decoder(nn.Module):
    def __init__(self, out_size, h_size=32, n_layers=4, dropout_rate=0.3, latent_size=8):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, h_size)
        layers = []
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(h_size, h_size))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.LeakyReLU())

        self.dec_layers = nn.Sequential(*layers)
        self.fc_last = nn.Linear(h_size, out_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dec_layers(x)
        out = self.fc_last(x)
        return out



def sample_z(args):
    mu, log_sigma, device = args
    eps = Variable(torch.randn(mu.size())).to(device)
    return mu + torch.exp(log_sigma / 2) * eps

class innerVAE(nn.Module):
    def __init__(self, in_size, h_size, latent_size, n_layers=4, dropout_rate=0.3, device='cpu'):
        super(innerVAE, self).__init__()

        self.device = device
        self.encoder = Encoder(in_size=in_size,
                               h_size=h_size,
                               n_layers=n_layers,
                               dropout_rate=dropout_rate,
                               latent_size=latent_size)

        self.decoder = Decoder(out_size=in_size,
                               h_size=h_size,
                               n_layers=n_layers,
                               dropout_rate=dropout_rate,
                               latent_size=latent_size)
    def forward(self, x):
        mu, log_sigma = self.encoder(x)
        z = sample_z([mu, log_sigma, self.device])
        return self.decoder(z), mu, log_sigma


class outerVAE(nn.Module):
    def __init__(self, in_size, h_size, latent_size, n_layers=4, dropout_rate=0.3, device='cpu'):
        super(outerVAE, self).__init__()

        self.device = device
        self.encoder = Encoder(in_size=in_size,
                               h_size=h_size,
                               n_layers=n_layers,
                               dropout_rate=dropout_rate,
                               latent_size=latent_size)

        self.decoder = Decoder(out_size=in_size,
                               h_size=h_size,
                               n_layers=n_layers,
                               dropout_rate=dropout_rate,
                               latent_size=latent_size)

    def forward(self, x):
        mu, log_sigma = self.encoder(x)
        z = sample_z([mu, log_sigma, self.device])
        return self.decoder(z), mu, log_sigma


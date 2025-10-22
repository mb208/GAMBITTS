from torch.utils.data import DataLoader, Dataset
import torch
import torch.utils
from torch import nn

import numpy as np
import random
import time
import pandas as pd

from transformers import BertTokenizer, BertModel
from tqdm import tqdm



# https://github.com/XMUBQ/dtr-text/blob/main/models/backbone/VAE.py


class VAE(nn.Module):
    def __init__(self, input_dim, n_components):
        super(VAE, self).__init__()
        self.encoder_mean = nn.Linear(input_dim, n_components)
        self.encoder_var = nn.Linear(input_dim, n_components)
        self.decoder = nn.Linear(n_components, input_dim)

    @staticmethod
    def reparameterize(z_mean, z_logvar):
        epsilon = torch.randn_like(z_logvar)
        return epsilon * ((0.5 * z_logvar).exp()) + z_mean

    def encode(self,x):
        encoded_mean = self.encoder_mean(x)
        encoded_var = self.encoder_var(x)
        return encoded_mean, encoded_var

    def forward(self, x):
        encoded_mean, encoded_var = self.encode(x)
        encoded = self.reparameterize(encoded_mean, encoded_var)
        decoded = self.decoder(encoded)
        return encoded_mean, encoded_var, decoded
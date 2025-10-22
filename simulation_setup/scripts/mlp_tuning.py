import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from torchvision.ops import MLP

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import pandas as pd 
import numpy as np

# import mlxp
import hydra
from omegaconf import DictConfig, OmegaConf 

import os
from datetime import date

from modelling.mlp import NeuralNet
from utils import Trainer
### For testing 
import yaml 
from types import SimpleNamespace
from collections import namedtuple
import itertools

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader



class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Pass input through the first linear layer and apply ReLU
        x = F.relu(self.fc1(x))
        # Pass the result through the second linear layer for output
        x = self.fc2(x)
        return x


def train(model, loader, optimizer, criterion):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb).view(-1).to(torch.float32), yb.view(-1).to(torch.float32))
        loss.backward()
        optimizer.step()

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for xb, yb in loader:
            total_loss += criterion(model(xb), yb).item()
    return total_loss / len(loader)

@hydra.main(version_base=None, config_path='../configs', config_name="tuning_agent_mlp")
def tune_mlp(cfg : DictConfig) -> None:
   
    dataset = pd.read_csv(cfg.dataset)

    config_file = open(cfg.agent_config, 'r')
    agent_cfg =  yaml.load(config_file, Loader=Loader)
    
    batch_size = agent_cfg['batch_size']

    dimensions = list(agent_cfg['covariates'].keys())

    X = dataset[dimensions].values
    y = dataset[cfg.outcome].values

    input_size = X.shape[1]
    output_size = y.shape[1] if len(y.shape) > 1 else 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)  # Change dtype to `torch.float32` for regression

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset   = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    


    hidden_sizes = [32, 64, 128]
    learning_rates = [1e-4, 1e-3, 1e-2]
    epochs = 20

    tuning_config = namedtuple('best_config',['hidden_size', 'lr'])
    best_loss = float('inf')

    for hidden_size, lr in itertools.product(hidden_sizes, learning_rates):
        model = MLP(input_size, hidden_size, output_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            train(model, train_loader, optimizer, criterion)

        val_loss = evaluate(model, val_loader, criterion)
        print(f"hidden_size={hidden_size}, lr={lr:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_config = tuning_config(hidden_size, lr)


    agent_cfg['learning_rate'] = best_config.lr
    agent_cfg['hidden_dimension'] = best_config.hidden_size

    agent_cfg['base_config_path'] = cfg.agent_config
    agent_cfg['run_date'] = date.today().strftime("%Y-%d-%m")

    file_path = "configs/tuned_mlp4ensemble.yaml"
    with open(file_path, 'w') as f:
        yaml.dump(agent_cfg, f, default_flow_style=False)
    

if __name__ == "__main__":
    tune_mlp()
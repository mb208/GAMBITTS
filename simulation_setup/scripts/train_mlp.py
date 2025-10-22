import torch
import torch.nn as nn
import torch.optim as optim
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

from modelling.mlp import NeuralNet
from torchvision.ops import MLP
from utils import Trainer
### For testing 
import yaml 
from types import SimpleNamespace

from utils import Trainer

# with open("configs/config.yaml", "r") as f:
#     cfg = yaml.safe_load(f)

# cfg = SimpleNamespace(**cfg)
# cfg.env_nn = SimpleNamespace(**cfg.env_nn)

@hydra.main(version_base=None, config_path='../configs', config_name="config")
def train_mlp(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    criteria = {
        'mse' : nn.MSELoss
    }

    optimizers = {
        'Adam' : optim.Adam
    }


    activation_functions = {
        'ReLU': nn.ReLU,
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh,
        'LeakyReLU': nn.LeakyReLU
    }
    
    
    dimensions = list(cfg.env_setup.dimensions)
    outcome = cfg.env_setup.outcome
    dataset = pd.read_csv(cfg.env_setup.data_path)

    X = dataset[dimensions].values
    y = dataset[outcome].values

    input_size = X.shape[1]
    output_size = y.shape[1] if len(y.shape) > 1 else 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)  # Change dtype to `torch.float32` for regression

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset   = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=cfg.env_setup.batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=cfg.env_setup.batch_size, shuffle=False)
    

    # Define model, loss function, and optimizer
    model = NeuralNet(input_size=input_size, hidden_sizes = cfg.env_setup.hidden_size,
                      output_size=output_size, act_fn = cfg.env_setup.activation)
    

    criterion = criteria[cfg.env_setup.loss_fn](reduction="mean")
    optimizer = optimizers[cfg.env_setup.optimizer](model.parameters(), float(cfg.env_setup.lr))

    # Initialize trainer
    trainer = Trainer(model, criterion, optimizer,device = cfg.env_setup.device)

    # Train the model
    trainer.train(train_loader, val_loader, epochs=cfg.env_setup.epochs)

    # Evaluate on validation set
    val_loss = trainer.evaluate(val_loader)

    # Evaluate on training set
    train_loss = trainer.evaluate(val_loader)

    file_path = os.path.join(cfg.env_setup.save_path, cfg.env_setup.model_id + '_' + cfg.env_setup.model_suffix)
    print(f"Saving model to {file_path}")
    trainer.save(file_path)

    cfg_dict = {
        'type' : 'nn',
        'dimensions': dimensions, 
        'model_path': file_path,
        'online_calc_necessary': False,
        'errors' : {
            'iid': True,
            'dist': 'normal',
            'parms': {
                'sd' : float(np.sqrt(val_loss))
            }
        },
        "train_rmse" : float(np.sqrt(train_loss)),
        "training_data" : cfg.env_setup.data_path,
        "seed" : cfg.env_setup.seed,
        "dim_batch" : cfg.env_setup.dim_batch
    }

    file_path = os.path.join(cfg.env_setup.cfg_path, cfg.env_setup.model_id + "_mlp_outcome_model.yaml")
    with open(file_path, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)

if __name__ == "__main__":
    
    train_mlp()
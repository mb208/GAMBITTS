from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from tqdm import tqdm

import pandas as pd 
import numpy as np

# import mlxp
import hydra
from omegaconf import DictConfig, OmegaConf 
import os

### For testing 
import yaml 
from types import SimpleNamespace


@hydra.main(version_base=None, config_path='../configs', config_name="config")
def train_ols(cfg : DictConfig) -> None:
    print(cfg.env_setup)

    dimensions = list(cfg.env_setup.dimensions)
    n_dim = len(dimensions)

    dataset = pd.read_csv(cfg.env_setup.data_path)

    X = dataset[dimensions].values
    y = dataset[cfg.env_setup.outcome].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    ols = linear_model.LinearRegression()

    ols.fit(X_train,y_train)

    # get error sd 
    train_rmse = np.sqrt(mean_squared_error(y_train,ols.predict(X_train)))

    test_rmse = np.sqrt(mean_squared_error(y_test,ols.predict(X_test)))

    # dimension coeffs
    dim_coefs = { 
        dimensions[i]: {
        'offline_calc': True,
         'one_hot': False,
         'coef': float(ols.coef_[i])
        }
        for i in range(n_dim)
    }
    
    dim_coefs['intercept'] = {
        'offline_calc': True,
        'one_hot': False,
        'coef': float(ols.intercept_)
    }

    cfg_dict = {
        'type' : 'linear',
        'past_contribution': False,
        'coefs': dim_coefs,
        'errors' : {
            'iid': True,
            'dist': 'normal',
            'parms': {
                'sd' : float(test_rmse)
            }
        },
        'train_rmse' : float(train_rmse),
        'training_data': cfg.env_setup.data_path,
        'seed' : cfg.env_setup.seed,
        "dim_batch" : cfg.env_setup.dim_batch
    }

    file_path = os.path.join(cfg.env_setup.cfg_path, 
                             f"{cfg.env_setup.filename}.yaml")
    with open(file_path, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)

if __name__ == "__main__":
    train_ols()


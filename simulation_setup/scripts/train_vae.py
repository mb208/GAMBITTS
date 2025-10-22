from pythae.pipelines import TrainingPipeline
from pythae.models import VAE, VAEConfig
from pythae.trainers import BaseTrainerConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

import pandas as pd 
import numpy as np

# import mlxp
import hydra
from omegaconf import DictConfig, OmegaConf

import os

from utils import VAEDataset 

# @mlxp.launch(config_path='../configs')

# def train_vae(ctx: mlxp.Context) -> None:

@hydra.main(version_base=None, config_path='../configs', config_name="config")
def train_vae(cfg : DictConfig) -> None:
    # cfg = ctx.config
    # logger = ctx.logger
    print(cfg)
    
    # Load data
    print('Load Dataset\n')
    text_dataset = VAEDataset(
        data_folder = cfg.vae.data_folder, # 'data/simulations/' 
        file =  cfg.vae.filename, 
        text_column="text")
    
    input_dim = text_dataset.text_embeddings.shape[1]
    
    output_dir = os.path.join('models/vae_model', cfg.vae.style_dim)
    # Define vae config    
    my_training_config = BaseTrainerConfig(
        output_dir=  output_dir, # 
        num_epochs=  cfg.vae.epochs, # 100
        learning_rate= cfg.vae.lr, #1e-4, 
        per_device_train_batch_size=200,
        per_device_eval_batch_size=200,
        train_dataloader_num_workers=2,
        eval_dataloader_num_workers=2,
        steps_saving=20,
        optimizer_cls="AdamW",
        optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.995)},
        )


    model_config = VAEConfig(
        input_dim=(1, input_dim),
        latent_dim=1
    )

    model = VAE(
        model_config=model_config
    )


    # Build the Pipeline
    pipeline = TrainingPipeline(
        training_config=my_training_config,
        model=model
        )
    
    
    # Test train split the dataset
    # if cfg.vae.eval_size is not None:
    #     text_train, text_test = train_test_split(text_dataset.text_embeddings, 
    #                                              test_size=cfg.vae.eval_size)
        
    #     # Train the model
    #     pipeline(
    #         train_data=text_train,
    #         eval_data=text_test
    #         )
    # else:
    #     # Train the model
    #     pipeline(train_data=text_dataset.text_embeddings)
    
    print('Train vae\n')
    pipeline(train_data=text_dataset.text_embeddings)
    


if __name__ == "__main__":
    
    train_vae()

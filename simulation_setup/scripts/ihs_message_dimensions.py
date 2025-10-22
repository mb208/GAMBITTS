from pythae.pipelines import TrainingPipeline
from pythae.models import VAE, VAEConfig
from pythae.trainers import BaseTrainerConfig
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

import pandas as pd 
import os
import numpy as np

from utils import VAEDataset
import torch

def get_most_recent_folder(directory):
    # Get all subdirectories
    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    
    # If no subfolders exist, return None
    if not subfolders:
        return None

    # Get the most recently modified subfolder
    most_recent_subfolder = max(subfolders, key=os.path.getmtime)
    
    return most_recent_subfolder



@hydra.main(version_base=None, config_path='../configs', config_name="config")
def ihs_message_dim(cfg : DictConfig) -> None:
    print(cfg.ihs)
    data_path = cfg.ihs.data_path
    filename = cfg.ihs.filename
    
    
    # vae prefix
    # save path     
    models = os.listdir(cfg.ihs.model_prefix)
    
    # prepare database for VA
    ihs_dataset = VAEDataset(
        data_folder = cfg.ihs.data_path,
        file =  cfg.ihs.filename, 
        text_column= cfg.ihs.text_col) # "llm_text")
    
    for style_dim in models:
            
            print(f"Processing {style_dim}")

            model_path = os.path.join(cfg.ihs.model_prefix, style_dim)
             # Take most recent vae model trained. Maybe this should be handled differently?
            model_path = get_most_recent_folder(model_path) 
            
            print(model_path)
            # load vae
            vae = VAE.load_from_folder(
                os.path.join(model_path, cfg.ihs.model_suffix)
                )
            
            vae.eval()
            with torch.no_grad():
                ihs_dataset.data[style_dim] = vae.encoder(ihs_dataset.text_embeddings)['embedding'].numpy() 
        
    
    
    message_path = os.path.join(cfg.ihs.save_folder, str(cfg.ihs.year), "message_dimensions.csv")
    final_data_path = os.path.join(cfg.ihs.save_folder, str(cfg.ihs.year), "ihs_complete.csv")
    print("Saving IHS messages with VAE embeddings to {message_path}\n")
    ihs_dataset.data.to_csv(message_path, index=False)
    
    imputed_data = pd.read_csv(cfg.ihs.imputed_data)

    ihs_dataset.data.columns = ihs_dataset.data.columns.str.lower().str.replace(" ", "")
    ihs_dataset.data = ihs_dataset.data[['notificationidentifier'] + models]

    print("\nMerging notifications sent with latent dimensions.")
    imputed_data = imputed_data.merge(ihs_dataset.data , on='notificationidentifier', how='inner')

    print("Saving IHS messages with VAE embeddings to {final_data_path}\n")
    imputed_data.to_csv(final_data_path, index=False)
    
    


if __name__ == "__main__":
    ihs_message_dim()
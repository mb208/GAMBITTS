"""
This script generates VAE scores for prompt database create in prompts4sim.py
Created by: Marc Brooks (MGB)
Date: 11/2024
Edit History: 5/21/2025 (MGB) - Changed the config file that this points to.
"""
from pythae.pipelines import TrainingPipeline
from pythae.models import VAE, VAEConfig
from pythae.trainers import BaseTrainerConfig
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime


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
def create_prompt_db(cfg : DictConfig) -> None:
    print(cfg.prompt_sim)
    data_path = cfg.prompt_sim.sim_path
    llm_name = cfg.prompt_sim.llm_type
    
    dimensions = cfg.prompt_sim.dimension

    model_prefix = cfg.prompt_sim.model_prefix
    model_suffix = cfg.prompt_sim.model_suffix


    df = pd.DataFrame([])
    for dim in dimensions:
        dim_filename =  dim + '_prompts.csv'
    
        # prepare database for VA
        print(dim_filename)
        prompt_dataset = VAEDataset(
            data_folder = data_path,
            file =  dim_filename, 
            text_column= cfg.prompt_sim.text_col) # "llm_text")
            
        print(f"Processing {dim}")
        

        for dim_2 in dimensions:
            # load vae
            vae_folder = os.path.join(cfg.prompt_sim.model_prefix, dim_2)
            vae_path = get_most_recent_folder(vae_folder)
            vae = VAE.load_from_folder(os.path.join(vae_path, cfg.prompt_sim.model_suffix))  #
            
            vae.eval()
            with torch.no_grad():
                prompt_dataset.data[dim_2] = vae.encoder(prompt_dataset.text_embeddings)['embedding'].numpy() 

        
        df = pd.concat([df,  prompt_dataset.data[['prompt_id','stepsprevday','currloc'] + dimensions]], ignore_index=True)
        

    
    df = (df.rename({'stepsprevday' : 'prev_steps', 
                  'currloc' : 'curr_loc', 
                  'prompt_id' : 'prompt'}, axis=1))

    df['curr_loc'] = df['curr_loc'].replace({'a location other than home or work' : 'other'})
    df['prev_steps'] = (df['prev_steps'].replace({'0-4,999' : 1, 
                                                   '5,000-9,999' : 2,
                                                   '10,000-15,000':3,
                                                   'more than 15,000' : 4 })
                                         )
    if df.prompt is not int:
        df['prompt_id'] = df['prompt']
        df['prompt'], _ = pd.factorize(df['prompt_id'])
        df['prompt'] = df['prompt'] +  1

    save_path = os.path.join(cfg.prompt_sim.save_path, f"{llm_name}_prompt_db_{len(dimensions)}dim.csv.gzip") 
    print(f"Saving prompt db with VAE embeddings to {save_path}\n")
    df.to_csv(save_path, index=False, compression='gzip')
    
    


if __name__ == "__main__":
    create_prompt_db()
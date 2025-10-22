"""
This script create text interventions used to train VAE models for the simulation.
Created by: Marc Brooks (MGB)
Date: 11/2024
Edit History: 5/21/2025 (MGB) - Changed the config file that this points to.
"""

import transformers 
from transformers import pipeline
import torch
import mlxp
import hydra
from omegaconf import DictConfig, OmegaConf
import ollama

import re
import numpy as np
import pandas as pd
import os
import time
import datetime

import utils


@hydra.main(version_base=None, config_path='../configs', config_name="config")
def sample_interventions(cfg : DictConfig) -> None:

    # Retrieving configuration parameters
    message_type = cfg.text_sim.style_dim # 'optimism'  
    style_path = cfg.text_sim.style_path # 'data/sim_params/Style Dimensions.xlsx - Raw.csv' 
    sim_path = cfg.text_sim.sim_path #'data/simulations' 
    n_messages = cfg.text_sim.n_messages # 2000 #cfg['n_messages']
    llm_type = cfg.text_sim.llm_type
    
    
    
    print(cfg)
    
    # Load model
    os.system("ollama serve > /dev/null 2>&1 &")
    
    print("Wait 5 seconds before trying to connect\n")
    time.sleep(5)
    
    # print(ollama.list())

    ollama.create(model='jitai' + llm_type, modelfile=utils.modelfile[llm_type])

    
    
    interventions = list()    
    x_stepsprevday = np.random.choice(["0-4,999", "5,000-9,999", "10,000-15,000", "more than 15,000"],
                                    size = n_messages)
    x_currloc = np.random.choice(["home", "work", "a location other than home or work"],
                               size = n_messages)

    # SAVE FILE FOR MESSAGES
    curr_date = datetime.date.today().strftime("%Y_%m_%d")
    if not os.path.exists(os.path.join(sim_path, llm_type + '_' + curr_date)):
        os.makedirs(os.path.join(sim_path, llm_type + '_' + curr_date))

    save_path = os.path.join(sim_path, llm_type + '_' + curr_date, f"{message_type}_interventions_{n_messages}.csv")
    # File for potential badly formatted messages
    discard_path = os.path.join(sim_path, llm_type + '_' + curr_date,  f"{message_type}_discards.txt")
    print(f"Initializing interventions file to {save_path}\n")
    interventions_df = pd.DataFrame(columns=['text', 'rating','stepsprevday', 'currloc'])
    interventions_df.to_csv(save_path, index=False)

    n_discards = 0
    print("Starting text generation\n")
    start_time = time.time()
    for i in range(n_messages):
        if i % 100 == 0:
            print(f"Generating message {i} of {n_messages}\n")


        llama_resp = ollama.generate(model="jitai", prompt = utils.ollama_prompt(x_stepsprevday[i], x_currloc[i], message_type)) 

        messages = [txt.strip() for txt in re.split(r'\d{1,2}\.', re.sub(r"\n{1,2}", "" ,llama_resp['response'])) if txt != '']

        print(f"Counting {len(messages)} messages.\n")
        if len(messages) == 11:
            pd.DataFrame({
                "text": messages,
                "rating": np.arange(0,11),
                "stepsprevday": x_stepsprevday[i],
                "currloc": x_currloc[i],
                },
                columns=['text', 'rating','stepsprevday', 'currloc']).to_csv(save_path, mode='a', header=False, index=False)

        else:
            with open(discard_path, "a") as f:
                f.write(llama_resp['response'])
                n_discards +=1

    end_time = time.time()
    print(f"Done generating messages. Elapsed time = {(end_time-start_time)/(60*60)} hours.\n")
    print(f"There were {n_discards} discarded iterations")
    
        

if __name__ == '__main__':
    sample_interventions()

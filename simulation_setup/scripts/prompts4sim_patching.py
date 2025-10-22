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
import itertools
import utils
from datetime import datetime




"""
This script generates samples of LLM-generatd treatments for all context combinations.
Created by: Marc Brooks (MGB)
Date: 11/2024
Edit History: 5/21/2025 (MGB) - Changed the config file that this points to.
"""

@hydra.main(version_base=None, config_path='../configs', config_name="config")
def sim_prompts(cfg : DictConfig) -> None:
   
    n_prompts = 1000  # 662
    dimension = "humor" # "threat" 
    llm_type = cfg.prompt_sim.llm_type

    sim_path = cfg.prompt_sim.sim_path
    model_name = 'jitai_' + dimension

    
    print(cfg)
    
    # Load model
    os.system("ollama serve > /dev/null 2>&1 &")
    
    print("Wait 5 seconds before trying to connect\n")
    time.sleep(5)
    
    print(ollama.list())

    # modelfile_sim is a dictionary in utils.py that maps llm_type to the model file (e.g. system prompt)
    ollama.create(model= model_name, modelfile=utils.modelfile_sim[llm_type])

    
    print(ollama.list())

    prompt_info = utils.prompt_template[dimension]
    dim_direction = ['neg']
    # steps =  "more than 15,000" # 10,000-15,000
    # loc = "other"
    
    
    # steps = ["0-4,999", "5,000-9,999", "10,000-15,000", "more than 15,000"]
    steps = ["more than 15,000"]
    currlocs = ["a location other than home or work"]
    
    
    # steps = ["more than 15,000"]
    
    prmpt_vls = itertools.product(dim_direction, steps, currlocs)
        
    # SAVE FILE FOR MESSAGES
    today_str = datetime.today().strftime('%Y%m')
    dir_path = os.path.join(sim_path, today_str)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    save_path = os.path.join(dir_path, dimension + '_prompts.csv')
    
    interventions_df = pd.DataFrame(columns=['prompt_id', 'stepsprevday', 'currloc', 'llm_text'])
    interventions_df.to_csv(save_path, index=False)

    print("Starting prompt db generation\n")
    start_time = time.time()
    for direction, steps, loc in prmpt_vls:
        prompt_id = direction + str(prompt_info['action_id'])
        print(prompt_id, steps, loc)
        print(f"Simulating responses for prompts from: prompt {prompt_id} with {steps} at {loc}\n")
    
        for i in range(n_prompts):
            llama_resp = ollama.generate(model = model_name, 
                                            prompt = utils.prompt_fn(steps,
                                                                    loc,
                                                                    dimension, 
                                                                    direction))
            
            pd.DataFrame({
                "prompt_id": [prompt_id],
                "stepsprevday": [steps],
                "currloc": [loc],
                "llm_text": [llama_resp['response']]
                },
                columns=['prompt_id', 'stepsprevday', 'currloc', 'llm_text']).to_csv(save_path, mode='a', 
                                                                                        header=False, index=False)

    end_time = time.time()
    print(f"Done generating messages. Elapsed time = {(end_time-start_time)/(60*60)} hours.\n")
    

if __name__ == '__main__':
    sim_prompts()

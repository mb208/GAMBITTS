import numpy as np 
import pandas as pd 


import os 
import re 
import sys 

import yaml
import utils


try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
    

# input

date = '2024_12_30'
tones = ['actionability', 'authenticity', 'authoritativeness', 'conciseness',
         'emotiveness', 'formality', 'politeness']
# tone = utils.tone_prompts.keys()
with open('configs/config.yaml', 'r') as f:
    sim_params = yaml.load(f, Loader=Loader)


n_messages = sim_params['text_sim']['n_messages'] 
sim_path = sim_params['text_sim']['sim_path'] 


for style in tones:
    with open(f'{sim_path}/{date}/{style}_discards.txt', 'r') as f:
        lines = f.readlines()
   
    ratings = []
    texts = []
    mismatch = []

    for line in lines:
        if re.match(r"^\d{1,2}[^\n]", line) is not None:
            # print(line)
            rating = re.match(r"^\d{1,2}", line).group()
            text = re.split(r"\d{1,2}", line, 1)[1]
            ratings.append(rating)
            texts.append(text)
        else:
            mismatch.append(line)
    
    ratings = np.array(ratings).astype(int)
    texts = np.array(texts)
    texts = texts[np.where(np.array(ratings).astype(int) <= 11)]
    ratings = ratings[np.where(np.array(ratings).astype(int) <= 11)]
    
    print(f"Number of added messages for {style}: {len(texts)}")
    
    text_df = pd.DataFrame({
                "text": texts,
                "rating": ratings-1,
                "stepsprevday": None,
                "currloc": None,
                },
                columns=['text', 'rating','stepsprevday', 'currloc'])
    
    text_df.to_csv(f'{sim_path}/{date}/{style}_interventions_{n_messages}.csv',
                    mode='a', header=False, index=False)

    

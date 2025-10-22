"""
Implementation of poNATS with ensemble learning: 


Author: Ki Hong (KH)
Created: 3/25
Edit History:
    - 5/12/25:  (MGB) Made the following changes 
                 - Doesn't update batch until 100 observations have been made (Line 125)
                 - Samples batchs with replacement  (Line 62)
                 - Changing all sampling to come from numpy:
                     Class takes in seed that is then given to create an rng object and is passed to any method that needs to sample
                 - Changed to use torch tensors instead of numpy arrays
    - 5/19/25:  (MGB) Added PerturbedEnsembleSampling class that adds noise to the reward signal
    - 5/22/25:  (MGB) Changing initialization of the weights
    - 10/13/25: (MGB) Weight initialization can be specified as Normal with inputed sd, otherwise it is default pytorch init (uniform)
"""

from src.agents.agent_base import AgentBase
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque, defaultdict

TEXT_CONTEXT_COLUMNS = ['curr_loc', 'prev_steps']
PROMPT_COLUMN = 'prompt'

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,init_sd=None):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        if init_sd:
            self._init_weights(init_sd)

    def _init_weights(self, init_sd):
        nn.init.normal_(self.fc1.weight, mean=0.0, std=init_sd)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=init_sd)
        nn.init.normal_(self.fc1.bias, mean=0.0, std=init_sd)
        nn.init.normal_(self.fc2.bias, mean=0.0, std=init_sd)

    def forward(self, x):
        # Pass input through the first linear layer and apply ReLU
        x = F.relu(self.fc1(x))
        # Pass the result through the second linear layer for output
        x = self.fc2(x)
        return x

class FeatureComputer:
    def __init__(self, config):
        self.config = config

    def compute_feature(self, context, action):
        return [1., 2, 3, 4]

    def get_dimension(self):
        return 4

    def get_action_space(self):
        return [0, 1, 2, 3]

class ReplayBuffer:
    def __init__(self, capacity, rng=None):
        if rng:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()
            
        self.capacity = capacity
        self.buffer = deque()

    # Marc Added Noise to be pus
    def push(self, feature, reward, noise):
        self.buffer.append((feature, reward, noise))
        if len(self.buffer) > self.capacity:
            self.buffer.popleft()

    def sample(self, batch_size):
        indices = self.rng.integers(0, len(self.buffer), size=batch_size) # Marc changed to use rng, np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in indices] 
        # batch = np.random.choice(self.buffer, size=batch_size, replace=False)
        features, rewards, noise = zip(*batch)

        # Convert to numpy arrays for convenience
        # 5/19/25: (MGB) Changed to return indices for petrubation of rewards in updates
        #                Changed these all to be turned into np arrays before tensors for efficiency
        return (
            torch.tensor(np.array(features), dtype=torch.float32),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(noise), dtype=torch.float32)
        )

    def __len__(self):
        """
        Returns the current size of the replay buffer.
        """
        return len(self.buffer)

class EnsembleSampling(AgentBase):
    def __init__(self, prompt_db, config, rng=None):
        # (Marc): Initialize the random number generator . 
        if rng:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()
            
        self.config = config
        self.M = self.config['ensemble_size']

        self.dimension_columns = [
            covariate for covariate in config['covariates'].keys()
                if config['covariates'][covariate]['text_based']
        ]

        self.representation_distribution = \
            self.__get_representation_distribution(prompt_db)
        self.action_space = self.__get_action_space(prompt_db)
        self.d = len(self.dimension_columns)

        self.decision_t = 0 # Keep track of time step

        # initialize neural network
        self.networks = [
            MLP(self.d, config['hidden_dimension'], 1) for _ in range(self.M)
        ]

        self.replay_buffer = ReplayBuffer(config['replay_buffer_capacity'], rng=self.rng) # Marc changed to use rng
        self.feature_computer = FeatureComputer(config)

    def sample_feature(self, context, action):
        text_context = tuple(context[c] for c in TEXT_CONTEXT_COLUMNS)
        text_representation = self.text_representation[(text_context, action)]
        return np.array(text_representation)

    def action(self, context):
        i = self.rng.integers(0, self.M).item() # Marc changed to use rng, from np.random.randint(0, self.M)
        max_reward = -np.inf
        max_action = None
        if self.decision_t < 100:
            
            # Select uniformly until enough data is collected
            # Update decison counter
            self.decision_t += 1

            return self.rng.choice(list(self.action_space), size=1).item() # Marc changed to use rng, from np.random.choice(list(self.action_space), size=1).item()
        else:
            for action in self.action_space:
                text_context = tuple(context[c] for c in TEXT_CONTEXT_COLUMNS)
                distribution = self.representation_distribution[(text_context, action)]
                num_samples = self.config['num_monte_carlo_samples']
                # reward = 0
                # for z in np.random.choice(distribution, size=num_samples, replace=True):
                #     reward += self.networks[i](torch.tensor(z, dtype=torch.float32))
                # reward /= num_samples
                sample_idx = self.rng.choice(len(distribution), size=num_samples, replace=True) # Marc changed to use rng, from np.random.choice(len(distribution), size=num_samples, replace=True)
                self.networks[i].eval()
                with torch.no_grad():
                    z_batch = np.array(distribution)[sample_idx]
                    reward = self.networks[i](torch.tensor(z_batch, dtype=torch.float32)).mean().item()

                if reward > max_reward:
                    max_reward = reward
                    max_action = action

            # Update decison counter
            self.decision_t += 1

            return max_action

    def update(self, context, action, text_representation, reward):
        representation = [text_representation[col] for col in self.dimension_columns]
        self.replay_buffer.push(representation, reward)
        if len(self.replay_buffer) < 100:
            return
        x, y,_ = self.replay_buffer.sample(self.config['batch_size'])
        criterion = nn.MSELoss()
        for network in self.networks:
            optimizer = optim.Adam(network.parameters(), lr=self.config['learning_rate'])
            y_hat = network(x).flatten()
            loss = criterion(y, y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    @staticmethod
    def __get_action_space(prompt_db):
        # action_space = set()
        # for _, row in prompt_db.iterrows():
        #     action_space.add(row[PROMPT_COLUMN])
        action_space = prompt_db[PROMPT_COLUMN].unique().tolist()
        return action_space

    def __get_representation_distribution(self, prompt_db):
        distribution = defaultdict(list)
        for _, row in prompt_db.iterrows():
            context = tuple([row[col] for col in TEXT_CONTEXT_COLUMNS])
            action = row[PROMPT_COLUMN]
            representation = [row[col] for col in self.dimension_columns]
            distribution[(context, action)].append(representation)
        return distribution
    
class PerturbedEnsembleSampling(AgentBase):
    def __init__(self, prompt_db, config, noise=1, rng=None):
        # (Marc): Initialize the random number generator . 
        if rng:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()
            
        self.config = config
        self.M = self.config['ensemble_size']
        
        self.weight_init_sd = None
        if 'weight_init_sd' in config:
            self.weight_init_sd = config['weight_init_sd']

        # Marc: Added noise parameter used for perturbations in training
        self.noise = noise
        
        

        self.dimension_columns = [
            covariate for covariate in config['covariates'].keys()
                if config['covariates'][covariate]['text_based']
        ]

        self.representation_distribution = \
            self.__get_representation_distribution(prompt_db)
        self.action_space = self.__get_action_space(prompt_db)
        self.K = len(self.action_space)
        self.d = len(self.dimension_columns)
        
        if self.config.get('use_prompts'):
            self.d += self.K-1

        self.decision_t = 0 # Keep track of time step

        # initialize neural network
        self.networks = [
            MLP(self.d, config['hidden_dimension'], 1, init_sd=self.weight_init_sd) for _ in range(self.M)
        ]

        self.replay_buffer = ReplayBuffer(config['replay_buffer_capacity'], rng=self.rng) # Marc changed to use rng
        self.feature_computer = FeatureComputer(config)

    def sample_feature(self, context, action):
        text_context = tuple(context[c] for c in TEXT_CONTEXT_COLUMNS)
        text_representation = self.text_representation[(text_context, action)]
        return np.array(text_representation)

    def action(self, context):
        i = self.rng.integers(0, self.M).item() # Marc changed to use rng, from np.random.randint(0, self.M)
        max_reward = -np.inf
        max_action = None
        if self.decision_t < 100:
            
            # Select uniformly until enough data is collected
            # Update decison counter
            self.decision_t += 1

            return self.rng.choice(list(self.action_space), size=1).item() # Marc changed to use rng, from np.random.choice(list(self.action_space), size=1).item()
        else:
            for action in self.action_space:
                text_context = tuple(context[c] for c in TEXT_CONTEXT_COLUMNS)
                distribution = self.representation_distribution[(text_context, action)]
                num_samples = self.config['num_monte_carlo_samples']
                # reward = 0
                # for z in np.random.choice(distribution, size=num_samples, replace=True):
                #     reward += self.networks[i](torch.tensor(z, dtype=torch.float32))
                # reward /= num_samples
                sample_idx = self.rng.choice(len(distribution), size=num_samples, replace=True) # Marc changed to use rng, from np.random.choice(len(distribution), size=num_samples, replace=True)
                self.networks[i].eval()
                with torch.no_grad():
                    z_batch = np.array(distribution)[sample_idx]
                    reward = self.networks[i](torch.tensor(z_batch, dtype=torch.float32)).mean().item()

                if reward > max_reward:
                    max_reward = reward
                    max_action = action

            # Update decison counter
            self.decision_t += 1

            return max_action

    def update(self, context, action, text_representation, reward):
        representation = [text_representation[col] for col in self.dimension_columns]

        noise = self.rng.normal(0, self.noise, size=self.M)

        self.replay_buffer.push(representation, reward, noise)

        if len(self.replay_buffer) < 100:
            return
        x, y, noise = self.replay_buffer.sample(self.config['batch_size'])
        criterion = nn.MSELoss()
        for ix, network in enumerate(self.networks):
            y_i = y + noise[ :,ix] # Adding in noise for particular neural net
            optimizer = optim.Adam(network.parameters(), lr=self.config['learning_rate'])
            y_hat = network(x).flatten()
            loss = criterion(y_i, y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    @staticmethod
    def __get_action_space(prompt_db):
        # action_space = set()
        # for _, row in prompt_db.iterrows():
        #     action_space.add(row[PROMPT_COLUMN])
        action_space = prompt_db[PROMPT_COLUMN].unique().tolist()
        return action_space

    def __get_representation_distribution(self, prompt_db):
        distribution = defaultdict(list)
        for _, row in prompt_db.iterrows():
            context = tuple([row[col] for col in TEXT_CONTEXT_COLUMNS])
            action = row[PROMPT_COLUMN]
            representation = [row[col] for col in self.dimension_columns]
            distribution[(context, action)].append(representation)
        return distribution



if __name__ == '__main__':
    import pandas as pd
    import yaml

    from Environment import Environment
    from simulator import BaseSimulator, TwoStageSimulator

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    yaml_parms = {}
    context = open("./configs/poster_context.yaml", 'r')
    yaml_parms["context"] = yaml.load(context, Loader=Loader)
    outcome_model = open("./configs/report_outcome_model_3comps.yaml", 'r')
    yaml_parms["outcome_model"] = yaml.load(outcome_model, Loader=Loader)

    config_file = open("configs/ensemble_sampling.yaml", 'r')
    config = yaml.load(config_file, Loader=Loader)

    prompt_db = pd.read_csv("data/sim_params/prompt_db_w_vae.csv")

    env = Environment(prompt_db, yaml_parms)

    K = prompt_db.prompt.nunique()
    # Remove context / prompt columns to get number of dimensions in db
    d = len(set(prompt_db.columns) - set(['prompt', 'prev_steps', 'curr_loc']))
    agent = EnsembleSampling(prompt_db, config)

    simulator = TwoStageSimulator(agent =  agent, env  = env, n_steps = 100, unique_id = 1)

    simulator.run_experiment()

    print(simulator.results)
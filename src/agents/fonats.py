"""
Implementation of FONATS: Fully Online Noisy Action Thompson Sampling

At the moment only includes linear 2nd model.

Author: Ki Hong (KH)
Create: 1/25
Edit History:
    - 5/10/25: (KH) Adding intercept 
    - 5/11/25: (MGB) Need to change all feature representations to be forced to be 2d to handle concatentation with intercept
                Edits made to all functions computing features or representations.
                Required changing concatenation axis explicitly to be 1.
    - 7/25/25: (MGB) Added option for adding prompt to the model.
"""


from src.agents.agent_base import AgentBase
from collections import defaultdict
import numpy as np
from scipy.stats import invgauss, wishart, multivariate_normal
import pandas as pd

PROMPT_COLUMN = 'prompt'

class FONATS(AgentBase):
    def __init__(self, prompt_db, config, rng=None):
        
        # Initiatilize random number generator
        if rng:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()
            
        self.conf = config
        self.dimension_columns = [
            covariate for covariate in config['covariates'].keys()
                if config['covariates'][covariate]['text_based']
        ]
        self.personal_features = {}
        for covariate, data in config['covariates'].items():
            if data['text_based']:
                continue
            if data['var_type'] == 'discrete':
                self.personal_features[covariate] = data['possible_values']
            else:
                self.personal_features[covariate] = None
                
        
            
        self.action_space = self.__get_action_space(prompt_db)
        self.K = len(self.action_space)

        d0 = len(self.dimension_columns)
        d = len(self.dimension_columns)
        for k, v in self.personal_features.items():
            if v is None:
                d += 1
            else:
                d += len(v) + 1
        self.d = d
        
        if config.get('intercept'):
            self.d += 1
            
        if config.get('use_prompts'):
            self.d += self.K-1


        # variables for computing first-stage posterior

        # track sum_i x_i x_i^T for each (c, a) pair where x_i is text representation
        self.XXT = defaultdict(lambda: np.zeros((d0, d0)))

        # track sum_i x_i for each (c, a) pair where x_i is text representation
        self.X = defaultdict(lambda: np.zeros(d0))

        # track number of occurences for each (c, a) pair
        self.n = defaultdict(int)

        self.mu0 = np.zeros(d0)
        self.Psi0 = 1 * np.eye(d0)
        self.kappa = 1
        self.nu0 = 1

        self.Psi = defaultdict(lambda: self.Psi0)

        # TODO: move these parameters to config
        self.v2 = config["v2"] # Before: 1
        self.B = np.eye(self.d)*config["precision"] #  Before: np.eye(self.d)
        self.S = config["S"] # Before:  0
        self.mu = np.ones(self.d)*config["mu"] #  Before: np.zeros(self.d)
        
        if config.get('intercept'):
            self.B[-1, -1] = config["precision_intercept"]
            self.mu[-1] = config["mu_intercept"]

    def __compute_personal_features(self, row):
        features = []
        for context_key, categories in self.personal_features.items():
            if categories:
                category_feature = [0] * (len(categories) + 1) # last entry for "other"
                if row[context_key] in categories:
                    category_feature[categories.index(row[context_key])] = 1
                else:
                    category_feature[-1] = 1
                features += category_feature
            else:
                features.append(row[context_key])
        return features

    @staticmethod
    def __get_action_space(prompt_db):
        # action_space = set()
        # for _, row in prompt_db.iterrows():
        #     action_space.add(row[PROMPT_COLUMN])
        action_space = prompt_db[PROMPT_COLUMN].unique().tolist()
        return action_space

    def compute_feature(self, context, action):
        return self.__compute_personal_features(context)

    def action(self, context):
        mu = self.rng.multivariate_normal(self.mu, self.v2 * np.linalg.inv(self.B)) # Marc changed from np.random.multivariate_normal(self.mu, self.v2 * np.linalg.inv(self.B))
        max_reward = -np.inf
        max_action = None
        for action in self.action_space:
            personal_feature = self.compute_feature(context, action)
            text_feature = self.__sample_mean_vector(context, action)
            # feature = np.atleast_2d(
            #     np.concatenate(([text_feature], np.array(personal_feature)))
            #     if personal_feature else np.array([text_feature])
            # )
            feature = np.atleast_2d(
                np.concatenate([text_feature, np.array(personal_feature)])
                if personal_feature else np.array([text_feature])
                )
            if self.conf.get('use_prompts'):
                feature =  np.concatenate((feature,
                                           1*(np.array(self.action_space[1:self.K], ndmin=2) == action)), axis=1)

            if self.conf.get('intercept'):
                feature = np.concatenate((feature, np.array([1.0], ndmin=2)), axis=1)
            reward = feature.dot(mu)
            if reward > max_reward:
                max_reward = reward
                max_action = action
        return max_action

    def update(self, context, action, text_representation, reward):
        c = self.__encode_context(context)
        representation = np.atleast_2d(
            [text_representation[covariate] for covariate in self.dimension_columns]
            + self.__compute_personal_features(context)
        )
        # (MGB 7-23-25: Adding option for prompt indicator 
        if self.conf.get('use_prompts'):
                representation =  np.concatenate((representation,
                                           1*(np.array(self.action_space[1:self.K], ndmin=2) == action)), axis=1)
        if self.conf.get('intercept'):
            # (MGB 5-11-25: Need to set concatenate axis=1 since both are assumed to be 2d arrays )
            representation = np.concatenate((representation, np.atleast_2d([1.0])),axis=1)
        self.B += np.outer(representation, representation)
        self.S += (reward * representation).ravel()  # Ensure correct dimention
        self.mu = (np.linalg.inv(self.B) @ self.S)

        x = [text_representation[p] for p in self.dimension_columns]
        self.n[(c, action)] += 1
        self.X[(c, action)] += x
        self.XXT[(c, action)] += np.outer(x, x)
        XXT = self.XXT[(c, action)]
        n = self.n[(c, action)]
        X_bar = self.X[(c, action)] / n
        self.Psi[(c, action)] = (
            self.Psi0 +
            XXT - n * np.outer(X_bar, X_bar) +
            self.kappa * n / (self.kappa + n) * np.outer(X_bar - self.mu0, X_bar - self.mu0)
        )

    def __sample_mean_vector(self, context, action):
        """
        Sigma ~ IW(Psi, nu)
        mu ~ N(mu_mean, Sigma / (kappa + n)
        """
        c = self.__encode_context(context)
        n = self.n[(c, action)]
        psi = self.Psi[(c, action)]
        nu = self.nu0 + n
        psi_inv = np.linalg.inv(psi)
        d = psi.shape[0]
        w_sample = wishart.rvs(df=nu, scale=psi_inv, size=1, random_state=self.rng) # Marc changed to include random_state=self.rng
        iw_sample = np.linalg.inv(w_sample) if d > 1 else 1 / w_sample
        mu_mean = self.X[(c, action)] / (self.kappa + n)
        return multivariate_normal.rvs(mean=mu_mean, cov=iw_sample / (self.kappa + n), random_state=self.rng) # Marc changed to include random_state=self.rng


    def __encode_context(self, context):
        return tuple([context[c] for c in self.conf['context']])

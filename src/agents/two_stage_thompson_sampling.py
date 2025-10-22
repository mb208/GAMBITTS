from src.agents.agent_base import AgentBase
from collections import defaultdict
import numpy as np
from scipy.stats import invgauss
import pandas as pd


"""

Edit history:
- 7/25/2025: (MGB) Added option for adding prompt to the model.
"""

# TODO: move these to config?
TEXT_CONTEXT_COLUMNS = ['curr_loc', 'prev_steps']

PERSONAL_FEATURES = {
    'curr_loc': ['work', 'home'], # categorical
    'prev_steps': None, # numeric
    'x1': None, # numeric
}

PROMPT_COLUMN = 'prompt'


class TwoStageThompsonSampling(AgentBase):
    def __init__(self, prompt_db, config, rng=None):
        
        # Initialize rng. If rng is None, use default rng
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng
            
        self.config = config
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
        


        # Get specified interaction terms
        self.interactions = config.get("interactions", [])

            
        self.text_representation = \
            self.__compute_representation_center(prompt_db)

        self.action_space = self.__get_action_space(prompt_db)
        self.K = len(self.action_space)
        
        
        d = len(self.dimension_columns)
        for k, v in self.personal_features.items():
            if v is None:
                d += 1
            else:
                d += len(v) + 1
        self.d = d

        # Add interaction term dimensions
        for pair in self.interactions:
            pf_name = pair["personal_feature"]
            if self.personal_features[pf_name] is None:
                self.d += 1
            else:
                self.d += len(self.personal_features[pf_name]) + 1

        if config.get('intercept'):
            self.d += 1
        
        if self.config.get('use_prompts'):
            self.d += self.K-1

        # TODO: move these parameters to config (DONE)?
        self.v2 = config["v2"] #1
        self.B = np.eye(self.d)*config["precision"] # 0.01
        self.S = config["S"] # 0
        self.mu = np.ones(self.d)*config["mu"] # 0
        
        if self.config.get('intercept'):
            self.B[-1, -1] = config["precision_intercept"]
            self.mu[-1] = config["mu_intercept"]
            

    def __compute_representation_center(self, prompt_db):
        """
        Integrate over the randomness of the first stage of the data
        generating process (c, a) -> z, where c is context, a is action,
        and z is low-dimensional representation.
        The integration gives expected representation for each (c, a) pair.
        TODO: This is a hack that only works when we assume that
        (i) the agent knows the true distribution of (c, a) -> z
           (via free access to simulator)
        (ii) and the reward model is linear in z.
        To relax (i), we need to maintain a posterior distribution on the
        expected representation. To relax (ii), we need to do MC integration.
        """
        agg_key_values = {
            column: (column, 'mean') for column in self.dimension_columns
        }
        groupby_columns = [PROMPT_COLUMN] + TEXT_CONTEXT_COLUMNS
        summary_df = (
            prompt_db
                .groupby(groupby_columns)
                .agg(**agg_key_values)
                .reset_index()
        )

        representation_centers = {}
        for _, row in summary_df.iterrows():
            context = tuple([row[col] for col in TEXT_CONTEXT_COLUMNS])
            action = row[PROMPT_COLUMN]
            representation = [row[col] for col in self.dimension_columns]
            representation_centers[(context, action)] = representation
        return representation_centers

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

    # (Added by MGB 8/7/2025)
    def __compute_interaction_terms(self, text_representation_dict, personal_feature_dict):
        interaction_terms = []
        for pair in self.interactions:
            dim_name = pair["dimension"]
            pf_name = pair["personal_feature"]
            t_val = text_representation_dict[dim_name]
            p_val = personal_feature_dict[pf_name]

            if isinstance(p_val, list):  # one-hot
                interaction_terms.extend([t_val * v for v in p_val])
            else:  # continuous
                interaction_terms.append(t_val * p_val)
    
        return interaction_terms


    @staticmethod
    def __get_action_space(prompt_db):
        # action_space = set()
        # for _, row in prompt_db.iterrows():
        #     action_space.add(row[PROMPT_COLUMN])
        action_space = prompt_db[PROMPT_COLUMN].unique().tolist()
        return action_space

    def compute_feature(self, context, action):
        text_context = tuple(context[c] for c in TEXT_CONTEXT_COLUMNS)
        text_representation = self.text_representation[(text_context, action)]
        personal_feature = self.__compute_personal_features(context)
        feature = text_representation + personal_feature
        if self.config.get('use_prompts'):
            feature += (1*(np.array(self.action_space[1:self.K]) == action)).tolist()
        if self.config.get('intercept'):
            feature += [1.0]
        return np.array(feature)

    def action(self, context):
        mu = self.rng.multivariate_normal(self.mu, self.v2 * np.linalg.inv(self.B)) # Marc changed from np.random.multivariate_normal(self.mu, self.v2 * np.linalg.inv(self.B))
        max_reward = -np.inf
        max_action = None
        for action in self.action_space:
            feature = self.compute_feature(context, action)
            reward = feature.dot(mu)
            if reward > max_reward:
                max_reward = reward
                max_action = action
        return max_action

    def update(self, context, action, text_representation, reward):
        # representation = np.array(
        #     [text_representation[covariate] for covariate in self.dimension_columns]
        #     + self.__compute_personal_features(context) + ([1.0] if self.config.get('intercept') else [])
        # )
        
        representation = [text_representation[covariate] for covariate in self.dimension_columns] \
            + self.__compute_personal_features(context) 
            
        if self.config.get('use_prompts'):
            representation += (1*(np.array(self.action_space[1:self.K]) == action)).tolist()
            
        if self.config.get('intercept'):
            representation += [1.0] 
            
        representation = np.array(representation)   
         
        self.B += np.outer(representation, representation)
        self.S += reward * representation
        self.mu = np.linalg.inv(self.B) @ self.S

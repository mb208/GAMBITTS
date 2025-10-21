################################
### FUNCTIONS_ENVIRONMENT.PY ###
################################

### PURPOSE: This code contains helper functions for the environment class
### PROGRAMMER: Gabriel Durham (GJD)
### CREATED ON: 25 NOV 2024
### EDIT HISTORY: 01 DEC 2024 (GJD): Reformatted regret returned from step()
###               08 DEC 2024 (GJD): Changed outcome generation to speed up step function
###               13 APR 2025 (GJD): Added NN ourtcome model capability
###                             Also added functionality to run calculate_exp_y_online() before initializing (to avoid re-running in simulations)
###               27 APR 2025 (MGB): Edit NN ourtcome model capability. It seems like torch model wasn't loaded in
###                                  so that is done in self.__init__ if outcome_model type is "nn"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from modelling.mlp import NeuralNet





class Environment:
    def __init__(self, response_dims, yaml_parms, calc_exp_y_offline=True, model = None, rng=None):

        if rng:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

        self.response_db = response_dims.copy()
        self.prompt_vars = [key for key, subdict in yaml_parms["context"].items() if subdict.get("prompt_var")]
        self.text_dim_names = list(set(response_dims.columns) - set(self.prompt_vars + ["prompt", "exp_y_offline"]))
        self.context = simulate_context(yaml_parms["context"], initial=True, rng=self.rng) # Marc changed to take rng
        self.context_parms = yaml_parms["context"]
        self.outcome_model = yaml_parms["outcome_model"]

        # Reading in neural net
        if self.outcome_model["type"] == "nn":
            # Load in NN according to spec given in yaml
            if model is None:
                model = torch.load(yaml_parms["outcome_model"]["model_path"], weights_only=False)
                self.outcome_model["model"] = NeuralNet(**model['kwargs'])
                self.outcome_model["model"].load_state_dict(state_dict=model["state_dict"]) 
            # Allow NN to be passed and enviornment instantiation to reduce overhead
            else:
                self.outcome_model["model"] = model

        self.y = np.nan
        self.t = 1
        self.history = pd.DataFrame(columns=list(self.context.keys()) + self.text_dim_names + ["t", "prompt", "y"])
        # Enable calculating expected Y offline once then feeding it back in rather than re-calculating it at each iteration

        if calc_exp_y_offline:
            self.response_db.loc[:,"exp_y_offline"] = calculate_exp_y_offline(self.response_db, self.outcome_model)
        if yaml_parms["outcome_model"]["type"]=="linear":
            self.online_contribution = (len([key for key, subdict in yaml_parms["outcome_model"]["coefs"].items() if not subdict.get("offline_calc")])>0)
        else:
            self.online_contribution = yaml_parms["outcome_model"]["online_calc_necessary"]
            
    def get_expected_outcomes(self):
        #Identify rows in response data which have same prompt covariates
        context_match = (self.response_db[self.prompt_vars] == pd.Series({key: self.context[key] for key in self.prompt_vars})).all(axis=1)
        possible_responses = self.response_db.loc[context_match,:].copy()
        if self.online_contribution:
            # Get online contribution
            possible_responses.loc[:,"exp_y_online"] = calculate_exp_y_online(possible_responses, self.outcome_model, self.context, self.prompt_vars)
        else:
            possible_responses.loc[:,"exp_y_online"] = 0
        possible_responses.loc[:,"expected_outcome"] = possible_responses.loc[:,"exp_y_offline"] + possible_responses.loc[:,"exp_y_online"]
        return(possible_responses)
    
    def step(self, action):
        # First grab all responses (note: get_expected_outcomes() filters for prompt context match)
        possible_responses = self.get_expected_outcomes()
        action_summary = possible_responses.groupby("prompt", as_index=False).agg({"expected_outcome":"mean"})
        #Calculate best action
        best_action = action_summary.sort_values("expected_outcome", ascending=False).head(1).iloc[0]
        taken_action = action_summary[action_summary["prompt"]==action].iloc[0]

        #Sample a response
        generated_llm_response = possible_responses[possible_responses["prompt"]==action].sample(1).iloc[0]

        #Generate Error
        error = generate_outcome_error(self.outcome_model, rng=self.rng) # Marc changed to take rng
        y = generated_llm_response.expected_outcome + error

        #Update history
        new_history_row = pd.DataFrame({"y":[y], "prompt":[action], "t":[self.t]})
        for var in self.context.keys():
            new_history_row.loc[:,var] = self.context[var]
        for var in self.text_dim_names:
            # Gotta be a cleaner way of setting these variables
            new_history_row.loc[:,var] = generated_llm_response[var]
        self.history = pd.concat([self.history, new_history_row], ignore_index=True)

        #text representation, to be revealed to the agent
        text_representation = {}
        for var in self.text_dim_names:
            text_representation[var] = generated_llm_response[var]

        #Update info and context
        self.context = simulate_context(self.context_parms, rng=self.rng) #Marc changed to take rng
        self.y = y
        self.t = self.t + 1
        #Generate output for regret tracking
        regret_data = {
            "observed_prompt":action,
            # Don't want expected reward given text, want expected reward given prompt
            #"observed_exp_reward":[generated_llm_response.expected_outcome],
            "observed_exp_reward":taken_action.expected_outcome,
            "observed_reward":y,
            "observed_text_representation":text_representation,
            "optimal_prompt":best_action.prompt,
            "optimal_exp_reward":best_action.expected_outcome,
        }
        return(regret_data)











# Simulate context variables
def simulate_context(context_parms, initial=False, rng=None):
    if not rng:
        rng = np.random.default_rng()
            
    if initial:
        init_status = "initial"
    else:
        init_status = "step"
    output = {}
    for var in context_parms.keys():
        var_parms = context_parms[var]
        assign_parms = var_parms[init_status]
        if var_parms["var_type"]=="discrete":
            if assign_parms["assign"]=="random":
                if assign_parms["probs"]=="balanced":
                    output[var] = rng.choice(var_parms["possible_values"]) # np.random.choice(var_parms["possible_values"])
        elif var_parms["var_type"]=="continuous":
            if assign_parms["assign"]=="random":
                if assign_parms["dist"]=="normal":
                    output[var] = rng.normal(assign_parms["parms"]["mean"], assign_parms["parms"]["sd"]) # np.random.normal(assign_parms["parms"]["mean"], assign_parms["parms"]["sd"])
    return(output)

# Generate Error Parameters
def generate_outcome_error(outcome_model_parms, rng=None):
    if not rng:
        rng = np.random.default_rng()
    error_parms = outcome_model_parms["errors"]
    if error_parms["iid"]:
        if error_parms["dist"]=="normal":
            return(rng.normal(0, error_parms["parms"]["sd"]))  # Before np.random.normal(0, error_parms["parms"]["sd"])

        
        
def calculate_exp_y_offline(response_dims, outcome_model_parms):
    if outcome_model_parms["type"]=="linear":
        exp_y_offline = get_exp_y_offline_lin(response_dims, outcome_model_parms)
    elif outcome_model_parms["type"]=="nn":
        exp_y_offline = get_exp_y_offline_nn(response_dims, outcome_model_parms)
    return(exp_y_offline)

def get_exp_y_offline_lin(response_dims, outcome_model_parms):
    coefs = outcome_model_parms["coefs"]
    vars_to_calc = [key for key, subdict in coefs.items() if subdict.get("offline_calc")]
    processed_response_db = response_dims.copy()
    processed_response_db.loc[:,"exp_y_offline"] = 0
    for var in vars_to_calc:
        if coefs[var]["one_hot"]:
            #Grab coefficient from the right level of the predictor
            covar_contribution = processed_response_db[var].apply(lambda x: coefs[var].get(x, None))
            processed_response_db.loc[:,"exp_y_offline"] = processed_response_db.loc[:,"exp_y_offline"] + covar_contribution
        else:
            if var=="intercept":
                covar_contribution = coefs[var]["coef"]
            else:
                covar_contribution = float(coefs[var]["coef"])*processed_response_db[var]
            processed_response_db.loc[:,"exp_y_offline"] = processed_response_db.loc[:,"exp_y_offline"] + covar_contribution
    return(processed_response_db.loc[:,"exp_y_offline"])


def get_exp_y_offline_nn(response_dims, outcome_model_parms):
    nn = outcome_model_parms["model"]
    predictors = response_dims.loc[:, outcome_model_parms["dimensions"]]
    processed_response_db = response_dims.copy()
    processed_response_db.loc[:,"exp_y_offline"] = 0
    predictors_tensor = torch.tensor(processed_response_db[outcome_model_parms["dimensions"]].to_numpy(), dtype=torch.float32)
    with torch.no_grad():
        nn.eval()
        processed_response_db.loc[:,"exp_y_offline"] = nn(predictors_tensor).float().numpy()
    return(processed_response_db.loc[:,"exp_y_offline"])

def calculate_exp_y_online(response_db_filtered, outcome_model_parms, current_context, prompt_vars):
    if outcome_model_parms["type"]=="linear":
        exp_y_online = get_exp_y_online_lin(response_db_filtered, outcome_model_parms, current_context, prompt_vars)
    return(exp_y_online)
def get_exp_y_online_lin(response_db_filtered, outcome_model_parms, current_context, prompt_vars):
    coefs = outcome_model_parms["coefs"]
    processed_response_db = response_db_filtered.copy()
    processed_response_db.loc[:,"exp_y_online"] = 0
    vars_to_calc = [key for key, subdict in coefs.items() if not subdict.get("offline_calc")]
    
    #Need to update if we start using interaction variables and/or history variables
    for var in vars_to_calc:
        if not coefs[var]["text_based"]:
            if coefs[var]["one_hot"]:
                #Grab coefficient from the right level of the predictor
                covar_contribution = float(coefs[var][current_context[var]])
            else:
                covar_contribution = float(coefs[var]["coef"])*current_context[var]
            processed_response_db.loc[:,"exp_y_online"] = processed_response_db.loc[:,"exp_y_online"] + covar_contribution
    return(processed_response_db.loc[:,"exp_y_online"])
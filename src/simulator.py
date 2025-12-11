""" Experiment wraps an agent with an environment and then runs the experiment.

We end up with several experiment variants since we might want to log different
elements of the agent/environment interaction. At the end of `run_experiment` we
save the key results for plotting in a pandas dataframe `experiment.results`.

Based on https://github.com/iosband/ts_tutorial/blob/master/src/base/experiment.py
"""
import numpy as np
import pandas as pd
import os

##############################################################################

class BaseSimulator(object):
  """Simple experiment that logs regret and action taken.
  
  Assume use of thomspon sampling agent found in src/agents/thompson_sampling.py

  If you want to do something more fancy then you should extend this class.
  """

  def __init__(self, agent, env, n_steps, rec_freq=1, unique_id='NULL'):
    """Setting up the experiment.

    Note that unique_id should be used to identify the job later for analysis.
    """
    self.agent = agent
    self.env = env
    self.n_steps = n_steps
    self.unique_id = unique_id

    self.results = []
    self.data_dict = {}
    self.rec_freq = rec_freq




  def run_step_maybe_log(self, t):
    # Evolve the bandit (potentially contextual) for one step and pick action
    action = self.agent.action(self.env.context)
    
    prev_ctx = self.env.context
    curr_loc=self.env.context['curr_loc']
    # Collect data for regret calculations
    reward_info  = self.env.step(action)
    
    # if t > 1000:
    #   print("ere")


    
    # Log whatever we need for the plots we will want to use.
    instant_regret = reward_info['optimal_exp_reward'] - reward_info['observed_exp_reward']
    self.cum_regret += instant_regret

    # Advance the environment (used in nonstationary experiment)
    # self.env.advance(action, reward)

    if (t + 1) % self.rec_freq == 0:
      self.data_dict = {'t': (t + 1),
                        'instant_regret': instant_regret,
                        'cum_regret': self.cum_regret,
                        'action': action,
                        'best_action': reward_info['optimal_prompt'],
                        'curr_loc' : curr_loc,
                        'observed_reward' : reward_info['observed_reward'],
                        'unique_id': self.unique_id}
      
      self.results.append(self.data_dict)
    
    

    # Update the agent using realized rewards + bandit learning
    self.agent.update(context = prev_ctx,  # self.env.context, 
                      action = action, 
                      reward = reward_info['observed_reward'])

    


  def run_experiment(self, seed = None):
    """Run the experiment for n_steps and collect data."""
    if seed is not None:
      np.random.seed(self.seed)
    self.cum_regret = 0
    self.cum_optimal = 0

    for t in range(self.n_steps):
      self.run_step_maybe_log(t)

    self.results = pd.DataFrame(self.results)
    
  
  def save_results(self, path, append, name):
    
    assert self.results is not None, "No experiment has been run."
    
    self.results['name'] = name
    
    if append is not None:
      assert os.path.exists(path), "No existing file to append to"
      self.results.to_csv(path, mode='a', header=False, index=False)
    else:
      self.results.to_csv(path, header=False, index=False)

    
    
  

class TwoStageSimulator(BaseSimulator):
  """Experiment that logs regret and action taken when using two stage thompson sampling.
  """

  def __init__(self, agent, env, n_steps, rec_freq=1, unique_id='NULL'):
    super().__init__(agent, env, n_steps, rec_freq, unique_id)

  def run_step_maybe_log(self, t):
    # Evolve the bandit (potentially contextual) for one step and pick action
    action = self.agent.action(self.env.context)
    
    # Collect data for regret calculations
    reward_info  = self.env.step(action)
    
    # Log whatever we need for the plots we will want to use.
    instant_regret = reward_info['optimal_exp_reward'] - reward_info['observed_exp_reward']
    self.cum_regret += instant_regret

    # Advance the environment (used in nonstationary experiment)
    # self.env.advance(action, reward)

    if (t + 1) % self.rec_freq == 0:
      self.data_dict = {'t': (t + 1),
                        'instant_regret': instant_regret,
                        'cum_regret': self.cum_regret,
                        'action': action,
                        'best_action': reward_info['optimal_prompt'],
                        'unique_id': self.unique_id}
      
      self.results.append(self.data_dict)
    
    

    # Update the agent using realized rewards + bandit learning
    self.agent.update(context = self.env.context, 
                      action = action, 
                      text_representation = reward_info['observed_text_representation'],
                      reward = reward_info['observed_reward'])
    
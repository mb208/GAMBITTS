import numpy as np
import pandas as pd
import copy
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import os
import yaml
import copy
from src.Environment import Environment
def quantile(x):
  return np.array([np.count_nonzero(x<i)/(len(x)-1) for i in x])




def get_empirical_quanile_values(x, percentiles=[0,.25,.5,.75,1]):
    """
    Returns actual values from x at the p_i percentiles.
    
   Returns the closest actual value in x.
   Min and max are exact.
    
    Args:
        x (array-like): Input vector
        
    Returns:
        dict: {'min', 'q1', 'median', 'q3', 'max'} values from x
    """
    x = np.array(x)
    # percentile_targets = np.quantile(x, percentiles)
    qs = quantile(x)
    # quantile_matches = np.array([np.argmin(np.abs(x - pt)) for pt in percentile_targets])
    quantile_matches = np.array([np.argmin(np.abs(qs - p)) for p in percentiles])
    return quantile_matches


@hydra.main(version_base=None, config_path='../configs', config_name="config")
def gen_prompt_cfg(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader    

    overrides = HydraConfig.get().overrides.task
    
    try:
        env_override = next(o for o in overrides if o.startswith("environment="))
        environment_name = env_override.split("=")[1]
    except StopIteration:
        raise ValueError("You must specify an environment using 'environment=...' on the command line.")

    # Load in prompt database 
    if "gzip" in cfg.simulation.prompt_db:
        prompt_db = pd.read_csv(cfg.simulation.prompt_db, compression="gzip")
    else:
        prompt_db = pd.read_csv(cfg.simulation.prompt_db)

    
    env_cfg = {}
    env_cfg["outcome_model"] = dict(cfg.environment)
    env_cfg["context"]  = dict(cfg.context)
    env =  Environment(prompt_db, env_cfg)
    prompt_means = (env.response_db.groupby(['prompt'])['exp_y_offline'].agg(expected_outcome='mean'))


    base_sim = OmegaConf.to_container(cfg.simulation, resolve=True)

    # May want to move this to a config
    prompts_to_vary = list(cfg.env_setup.n_prompts) # [3,5,15,30,40]
    for n_prompt in prompts_to_vary:
        new_sim = copy.deepcopy(base_sim)

        percentiles = np.arange(n_prompt)/(n_prompt-1)

        qs = get_empirical_quanile_values(prompt_means.values, percentiles=percentiles) 
        prompts = prompt_means.reset_index().iloc[qs]['prompt'].tolist()   
        new_sim["n_prompts"] = prompts
        new_sim['env_used'] = environment_name
        if 'sd_id' in cfg.environment:
            sd_id = '_' + cfg.environment.sd_id if cfg.environment.sd_id else None
        else: 
            sd_id='' 
        if 'vrs' in cfg.simulation:
            vrs = cfg.simulation.vrs if cfg.simulation.vrs else ""
        else:
            vrs=''

        cfg_path = os.path.join(cfg.simulation.cfg_path, f"{vrs}{cfg.environment.type}_{cfg.environment.dim_batch}{sd_id}_sim_{n_prompt}_prompts.yaml")
        print(f"Saving config to {cfg_path}.")
        with open(cfg_path, 'w') as f:
            yaml.dump(new_sim, f, default_flow_style=False)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", category=DeprecationWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    gen_prompt_cfg()

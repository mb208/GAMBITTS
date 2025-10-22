import numpy as np
import pandas as pd
import copy
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import os
import yaml

@hydra.main(version_base=None, config_path='../configs', config_name="config")
def gen_variance_cfg(cfg : DictConfig) -> None:
    
    overrides = HydraConfig.get().overrides.task
    
    try:
        env_override = next(o for o in overrides if o.startswith("environment="))
        environment_name = env_override.split("=")[1]
    except StopIteration:
        raise ValueError("You must specify an environment using 'environment=...' on the command line.")

    # Load in prompt database 
    if "gzip" in cfg.env_setup.prompt_db:
        prompt_db = pd.read_csv(cfg.env_setup.prompt_db, compression="gzip")
    else:
        prompt_db = pd.read_csv(cfg.env_setup.prompt_db)


    main_dims =  list(cfg.environment.dimensions) if cfg.environment.type == "nn" else list(cfg.environment.coefs.keys())
    if 'intercept' in main_dims:
        main_dims.remove('intercept')
    
    starting_sd = ((prompt_db[main_dims] - prompt_db[main_dims].mean(axis=0))**2).sum(axis=1).mean()
    

    base_env = OmegaConf.to_container(cfg.environment, resolve=True)

    range_sds = np.linspace(starting_sd, base_env["errors"]["parms"]["sd"], cfg.env_setup.grid_num)

    for ix, sd in enumerate(range_sds):
        new_env = copy.deepcopy(base_env)
        new_env["errors"]["parms"]["sd"] = float(sd)
        new_env["sd_id"] = f"sd_{ix}"

        cfg_path = os.path.join(cfg.env_setup.cfg_path, f"{environment_name}_sd_{ix}.yaml")
        print(f"Saving config to {cfg_path}.")
        with open(cfg_path, 'w') as f:
            yaml.dump(new_env, f, default_flow_style=False)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", category=DeprecationWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    gen_variance_cfg()

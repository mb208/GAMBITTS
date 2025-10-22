################################
### FUNCTIONS_ENVIRONMENT.PY ###
################################

### PURPOSE: Runs simulation for any configuration
### PROGRAMMER: (MGB)
### CREATED ON: 27 APR 2025 
### EDIT HISTORY:
"""
07 AUG 2025 (MGB): Added interaction effect capability

"""


import numpy as np
import pandas as pd
import plotnine as gg
# import multiprocessing
import time
import sys
import os
import yaml
import hydra 
from omegaconf import DictConfig, OmegaConf
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from joblib import Parallel, delayed
import os


from src.agents import one_stage_thompson_sampling, ensemble_sampling, two_stage_thompson_sampling, fonats
# from src.Environment import Environment
from src.Environment_Int import Environment

from src.simulator import BaseSimulator, TwoStageSimulator


def process_prompt_db(prompt_db, n_prompts=None, dimensions = None, samples=None, ctx = None) -> pd.DataFrame:
    """
    Filters and samples a prompt database DataFrame according to the number of prompts, specific dimensions, 
    and contextual grouping.

    Args:
        prompt_db (pd.DataFrame): A DataFrame containing prompt data. Must include a 'prompt' column.
        n_prompts (int, optional): The number of unique prompts to randomly sample. If None and `dimensions` 
                                   is also None, the function exits early.
        dimensions (list, optional): A list of column names to retain alongside the sampled prompts. If provided,
                                     only these dimensions will be retained in the filtered DataFrame.
        samples (int, optional): Number of samples to draw from each group defined by 'prompt' and context keys.
        ctx (dict, optional): A dictionary representing context variables to be used as additional group keys 
                              for sampling.

    Returns:
        pd.DataFrame: A filtered and/or sampled DataFrame based on the specified criteria.

    Notes:
        - The action space is implicitly defined as integers from 0 to (K - 1), where K is the number of unique prompts.
        - If neither `n_prompts` nor `dimensions` are provided, the function exits early (currently with a `continue`, 
          which will raise a SyntaxError unless inside a loop — this may need revision).
        - Sampling is applied after filtering prompts and selecting dimensions.
    """
    available_prompts = prompt_db.prompt.unique()
    prompts = None
    if n_prompts is not None:
        if isinstance(n_prompts, (int, np.integer)):
            if n_prompts > len(available_prompts):
                raise ValueError(f"Requested {n_prompts} prompts, but only {len(available_prompts)} available.")
            prompts = np.random.choice(available_prompts, size=n_prompts, replace=False)
        else:
            if not isinstance(n_prompts, (list, np.ndarray)):
                raise TypeError("n_prompts must be int, np.ndarray, or list.")
            prompts = np.array(n_prompts)

        prompt_db = prompt_db.loc[prompt_db.prompt.isin(prompts)].copy() 
        # Action space should be 1-K
        prompt_db['prompt'], _ = pd.factorize(prompt_db['prompt_id'])
        prompt_db.loc[:,'prompt']  = prompt_db['prompt'] + 1

    if dimensions:
        columns = ['prompt', 'prompt_id'] + list(ctx.keys()) + dimensions
        prompt_db = prompt_db[columns].copy() 
    
    if samples:
        group_idx = ['prompt'] + list(ctx.keys())
        prompt_db = prompt_db.groupby(group_idx).sample(samples).copy()

    return prompt_db, prompts

    
def run_iteration(cfg, env_base, prompt_db, seed_i, iter):
    from omegaconf import DictConfig, OmegaConf
    import pandas as pd
    import numpy as np

    from src.agents import one_stage_thompson_sampling, ensemble_sampling, two_stage_thompson_sampling, fonats
    # from src.Environment import Environment
    from src.Environment_Int import Environment 
    from src.simulator import BaseSimulator, TwoStageSimulator

    cfg = OmegaConf.create(cfg)

    # np.random.seed(seed_i.generate_state(1)[0])
    # seed_entropy = seed_i.entropy
    # np.random.seed(seed_entropy)
    
    int_seed = seed_i.generate_state(1)[0]
    rng = np.random.default_rng(seed_i)

    agent_cfg = dict(cfg.agent)
    sim_cfg = cfg.simulation

    # Setting up environment configuration
    env_cfg = {}
    env_cfg["outcome_model"] = dict(cfg.environment)
    env_cfg["context"]  = dict(cfg.context)


    if sim_cfg.n_prompts is not None:
        # Process prompt_db according to simulation spec
        prompts = sim_cfg.n_prompts if isinstance(sim_cfg.n_prompts , int) else list(sim_cfg.n_prompts)  
        prompt_db, prompts = process_prompt_db(prompt_db, n_prompts=prompts)

        # Changing environment response db to match number of prompts
        env_base.response_db, _ = process_prompt_db(env_base.response_db, n_prompts=prompts)

        
    if sim_cfg.n_samples is not None:
        prompt_db,_ = process_prompt_db(prompt_db, samples=sim_cfg.n_samples, ctx = cfg.context)

    if cfg.environment.type=="nn":
        env = Environment(env_base.response_db, env_cfg,
                           calc_exp_y_offline=False,  model=env_base.outcome_model["model"], rng=rng)  # Marc inluded rng
    else:
        env = Environment(env_base.response_db, env_cfg, calc_exp_y_offline=False, rng=rng)  # Marc inluded rng

    # Handle the different agent scenarios
    if "K" in agent_cfg: 
        agent_cfg["K"] = int(prompt_db.prompt.nunique())

    if cfg.agent.type == "foNATS-linear":
        agent = fonats.FONATS(prompt_db, agent_cfg, rng=rng) # Marc added rng
        agent.nu0 = len(cfg.agent.covariates) # Need df for inverse wishart to > dim(scale_matrix) -1
    elif cfg.agent.type == "poNATS-ensemble":
        # agent = ensemble_sampling.EnsembleSampling(prompt_db, agent_cfg, rng=rng) # Marc added rng
        agent = ensemble_sampling.PerturbedEnsembleSampling(prompt_db,
                                                            agent_cfg,
                                                            noise= cfg.environment["errors"]["parms"]["sd"], 
                                                            rng=rng) # Marc added rng
    elif cfg.agent.type  == "poNATS-linear":
        agent = two_stage_thompson_sampling.TwoStageThompsonSampling(prompt_db, agent_cfg, rng=rng) # Marc added rng
    elif cfg.agent.type  == "LB":
        agent = one_stage_thompson_sampling.OneStageLinearThompsonSampling(agent_cfg, rng=rng) # Marc added rng
    else:
        agent = one_stage_thompson_sampling.OneStageThompsonSampling(agent_cfg, rng=rng) # Marc added rng

    if "NATS" in cfg.agent.type:
        sim = TwoStageSimulator(env = env, agent = agent, n_steps = sim_cfg.horizon, unique_id = cfg.agent.type)
    else:
        sim = BaseSimulator(env = env, agent = agent, n_steps = sim_cfg.horizon, unique_id = cfg.agent.type)

    sim.run_experiment()
    # Save number of actions (e.g. prompts) used
    sim.results['n_prompts'] = prompt_db.prompt.nunique()
    sim.results['mc_iter'] = iter
    sim.results['int_seed'] = int_seed
    return sim.results 

# @hydra.main(version_base=None, config_path='../configs', config_name="config")
def run_sim(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Setting up sets for parallel computation
    ss = np.random.SeedSequence(cfg.simulation.seed)
    child_seeds = ss.spawn(cfg.simulation.mc_iterations)
    

    # Load in prompt database 
    if "gzip" in cfg.simulation.prompt_db:
        prompt_db = pd.read_csv(cfg.simulation.prompt_db, compression="gzip")
    else:
        prompt_db = pd.read_csv(cfg.simulation.prompt_db)


    # Hard coding interaction 

    prompt_db['optimism_work'] = prompt_db['optimism']*(prompt_db['curr_loc']=="work").astype(int)
    prompt_db['optimism_home'] = prompt_db['optimism']*(prompt_db['curr_loc']=="home").astype(int)
    prompt_db['optimism_other'] = prompt_db['optimism']*(prompt_db['curr_loc']=="other").astype(int)
    

    
    # Instatiate environment to pass on env dependent parameters the we don't want to recompute each iteration
    # Setting up environment configuration
    env_cfg = {}
    env_cfg["outcome_model"] = dict(cfg.environment)
    env_cfg["context"]  = dict(cfg.context)

    # Taking into seed into account now
    rng = np.random.default_rng(cfg.simulation.seed) if cfg.simulation.seed else np.random.default_rng() # Marc added rng
    env =  Environment(prompt_db, env_cfg, rng=rng) # Marc added rng
    if cfg.environment.type == "nn":
        model = env.outcome_model["model"]
    else:
        model = None

    # Initialize dataframe to store simulation results
    sim_results = pd.DataFrame(columns=['t','instant_regret','cum_regret', 'action', 'best_action','unique_id'])
        
    print("Running baseline simulation in Parallel\n")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    # with ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(run_iteration, cfg_dict, prompt_db=env.response_db, model=None, seed_i= child_seeds[i]) #env.response_db=None, None, child_seeds[i])
    #                for i in range(cfg.simulation.mc_iterations)]

    # For debugging
    # run_iteration(cfg_dict, env, prompt_db, child_seeds[0], 0)

    n_cpus = cfg.simulation.n_cpus if cfg.simulation.n_cpus else max(1, os.cpu_count() - 1)
    n_cpus = min(n_cpus, os.cpu_count() - 1)  # Leave one CPU free for the system
    results = Parallel(n_jobs=n_cpus)(
        delayed(run_iteration)(cfg_dict, env, prompt_db, child_seeds[i], i)
        for i in range(cfg.simulation.mc_iterations)
        )

    # Collect and combine all results
    sim_results = pd.concat(results, ignore_index=True)
    

    env_dim = len(cfg.environment.dimensions) if cfg.environment.type == "nn" else len(cfg.environment.coefs)
    agent_dim = len(cfg.agent.covariates) if "NATS" in cfg.agent.type else None
    
    sim_results['agent_type'] =str( cfg.agent.type)
    sim_results['env_type'] = str(cfg.environment.type)
    sim_results['env_dim'] = env_dim
    sim_results['agent_dim'] = agent_dim
    sim_results['sim_date'] = date.today().strftime("%Y-%d-%m")
    sim_results['sim_setup'] = f"{str(cfg.environment.type)} ({env_dim}) - {str(cfg.agent.type)} ({agent_dim})"
    if cfg.simulation.n_samples:
        sim_results['gambitt_samples'] = sim_results['agent_type'] + ": " + str(cfg.simulation.n_samples) 

   

    save_root = os.path.join(cfg.simulation.proj_root, cfg.simulation.sim_id) # "data/results"
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # id for which dimensions were used in agent model
    if "NATS" not in cfg.agent.type:
        dim_batch = "" 
    else:
        # If NATs but dim_batch is null we can assume that one variable was specified
        agent_dimensions = [var for var in cfg.agent.covariates.keys() if var in cfg.possible_dimensions]
        dim_batch = "_" + agent_dimensions[0] if not cfg.agent.dim_batch else "_" + cfg.agent.dim_batch
        
     # agent_dim is only non null if it uses "covariates" e.g. latent dimensions 
    if cfg.simulation.store_dims and agent_dim:
        sim_results['dims_used'] = ",".join(agent_dimensions)
    
    # Create file name for results
    ## Initalize filename
    filename = filename = f"{cfg.environment.type}_{cfg.environment.dim_batch}_{cfg.agent.type}{dim_batch}"
    
    # Append with prompts used
    if cfg.simulation.n_prompts:
        n_prompts = cfg.simulation.n_prompts if isinstance(cfg.simulation.n_prompts, (int, np.integer)) else len(cfg.simulation.n_prompts)
        filename = f"{filename}_{n_prompts}prompts"

    # Append with samples used
    if cfg.simulation.n_samples:
        filename = f"{filename}_{cfg.simulation.n_samples}samples"

    # Should append with id for level of variance used   
    if "sd_id" in cfg.environment:
        sim_results["env_sd"] = cfg.environment["errors"]["parms"]["sd"]
        sim_results["sd_lvl"] = cfg.environment.sd_id.split('_')[1]
        filename = f"{filename}_{cfg.environment.sd_id}"   
            
    if cfg.simulation.vrs:
        filename = f"{filename}_{cfg.simulation.vrs}"

    
    datapath = os.path.join(save_root,f"{filename}.csv")

    # Save simulation results
    print(f"Saving simulation results to {datapath}")
    sim_results.to_csv(datapath, index=False)
    
    # Save logs
    if not os.path.exists(os.path.join(cfg.simulation.log_path, cfg.simulation.sim_id)):
        os.makedirs(os.path.join(cfg.simulation.log_path, cfg.simulation.sim_id))

    cfg_path = os.path.join(cfg.simulation.log_path, cfg.simulation.sim_id, f"{filename}_config.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)
    print(f"Saving configuration to {cfg_path}")
    
if __name__ == "__main__":
    import warnings

    # import multiprocessing
    # multiprocessing.set_start_method("spawn", force=True)
    warnings.simplefilter("ignore", category=DeprecationWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    start = time.time()
    # run_sim()
    hydra.main(
        config_path="../configs",
        config_name="config",
        version_base=None
    )(run_sim)()
    end = time.time()
    print(f"Sim took {(end - start)/(60*60)}")
    
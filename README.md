# GAMBITTS
This repository contains the code used to reproduce the simulation results from the paper: "Generator-Mediated Bandits: Thompson Sampling for GenAI-Powered Adaptive Interventions."


## Reproducing Simulations

To recreate the simulation results presented in the paper, run:

```bash
bash run_all_simulations.sh
```

Following this command, to generate the plots presented in the paper run the following command 

```bash 
bash get_all_simulation_plots.sh
```
which will save the relevant plots to `results/plots`. The bash script executes a jupyter notebook to generate the plots corresponding 
to each simulation. Should any execution fail, one can generate these plots directly with the corresponding jupyter notebook located in `notebooks`.

# System requirements and other technical details
Ensure that you have created the conda environment provided in the `environment.yaml` file using 
```bash
conda env create -f environment.yaml
```

Next, activate the environment using 

```bash
conda activate nats 
```

Following this, additional required packages should be installed by executing

```bash 
pip install -r requirements.txt
```
Our agents, data generating environment, and simulator are implemented as a python library  `src`. Make sure this library is installed in your conda environment before running 
simulations by executing 

```bash
pip install -e .
```

These simulations were originally run on a high-performance compute cluster using a SLURM scheduler. SLURM job arrays were used to parallelize the simulation configurations. To run locally (e.g., for testing or smaller runs), comment out the following line in `configs/config.yaml`: 

```
# /hydra/launcher: submitit_slurm
```

# Configuration files and data
All the simulation configurations for agents, environments, and simulation design can be found in `configs`. The configuration files provided as well as the data provided in the `data` folder are sufficient for recreatng the results in the article.

# GAMBITTS: Generator-Mediated Bandits: Thompson Sampling for GenAI-Powered Adaptive Interventions


## Reproducing Simulations

To recreate the simulation results presented in the paper, run:

```bash
bash run_all_simulations.sh [FLAGS]
```
where `[FLAGS]` take the following values
| Flag | Description |
|------|--------------|
| `--all` | Run **all** experiment blocks. |
| `--one_d_sim` | Run 1-D baseline and misspecification experiments. |
| `--scale_arms` | Vary the number of prompts (arms) in the 3-D linear environment. |
| `--var_decomp_linear` | Scale the 2nd-stage variance in the **linear** outcome environment. |
| `--var_decomp_nonlinear` | Scale the 2nd-stage variance in the **non-linear (NN)** outcome environment. |
| `--db_access_sim` | Run database-access simulations that vary the number of samples from the generator. |
| `--scale_dimensions` | Compare performance as latent dimensionality increases. |
| `--scale_fogambitt_arms` | Evaluate **foGAMBITT** with extended horizons and varying prompts. |
| `--hedged_1d_sim` | Run 1-D hedged simulations with prompt flags. |
| `--covariates` | Run 3-D simulations that include covariates in the outcome model. |
| `--help`, `-h` | Show usage information. |


> **Note:** Using `--all` will launch **all experiments**.  
> If you are not running on an HPC or a job scheduler such as **SLURM**, this will run sequentially and
> the full suite may take a **prohibitively long time** to complete on a standard machine.

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


# Citation 
Marc Brooks, Gabriel Durham, Kihyuk Hong, and Ambuj Tewari. *Generator-Mediated Bandits: Thompson Sampling for GenAI-Powered Adaptive Interventions.*  In *Advances in Neural Information Processing Systems 39 (NeurIPS 2025)*, 2025.  
 [https://arxiv.org/abs/2505.16311](https://arxiv.org/abs/2505.16311)

```bibtex
@inproceedings{GAMBITTS2025,
  author    = {Marc Brooks and Gabriel Durham and Kihyuk Hong and Ambuj Tewari},
  title     = {Generator-Mediated Bandits: Thompson Sampling for GenAI-Powered Adaptive Interventions},
  booktitle = {Advances in Neural Information Processing Systems 39 (NeurIPS 2025)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2505.16311}
}

from src.agents.agent_base import AgentBase
from collections import defaultdict
import numpy as np
from scipy.stats import invgamma

class OneStageThompsonSampling(AgentBase):
    def __init__(self, conf, rng=None):
        if rng:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()
            
        # TODO: move these parameters to config
        self.conf = conf
        self.K = conf['K']
        self.m = conf['m']
        self.nu = conf['nu']
        self.alpha = conf['alpha']
        self.beta = conf['beta']

        self.N = defaultdict(int) # number of times (arm, context) was pulled.
        self.reward_sum = defaultdict(float) # sum of rewards for each (arm, context)
        self.reward_squares_sum = defaultdict(float) # sum of reward squares for each (arm, context)
        self.betas = {}

    def action(self, context):
        max_mu = -np.inf
        max_action = None
        c = self.__encode_context(context)
        for action in range(1, self.conf['K'] + 1):
            pair = (c, action)
            mean_reward = self.reward_sum[pair] / self.N[pair] if self.N[pair] else 0
            s = self.reward_squares_sum[pair] - (mean_reward  ** 2) * self.N[pair]
            self.betas[pair] = self.beta + s / 2 + (
                (self.N[pair] * self.nu * ((mean_reward - self.m) ** 2)) /
                (2 * (self.N[pair] + self.nu))
            )
            sigma = invgamma.rvs(
                self.N[pair] / 2 + self.alpha,
                scale=self.betas[pair],
                size=1, random_state=self.rng) # Marc added random_state=self.rng
            zeta = sigma / (self.N[pair] + self.nu)
            rho = (
                (self.nu * self.m + self.N[pair] * mean_reward) /
                (self.nu + self.N[pair])
            )
            # print(f"action = {action}")
            # print(f"rho = {rho}")
            # print(f"zeta = {zeta}")
            # print(f"Posterior beta {self.betas[pair]}")
            # print(f"Posterior alpha {self.N[pair] / 2 + self.alpha}")
            mu = self.rng.normal(rho, np.sqrt(zeta)) # Mar changed from np.random.normal(rho, np.sqrt(zeta))
            if mu > max_mu:
                max_action = action
                max_mu = mu
        return max_action

    def update(self, context, action, reward):
        c = self.__encode_context(context)
        self.N[(c, action)] += 1
        self.reward_sum[(c, action)] += reward
        self.reward_squares_sum[(c, action)] += reward ** 2


    def get_all_posterior_means(self):
        estimates = {}
        for key in self.N:
            context, action = key
            N = self.N[key]
            if N > 0:
                mean_reward = self.reward_sum[key] / N
                rho = (self.nu * self.m + N * mean_reward) / (self.nu + N)
            else:
                rho = self.m  # fallback to prior mean
            estimates[(context, action)] = rho
        return estimates


    def get_posterior_means_and_variances(self):
        estimates = {}
        for key in self.N:
            N = self.N[key]
            sum_r = self.reward_sum[key]
            sum_r2 = self.reward_squares_sum[key]
            if N > 0:
                # Empirical mean
                mean_r = sum_r / N
                # Posterior mean of reward
                rho = (self.nu * self.m + N * mean_r) / (self.nu + N)
                # Unbiased sum of squares
                s = sum_r2 - (sum_r ** 2) / N
                s = max(0, s)  # guard against negative due to floating point error
                # Posterior parameters
                alpha_post = self.alpha + N / 2
                beta_post = self.beta + s / 2 + (N * self.nu * (mean_r - self.m) ** 2) / (2 * (N + self.nu))
                # Posterior variance of the mean (not predictive variance)
                var_rho = (beta_post / alpha_post) / (N + self.nu)
            else:
                # Prior
                rho = self.m
                var_rho = (self.beta / self.alpha) / self.nu
            estimates[key] = {'mean': rho, 'var': var_rho, "N": N}
        return estimates

    def __encode_context(self, context):
        return tuple([context[c] for c in self.conf['context']])
    

class OneStageLinearThompsonSampling(AgentBase):
    def __init__(self, conf, rng=None):
        # Marc Changed to include np random number generator for reproducibility
        if rng:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()
        # TODO: move these parameters to config
        self.conf = conf
        self.K = conf['K']
        self.m = conf['m']
        self.nu = conf['nu']
        self.alpha = conf['alpha']
        self.beta = conf['beta']

        self.m_c = conf['m_c']
        self.nu_c = conf['nu_c']

        self.N = defaultdict(int) # number of times arm was pulled.
        self.reward_sum = defaultdict(int) # sum of rewards for each arm
        self.reward_squares_sum = defaultdict(int) # sum of reward squares for each arm
        self.betas = {}


    def action(self, context):
        max_mu = -np.inf
        max_action = None
        c = self.__encode_context(context)
        N_c = 0
        for ctx in c:
            mean_reward = self.reward_sum[ctx] / self.N[ctx] if self.N[ctx] else 0
            s = self.reward_squares_sum[ctx] - (mean_reward  ** 2) * self.N[ctx]
            self.betas[ctx] = s / 2 + (
                (self.N[ctx] * self.nu_c * ((mean_reward - self.m_c) ** 2)) /
                (2 * (self.N[ctx] + self.nu_c))
            )
            N_c += self.N[ctx]
    
        for action in range(1, self.conf['K'] + 1):
            action_key = ('a', action)
            mean_reward = self.reward_sum[action_key] / self.N[action_key] if self.N[action_key] else 0
            s = self.reward_squares_sum[action_key] - (mean_reward  ** 2) * self.N[action_key]
            self.betas[action_key] =  s / 2 + (
                (self.N[action_key] * self.nu * ((mean_reward - self.m) ** 2)) /
                (2 * (self.N[action_key] + self.nu))
            )
            features = [action_key]+ [ctx for ctx in c]
            post_beta = self.beta + sum(self.betas[feature] for feature in features)
            sigma = invgamma.rvs(
                (self.N[action_key] + N_c)/ 2 + self.alpha,
                scale=post_beta,    
                size=1, random_state=self.rng) # Marc added random_state=self.rng
            zeta = sigma / (self.N[action_key] + self.nu)
            rho = (
                (self.nu * self.m + self.N[action_key] * mean_reward) /
                (self.nu + self.N[action_key])
            )
            # print(f"action = {action}")
            # print(f"rho = {rho}")
            # print(f"zeta = {zeta}")
            mu = self.rng.normal(rho, np.sqrt(zeta)) # Mar changed from np.random.normal(rho, np.sqrt(zeta))
            if mu > max_mu:
                max_action = action
                max_mu = mu
        return max_action

    def update(self, context, action, reward):
        c = self.__encode_context(context)
        self.N[('a', action)] += 1
        self.reward_sum[('a', action)] += reward
        self.reward_squares_sum[('a', action)] += reward ** 2
        for ctx in c:
            self.N[ctx] +=1
            self.reward_sum[ctx] += reward
            self.reward_squares_sum[ctx] += reward ** 2


    def __encode_context(self, context):
        return tuple([(c, context[c]) for c in self.conf['context']])
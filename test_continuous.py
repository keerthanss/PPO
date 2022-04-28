import gym
import numpy as np
import matplotlib.pyplot as plt
from ppo_continuous import *
from utils_continuous import *

np.random.seed(42)

ContinuousEnvs = ["MountainCarContinuous-v0", "Pendulum-v1"]#, "BipedalWalker-v3"]
choice = 0
env=gym.make(ContinuousEnvs[choice])
s = env.reset()
hyperparams = HyperParameters(
                learning_rate=3e-4,
                gamma=0.99,
                lambd=0.95,
                epsilon=0.1,
                VF_c1=1,
                Entropy_c2=10,
                annealing_period=80,
                annealing_rate=0.7
)

model = ActorCritic([512])
tf_env = EnvHelper(env, 256)
# tf_env.run_episode(model)
# tf_env.run_and_save_episode(model)
num_actors = 2
num_epochs = 2500
K_epochs = 2

algo = PPO(hyperparams, model, tf_env, num_actors, num_epochs, K_epochs)
reward_per_iteration, running_reward_per_iteration = algo.train()
plot_training_results(ContinuousEnvs[choice], reward_per_iteration, running_reward_per_iteration)

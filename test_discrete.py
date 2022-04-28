import gym
import numpy as np
import matplotlib.pyplot as plt
from ppo_discrete import *
from utils_discrete import *

# from ppo import *
# from utils import *

np.random.seed(42)

DiscreteEnvs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
choice = 2
env=gym.make(DiscreteEnvs[choice])
s = env.reset()
num_actions = env.action_space.n
hyperparams = HyperParameters(
                learning_rate=3e-4,
                gamma=0.99,
                lambd=0.95,
                epsilon=0.1,
                VF_c1=1,
                Entropy_c2=1,
                annealing_period=20,
                annealing_rate=0.9
)
model = ActorCritic(num_actions, [512])
# model = ActorCritic(num_actions, 32)
tf_env = EnvHelper(env, 100)
# tf_env.run_episode(model)
num_actors = 8
num_epochs = 1000
K_epochs = 2

algo = PPO(hyperparams, model, tf_env, num_actors, num_epochs, K_epochs)
reward_per_iteration, running_reward_per_iteration = algo.train()
plot_training_results(DiscreteEnvs[choice], reward_per_iteration, running_reward_per_iteration)

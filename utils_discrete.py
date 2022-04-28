import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym

from gym.wrappers.monitoring.video_recorder import VideoRecorder
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
from datetime import datetime

class HyperParameters:
    "A separate distinction for PPO's algorithmic hyperparameters"

    def __init__(self, learning_rate=1e-3, gamma=0.99, lambd=0.95, epsilon=0.2, VF_c1=1, Entropy_c2=0.01, annealing_period = 0, annealing_rate=0):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambd = lambd
        self.epsilon = epsilon
        self.VF_c1 = VF_c1
        self.Entropy_c2 = Entropy_c2
        self.annealing_period = annealing_period
        self.annealing_rate = annealing_rate

    def anneal(self, iter):
        if self.annealing_period == 0 or self.annealing_rate == 0:
            return
        if iter % self.annealing_period == 0:
            # self.learning_rate = self.learning_rate * self.annealing_rate
            self.Entropy_c2 = self.Entropy_c2 * self.annealing_rate
            # self.epsilon = self.epsilon * self.annealing_rate
        return


class ActorCritic(tf.keras.Model):
    "The shared neural network definition for policy and value networks"

    def __init__(self, num_actions, num_hidden_units):
        super().__init__()
        self.common = [layers.Dense(h, activation='relu', name='common') for h in num_hidden_units]

        self.pi_layer = layers.Dense(num_actions, name='pi')
        self.value_layer = layers.Dense(1, name='value')
        return

    def _common(self, x: tf.Tensor) -> tf.Tensor:
        for common_layer in self.common:
            x = common_layer(x)
        return x

    def pi(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self._common(inputs)
        return self.pi_layer(x)

    def v(self, inputs:tf.Tensor) -> tf.Tensor:
        x = self._common(inputs)
        return self.value_layer(x)

    def predict(self, inputs:tf.Tensor):
        x = self._common(inputs)
        pi = self.pi_layer(x)
        value = self.value_layer(x)
        return pi, value

class Trajectory:

    def __init__(self):
        self.states, self.actions, self.action_probs, self.rewards, self.values = [], [], [], [], []

        self.states_tensor = None
        self.actions_tensor = None
        self.action_probs_tensor = None
        self.rewards_tensor= None
        self.values_tensor = None
        self.advantages_tensor = None
        self.value_targets_tensor = None
        return

    def add_transition(self, state, action, action_prob, reward, value):
        self.states.append(state)
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.rewards.append(reward)
        self.values.append(value)
        return

    def _make_transition_tensors(self):
        self.states_tensor = tf.constant(self.states, dtype=tf.float32)
        self.actions_tensor = tf.constant(self.actions, dtype=tf.int32)
        self.action_probs_tensor = tf.constant(self.action_probs, dtype=tf.float32)
        self.rewards_tensor= tf.constant(self.rewards, dtype=tf.float32)
        self.values_tensor = tf.constant(self.values, dtype=tf.float32)
        return

    def _calculate_advantages_targets(self, model: tf.keras.Model, gamma, lambd):
        length_of_episode = len(self.rewards)
        advantages = np.zeros(length_of_episode)
        value_targets = np.zeros(length_of_episode)

        discounted_sum = 0
        for t in range(length_of_episode-2, -1, -1):
            r = self.rewards[t]
            value_s_prime = self.values[t+1]

            value_targets[t] = r + gamma * value_s_prime

            delta = value_targets[t] - self.values[t]
            discounted_sum = delta + gamma * lambd * discounted_sum
            advantages[t] = discounted_sum

        self.advantages_tensor = tf.constant(advantages, dtype=tf.float32)
        self.value_targets_tensor = tf.constant(value_targets, dtype=tf.float32)
        return

    def process(self, model: tf.keras.Model, gamma, lambd):
        self._make_transition_tensors()
        self._calculate_advantages_targets(model, gamma, lambd)
        return


class EnvHelper:

    def __init__(self, env, horizon):
        self.env = env
        self.horizon = horizon # maximum length any episode is allowed to run
        self.env_name = self.env.unwrapped.spec.id
        return

    def gaussian(self, mu, std, x):
        coeff = 1.0 / np.sqrt(2 * np.pi) * std
        diff = mu - x
        numr = np.dot(diff, diff)
        denr = 2 * std * std
        exp_value = np.exp(numr / denr)
        final = coeff * exp_value
        return final

    #@tf.function
    def run_episode(self, model:tf.keras.Model):
        initial_state = self.env.reset()
        traj = Trajectory()
        actions = np.arange(self.env.action_space.n)
        state = initial_state
        for t in range(self.horizon):
            action_logits_t, value_t = model.predict(tf.constant(np.expand_dims(state,0), dtype=tf.float32))
            action_logits_t = action_logits_t[0]
            action_probs_t = np.exp(action_logits_t) / np.sum(np.exp(action_logits_t), axis=0) # softmax
            action_t = np.random.choice(actions, p=action_probs_t)
            action_prob = action_probs_t[action_t]
            # print(action_t)

            new_state, reward, done, _ = self.env.step(action_t)#np.expand_dims(action_t,0))
            # self.env.render()
            value_t = np.squeeze(value_t)
            # print(state, action_t, action_prob, reward, value_t)
            traj.add_transition(state, action_t, action_prob, reward/10, value_t)
            if done:
                break
            state = new_state

        return traj

    def run_and_save_episode(self, model, video_path):
        video_recorder = VideoRecorder(self.env, video_path)
        state = self.env.reset()
        done = False
        actions = np.arange(self.env.action_space.n)
        while not done:
            self.env.render()
            video_recorder.capture_frame()
            action_logits_t, value_t = model.predict(tf.constant(np.expand_dims(state,0), dtype=tf.float32))
            action_logits_t = action_logits_t[0]
            action_probs_t = np.exp(action_logits_t) / np.sum(np.exp(action_logits_t), axis=0) # softmax
            action_t = np.random.choice(actions, p=action_probs_t)
            new_state, reward, done, _ = self.env.step(action_t)#np.expand_dims(action_t,0))

        video_recorder.close()
        # self.env.close()
        return


def plot_training_results(env_name,reward_per_iteration, running_reward_per_iteration):
    x = np.arange(len(reward_per_iteration))
    plt.figure(layout='constrained')
    plt.plot(x, reward_per_iteration, label='Reward Per Iteration', color='blue')
    plt.plot(x, running_reward_per_iteration, label='Running Average over 20 iterations', color='orange')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('PPO on {}'.format(env_name))
    plt.legend()
    now = datetime.now()
    str_now = now.strftime("%b%d_%H%M%S") # Eg: Apr28_120043
    plt.savefig('PPO_on_{}_{}.png'.format(env_name, str_now))
    plt.show()

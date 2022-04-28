import tensorflow as tf
import tqdm
import statistics
import collections
import os

from typing import Any, List, Sequence, Tuple
from datetime import datetime
from utils import *
import tensorflow_probability as tfp

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


class PPO:

    def __init__(self, hyperparams: HyperParameters, model:tf.keras.Model, tf_env: EnvHelper, num_actors, num_epochs, K_epochs):
        self.hyperparams = hyperparams
        self.model = model
        self.tf_env = tf_env
        self.num_actors = num_actors
        self.num_epochs = num_epochs
        self.K_epochs = K_epochs
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.hyperparams.learning_rate)
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM) # better than L2 in handling outliers


    @tf.function
    def loss(self, states: tf.Tensor, actions: tf.Tensor, action_probs: tf.Tensor, advantages: tf.Tensor, value_targets: tf.Tensor) -> tf.Tensor:
        # run model with current parameters to obtain action probabilities
        new_action_logits = self.model.pi(states)
        # calculate new probabilites
        new_action_probs = tf.nn.softmax(new_action_logits, 1)
        action_indices = tf.expand_dims(actions, 1)
        respective_action_probs = tf.gather_nd(params=new_action_probs, indices=action_indices, batch_dims=1)

        # calculate policy loss
        # ratio = tf.divide(respective_action_probs, action_probs)
        ratio = tf.math.exp(tf.math.log(respective_action_probs) - tf.math.log(action_probs))
        clip_value = tf.clip_by_value(ratio, 1 - self.hyperparams.epsilon, 1 + self.hyperparams.epsilon)
        normal_loss = tf.math.multiply(ratio, advantages)
        clipped_loss = tf.math.multiply(clip_value, advantages)

        lower_bound = tf.math.minimum(normal_loss, clipped_loss)
        policy_loss = tf.math.reduce_mean(lower_bound)

        # calculate value loss
        values = self.model.v(states)
        # value_loss = self.huber_loss(values, value_targets) # value targets have been computed using current model parameters
        value_diff = tf.square(values - value_targets)
        value_loss = tf.math.reduce_mean(value_diff)
        # calculate entropy loss
        log_action_probs = tf.math.log(new_action_probs)
        entropy = -tf.math.multiply(new_action_probs, log_action_probs)
        entropy_loss = tf.math.reduce_mean(entropy)
        # total
        total_loss = policy_loss - self.hyperparams.VF_c1 * value_loss + self.hyperparams.Entropy_c2 * entropy_loss
        total_loss = -total_loss # we wish to maximize the above value, that's why negating
        # tf.print(policy_loss, value_loss, entropy_loss)
        return total_loss

    @tf.function(experimental_relax_shapes=True)
    def gradients_one_surrogate_actor(self, states, actions, action_probs, advantages, value_targets):
        with tf.GradientTape() as tape:
            loss_k = self.loss(states,
                            actions,
                            action_probs,
                            advantages,
                            value_targets)
        gradient = tape.gradient(loss_k, self.model.trainable_variables,unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return gradient, loss_k


    def generate_batch(self):
        batch = []
        for k in range(self.num_actors):
            traj = self.tf_env.run_episode(self.model)
            traj.process(self.model, self.hyperparams.gamma, self.hyperparams.lambd) # calculates advantages and targets and stores everything as tensor
            batch.append(traj)
        return batch


    def train_one_step(self, batch):
        # Optimize the surrogate for K steps
        num_vars = len(self.model.trainable_variables)
        for iter in range(self.K_epochs):

            gradients = [0*v for v in self.model.trainable_variables] # gets list of zero tensors in corresponding shapes
            total_loss = 0.0

            for k in range(self.num_actors):
                traj = batch[k]
                # collect training data
                advantages, value_targets = traj.advantages_tensor, traj.value_targets_tensor
                states, actions, action_probs = \
                    traj.states_tensor, traj.actions_tensor, traj.action_probs_tensor

                gradient_k, loss_k = self.gradients_one_surrogate_actor(states,
                                actions,
                                action_probs,
                                advantages,
                                value_targets)

                #consolidate
                for i in range(num_vars):
                    gradients[i] = gradients[i] + gradient_k[i]
                total_loss = np.sum(loss_k.numpy()) + total_loss

            # Note that applying sum of gradients for each loss is same as
            # calculating gradient over sum of losses
            for i in range(num_vars):
                gradients[i] = gradients[i]/self.num_actors

            # clip gradients for stability
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            total_loss = total_loss / self.num_actors

        return total_loss


    def train(self):
        print_interval = 20
        episodes_reward: collections.deque = collections.deque(maxlen=print_interval)
        reward_per_iteration = []
        running_reward_per_iteration = []

        now = datetime.now()
        str_now = now.strftime("%b%d_%H%M%S") # Eg: Apr28_120043
        env_name = self.tf_env.env_name
        video_folder = f'./video/{env_name}_{str_now}'
        os.mkdir(video_folder)

        for i in range(self.num_epochs):
            # train one step
            batch = self.generate_batch()
            total_loss = self.train_one_step(batch)

            # anneal hyperparams according to iteration
            self.hyperparams.anneal(i+1)
            self.optimizer.learning_rate = self.hyperparams.learning_rate

            # calculate statistics
            average_reward = np.average([np.sum(traj.rewards) for traj in batch])
            episodes_reward.append(average_reward)
            running_reward = statistics.mean(episodes_reward)

            # print or store video accordingly
            if (i+1) % print_interval == 0:
                print(f'Iteration {i+1}: Batch average reward = {average_reward}, Running average = {running_reward}, Batch average loss = {total_loss}')#'.numpy()}')
            if (i+1) % 100 == 0:
                video_title = f'{env_name}_iteration{i+1}'
                video_path = f'{video_folder}/{video_title}.mp4'
                self.tf_env.run_and_save_episode(self.model, video_path)

            # save statistics for plotting
            reward_per_iteration.append(average_reward)
            running_reward_per_iteration.append(running_reward)

        return reward_per_iteration, running_reward_per_iteration

import tensorflow as tf
from collections import deque, namedtuple
import numpy as np


class QAgent:
    def __init__(self, aid, policy, model, target_model, timesteps=1):
        self.aid = aid
        self.policy = policy
        self.model = model
        self.target_model = target_model
        self.prev_action = 0
        self.timesteps = timesteps
        self.trajectory = None
        self.last_q_values = None

    def _init_deque(self, observation):
        trajectory = deque(maxlen=self.timesteps)
        for i in range(self.timesteps):
            trajectory.append(observation)
        return trajectory

    def act(self):
        action = self._forward()
        self.prev_action = action
        return action

    def observe(self, observation):
        self.state = observation
        self.trajectory.append(observation)

    def _forward(self):
        q_values = self._compute_q_values(self.trajectory)
        self.last_q_values = q_values
        action_id = self.policy.select_action(
            q_values=q_values, is_training=True)
        self.recent_action_id = action_id

        return action_id

    def _compute_q_values(self, state):
        inputs = tf.convert_to_tensor(list(self.trajectory))
        inputs = tf.expand_dims(inputs, 0)
        q_values = self.target_model.predict(inputs)
        return q_values[0]

    def _hard_update_target_model(self):
        """ for hard update """
        self.target_model.set_weights(self.model.get_weights())

    def _soft_update_target_model(self):
        target_model_weights = np.array(self.target_model.get_weights())
        model_weights = np.array(self.model.get_weights())
        new_weight = (1. - self.update_interval) * target_model_weights \
            + self.update_interval * model_weights
        self.target_model.set_weights(new_weight)

    def reset(self, observation):
        self.observation = observation
        self.prev_observation = observation
        self.trajectory = self._init_deque(observation)

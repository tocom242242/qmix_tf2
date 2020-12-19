import random
from collections import deque, namedtuple
import numpy as np

Experience = namedtuple(
    'Experience',
    'state0, observations0, action, reward,observations1, state1, terminal')


def sample_batch_indexes(low, high, size):
    r = range(low, high)
    batch_idxs = random.sample(r, size)

    return batch_idxs


class Memory():

    def sample(self, **kwargs):
        raise NotImplementedError()

    def append(self, **kwargs):
        raise NotImplementedError()


class RandomMemory(Memory):
    def __init__(self, limit, agent_num=2):
        super(Memory, self).__init__()
        self.experiences = deque(maxlen=limit)
        self.agent_num = agent_num

    def sample(self, batch_size):
        assert batch_size > 1, "batch_size must be positive integer"

        batch_size = min(batch_size, len(self.experiences))
        mini_batch = random.sample(self.experiences, batch_size)
        state_batch = []
        observation_batch = [[] for _ in range(self.agent_num)]
        action_batch = []
        reward_batch = []
        next_state_batch = []
        next_observation_batch = [[] for _ in range(self.agent_num)]
        terminal_batch = []
        for state, observation, action, reward, next_state, next_observation, done in mini_batch:
            state_batch.append(state)
            for i in range(self.agent_num):
                observation_batch[i].append(observation[i])
                next_observation_batch[i].append(next_observation[i])
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            # next_observation_batch.append(next_observation)
            terminal_batch.append(0. if done else 1.)
        state_batch = np.array(state_batch)
        observation_batch = np.array(observation_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_observation_batch = np.array(next_observation_batch)
        next_state_batch = np.array(next_state_batch)
        terminal_batch = np.array(terminal_batch)

        assert len(state_batch) == batch_size

        return state_batch, observation_batch, action_batch, reward_batch, next_state_batch, next_observation_batch, terminal_batch

    def append(
            self,
            state,
            observation,
            action,
            reward,
            next_state,
            next_observation,
            terminal=False):
        self.experiences.append(
            (state,
             observation,
             action,
             reward,
             next_state,
             next_observation,
             terminal))

# class Memory:
#     def __init__(self, limit, maxlen):
#         self.actions = deque(maxlen=limit)
#         self.rewards = deque(maxlen=limit)
#         self.terminals = deque(maxlen=limit)
#         self.observations = deque(maxlen=limit)
#         self.states = deque(maxlen=limit)
#         self.maxlen = maxlen
#         self.recent_observations = deque(maxlen=maxlen)
#
#     def sample(self, batch_size):
#         batch_idxs = sample_batch_indexes(
#             0, len(self.observations) - 1, size=batch_size)
#         for (i, idx) in enumerate(batch_idxs):
#             terminal = self.terminals[idx-1]
#             while terminal:
#                 idx = sample_batch_indexes(
#                     0, len(self.observations)-1, size=1)[0]
#                 terminal = self.terminals[idx-1]
#             batch_idxs[i] = idx
#
#         experiences = []
#         for idx in batch_idxs:
#             state0 = self.states[idx]
#             observations0 = self.observations[idx]
#             action = self.actions[idx]
#             reward = self.rewards[idx]
#             terminal = self.terminals[idx]
#             state1 = self.states[idx+1]
#             observations1 = self.observations[idx]
#             experiences.append(Experience(state0=state0,
#                                           observations0=observations0,
#                                           action=action,
#                                           reward=reward,
#                                           observations1=observations1,
#                                           state1=state1,
#                                           terminal=terminal))
#
#         return experiences
#
#     def append(self, state, observations, action, reward, terminal=False):
#         self.observations.append(observations)
#         self.states.append(state)
#         self.actions.append(action)
#         self.rewards.append(reward)
#         self.terminals.append(terminal)

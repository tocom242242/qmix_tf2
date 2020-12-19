import numpy as np


class Policy():

    def select_action(self, **kwargs):
        raise NotImplementedError()


class EpsGreedyQPolicy(Policy):
    def __init__(self, eps=1., eps_decay_rate=.999, min_eps=.1):
        super(EpsGreedyQPolicy, self).__init__()
        self.eps = eps
        self.eps_decay_rate = eps_decay_rate
        self.min_eps = min_eps

    def select_action(self, q_values, is_training=True):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if is_training:
            if np.random.uniform() < self.eps:
                action = np.random.randint(0, nb_actions)
            else:
                action = np.argmax(q_values)
        else:
            action = np.argmax(q_values)
        self.decay_eps_rate()
        return action

    def decay_eps_rate(self):
        self.eps = self.eps * self.eps_decay_rate
        if self.eps < self.min_eps:
            self.eps = self.min_eps


class BoltzmannPolicy(Policy):
    def __init__(self, tau=1., clip=(-500., 500.)):
        super(BoltzmannQPolicy, self).__init__()
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values):

        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]

        exp_values = np.exp(
            np.clip(
                q_values /
                self.tau,
                self.clip[0],
                self.clip[1]))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)
        return action

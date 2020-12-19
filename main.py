import tqdm
import tensorflow as tf
from qagent import QAgent
from qmix import QMix
from memory import RandomMemory
from policy import EpsGreedyQPolicy
from two_step_env import TwoStepEnv
import copy
import numpy as np
import matplotlib.pyplot as plt


def build_q_network(input_shape, nb_output):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64, activation='relu')(input_layer)
    # x = tf.keras.layers.GRU(32, activation='relu')(x)
    # x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(nb_output, activation='linear')(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output)

    return model


agents_num = 2
max_episodes = 1000
actions = [0, 1]
actions_num = len(actions)
trajectory_len = 1
observation_dim = 3

env = TwoStepEnv()
memory = RandomMemory(limit=500)
policy = EpsGreedyQPolicy(eps=1., eps_decay_rate=.999, min_eps=.01)

agent_input_shape = (trajectory_len, observation_dim)

agents = []
for aid in range(agents_num):
    model = build_q_network(agent_input_shape, actions_num)
    target_model = build_q_network(agent_input_shape, actions_num)
    agent = QAgent(
        aid=aid,
        policy=policy,
        model=model,
        target_model=target_model)
    agent.target_model.set_weights(model.get_weights())
    init_state = tf.one_hot(0, observation_dim)
    agents.append(agent)

# init qmix
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.RMSprop()
batch_size = 32
qmix = QMix(
    agents=agents,
    memory=memory,
    batch_size=batch_size,
    loss_fn=loss_fn,
    optimizer=optimizer)

episode_reward_history = []
loss_history = []
episode_reward_mean = 0
loss_mean = 0
with tqdm.trange(max_episodes) as t:
    for episode in t:
        for agent in agents:
            agent.reset(init_state)
        env.reset()
        rewards = []
        for step in range(3):
            actions = []
            for agent in agents:
                action = agent.act()
                actions.append(action)
            state, reward, done = env.step(actions)
            state = tf.one_hot(state, observation_dim)
            rewards.append(reward)

            trajectories = []
            for agent in agents:
                agent.observe(state)
                trajectory = copy.deepcopy(agent.trajectory)
                trajectories.append(trajectory)

            one_hot_actions = []
            for action in actions:
                action = tf.one_hot(action, depth=actions_num)
                one_hot_actions.append(action)
            qmix.save(state, trajectories, one_hot_actions, reward, done)

            if episode > batch_size:
                loss = qmix.train()
                loss_history.append(loss)

            if done:
                break

        episode_reward = np.sum(rewards)
        episode_reward_history.append(episode_reward)
        episode_reward_mean = 0.01 * episode_reward + 0.99 * episode_reward_mean
        t.set_description(
            f"Episode:{episode},state:{env.prev_state},qmix:{qmix.get_qmix_output()}, reward:{episode_reward}")
        t.set_postfix(episode_reward_mean=episode_reward_mean)

fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))
axL.plot(
    np.arange(
        len(episode_reward_history)),
    episode_reward_history,
    label="episode_reward")
axL.set_xlabel('episode')
axL.set_title("episode reward history")

axR.plot(np.arange(len(loss_history)), loss_history, label="loss")
axR.set_title("qmix's loss history")

axR.legend()
axL.legend()
plt.savefig("result.png")

import tensorflow as tf
import numpy as np
import copy


class MixingNet(tf.keras.Model):
    def __init__(self, agent_nets, embed_shape):
        super(MixingNet, self).__init__()
        self.agent_nets = agent_nets
        self.agent_num = len(agent_nets)
        self.embed_shape = embed_shape
        self.timesteps = agent_nets[0].input_shape[1]
        self.agent_output_dim = agent_nets[0].output_shape[-1]

        # self.hyper_w1 = tf.keras.layers.Dense(embed_shape*self.agent_num*self.agent_output_dim, activation='linear',use_bias=True)

        self.hyper_w1_1 = tf.keras.layers.Dense(
            embed_shape, activation='relu', use_bias=True)
        self.hyper_w1_2 = tf.keras.layers.Dense(
            embed_shape *
            self.agent_num *
            self.agent_output_dim,
            activation='linear',
            use_bias=True)

        self.hyper_b1 = tf.keras.layers.Dense(self.embed_shape)

        # self.hyper_w2 = tf.keras.layers.Dense(self.embed_shape, activation='linear', use_bias=True)

        self.hyper_w2_1 = tf.keras.layers.Dense(
            self.embed_shape, activation='relu', use_bias=True)
        self.hyper_w2_2 = tf.keras.layers.Dense(
            self.embed_shape, activation='linear', use_bias=True)

        self.hyper_b2 = tf.keras.layers.Dense(1, activation="relu")

    def call(self, inputs):
        agents_inputs = inputs[0]
        states = inputs[1]
        masks = inputs[2]
        batch_size = states.shape[0]

        agents_outputs = []
        for agent_net, agent_input, mask in zip(
                self.agent_nets, agents_inputs, masks):
            agent_out = agent_net(agent_input)
            agent_out = tf.multiply(agent_out, mask)
            agents_outputs.append(agent_out)

        # w1 = tf.abs(self.hyper_w1(states))
        w1 = tf.abs(self.hyper_w1_2(self.hyper_w1_1(states)))

        agents_outputs = tf.concat(agents_outputs, 1)
        agents_outputs = tf.expand_dims(agents_outputs, 1)

        w1 = tf.reshape(w1, [
            batch_size, self.agent_output_dim * self.agent_num, -1])
        b1 = self.hyper_b1(states)
        b1 = tf.reshape(b1, [batch_size, 1, -1])
        hidden = tf.keras.activations.elu(tf.matmul(agents_outputs, w1) + b1)

        # w2 = tf.abs(self.hyper_w2(states))
        w2 = tf.abs(self.hyper_w2_2(self.hyper_w2_1(states)))

        w2 = tf.reshape(w2, [batch_size, self.embed_shape, 1])
        b2 = self.hyper_b2(states)
        b2 = tf.reshape(b2, [batch_size, 1, 1])
        y = tf.matmul(hidden, w2) + b2
        q_tot = tf.reshape(y, [-1, 1])
        return q_tot


class QMix:
    def __init__(
            self,
            agents=None,
            memory=None,
            gamma=0.99,
            batch_size=32,
            loss_fn=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.RMSprop(),
            is_ddqn=False,
            update_interval=200,
            embed_shape=60,
            lr=0.0005,
            agent_action_num=2):
        self.agents = agents
        self.memory = memory
        self.gamma = gamma
        self.batch_size = batch_size
        self.is_ddqn = is_ddqn
        self.update_interval = update_interval
        self.step = 0
        self.train_interval = 1
        self.warmup_steps = 60
        self.prev_state = None
        self.prev_observations = None
        self.agent_action_num = agent_action_num
        self.last_q_values = [0]  # @todo
        self.last_targets = [0]  # @todo

        models = []
        target_models = []
        self.trainable_variables = None
        self.target_trainable_variables = None
        for agent in agents:
            models.append(agent.model)
            target_models.append(agent.target_model)
            if self.trainable_variables is None:
                self.trainable_variables = agent.model.trainable_variables
                self.target_trainable_variables = agent.target_model.trainable_variables
            else:
                self.trainable_variables += agent.model.trainable_variables
                self.target_trainable_variables += agent.target_model.trainable_variables

        self.model = MixingNet(models, embed_shape)
        self.target_model = MixingNet(target_models, embed_shape)
        self.trainable_variables += self.model.trainable_variables
        self.target_trainable_variables += self.target_model.trainable_variables

        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def save(self, state, observations, actions, reward, is_terminal):
        if self.prev_state is None:
            self.prev_state = copy.deepcopy(state)
            self.prev_observations = observations

        self.memory.append(self.prev_state,
                           self.prev_observations,
                           actions,
                           reward,
                           state,
                           observations,
                           terminal=is_terminal)
        self.prev_state = copy.deepcopy(state)
        self.prev_observations = copy.deepcopy(observations)
        self.step += 1

    def train(self):
        loss = self._experience_replay()
        return loss

    def _experience_replay(self):
        loss = 0
        if self.step > self.warmup_steps \
                and self.step % self.train_interval == 0:
            states, observations, actions, rewards, next_states, next_observations, terminals = self.memory.sample(
                self.batch_size)

            rewards = np.array(rewards).reshape(-1, 1)
            terminals = np.array(terminals).reshape(-1, 1)
            next_observations = np.array(next_observations)
            next_states = np.array(next_states)

            masks, target_masks = [], []
            for idx, (agent, next_observation) in enumerate(
                    zip(self.agents, next_observations)):
                agent_out = agent.target_model(next_observation)
                argmax_actions = tf.keras.backend.argmax(agent_out)
                target_mask = tf.one_hot(
                    argmax_actions, depth=self.agent_action_num)
                target_masks.append(target_mask)
                masks.append(actions[:, idx, :])

            masks = tf.convert_to_tensor(masks)
            target_masks = tf.convert_to_tensor(target_masks)

            target_q_values = self._predict_on_batch(
                next_states, next_observations, target_masks, self.target_model)
            discounted_reward_batch = self.gamma * target_q_values * terminals
            targets = rewards + discounted_reward_batch

            # Set up logging.
            # from  datetime import datetime
            # stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            # logdir = 'logs/func/%s' % stamp
            # writer = tf.summary.create_file_writer(logdir)
            # tf.summary.trace_on(graph=True, profiler=True)

            observations = np.array(observations)
            states = np.array(states)
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            observations = tf.convert_to_tensor(
                observations, dtype=tf.float32)

            loss = self._train_on_batch(
                states, observations, masks, targets)

            # with writer.as_default():
            #     tf.summary.trace_export(
            #             name="my_func_trace",
            #             step=0,
            #             profiler_outdir=logdir)

        if self.update_interval > 1:
            # hard update
            self._hard_update_target_model()
        else:
            # soft update
            self._soft_update_target_model()
        self.step += 1
        return loss

    def _predict_on_batch(
            self,
            states,
            observations,
            masks,
            model):
        q_values = model([observations, states, masks])
        return q_values

    def _compute_q_values(self, state):
        q_values = self.target_model.predict(np.array([state]))
        return q_values[0]

    # @tf.function
    def _train_on_batch(self, states, observations, masks, targets):
        with tf.GradientTape() as tape:
            tape.watch(observations)
            tape.watch(states)
            y_preds = self.model([observations, states, masks])
            loss_value = self.loss_fn(targets, y_preds)

        self.last_q_values = y_preds  # @todo
        self.last_targets = targets  # @todo
        grads = tape.gradient(loss_value, self.trainable_variables)

        self.optimizer.apply_gradients(
            zip(grads, self.trainable_variables))

        return loss_value.numpy()

    def _hard_update_target_model(self):
        """ for hard update """
        if self.step % self.update_interval == 0:
            self.target_model.set_weights(self.model.get_weights())
        for agent in self.agents:
            agent._hard_update_target_model()

    def _soft_update_target_model(self):
        target_model_weights = np.array(self.target_model.get_weights())
        model_weights = np.array(self.model.get_weights())
        new_weight = (1. - self.update_interval) * target_model_weights \
            + self.update_interval * model_weights
        self.target_model.set_weights(new_weight)
        for agent in self.agents:
            agent._soft_update_target_model()

    def get_qmix_output(self):
        """
            for debug
        """
        obs = np.array([[[[0., 0., 1.]]], [[[0., 0., 1.]]]])
        st = np.array([[0., 0., 1.]])
        mk = np.array([[[1., 0.]], [[1., 0.]]])

        obs = tf.convert_to_tensor(obs, dtype=np.float32)
        st = tf.convert_to_tensor(st, dtype=np.float32)
        mk = tf.convert_to_tensor(mk, dtype=np.float32)

        result = {}
        result[(0, 0)] = round(self.model([obs, st, mk]).numpy()[0][0], 2)

        mk = np.array([[[1., 0.]], [[0., 1.]]])
        mk = tf.convert_to_tensor(mk, dtype=np.float32)
        result[(0, 1)] = round(self.model([obs, st, mk]).numpy()[0][0], 2)

        mk = np.array([[[0., 1.]], [[1., 0.]]])
        mk = tf.convert_to_tensor(mk, dtype=np.float32)
        result[(1, 0)] = round(self.model([obs, st, mk]).numpy()[0][0], 2)

        mk = np.array([[[0., 1.]], [[0., 1.]]])
        mk = tf.convert_to_tensor(mk, dtype=np.float32)
        result[(1, 1)] = round(self.model([obs, st, mk]).numpy()[0][0], 2)

        return result

import copy


class TwoStepEnv:

    def __init__(self):
        self.step_num = 0
        self.state = 0
        self.prev_state = 0

    def step(self, actions):
        self.prev_state = copy.deepcopy(self.state)
        if self.state == 0:
            if actions[0] == 0:
                self.state = 1
                return 1, 0, False
            else:
                self.state = 2
                return 2, 0, False
        elif self.state == 1:
            self.state = 0
            return self.state, 7, True
        elif self.state == 2:
            self.state = 0
            if actions[0] == 0 and actions[1] == 0:
                reward = 0
            elif actions[0] == 0 and actions[1] == 1:
                reward = 1
            elif actions[0] == 1 and actions[1] == 0:
                reward = 1
            elif actions[0] == 1 and actions[1] == 1:
                reward = 8
            return self.state, reward, True
        else:
            raise Exception('invalid state:{}'.format(self.state))

    def reset(self):
        self.state = 0

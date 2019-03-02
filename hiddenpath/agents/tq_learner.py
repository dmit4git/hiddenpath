import random
import time
import sys
import pickle
import numpy
from gym.spaces import Discrete, MultiDiscrete, Box

class TabularQ:

    def __init__(self, env):
        self.environment = env
        self.action_space = env.action_space
        self.settings = {'angle_step': 1, 'force_step': 1,
                         'save_epoch': 0, 'loaded_episode': 0,
                         'performance_run': 100,
                         'save_directory': '',
                         'mode': 'exploration'}
        self.action_ranges = []
        self.dtype = numpy.int32
        self.chosen_action = self.np_array_action
        self.choose_random_action = self.sample
        if isinstance(self.action_space, Discrete):
            self.action_ranges.append(range(0, self.action_space.n))
            self.chosen_action = self.integer_action
            self.choose_random_action = self.sample_list
        elif isinstance(self.action_space, MultiDiscrete):
            nvec = self.action_space.nvec
            for i in range(0, len(nvec)):
                self.action_ranges.append(range(0, nvec[i]))
        elif isinstance(self.action_space, Box):
            low, high = self.action_space.low, self.action_space.high
            for i in range(0, len(low)):
                lw = numpy.ceil(low[i])
                hg = numpy.floor(high[i])
                self.action_ranges.append(range(int(lw), int(hg + 1)))
            self.dtype = numpy.float32
        self.an_action = [0] * len(self.action_ranges)
        #
        self.q_learner = {'epsilon': 0.2, 'alpha': 1.0, 'gamma': 1.0}
        self.q_values = {}
        self.max_q = 0.0
        #
        self.episode = 0
        #
        self.state = []
        self.prev_state = []
        self.action = [0, 0]
        self.last_max_q_action = [0, 0]
        self.reward = 0.0
        self.total_reward = 0.0
        self.best_total_reward = sys.float_info.max * (-1)
        #
        random.seed()

    def reset(self, observation):
        if self.episode > 0:
            if self.total_reward > self.best_total_reward:
                self.best_total_reward = self.total_reward
            print(f"(learner) {time.strftime('%H:%M:%S')} : episode #{self.episode}; "
                  f"total reward = {self.total_reward}, best so far is {self.best_total_reward}")
        self.episode += 1
        self.performance_run()
        self.save()
        self.state = observation
        self.total_reward = 0.0
        return self.choose_action(True)

    def performance_run(self):
        nth = self.episode % self.settings['performance_run']
        if nth == 1:
            self.settings['mode'] = 'performance'
        elif nth == 2:
            self.settings['mode'] = 'exploration'

    def choose_action(self, initial=False):
        if self.settings['mode'] == 'exploration':
            epsilon = random.random()
            if epsilon > self.q_learner['epsilon']:
                if initial:
                    self.get_max_q(self.state)
                else:
                    self.action = self.last_max_q_action
            else:
                self.choose_random_action()
        elif self.settings['mode'] == 'performance':
            self.get_max_q(self.state)
        return self.chosen_action()

    def np_array_action(self):
        return numpy.asarray(self.action, dtype=self.dtype)

    def integer_action(self):
        return self.action[0]

    def step(self, observation, reward):
        self.prev_state = self.state
        self.state = observation
        self.reward = reward
        self.total_reward += reward
        if self.settings['mode'] == 'exploration':
            self.update_q_value()
        return self.choose_action()

    def get_max_q(self, state: list):
        self.max_q = sys.float_info.max * (-1)
        self.rec_max_q(0)
        self.last_max_q_action = self.action
        return self.max_q

    def rec_max_q(self, rec):
        if rec < len(self.action_ranges):
            for a in self.iter_from_middle(self.action_ranges[rec]):
                self.an_action[rec] = a
                self.rec_max_q(rec + 1)
        else:
            q = self.q_values.get((tuple(self.state), tuple(self.an_action)), 0.0)
            if q > self.max_q:
                self.max_q = q
                self.action = list(self.an_action)

    def iter_from_middle(self, lst):
        try:
            middle = int(numpy.floor(len(lst) / 2))
            yield lst[middle]

            for shift in range(1, middle + 1):
                # order is important!
                yield lst[middle - shift]
                yield lst[middle + shift]

        except IndexError:  # occures on lst[len(lst)] or for empty list
            raise StopIteration

    def sample(self):
        self.action = self.action_space.sample()

    def sample_list(self):
        self.sample()
        self.action = [self.action]

    def update_q_value(self):
        alpha = self.q_learner['alpha']
        gamma = self.q_learner['gamma']
        action = tuple(self.action)
        current_q = self.q_values.get((tuple(self.prev_state), action), 0.0)
        next_max_q = self.get_max_q(self.state)
        new_q = current_q + alpha * (self.reward + gamma * next_max_q - current_q)
        self.q_values[(tuple(self.prev_state), action)] = new_q

    def save(self):
        if self.settings['save_epoch'] <= 0:
            return
        if self.episode % self.settings['save_epoch'] == 0 and \
                self.episode > self.settings['loaded_episode']:
            path = self.settings['save_directory']
            if path == '':
                print(f"save_directory is not set")
                return
            file_name = "E" + str(self.episode) + "-T" + time.strftime('%H%M') + '.ptq'
            print("(learner)" + time.strftime('%H:%M:%S') + ": Saving " + path + file_name)
            save_file = path + file_name
            with open(save_file, 'wb') as save_file:
                ple = (self.q_values, self.episode, self.settings, self.last_max_q_action)
                pickle.dump(ple, save_file, pickle.HIGHEST_PROTOCOL)

    def load(self, file_name):
        path = self.settings['save_directory']
        if path == '':
            print(f"save_directory is not set")
            return
        print(f"(learner) Loading knowledge '{path}{file_name}'")
        file_name = path + file_name
        with open(file_name, 'rb') as load_file:
            (self.q_values, self.episode, self.settings, self.last_max_q_action) = pickle.load(load_file)
        self.settings['loaded_episode'] = self.episode

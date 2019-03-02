"""
HiddenPath
"""

import gym
from gym import spaces
import math
import sys
import numpy as np


class HiddenPath(gym.Env):

    def __init__(self):
        # gym environment variables
        self.state = [[], []]  # previous and current state (for reward calculation)
        self.observation = []
        self.move = []
        self.action = {}
        self.walk = {'last': [], 'best': []}
        self.reward = 0
        self.total_reward = {'last': sys.float_info.max * (-1), 'best': sys.float_info.max * (-1)}
        self.done = False
        self.info = {}
        self.episode = 0
        self.act = 0  # time step number
        self.settings = {'action_space': 'MultiDiscrete',  # 'Discrete', 'MultiDiscrete', 'Box'
                         'observation_space': 'MultiDiscrete',  # 'MultiDiscrete', 'Box'
                         'max_angle': 75, 'angle_space_size': 3,
                         'min_force': 2, 'max_force': 10, 'force_space_size': 2,
                         'visualization': 'multiprocess',  # None, 'classic', 'multiprocess'
                         'boundaries_color': (0.0, 0.0, 0.0, 1.0),  # RGBA
                         'path_color': (0.0, 0.8, 0.0, 1.0),
                         'last_walk_color': (0.8, 0.0, 0.0, 0.5),
                         'best_walk_color': (0.8, 0.8, 0.0, 0.5),
                         'reward_mode': 'steps_only',  # see comments in calculate_reward()
                         'draw_mode': 'act'}  # 'exploration' 'performance'
        self.transform = {}
        self.set_action_space()
        self.observation_space = spaces.MultiDiscrete([1, 1])  # for wrappers only, actually set after world is loaded
        # world variables
        self.world_image = {}  # stores world image properties
        self.pixel = {}  # stores color components of a pixel during world image parsing
        self.path_coords = []  # 2D list [[x, y]], stores coords of the path
        self.path_y_values = []  # [[]], stores y values of path for every x value
        # rendering variables
        self.renderer = None  # initiated in reset()
        self.queue = None
        self.semaphore = None
        self.timer = 0.0
        self.act_time = 0.0
        self.drawings = []

    def set_action_space(self, space=None):
        if space is not None:
            self.settings['action_space'] = space
        if self.settings['action_space'] == 'MultiDiscrete':
            self.action_space = spaces.MultiDiscrete([self.settings['angle_space_size'],
                                                      self.settings['force_space_size']])
            self.get_angle_force = self.get_multidiscrete_action
        elif self.settings['action_space'] == 'Discrete':
            self.action_space = spaces.Discrete(self.settings['angle_space_size'])
            self.get_angle_force = self.get_discrete_action

        elif self.settings['action_space'] == 'Box':
            self.action_space = spaces.Box(low=np.array([0, 0]),
                                           high=np.array([self.settings['angle_space_size'],
                                                          self.settings['force_space_size']]),
                                           dtype=np.float32)
            self.get_angle_force = self.get_box_action
        self.make_transformers()

    def get_dict_action(self):
        return self.int_values_to_action(self.action['angle'], self.action['force'])

    def get_discrete_action(self):
        return self.int_values_to_action(self.action, self.settings['force_space_size'] - 1)

    def get_multidiscrete_action(self):
        return self.int_values_to_action(self.action[0], self.action[1])

    def get_box_action(self):
        return self.float_values_to_action(self.action[0], self.action[1])

    def int_values_to_action(self, av, fv):
        angle = (av - self.transform['angle_mid']) * self.transform['angle_step']
        force = self.settings['min_force'] + fv * self.transform['force_step']
        return angle, force

    def float_values_to_action(self, av, fv):
        angle = av * self.transform['angle_factor'] - self.settings['max_angle']
        force = self.settings['min_force'] + fv * self.transform['force_factor']
        return angle, force

    def set_observation_space(self, space=None):
        if space is not None:
            self.settings['observation_space'] = space
        width, height = self.world_image['width'], self.world_image['height']
        if self.settings['observation_space'] == 'MultiDiscrete':
            self.observation_space = spaces.MultiDiscrete([width, height])
            self.get_distance_to_path = self.discrete_deviation
        elif self.settings['observation_space'] == 'Box':
            self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([width, height]),
                                                dtype=np.float32)
            self.get_distance_to_path = self.continuous_deviation

    def make_transformers(self):
        angle_spsz = self.settings['angle_space_size']
        if self.settings['action_space'] == 'Discrete' or self.settings['action_space'] == 'MultiDiscrete':
            self.transform['angle_mid'] = math.floor(angle_spsz / 2)
            if angle_spsz > 1:
                self.transform['angle_step'] = (self.settings['max_angle'] * 2) / (angle_spsz - 1)
            else:
                self.transform['angle_step'] = 1
            if self.settings['force_space_size'] == 1:
                self.settings['min_force'] = self.settings['max_force']
                self.transform['force_step'] = 0
            else:
                force_space = self.settings['max_force'] - self.settings['min_force']
                self.transform['force_step'] = force_space / (self.settings['force_space_size'] - 1)
        elif self.settings['action_space'] == 'Box':
            self.transform['angle_factor'] = (self.settings['max_angle'] * 2) / float(angle_spsz)
            force_space = self.settings['max_force'] - self.settings['min_force']
            self.transform['force_factor'] = force_space / float(self.settings['force_space_size'])

    # parse world image and store path coordinates in self.path_coords
    # also sets self.world_image and self.state_space
    def set_world(self, image_file, pixel_format='RGBA', is_path=None):
        print(f"(environment) Processing world '{image_file}'")
        import pyglet
        image = pyglet.image.load(image_file)
        width, height = image.width, image.height
        pixel_len = len(pixel_format)

        # https://pyglet.readthedocs.io/en/pyglet-1.3-maintenance/programming_guide/image.html#accessing-or-providing-pixel-data
        # get image data and read components offsets in pixel data
        image = image.get_image_data().get_data(pixel_format, width * pixel_len)
        color_components = ('Red', 'Green', 'Blue', 'Alpha', 'Luminance', 'Intensity')
        offsets = {}
        for i in range(pixel_len):
            for color in color_components:
                if pixel_format[i] == color[0]:
                    offsets[color] = i

        # parse all pixels and save coords if a pixel is path
        if not is_path:
            is_path = lambda: self.pixel['Green'] == 255
        for x in range(width):
            self.path_y_values.append([])
            for y in range(height):
                pixel_num = (y * width + x) * pixel_len
                for color, offset in offsets.items():
                    self.pixel[color] = image[pixel_num + offset]
                try:
                    path_spotted = is_path()
                except KeyError:
                    raise KeyError("Probably \
                    no such color component was specified in 'format' argument for 'set_world()'")
                if path_spotted:
                    self.path_coords.append([x, y])
                    self.path_y_values[x].append(y)

        # save world properties and set self.state_space
        self.world_image['width'], self.world_image['height'] = width, height
        self.world_image['size'] = max(width, height)
        self.set_action_space()
        self.set_observation_space()
        self.draw_path()

    # Step the environment by one timestep. Returns observation, reward, done, info.
    def step(self, action):
        self.process_action(action)
        self.act += 1
        self.reward = 0.0
        self.action_to_vector()
        self.environment_effect()
        self.transfer_to_new_state()
        self.calculate_reward()
        self.total_reward['last'] += self.reward
        self.make_observation()
        self.draw_progress()
        return self.observation, self.reward, self.done, self.info

    def process_action(self, action):
        if isinstance(action, tuple):  # tensorforce gives tuples for action
            action = np.array(action)
        self.action = action
        if not self.action_space.contains(self.action):
            raise ValueError('Invalid action. Check action_space of the environment.')

    # translate angle and force into vector
    def action_to_vector(self):
        angle, force = self.get_angle_force()
        dx = math.cos(math.radians(angle)) * force
        dy = math.sin(math.radians(angle)) * force * (-1)  # angle is clockwise
        if self.settings['observation_space'] == 'MultiDiscrete':
            dx, dy = round(dx), round(dy)
        self.move = [dx, dy]

    # affect outcome of an action somehow, makes learning more difficult
    def environment_effect(self):
        pass

    # transfer to new state, check if move leads to out of bounds
    def transfer_to_new_state(self):
        self.state[0] = self.state[1]
        self.state[1] = [self.state[0][0] + self.move[0], self.state[0][1] + self.move[1]]
        self.check_out_of_bounds()

    def check_out_of_bounds(self):
        if self.state[1][0] < 0:  # check left boundary
            self.reward -= (self.world_image['width'] - self.state[1][0]) * self.world_image['height']
            self.state[1][0] = 0
            self.done = True
        elif self.state[1][1] < 0 or self.state[1][1] >= self.world_image['height']:  # check top and bottom boundary
            self.reward -= (self.world_image['width'] - self.state[1][0]) * self.world_image['height']
            if self.state[1][1] < 0:
                self.state[1][1] = 0
            if self.state[1][1] >= self.world_image['height']:
                self.state[1][1] = self.world_image['height'] - 1
            self.done = True
        elif self.state[1][0] >= self.world_image['width']:  # check right boundary (finish)
            self.state[1][0] = self.world_image['width'] - 1
            self.done = True
        self.reward += self.move[0]

    def calculate_reward(self):
        if self.move == [0, 0]:
            return self.reward  # no move - no punishment
        if self.settings['reward_mode'] == 'steps_only':  # calculate reward only for state that agent ends up in
            self.reward -= self.get_distance_to_path(self.state[1])
        else:  # calculate reward for every state on the line from previous state to the new one
            if abs(self.move[0]) > abs(self.move[1]):
                i_ind, h_ind = 0, 1
            else:
                i_ind, h_ind = 1, 0
            dh = self.move[h_ind] / abs(self.move[i_ind])
            coord = [0, 0]
            i, i_step = 0, self.move[i_ind] / abs(self.move[i_ind])
            while i != self.move[i_ind]:
                i += i_step
                coord[i_ind] = round(self.state[0][i_ind] + i)
                coord[h_ind] = round(self.state[0][h_ind] + abs(i) * dh)
                r = self.get_distance_to_path(coord)
                self.reward -= r

    def discrete_deviation(self, coord: list):
        x, y = int(coord[0]), coord[1]
        if 0 <= x < self.world_image['size']:
            y_dist_min = distance = self.get_y_dist(y, self.path_y_values[x])
        else:
            y_dist_min = distance = self.world_image['size']
        search_x = 1
        while search_x <= distance:
            if 0 <= x + search_x < len(self.path_y_values):  # check right world bound
                y_dist_on_right = self.get_y_dist(y, self.path_y_values[x + search_x])
            else:
                y_dist_on_right = self.world_image['size']
            if 0 <= x - search_x < len(self.path_y_values):  # check left world bound
                y_dist_on_left = self.get_y_dist(y, self.path_y_values[x - search_x])
            else:
                y_dist_on_left = self.world_image['size']
            y_dist = min(y_dist_on_right, y_dist_on_left)
            if y_dist < y_dist_min:  # there can't be shorter distance on larger y_dist
                y_dist_min = y_dist
                found_dist = math.hypot(search_x, y_dist)
                if found_dist < distance:
                    distance = found_dist  # save found distance
            search_x += 1
        return distance

    def continuous_deviation(self, coord: list):
        x, y = coord[0], coord[1]
        x_int = int(round(x))
        x_off = x - x_int
        if 0 <= x_int < self.world_image['size']:
            y_dist = self.get_y_dist(y, self.path_y_values[x_int])
            distance = math.hypot(x_off, y_dist)
        else:
            distance = self.world_image['size']
        search_x = 1
        while search_x <= distance:
            if 0 <= x_int + search_x < len(self.path_y_values):  # check right world bound
                y_dist_on_right = self.get_y_dist(y, self.path_y_values[x_int + search_x])
                dist_on_right = math.hypot(search_x - x_off, y_dist_on_right)
            else:
                dist_on_right = self.world_image['size']
            if 0 <= x_int - search_x < len(self.path_y_values):  # check left world bound
                y_dist_on_left = self.get_y_dist(y, self.path_y_values[x_int - search_x])
                dist_on_left = math.hypot(search_x + x_off, y_dist_on_left)
            else:
                dist_on_left = self.world_image['size']
            found_dist = min(dist_on_right, dist_on_left)
            if found_dist < distance:
                distance = found_dist  # save found distance
            search_x += 1
        return distance

    # find closest value in list
    def get_y_dist(self, y, a_list: list):
        # lists of y coords are sorted but expected to be short, no need for binary search
        dist = self.world_image['size']  # max value in case no path on this x (list is empty)
        for y_path in a_list:
            diff = abs(y - y_path)
            if diff < dist:
                dist = diff
        return dist

    # Reset the environment's state. Returns observation.
    def reset(self):
        self.reset_renderer()
        self.show_best_walk()
        self.walk['last'].clear()
        self.set_initial_state()
        self.render()
        self.done = False
        self.episode += 1
        self.act = 0
        return self.observation

    def show_best_walk(self):
        if self.total_reward['last'] > self.total_reward['best']:
            self.total_reward['best'] = self.total_reward['last']
            self.add_to_drawings([], 'reset_batch', (), 'best_walk')
            self.add_walk_to_drawings('best')
        self.total_reward['last'] = 0
        self.walk['best'].clear()

    def reset_renderer(self):
        if self.settings['visualization']:  # reset to constant drawings (delete walk)
            if not self.renderer:
                self.initialize_renderer()
                self.draw_path()
            if self.settings['draw_mode'] == 'act':
                self.add_to_drawings([], 'reset_batch', (), 'last_walk')

    def initialize_renderer(self):
        print(f"(environment) Starting {self.settings['visualization']} renderer")
        if self.settings['visualization'] == 'multiprocess':
            import multiprocessing
            # without 'spawn' context instantiation of pyglet.window.Window hangs
            # if pyglet was imported before in another process as in set_world()
            # multiprocessing.set_start_method('spawn')
            mp_context = multiprocessing.get_context('spawn')
            self.queue = mp_context.Queue(1)
            self.semaphore = mp_context.Semaphore(0)
            self.renderer = mp_context.Process(target=self.mp_renderer)
            self.renderer.start()
            self.semaphore.acquire()  # wait for renderer process to get ready
        elif self.settings['visualization'] == 'classic':
            # initialize renderer with window size same as world size
            from hiddenpath.envs.renderer import PygletRenderer
            self.renderer = PygletRenderer(width=self.world_image['width'], height=self.world_image['height'],
                                           pathfinder=self)

    def mp_renderer(self):
        from hiddenpath.envs.renderer import PygletRenderer
        self.renderer = PygletRenderer(width=self.world_image['width'], height=self.world_image['height'],
                                       pathfinder=self, queue=self.queue, semaphore=self.semaphore)
        self.renderer.app_run()

    def draw_path(self):
        if not self.renderer:
            return
        boundaries = [[-1, -1], [-1, self.world_image['height']],
                      [self.world_image['width'], self.world_image['height']], [self.world_image['width'], -1]]
        self.add_to_drawings(boundaries, 'line_loop', self.settings['boundaries_color'], 'const')
        self.add_to_drawings(self.path_coords, 'square_points', self.settings['path_color'], 'const')
        self.render()

    # transforms current state into observation, can distort it to emulate POMDP
    def make_observation(self):
        observ = self.state_to_observation()
        if self.settings['visualization']:
            for _, walk in self.walk.items():
                walk.append(list(observ))
        if self.settings['observation_space'] == 'MultiDiscrete':
            self.observation = [int(observ[0]), int(observ[1])]
        elif self.settings['observation_space'] == 'Box':
            self.observation = np.array(observ, dtype=np.float32)

    def state_to_observation(self):
        return self.state[1]  # no distortion, agent observes state as it is

    # sets initial state in the beginning of each episode (used in reset())
    def set_initial_state(self):
        y = round(self.world_image['height'] / 2)
        y = float(y)
        self.state = [[0.0, y], [0.0, y]]
        self.make_observation()

    # Visualise current state of the environment
    def render(self, mode='human'):
        if self.settings['visualization'] == 'multiprocess':
            if not self.queue.full():
                self.queue.put(list(self.drawings))
                self.drawings.clear()
        elif self.settings['visualization'] == 'classic':
            self.renderer.switch_to()
            self.renderer.dispatch_events()
            self.renderer.draw_frame()

    def add_to_drawings(self, coords: list, draw_as: str, color: tuple, batch_name):
        if self.settings['visualization'] == 'multiprocess':
            self.drawings.append((coords, draw_as, color, batch_name))
        elif self.settings['visualization'] == 'classic':
            if draw_as == 'new_batch' or draw_as == 'reset_batch':
                self.renderer.new_batch(batch_name)
            elif draw_as == 'reset_to_const':
                self.renderer.reset_to_const()
            else:
                self.renderer.add_to_drawings(coords, draw_as, color, batch_name)

    def add_walk_to_drawings(self, walk_name='last'):
        walk_color = self.settings[walk_name + '_walk_color']
        if self.settings['visualization'] == 'multiprocess':
            self.drawings.append((list(self.walk[walk_name]), 'walk', walk_color, walk_name + '_walk'))
            self.walk[walk_name].clear()
        elif self.settings['visualization'] == 'classic':
            self.renderer.add_to_drawings(list(self.walk[walk_name]), 'walk', walk_color, walk_name + '_walk')
            self.walk[walk_name].clear()

    def draw_progress(self):
        if self.settings['visualization'] is None:
            return
        if self.settings['draw_mode'] == 'act':
            self.add_walk_to_drawings()
        elif self.settings['draw_mode'] == 'episode' and self.done:
            self.add_to_drawings([], 'reset_to_const', (), None)
            self.add_walk_to_drawings()
        self.render()

    def save_path(self, filename):
        print(f"(environment) Saving world to {filename}")
        import pickle
        with open(filename, 'wb') as save_file:
            pickle.dump((self.world_image, self.path_coords, self.path_y_values),
                        save_file, pickle.HIGHEST_PROTOCOL)

    def load_path(self, filename):
        print(f"(environment) Loading world from {filename}")
        import pickle
        with open(filename, 'rb') as load_file:
            self.world_image, self.path_coords, self.path_y_values = pickle.load(load_file)
        self.set_action_space()
        self.set_observation_space()
        self.draw_path()

    def close_renderer(self):
        if self.settings['visualization'] == 'multiprocess':
            self.add_to_drawings([], 'close', (), '')
            self.render()
            self.renderer.terminate()


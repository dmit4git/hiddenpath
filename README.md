# HiddenPath

Highly customizable OpenAI Gym Environment with asynchronous visualisation.
An agent learns to follow unobserved path, reward is given for advancing forward and penalty is provided for deviating from the path.
This short [demo](https://www.youtube.com/watch?v=-EQypmIiM2E&feature=youtu.be)  shows the environment in action.
![Colors meaning](farm8.staticflickr.com/7848/33376705138_0943a4e510_b.jpg  "Colors meaning")

## Features

- Visualization runs in a separate process. If execution of environment or learner is stopped for debugging, visualization window remains responsive and lets you resize pan and zoom.
- World where agent is to learn can be loaded from an image or a file saved by the environment.
- Environment supports various spaces for compatibility with different agents. Action space can be set as `Discrete`, `MultiDiscrete` or `Box`; observation space can be set as `MultiDiscrete` or `Box`.

## User guide

These instructions will get you a copy of the project up and running.

### Prerequisites

Environment was written using python 3.6. The only dependency is pure pythonian [pyglet](https://bitbucket.org/pyglet/pyglet/wiki/Home) .

```
pip3 install pyglet
```

### Installing

To use the environment you need to clone this git project 

```
git clone https://github.com/dmit4git/hiddenpath
```

and install it with pip

```
cd hiddenpath
pip3 install -e .
```

### Environment mechanics

The essential goal of a learner in RL concept is to approximate an optimal (in terms of total reward) function that maps observations to actions, this environment is somewhat intuitive illustration to that principle. Agent approximates path that can be easily made by a user.
The environment is a 2D world with path unobservable to the agent. At every episode the agent starts in the midle of the left border. At every step the environment provides the agent with coordinates of its current location, the agent choses direction where to move and size of the step. Agents goal is to travel from left to right following the path as close as possible. The environment rewards the agent for advancing forward (horizontally to the right on visualization) and punishes for deviating from the path.

#### Reward

The reward formula is **dx** - **distance** where **dx** is x (horizontal) component of move vector and **distance** is the distance. The agent essentially learns how to translate given coordinates of its location into move to minimize punishment for deviation from the path.
Agent receives heavy penalty for going outside world's boundaries, the penalty is horizontal distance to the right border multiplied by world's height.

### Using the environment

Due to implementation of spawning visualisation in a separate process, there must be explicit entry point in the script:
```
if __name__ == '__main__':
```
In order to get ready-to-use environment you need to import the package
```
import hiddenpath
```
make an instance of the environment
```
env = gym.make('HiddenPath-v0')
```
and load a world, 1024x384.png is available in hiddenpath/worlds/
```
env.set_world('/home/dmitry/Documents/projects/hiddenpath/hiddenpath/worlds/1024x384.png')
```
#### Examples

hiddenpath/examples/school.py gives three examples of using the environment with different agents

##### Custom tabular Q-Learner

First example uses a custom tabular Q-Learner agent included in hiddenpath (hiddenpath/agents/tq_learner.py). This example comprises classic OpenAI gym learning:

```
if __name__ == '__main__':
	from hiddenpath.agents.tq_learner import TabularQ # import agent
	import hiddenpath  # import and register environment
	import gym
	env = gym.make('HiddenPath-v0')  # make environment
	# set world (world size and path) from image
	env.set_world('/home/dmitry/Documents/projects/hiddenpath/hiddenpath/worlds/1024x384.png')
	tq = TabularQ(env)  # make agent 
	while episode <= 2500:
		episode += 1
		done = False
		observation = env.reset()
		action = tq.reset(observation)
		while not done:
			observation, reward, done, info = env.step(action)
			action = tq.step(observation, reward)
	env.close_renderer()  # close visualization window
```

##### Tensorforce

Second example uses [tensorforce](https://github.com/tensorforce/tensorforce)  PPO implementation (requires [tensorflow](https://www.tensorflow.org/))

```
if __name__ == '__main__':
    from tensorforce.agents import PPOAgent
    from tensorforce.execution import Runner
    from tensorforce.contrib.openai_gym import OpenAIGym
    import numpy as np
    import hiddenpath  # registers the environment
    env = OpenAIGym('HiddenPath-v0')
    # configure spaces befor world loading
    env.gym.settings['action_space'] = 'Discrete'
    env.gym.settings['observation_space'] = 'Box'
    env.gym.set_world('/home/dmitry/Documents/projects/hiddenpath/hiddenpath/worlds/1024x384.png')
    # tensorforce used default spaces during OpenAIGym construction, next 2 lines set actual spaces
    env._states = OpenAIGym.state_from_space(space=env.gym.observation_space)
    env._actions = OpenAIGym.action_from_space(space=env.gym.action_space)
    agent = PPOAgent(
        states=env.states,
        actions=env.actions,
        network=[
            dict(type='dense', size=32, activation='tanh'),
            dict(type='dense', size=32, activation='tanh'),
            dict(type='dense', size=32, activation='tanh'),
            dict(type='dense', size=32, activation='tanh'),
            dict(type='dense', size=32, activation='tanh'),
            dict(type='dense', size=32, activation='tanh')
        ],
    )
    r = Runner(agent=agent, environment=env)
    def episode_finished(r):
        print(
            "Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                   reward=r.episode_rewards[-1]))
        return True
    r.run(episodes=100001, episode_finished=episode_finished)
    r.close()
    env.close_renderer()
    print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
        ep=r.episode,
        ar=np.mean(r.episode_rewards[-100:]))
    )
```

##### OpenAI baseline

Last example uses [OpenAI baselines](https://github.com/openai/baselines) implementation of Deep Q-Learner ()
```
if __name__ == '__main__':
    import gym
    from baselines import deepq
    import hiddenpath
    env = gym.make("HiddenPath-v0")
    env.settings['action_space'] = 'Discrete'
    env.settings['observation_space'] = 'Box'
    env.set_world('/home/dmitry/Documents/projects/hiddenpath/hiddenpath/worlds/1024x384.png')
    act = deepq.learn(
        env,
        network='mlp',
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10
    )
    env.close_renderer()
```

### Customization and configuration

This environment is more flexible than most. Keep in mind a general rule: edit settings before loading world and load world before using the environment.

#### Loading world

A world can be loaded from image using method `set_world(file, pixel_format='RGBA', is_path=None)` of the environment. The supported image file types include png, bmp, gif, jpg, and many more, somewhat depending on the operating system. The method traverses given image pixel by pixel and stores coordinates of path pixels, it uses `is_path` method to distinguish path from everything else.

- **file** is file name including path
- **pixel_format** is order of color components in the image, png default is 'RGBA'
- **is_path** is method that determines if a pixel is a path pixel. If no method is given, default criteria is used - path pixels are those which have maximum (255) green color component `is_path = lambda: self.pixel['Green'] == 255`. Any custom method can be used, you can check pixel color components using `env.pixel[component]` entry where `component` is a string key from following set: `'Red', 'Green', 'Blue', 'Alpha', 'Luminance', 'Intensity'`, after analysing a pixel return `True` if it's a path pixel.

Alternitevly, world can be loaded from saved file with `load_path(file)`, it's faster and does not require pyglet. To save world after loading from image, use `save_path(file)` method. The package includes a png image of a world and its saved to ppe file counterpart in hiddenpath/worlds directory.

#### Visualization

Visualization mode can be changed through 'env.settings['visualization']' entry, which have three valid values:

- **'multiprocess'** : (default) visualiztion runs in a separate spawned process. Requires explicit entry point `if __name__ == '__main__':` in the script that uses the environment.
- **'classic'** : environment and visualisation run in a single process. Performance in this mode is significantly inferior comparing to multiprocess. Can be used for testing purposes.
- **None** : no visualization. Environment in this mode with world loaded from saved file does not use pyglet. Can be used on headless systems.

#### Action and observation spaces

For better compatibility with various agents the action and observation spaces can be set as various `gym.spaces` types, action space is particularly flexible.

##### Action space

Action space have six settings: **space type**, **max angle** , **angle space**, **min force**, **max force**, **force space** where angle is the direction of move and force is the step size. **max angle**, **min force** and **max force** define diapasone of angle and force values that are actually available for a move; **angle space** and **force space** define available action values that are translated into move. **max angle** is set through `env.settings['max_angle']`, **angle space** is set through `env.settings['angle_space_size']`, **min force** is set through `env.settings['min_force']`,  **max force** is set through `env.settings['max_force']`, **force space** is set through `env.settings['force_space_size']`.

action **space type** is set through `env.settings['action_space']` entry, valid values are:

- `'MultiDiscrete'` : (default) `gym.spaces.MultiDiscrete` type is used with following `nvec` for possible actions: [**angle space**, **force space**]. Those action values are evenly translated into possible angle values: [**max angle**, **max angle** * (-1)] and force values [**min force**, **max force**]. With default values (**max angle**=75 , **angle space**=3, **min force**=2, **max force**=10, **force space**=2), angle value 0 translates into 75 degrees, 1 into 0, 2 into -75; force value 0 translates into 2, 1 into 10. So action [0, 1] translates into move 75 degres to the left (up on visualization) and  step of 10 units long.
- `'Discrete'` : `gym.spaces.Discrete` type is used. Same as `'MultiDiscrete'`. Only angle choice is required, force is always **max force**.
- `'Box'` : `gym.spaces.Box` type is used with following shape: (low=[0, 0], high=[**angle space**, **force space**]). Real values from that diapasone are evenly translated into possible angle values: [**max angle**, **max angle** * (-1)] and force values [**min force**, **max force**].

##### Observation space

Observation space is simpler, it only has space type setting in `env.settings['observation_space']` entry, valid values are:

- `'MultiDiscrete'` : (default) `gym.spaces.MultiDiscrete` type is used. Agent location coorditaes are rounded.
- `'Box'` : (default) `gym.spaces.Box` type is used. Agent location coorditaes are not rounded.

Visualization is consistent with either type.
 
## License

This project is licensed under the MIT License


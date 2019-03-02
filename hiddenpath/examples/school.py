
if __name__ == '__main__':

    # Basic example
    from hiddenpath.agents.tq_learner import TabularQ
    import hiddenpath  # registers the environment
    import gym
    env = gym.make('HiddenPath-v0')
    # env.settings['visualization'] = 'multiprocess'  # default
    # env.settings['action_space'] = 'MultiDiscrete'  # default
    # env.settings['observation_space'] = 'MultiDiscrete'  # default
    # env.set_world('/home/dmitry/Documents/projects/hiddenpath/hiddenpath/worlds/1024x384.png')
    env.set_world('/home/dmitry/Documents/projects/hiddenpath/hiddenpath/worlds/1024x384.png')
    # env.save_path('/home/dmitry/Documents/projects/hiddenpath/hiddenpath/worlds/1024x384.ppe')
    # env.load_path('/home/dmitry/Documents/projects/hiddenpath/hiddenpath/worlds/1024x384.ppe')
    tq = TabularQ(env)
    # tq.settings['performance_run'] = 100  # default. No exploration (best moves only) every 100th episode
    tq.settings['save_epoch'] = 1500  # default, no progress saving. Save progress after every n-th episode
    # tq.settings['save_directory'] = '/home/dmitry/Documents/'  # save dir must exist for save to work
    # tq.load('E2000-T1748.ptq')
    episode = 0
    while episode <= 2500:
        episode += 1
        done = False
        observation = env.reset()
        action = tq.reset(observation)
        while not done:
            observation, reward, done, info = env.step(action)
            action = tq.step(observation, reward)
    env.close_renderer()

    # # tensorforce example
    # from tensorforce.agents import PPOAgent
    # from tensorforce.execution import Runner
    # from tensorforce.contrib.openai_gym import OpenAIGym
    # import numpy as np
    # import hiddenpath  # registers the environment
    # env = OpenAIGym('HiddenPath-v0')
    # # spaces must be set before world set_world/load_path
    # env.gym.settings['action_space'] = 'Discrete'
    # env.gym.settings['observation_space'] = 'Box'
    # env.gym.set_world('/home/dmitry/Documents/projects/hiddenpath/hiddenpath/worlds/1024x384.png')
    # # tensorforce used default spaces during OpenAIGym construction, nex 2 lines set actual spaces
    # env._states = OpenAIGym.state_from_space(space=env.gym.observation_space)
    # env._actions = OpenAIGym.action_from_space(space=env.gym.action_space)
    # agent = PPOAgent(
    #     states=env.states,
    #     actions=env.actions,
    #     network=[
    #         dict(type='dense', size=32, activation='tanh'),
    #         dict(type='dense', size=32, activation='tanh'),
    #         dict(type='dense', size=32, activation='tanh'),
    #         dict(type='dense', size=32, activation='tanh'),
    #         dict(type='dense', size=32, activation='tanh'),
    #         dict(type='dense', size=32, activation='tanh')
    #     ],
    # )
    # r = Runner(agent=agent, environment=env)
    # def episode_finished(r):
    #     print(
    #         "Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
    #                                                                                reward=r.episode_rewards[-1]))
    #     return True
    # r.run(episodes=100001, episode_finished=episode_finished)
    # r.close()
    # env.close_renderer()
    # print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    #     ep=r.episode,
    #     ar=np.mean(r.episode_rewards[-100:]))
    # )

    # openai baseline example
    # import gym
    # from baselines import deepq
    # import hiddenpath
    # env = gym.make("HiddenPath-v0")
    # env.settings['action_space'] = 'Discrete'
    # env.settings['observation_space'] = 'Box'
    # env.set_world('/home/dmitry/Documents/projects/hiddenpath/hiddenpath/worlds/1024x384.png')
    # act = deepq.learn(
    #     env,
    #     network='mlp',
    #     lr=1e-3,
    #     total_timesteps=100000,
    #     buffer_size=50000,
    #     exploration_fraction=0.1,
    #     exploration_final_eps=0.02,
    #     print_freq=10
    # )
    # env.close_renderer()




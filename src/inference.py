import retro
from .agent import Agent

env = retro.make(game='Airstriker-Genesis', state='Level1')

TRAINING_STEPS = 100

def test_env():
    step = 0
    observation = env.reset()
    # print(env.observation_space)
    # print(env.observation_space.shape)
    # print(env.observation_space.high)
    # print(env.observation_space.high.shape)
    # print(env.observation_space.low)
    while True:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("live steps: ", step)
            step = 0
            env.reset()
        env.render()
        step += 1

def train(sess):
    # test_env()

    agent = Agent(sess)
    step = 0
    for episode in range(TRAINING_STEPS):
        print('rest env at episode: ', episode)
        observation = env.reset()
        while True:
            action = agent.epsilon_action(observation)
            next_observation, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward, next_observation, done)
            if (step > 1000) and (step % 50 == 0):
                print("step: %s, episode: %s, learning..."%(step, episode))
                agent.learn(step)
            if (step > 1000) and (step % 500 == 0):
                agent.save_model()
            observation = next_observation
            step += 1
            if done: break
    env.close()

def eval_play(sess):
    print('testing')
    agent = Agent(sess, False)
    observation = env.reset()
    while True:
        action = agent.greedy_action(observation)
        next_observation, reward, done, info = env.step(action)
        if done:
            break
        observation = next_observation
        env.render()
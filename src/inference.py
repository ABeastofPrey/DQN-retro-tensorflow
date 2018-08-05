import retro
from .agent import Agent

env = retro.make(game='Airstriker-Genesis', state='Level1')

# IMAGE_HEIGHT = 224
# IMAGE_WIDTH = 320
# IMAGE_CHANNELS = 3
# ACTION_SPACE = 12

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
            env.reset()
        env.render()
        step += 1
        if step > 1000:
            env.close()
            break

def train(sess):
    # test_env()
    agent = Agent(sess)
    agent.test()
    # step = 0
    # for episode in range(TRAINING_STEPS):
    #     observation = env.reset()
    #     while True:
    #         action = agent.epsilon_action(observation[np.newaxis,:])
    #         next_observation, reward, done, info = env.step(action)
    #         agent.store_transition(observation, action, reward, next_observation)
    #         if (step > 200) and (step % 5 == 0):
    #             agent.learn()
    #         observation = next_observation
    #         step += 1
    #         if done: break
    # env.close()

def eval(sess):
    print('testing')
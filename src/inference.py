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
            print("live steps: ", step)
            env.close()
            break

def train(sess):
    # test_env()

    agent = Agent(sess)
    # agent.test(env)
    step = 0
    for episode in range(TRAINING_STEPS):
        print(rest env at episode: ', episode)
        observation = env.reset()
        while True:
            action = agent.epsilon_action(observation)
            next_observation, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward, next_observation, done)
            if (step > 200) and (step % 20 == 0):
                print("step: %s, episode: %s, learning..."%(step, episode))
                agent.learn(step)
            if (step > 0) and (step % 500 == 0):
                agent.save_model()
            observation = next_observation
            step += 1
            if done: 
                print('done')
                break
    env.close()

def eval(sess):
    print('testing')
    agent = Agent(sess, False)
import retro
from .agent import Agent
# [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] => right
# [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] => left

env = retro.make(game='Airstriker-Genesis', state='Level1')

TRAINING_ROUNDS = 100
PLAYING_ROUNDS = 10

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
        time.sleep(0.1)
            

def train(sess):
    # test_env()

    agent = Agent(sess, env)
    agent.training(TRAINING_ROUNDS)

def eval_play(sess):
    agent = Agent(sess, env)
    agent.eval_play(PLAYING_ROUNDS)

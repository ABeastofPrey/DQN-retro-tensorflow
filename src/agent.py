import os
import cv2
import random
import numpy as np
import tensorflow as tf
from src import inference
from collections import deque

# 第一层卷积层的尺寸和深度。
CONV1_SIZE = 8
CONV1_DEEP = 32

# 第二层卷积层的尺寸和深度。
CONV2_SIZE = 4
CONV2_DEEP = 64

# 第三层卷积层的尺寸和深度。
CONV3_SIZE = 3
CONV3_DEEP = 64

# 全连接节点的个数
FC1_SIZE = 512

RAW_IMAGE_HEIGHT = 224
RAW_IMAGE_WIDTH = 320
RAW_IMAGE_CHANNELS = 3

IMAGE_HEIGHT = 80
IMAGE_WIDTH = 80
IMAGE_CHANNELS = 1
ACTION_SPACE = 12

GAMMA = 0.99
EPSILON = 0.78
LEARN_RATE = 1e-8

MEMORY_SIZE = 4000
BATCH_SIZE = 200

LOG_PATH = 'logs'
# 模型的存储路径和文件名
MODEL_PATH = "models"
MODEL_NAME = "model.ckpt"

class Agent(object):
    def __init__(self, sess, env):
        self.sess = sess
        self.env = env
        self.memory = deque()
        # 定义存储训练论数的变量。
        # 这个变量不需要计算滑动平均值，所以这里指定这个变量为不可训练的变量（trainable=False）。
        # 在使用tensorflow训练神经网络时，一般会将代表训练轮数的参数指定为不可训练的参数。
        self.global_step = tf.Variable(0, trainable=False, name='global_steps')

        with tf.name_scope('input'):
            self.observations = tf.placeholder(name='observations', shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS], dtype=tf.float32)
            self.actions = tf.placeholder(name='actions', shape=[None, ACTION_SPACE], dtype=tf.float32)
            self.q_target = tf.placeholder(name='q_target', shape=[None], dtype=tf.float32)
        
        self.build_deep_q_network()
    
    def build_deep_q_network(self):
        with tf.variable_scope('layer1_conv1', reuse=False):
            filter = tf.get_variable(name='filter', shape=[CONV1_SIZE, CONV1_SIZE, IMAGE_CHANNELS, CONV1_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable(name='biases', shape=[CONV1_DEEP], initializer=tf.constant_initializer(0.0))
            # self.variable_summaries(filter)
            # self.variable_summaries(biases)
            conv = tf.nn.conv2d(name='conv', input=self.observations, filter=filter, strides=[1, 4, 4, 1], padding='SAME')
            relu1 = tf.nn.relu(tf.nn.bias_add(conv, biases))

        with tf.variable_scope('layer2_conv2', reuse=False):
            filter = tf.get_variable(name='filter', shape=[CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable(name='biases', shape=[CONV2_DEEP], initializer=tf.constant_initializer(0.0))
            # self.variable_summaries(filter)
            # self.variable_summaries(biases)
            conv = tf.nn.conv2d(name='conv', input=relu1, filter=filter, strides=[1, 2, 2, 1], padding="SAME")
            relu2 = tf.nn.relu(tf.nn.bias_add(conv, biases))

        with tf.variable_scope('layer3_conv3', reuse=False):
            filter = tf.get_variable(name='filter', shape=[CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable(name='biases', shape=[CONV3_DEEP], initializer=tf.constant_initializer(0.0))
            # self.variable_summaries(filter)
            # self.variable_summaries(biases)
            conv = tf.nn.conv2d(name='conv', input=relu2, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
            relu3 = tf.nn.relu(tf.nn.bias_add(conv, biases))
                
        with tf.name_scope('reshape_op'):
            shape = relu3.get_shape().as_list() # [1, 28, 40, 64]
            nodes = shape[1] * shape[2] * shape[3] # 71680
            reshaped = tf.reshape(relu3, [-1, nodes]) # Tensor("Reshape:0", shape=(1, 71680), dtype=float32)
        
        with tf.variable_scope('layer4_fc1', reuse=False):
            weights = tf.get_variable(name='weights', shape=[nodes, FC1_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable(name='biases', shape=[FC1_SIZE], initializer=tf.constant_initializer(0.0))
            # self.variable_summaries(weights)
            # self.variable_summaries(biases)
            fc1 = tf.nn.relu(tf.matmul(reshaped, weights) + biases)

        with tf.variable_scope('layer5_fc2'):
            weights = tf.get_variable(name='weights', shape=[FC1_SIZE, ACTION_SPACE], initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable(name='biase', shape=[ACTION_SPACE], initializer=tf.constant_initializer(0.0))
            with tf.name_scope('weights'):
                self.variable_summaries(weights)
            with tf.name_scope('biases'):
                self.variable_summaries(biases)
            self.logit = tf.matmul(fc1, weights) + biases # q_value

        with tf.name_scope('cross_entropy'):
            self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logit, labels=tf.argmax(self.actions, 1)))

        with tf.name_scope('loss'):
            # q_action = tf.reduce_sum(tf.multiply(self.logit, self.actions), 1)
            loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.cross_entropy))
            tf.summary.scalar('loss',loss)

        with tf.name_scope('train_op'):
            self.train = tf.train.AdamOptimizer(LEARN_RATE).minimize(loss, global_step=self.global_step)

        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(LOG_PATH, self.sess.graph)

        # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名。
        ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Successfully loaded:", ckpt.model_checkpoint_path)
        else:
            tf.global_variables_initializer().run()
            print("Could not find old network weights")
    
    def epsilon_action(self, observation):
        action = np.zeros(ACTION_SPACE)
        if random.random() < EPSILON:
            q_value = self.logit.eval(feed_dict={self.observations: observation[np.newaxis,:]})
            index = np.argmax(q_value[0])
        else:
            index = random.randrange(ACTION_SPACE)
        action[index] = 1
        return action

    def greedy_action(self, observation):
        action = np.zeros(ACTION_SPACE)
        q_value = self.logit.eval(feed_dict={self.observations: observation[np.newaxis,:]})
        index = np.argmax(q_value[0])
        action[index] = 1
        return action
    
    def store_transition(self, observation, action, reward, next_observation, terminal):
        self.memory.append((observation, action, reward, next_observation, terminal))
        if len(self.memory) > MEMORY_SIZE: self.memory.popleft()
    
    def training(self, rounds=100):
        step = 0
        for episode in range(rounds):
            count = 0
            print('rest env at steps: %s,  episode: %s'%(step, episode))
            _observation = self.env.reset()
            observation = self.grayed_resized_process(_observation)
            while True:
                print("step: %s, episode: %s."%(step, episode))
                action = self.epsilon_action(observation)
                _next_observation, reward, done, info = self.env.step(action)
                next_observation = self.grayed_resized_process(_next_observation) # handle raw observation
                self.store_transition(observation, action, reward, next_observation, done)
                if (count > 200) and (step % 20 == 0):
                    print("step: %s, episode: %s, learning..."%(step, episode))
                    self.learn(step)
                if (count > 200) and (step % 100 == 0):
                    self.save_model()
                observation = next_observation
                step += 1
                count += 1
                self.env.render()
                if done: break
        self.env.close()
    
    def eval_play(self, rounds=3):
        for episode in range(rounds):
            live_steps = 0
            _observation = self.env.reset()
            observation = self.grayed_resized_process(_observation)
            while True:
                action = self.greedy_action(observation)
                _next_observation, reward, done, info = self.env.step(action)
                next_observation = self.grayed_resized_process(_next_observation)
                if done:
                    print("Live %s steps at episode %s"%(live_steps, episode))
                    break
                observation = next_observation
                self.env.render()
                live_steps += 1

    def learn(self, step):
        batch_count = BATCH_SIZE
        if BATCH_SIZE > len(self.memory):
            batch_count = len(self.memory)
        batch_data = random.sample(self.memory, batch_count)
        cu_obs_batch = [data[0] for data in batch_data]
        action_batch = [data[1] for data in batch_data]
        reward_batch = [data[2] for data in batch_data]
        ne_obs_batch = [data[3] for data in batch_data]

        t_value_batch = []
        q_value_batch = self.logit.eval(feed_dict={self.observations: ne_obs_batch})
        for i in range(batch_count):
            terminal = batch_data[i][4]
            if terminal:
                t_value_batch.append(reward_batch[i])
            else:
                t_value_batch.append(reward_batch[i] + GAMMA * np.max(q_value_batch[i]))

        _, rs = self.sess.run([self.train, self.merged], feed_dict={
            self.observations: cu_obs_batch,
            self.actions: action_batch,
            self.q_target: t_value_batch
        })
        self.writer.add_summary(rs, step)

    def save_model(self):
        print('save model after train times: ', self.sess.run(self.global_step))
        # 保存当前的模型。
        # 这里给出了global_step参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数。
        # 比如"model.ckpt-1000"表示1000轮训练之后得到的模型。
        self.saver.save(self.sess, os.path.join(MODEL_PATH, MODEL_NAME), global_step=self.global_step)

    def variable_summaries(self, var):
        """对一个张量添加多个描述。
        
        Arguments:
            var {[Tensor]} -- 张量
        """
        
        with tf.name_scope('summaries'):
            # mean = tf.reduce_mean(var)
            # tf.summary.scalar('mean', mean) # 均值
            # with tf.name_scope('stddev'):
            #     stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            # tf.summary.scalar('stddev', stddev) # 标准差
            # tf.summary.scalar('max', tf.reduce_max(var)) # 最大值
            # tf.summary.scalar('min', tf.reduce_min(var)) # 最小值
            tf.summary.histogram('histogram', var)

    def grayed_resized_process(self, raw_input):
        resized = cv2.resize(raw_input, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC)
        grayed = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, thsh_img = cv2.threshold(grayed,1, 255, cv2.THRESH_BINARY)
        processed_image = np.reshape(thsh_img, (80, 80, 1))
        return processed_image

    def test(self, env):
        step = 0
        for episode in range(100):
            print('*******episode: ', episode)
            # observation = env.reset()
            observation = np.random.random((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)) 
            while True:
                # action = agent.epsilon_action(observation)
                action = self.epsilon_action(observation)
                # next_observation, reward, done, info = env.step(action)

                next_observation = np.random.random((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)) 
                done = random.choice([True, False])
                reward = random.random()

                # agent.store_transition(observation, action, reward, next_observation, done)
                self.store_transition(observation, action, reward, next_observation, done)
                if (step > 100) and (step % 20 == 0):
                    print("step: %s, ep: %s "%(step, episode))
                    # agent.learn(step)
                    self.learn(step)
                if (step > 0) and (step % 100 == 0): # step should be 500
                    self.save_model()

                observation = next_observation
                step += 1
                if done: 
                    print('done and break')
                    break
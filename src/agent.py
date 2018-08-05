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

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 320
IMAGE_CHANNELS = 3
ACTION_SPACE = 12

GAMMA = 0.99
EPSILON = 0.8
LEARN_RATE = 0.01

MEMORY_SIZE = 500
BATCH_SIZE = 200

LOG_PATH = 'logs'

class Agent(object):
    def __init__(self, sess, is_train=True):
        self.sess = sess
        self.memory = deque()
        self.is_train = is_train

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
            tf.summary.histogram('bias', biases)
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

        with tf.name_scope('loss'):
            # q_action = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=actions))
            q_action = tf.reduce_sum(tf.multiply(self.logit, self.actions), 1)
            loss = tf.reduce_mean(tf.square(self.q_target - q_action))
            tf.summary.scalar('loss',loss)

        with tf.name_scope('train_op'):
            self.train = tf.train.AdamOptimizer(LEARN_RATE).minimize(loss)
        
        if sefl.is_train:
            tf.global_variables_initializer().run()

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(LOG_PATH, self.sess.graph)
            
    
    def epsilon_action(self, observation):
        action = np.zeros(ACTION_SPACE)
        if random.random() < EPSILON:
            q_value = self.logit.eval(feed_dict={self.observations: observation})
            index = np.argmax(q_value[0])
        else:
            index = random.randrange(ACTION_SPACE)
        action[index] = 1
        return action
    
    def store_transition(self, observation, action, reward, next_observation, terminal):
        self.memory.append((observation, action, reward, next_observation, terminal))
        if len(self.memory) > MEMORY_SIZE: self.memory.popleft()
    
    def learn(self):
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

        # self.train.run(feed_dict={
        #     self.observations: cu_obs_batch,
        #     self.actions: action_batch,
        #     self.q_target: t_value_batch
        # })

        _, rs = self.sess.run([self.train, self.merged], feed_dict={
            self.observations: cu_obs_batch,
            self.actions: action_batch,
            self.q_target: t_value_batch
        })
        self.writer.add_summary(rs, 1)

    def variable_summaries(self, var):
        """对一个张量添加多个描述。
        
        Arguments:
            var {[Tensor]} -- 张量
        """
        
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean) # 均值
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev) # 标准差
            tf.summary.scalar('max', tf.reduce_max(var)) # 最大值
            tf.summary.scalar('min', tf.reduce_min(var)) # 最小值
            tf.summary.histogram('histogram', var)

    def test(self):
        observation = np.random.random((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)) 
        for i in range(100):
            print(i)
            action = self.epsilon_action(observation[np.newaxis,:])
            observation_ = np.random.random((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)) 
            done = random.choice([True, False])
            reward = random.random()
            self.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            if i % 5 == 0:
                self.learn()
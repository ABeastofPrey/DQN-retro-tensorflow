import tensorflow as tf
from src.inference import train, eval_play

#第一个是参数名称，第二个参数是默认值，第三个是参数描述
tf.app.flags.DEFINE_integer('train', 1, "whether do training(1) or testing(0)")

FLAGS = tf.app.flags.FLAGS

def main(_):
    with tf.Session() as sess:
        if FLAGS.train == 1: 
            train(sess)
        elif FLAGS.train == 0: 
            eval_play(sess)
        else:
            raise ValueError('train parameter should be 0 or 1')

if __name__ == '__main__':
    tf.app.run()
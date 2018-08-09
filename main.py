import tensorflow as tf
from src.inference import train, eval_play

#第一个是参数名称，第二个参数是默认值，第三个是参数描述
tf.app.flags.DEFINE_boolean('train', True, "whether do training or testing")

FLAGS = tf.app.flags.FLAGS

def main(_):
    with tf.Session() as sess:
        if FLAGS.train: 
            train(sess)
        else: 
            eval_play(sess)

if __name__ == '__main__':
    tf.app.run()

# python3 main.py --train True
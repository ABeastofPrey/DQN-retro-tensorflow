import tensorflow as tf
from src.inference import train, eval_play

#第一个是参数名称，第二个参数是默认值，第三个是参数描述
# tf.app.flags.DEFINE_string('str_name', 'def_v_1', "descrip1")
# tf.app.flags.DEFINE_integer('int_name', 10, "descript2")
# tf.app.flags.DEFINE_boolean('bool_name', False, "descript3")

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

# python3 main.py --str_name test_str --int_name 99 --bool_name True
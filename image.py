# import retro
# import numpy as np
# import tensorflow as tf
# from src.utils import Util

# env = retro.make(game='Airstriker-Genesis', state='Level1')
# observation = env.reset()

# def grayed_resized_process(raw_input, out_shape=[80, 80]):
#     resized_image = tf.image.resize_images(ob_img, out_shape, method=tf.image.ResizeMethod.AREA)
#     grayed_resized_image = tf.image.rgb_to_grayscale(resized_image)
#     return grayed_resized_image

# ob_img = tf.placeholder(name='ob_img', shape=[224, 320, 3], dtype=tf.float32)
# tf.summary.image('raw image', tf.expand_dims(input=ob_img, axis=0))

# resized_image = tf.image.resize_images(ob_img, [200, 200], method=tf.image.ResizeMethod.AREA)
# tf.summary.image('resized image', tf.expand_dims(resized_image, 0))

# grayed_image = tf.image.rgb_to_grayscale(ob_img)
# tf.summary.image('grayed image', tf.expand_dims(grayed_image, 0))

# grayed_resized_image = tf.image.rgb_to_grayscale(resized_image)
# grayed_resized_image = grayed_resized_process(ob_img)

# coord = tf.train.Coordinator()
# merge_all = tf.summary.merge_all()
# with tf.Session() as sess:
#     summary_writer = tf.summary.FileWriter('./tmp', sess.graph)
#     tf.global_variables_initializer().run()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#     feed_dict = { ob_img: observation }

#     # print(sess.run(ob_img, feed_dict=feed_dict).shape)
#     # print(sess.run(resized_image).shape)
#     # print(sess.run(grayed_image).shape)
#     # print(sess.run(grayed_resized_image, feed_dict=feed_dict).shape)
#     # filename_queue.close(cancel_pending_enqueues=True)

#     coord.request_stop()
#     coord.join(threads, stop_grace_period_secs=10)
#     summary_writer.add_summary(sess.run(merge_all, feed_dict=feed_dict), 0)
#     summary_writer.close()
#     env.close()

# print("------------------------------------------------------")

import cv2
import numpy as np

def iminfo(img):
    print("------------------------------------------------------")
    print(img.shape)
    print(img.size)

def main():
    img = cv2.imread('./assets/a-g.png')
    # grayed_resized_process
    resized = cv2.resize(img, (80, 80), interpolation=cv2.INTER_CUBIC)
    grayed = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, thsh_img = cv2.threshold(grayed,1,255,cv2.THRESH_BINARY)
    observation = np.reshape(thsh_img, (80, 80, 1))

    while True:
        # cv2.imshow('raw', img)
        # cv2.imshow('res', resized)
        # cv2.imshow('res', grayed)
        # cv2.imshow('bin', thsh_img)
        cv2.imshow('bin', observation)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # iminfo(img)
            # iminfo(resized)
            # iminfo(grayed)
            # iminfo(thsh_img)
            iminfo(observation)
            print("I'm done")
            break

if __name__ == '__main__':
    main()

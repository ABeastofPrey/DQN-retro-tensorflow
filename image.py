import tensorflow as tf

file_names = ['./assets/a-g.png']
filename_queue = tf.train.string_input_producer(file_names)
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

image = tf.image.decode_png(value)
tf.summary.image('raw image', tf.expand_dims(image, 0))

resized_image = tf.image.resize_images(image, [256, 256], method=tf.image.ResizeMethod.AREA)
tf.summary.image('resized image', tf.expand_dims(resized_image, 0))

grayed_image = tf.image.rgb_to_grayscale(image)
tf.summary.image('grayed image', tf.expand_dims(grayed_image, 0))


grayed_resized_image = tf.image.rgb_to_grayscale(resized_image)
tf.summary.image('grayed_resized image', tf.expand_dims(grayed_resized_image, 0))

coord = tf.train.Coordinator()
merge_all = tf.summary.merge_all()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('./tmp', sess.graph)
    # tf.global_variables_initializer().run()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    print(sess.run(image).shape)
    print(sess.run(resized_image).shape)
    print(sess.run(grayed_image).shape)
    print(sess.run(grayed_resized_image).shape)
    filename_queue.close(cancel_pending_enqueues=True)
    coord.request_stop()
    coord.join(threads)

    summary_writer.add_summary(sess.run(merge_all), 0)
    summary_writer.close()

print("------------------------------------------------------")
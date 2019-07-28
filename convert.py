import numpy as np
import imageio
import tensorflow as tf
import os

from glob import glob
from tqdm import tqdm


size = 64

tf.reset_default_graph()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
    paths = glob("data/half/*.png")

    def load(path):
        image = tf.image.decode_image(tf.read_file(path), 4)
        return tf.data.Dataset.from_tensor_slices([
            tf.random_crop(image, [size + 10, size + 10, 4])
            for _ in range(8)
        ])

    d = tf.data.Dataset.from_tensor_slices(tf.constant(paths))
    d = d.flat_map(load)

    iterator = d.make_one_shot_iterator()
    image = iterator.get_next()

    count = tf.Variable(0, name = 'count')
    count_incrementer = tf.assign(count, count + 1)

    with tf.control_dependencies([count_incrementer]):
        variance = tf.nn.moments(tf.cast(image, tf.float32), [0, 1])[1]
        saver = tf.cond(
            tf.reduce_mean(variance) > 10**2,
            lambda: tf.write_file(
                tf.string_join([
                    "data/cropped/", 
                    tf.as_string(count // 8), "_", tf.as_string(count % 8), 
                    ".png"
                ]),
                tf.image.encode_png(image)
            ),
            lambda: tf.no_op()
        )

    session.run(tf.global_variables_initializer())

    for _ in tqdm(range(len(paths) * 8)):
        try:
            session.run(saver)
        except tf.errors.OutOfRangeError:
            break
        except tf.errors.InvalidArgumentError as e:
            print(e.message)

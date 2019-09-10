import numpy as np
import imageio
import tensorflow as tf
import os

from math import floor, sin, pi
from glob import glob
from tqdm import tqdm

class GANSuperResolution:
    def __init__(
        self, session, continue_train = True, 
        learning_rate = 1e-4,
        batch_size = 32
    ):
        self.session = session
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.continue_train = continue_train
        self.rank = 128
        self.filters = 256
        self.checkpoint_path = "checkpoints"
        self.size = 64
        self.latent_dimensions = 12
        
        self.global_step = tf.Variable(0, name = 'global_step')

        print("lookup training data...")
        self.paths = glob("data/cropped/*.png")
                    
        def load(path):
            image = tf.image.decode_image(tf.read_file(path), 4)
            return tf.random_crop(image, [self.size + 10, self.size + 10, 4])
            
            
        lanczos3 = [
            3 * sin(pi * x) * sin(pi * x / 3) / pi**2 / x**2
            for x in np.linspace(-2.75, 2.75, 12)
        ]
        lanczos3 = [x / sum(lanczos3) for x in lanczos3]
        self.lanczos3_horizontal = tf.constant(
            [
                [[
                    [a if o == i else 0 for o in range(4)]
                    for i in range(4)
                ]] 
                for a in lanczos3
            ]
        )
        self.lanczos3_vertical = tf.constant(
            [[
                [
                    [a if o == i else 0 for o in range(4)]
                    for i in range(4)
                ]
                for a in lanczos3
            ]]
        )   
        
        d = tf.data.Dataset.from_tensor_slices(tf.constant(self.paths))
        d = d.map(load, num_parallel_calls = 16).cache()
        d = d.shuffle(100000).repeat()
        d = d.batch(self.batch_size).prefetch(100)
        
        iterator = d.make_one_shot_iterator()
        self.real = iterator.get_next()
        
        real = self.srgb2xyz(self.real)
        downscaled = self.lanczos3_downscale(real, "VALID")

        # remove padding
        real = real[:, 5:-5, 5:-5, :]
        self.real = self.real[:, 5:-5, 5:-5, :]

        self.downscaled = self.xyz2srgb(downscaled)

        self.tampered = tf.concat(
            [
                tf.map_fn(
                    lambda x: tf.image.random_jpeg_quality(x, 80, 100),
                    self.downscaled[:, :, :, :3]
                ),
                self.downscaled[:, :, :, 3:4] # keep alpha channel
            ], -1
        )
        tampered = self.srgb2xyz(self.tampered)
        
        self.nearest_neighbor = tf.image.resize_nearest_neighbor(
            self.downscaled, [self.size] * 2
        )
        
        
        encoded = self.encode(real)
        quantized = sum([
            tf.clip_by_value(encoded, -1, 1),
            tf.random_normal(tf.shape(encoded), 0.0, 0.125)
        ])
        decoded = self.decode(downscaled, quantized)
        
        # losses
        self.loss = sum([
            tf.reduce_mean(tf.abs(real - decoded)),
        ])
        
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        
        self.optimizer = optimizer.minimize(self.loss, self.global_step)
        
        self.saver = tf.train.Saver(max_to_keep = 2)


        example_path = "example.png"
        example = self.srgb2xyz([tf.random_crop(
            tf.image.decode_image(tf.read_file(example_path), 4), 
            [175, 175, 4]
        )])
        example_latent_size = tf.concat(
            [example.shape[:-1], [self.latent_dimensions]], -1
        )
        example = self.xyz2srgb(self.decode(
            tf.concat([example, example], 0), 
            tf.concat([
                #tf.random_normal(example_latent_size, 0, 0.125) + 
                tf.random_uniform(example_latent_size, -1, 1),
                tf.zeros(example_latent_size)
            ], 0)
        ))
        
        
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar(
            'latent variance', 
            tf.reduce_mean(tf.sqrt(tf.nn.moments(encoded, [1, 2])[1]))
        )
        tf.summary.image(
            'kernel', 
            tf.transpose(
                tf.trainable_variables(
                    "transform/conv0/kernel"
                )[0], 
                [3, 0, 1, 2]
            )[:, :, :, :3],
            24
        )
        tf.summary.image('example', example)

        tf.summary.image('decoded', self.xyz2srgb(decoded))
        tf.summary.image('real', self.xyz2srgb(real))
        
        self.summary_writer = tf.summary.FileWriter('logs', self.session.graph)
        self.summary = tf.summary.merge_all()

        self.session.run(tf.global_variables_initializer())

        # load checkpoint
        if self.continue_train:
            print(" [*] Reading checkpoint...")

            checkpoint = tf.train.get_checkpoint_state(self.checkpoint_path)

            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(
                    self.session,
                    self.checkpoint_path + "/" + os.path.basename(
                        checkpoint.model_checkpoint_path
                    )
                )
                print(" [*] before training, Load SUCCESS ")

            else:
                print(" [!] before training, failed to load ")
        else:
            print(" [!] before training, no need to load ")
            
            
    def lanczos3_downscale(self, x, padding = "SAME"):
        return tf.nn.conv2d(
            tf.nn.conv2d(
                x, 
                self.lanczos3_horizontal, [1, 2, 1, 1], padding
            ), 
            self.lanczos3_vertical, [1, 1, 2, 1], padding
        )
    def lanczos3_upscale(self, x):
        result = tf.nn.conv2d_transpose(
            tf.nn.conv2d_transpose(
                x * 4,
                self.lanczos3_horizontal, 
                tf.shape(x) * [1, 2, 1, 1],
                [1, 2, 1, 1], "SAME"
            ), 
            self.lanczos3_vertical, 
            tf.shape(x) * [1, 2, 2, 1], 
            [1, 1, 2, 1], "SAME"
        )
        result.set_shape(
            [x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]]
        )
        return result
            
    def srgb2xyz(self, c):
        c = (tf.cast(c, tf.float32) + tf.random_uniform(tf.shape(c))) / 256
        c, alpha = tf.split(c, [3, 1], -1)
        c = c * alpha # pre-multiply
        linear = tf.where(
            c <= 0.04045,
            c / 12.92,
            ((c + 0.055) / 1.055)**2.4
        )
        return tf.concat(
            [
                linear * [[[[0.4124564, 0.2126729, 0.0193339]]]] +
                linear * [[[[0.3575761, 0.7151522, 0.1191920]]]] +
                linear * [[[[0.1804375, 0.0721750, 0.9503041]]]],
                alpha
            ], -1
        )
        
    def xyz2srgb(self, xyza):
        xyz, alpha = tf.split(xyza, [3, 1], -1)
        linear = (
            xyz * [[[[3.2404542, -0.9692660, 0.0556434]]]] +
            xyz * [[[[-1.5371385, 1.8760108, -0.2040259]]]] +
            xyz * [[[[-0.4985314, 0.0415560, 1.0572252]]]]
        )
        srgb = tf.where(
            linear <= 0.003131,
            12.92 * linear,
            1.055 * linear**(1 / 2.4) - 0.055
        )
        #srgb = nice_power(linear, 1 / 2.4, 0.003131, 2) - 0.055
        srgb = srgb / tf.maximum(alpha, 1 / 256)
        srgb = tf.concat([srgb, alpha], -1)
        return tf.cast(tf.minimum(tf.maximum(
            srgb * 256, 0
        ), 255), tf.uint8)

    def decode(self, small_images, latent):
        with tf.variable_scope(
            'transform', reuse = tf.AUTO_REUSE
        ):
            x = small_images * 2 - 1
            y = latent
            
            x = tf.concat([x, y], -1)

            x = tf.layers.conv2d(
                x, self.rank,
                [11, 11], [1, 1], 'same', name = 'conv0', use_bias = False
            )
            x = tf.layers.dense(
                x, self.filters, name = 'dense0'
            )

            x = tf.nn.relu(x)

            x = tf.layers.conv2d_transpose(
                x, 4,
                [2, 2], [2, 2], 'same', name = 'conv1'
            )

            return x * 0.5 + 0.5
            
    def encode(self, large_images):
        with tf.variable_scope(
            'discriminate', reuse = tf.AUTO_REUSE
        ):
            x = large_images * 2 - 1

            x = tf.layers.conv2d(
                x, self.rank,
                [22, 22], [2, 2], 'same', name = 'conv0', use_bias = False
            )
            x = tf.layers.dense(
                x, self.filters, name = 'dense0'
            )
            
            x = tf.nn.relu(x)
            
            x = tf.layers.dense(
                x, self.latent_dimensions, name = 'dense1'
            )

            return x
 
    def train(self):
        step = self.session.run(self.global_step)
        
        while True:
            while True:
                try:
                    summary = self.session.run(self.summary)
                    break
                except tf.errors.InvalidArgumentError as e:
                    print(e.message)
        
            self.summary_writer.add_summary(summary, step)
                
            for _ in tqdm(range(1000)):
                while True:
                    try:
                        _, step = self.session.run([
                            [
                                self.optimizer
                            ],
                            self.global_step
                        ])
                        break
                    except tf.errors.InvalidArgumentError as e:
                        print(e.message)
                
            #print("saving iteration " + str(step))
            self.saver.save(
                    self.session,
                    self.checkpoint_path + "/gansr",
                    global_step=step
                )

    def scale_file(self, filename):
        image = tf.Variable(
            tf.image.decode_image(tf.read_file(filename), 3),
            validate_shape = False
        )
        
        tiles = tf.Variable(
            tf.reshape(
                tf.extract_image_patches(
                    [tf.pad(
                        image, [[0, 0], [0, 0], [0, 1]], 
                        constant_values = 255
                    )], 
                    [1, 128, 128, 1], [1, 128, 128, 1], [1, 1, 1, 1], "SAME"
                ), 
                [-1, 128, 128, 4]
            ),
            validate_shape = False
        )
        
        result_tiles = tf.Variable(
            tf.zeros(tf.shape(tiles) * [1, 2, 2, 1], tf.uint8),
            validate_shape = False
        )
        
        index = tf.Variable(0)
        
        self.session.run(image.initializer)
        self.session.run([
            tiles.initializer, result_tiles.initializer, index.initializer
        ])
        
        size = self.session.run(tf.shape(image)[:2])
        print(size)
        
        step = tf.scatter_update(
            result_tiles, [index], 
            self.xyz2srgb(
                self.decode(
                    self.srgb2xyz(
                        tf.reshape([tiles[index, :, :, :]], [1, 128, 128, 4])
                    ), 
                    tf.random_uniform(
                        [1, 128, 128, self.latent_dimensions], -1, 1
                    )
                )
            )
        )
        
        with tf.control_dependencies([step]):
            step = tf.assign_add(index, 1)
    
        tile_count = self.session.run(tf.shape(tiles)[0])
        
        for i in tqdm(range(tile_count)):
            self.session.run(step)
            
        height = ((size[0] - 1) // 128 + 1) * 256
        width =  ((size[1] - 1) // 128 + 1) * 256
            
        r = self.session.run(
            tf.reshape(
                tf.transpose(
                    tf.reshape(
                        tf.transpose(result_tiles, [0, 2, 1, 3]),
                        [-1, width, 256, 4]
                    ), 
                    [0, 2, 1, 3]
                ),
                [-1, width, 4]
            )
        )

        imageio.imwrite("{}_scaled.png".format(filename), r)
        
    
import numpy as np
import scipy.misc as scm
import tensorflow as tf
import os

from math import floor, sin, pi
from glob import glob
from tqdm import tqdm

class GANSuperResolution:
    def __init__(
        self, session, continue_train = True, 
        learning_rate = 2.5e-4, batch_size = 8
    ):
        self.session = session
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.continue_train = continue_train
        self.filters = 64
        self.max_filters = 128
        self.dimensions = 32
        self.checkpoint_path = "checkpoints"
        self.size = 128
        
        # values calculated from 1200 random training tiles
        self.lab_shift = tf.constant(
            [[[[58.190086, 22.409063, -4.994123, 0]]]]
        )
        self.lab_scale = tf.constant(
            [[[[34.140823, 83.433235, 31.84137, 1]]]]
        )

        self.global_step = tf.Variable(0, name = 'global_step')

        # build model
        print("lookup training data...")
        self.paths = glob("data/half/*.png")
                    
        def load(path):
            image = tf.image.decode_image(tf.read_file(path), 4)
            return tf.data.Dataset.from_tensor_slices([
                tf.random_crop(image, [self.size, self.size, 4])
                for _ in range(10)
            ])
            #return tf.data.Dataset.from_tensor_slices(
            #    tf.reshape(
            #        tf.extract_image_patches(
            #            [tf.image.decode_image(tf.read_file(path), 3)],
            #            [1, self.size * 2, self.size * 2, 1], 
            #            [1, self.size // 1, self.size // 1, 1], 
            #            [1, 1, 1, 1], "VALID"
            #        ),
            #        [-1, self.size * 2, self.size * 2, 3]
            #    )
            #)
            
        def tamper(images):
            # operates on batches
            # apply random blurring or sharpening
            blur = [0.06136, 0.24477, 0.38774, 0.24477, 0.06136]
            blur_kernel = tf.constant(
                [[[[a * b] for c in range(3)] for a in blur] for b in blur]
            )
            
            float_images = tf.to_float(images)
            
            blurred = tf.nn.depthwise_conv2d(
                float_images, blur_kernel, [1, 1, 1, 1], "SAME"
            )
            ratio = tf.random_uniform([self.batch_size, 1, 1, 1], -1, 1)
            float_images = ratio * float_images + (1 - ratio) * blurred
            
            float_images = tf.minimum(tf.maximum(float_images, 0), 255)
            
            # apply random jpg compression
            return tf.convert_to_tensor([
                tf.reshape(tf.image.decode_jpeg(tf.image.encode_jpeg(
                    tf.cast(float_images[i, :, :, :], tf.uint8),
                    quality = np.random.randint(10, 100)
                ), 3), [self.size, self.size, 3])
                for i in range(self.batch_size)
            ])
            
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
        d = d.shuffle(100000).repeat()
        d = d.flat_map(load).shuffle(1000)
        d = d.batch(self.batch_size).prefetch(10)
        
        
        iterator = d.make_one_shot_iterator()
        self.real = iterator.get_next()
        #self.tampered = tamper(self.real)
        
        self.moments = tf.nn.moments(
            self.preprocess(self.real), [0, 1, 2]
        )
        self.moments = tf.stack([
            self.moments[0], tf.sqrt(self.moments[1])
        ])
        
        #moments, error1, error2 = self.session.run([
        #    self.moments,
        #    tf.reduce_mean(tf.abs(
        #        tf.to_float(self.real) -
        #        tf.to_float(self.xyz2srgb(self.srgb2xyz(self.real)))
        #    )),
        #    tf.reduce_mean(tf.abs(
        #        tf.to_float(self.real) -
        #        tf.to_float(self.postprocess(self.preprocess(self.real)))
        #    ))
        #])
        #print(moments, "error:", error1, error2)
        #return
        
        real_xyz = self.srgb2xyz(self.real)
        real = self.xyz2ulab(real_xyz)
        #tampered = self.preprocess(self.tampered)
        downscaled_xyz = self.lanczos3_downscale(real_xyz)
        downscaled = self.xyz2ulab(downscaled_xyz)
        
        self.downscaled = self.postprocess(downscaled)
        
        encoded_mean, encoded_deviation = self.encode(real)
        encoded = encoded_mean + encoded_deviation * tf.random_normal(
            tf.shape(encoded_mean)
        )
        reconstructed = self.scale(downscaled, encoded)
        
        #cleaned = self.denoise(tampered)
        scaled_distribution = self.scale(
            downscaled, tf.random_normal(tf.shape(encoded))
        )
        scaled = self.scale(
            downscaled, tf.zeros_like(encoded)
        )
        
        #self.cleaned = self.postprocess(cleaned)
        self.scaled = self.postprocess(scaled)
        self.scaled_distribution = self.postprocess(scaled_distribution)
        self.reconstructed = self.postprocess(reconstructed)
        
        # losses
        def square(x):
            return tf.maximum(tf.square(x), 1e-5)
        
        def divergence(mean, deviation):
            # from
            # https://github.com/shaohua0116/VAE-Tensorflow/blob/master/demo.py
            variance = square(deviation)
            return tf.reduce_mean(
                0.5 * tf.reduce_sum(
                    square(mean) +
                    variance -
                    tf.log(variance) - 1,
                    3
                )
            )
        
        divergence_loss = divergence(encoded_mean, encoded_deviation)
        self.divergence_loss = divergence_loss
        
        self.g_loss = sum([
            1e-2 * tf.maximum(divergence_loss, 1e-2),
            1e-0 * tf.reduce_mean(square(
                (real - reconstructed) * 
                self.lab_scale / tf.reduce_mean(self.lab_scale)
            ))
        ])
        
        self.g_optimizer = tf.train.AdamOptimizer(
            self.learning_rate
        ).minimize(self.g_loss, self.global_step)


        self.saver = tf.train.Saver(max_to_keep = 2)
        
        tf.summary.scalar('generator loss', self.g_loss)
        
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
            
            
    def lanczos3_downscale(self, x):
        return tf.nn.conv2d(
            tf.nn.conv2d(
                x, 
                self.lanczos3_horizontal, [1, 2, 1, 1], "SAME"
            ), 
            self.lanczos3_vertical, [1, 1, 2, 1], "SAME"
        )
    def lanczos3_upscale(self, x):
        return tf.nn.conv2d_transpose(
            tf.nn.conv2d_transpose(
                x * 4,
                self.lanczos3_horizontal, tf.shape(x) * [1, 2, 1, 1],
                [1, 2, 1, 1], "SAME"
            ), 
            self.lanczos3_vertical, tf.shape(x) * [1, 2, 2, 1], 
            [1, 1, 2, 1], "SAME"
        )
            
    def srgb2xyz(self, c):
        c = tf.cast(c, tf.float32) / 255
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
                tf.ones_like(alpha)
            ], -1
        )
        
    def xyz2srgb(self, c):
        c, alpha = tf.split(c, [3, 1], -1)
        linear = (
            c * [[[[3.2404542, -0.9692660, 0.0556434]]]] +
            c * [[[[-1.5371385, 1.8760108, -0.2040259]]]] +
            c * [[[[-0.4985314, 0.0415560, 1.0572252]]]]
        )
        srgb = tf.where(
            linear <= 0.003131,
            12.92 * linear,
            1.055 * linear**(1 / 2.4) - 0.055
        )
        srgb = srgb / tf.maximum(alpha, 1 / 255)
        srgb = tf.concat([srgb, alpha], -1)
        return tf.cast(tf.minimum(tf.maximum(
            srgb * 255, 0
        ), 255), tf.int32)
        
    def xyz2ulab(self, c):
        c, alpha = tf.split(c, [3, 1], -1)
        # https://en.wikipedia.org/wiki/Lab_color_space
        def f(t):
            return tf.where(
                t > 6**3 / 29**3, 
                t**(1 / 3), 
                t / (3 * 6**2 / 29**2) + 4 / 29
            )
        x, y, z = (c[:, :, :, i] for i in range(3))
        x_n, y_n, z_n = 0.9505, 1.00, 1.089
        lab = tf.stack(
            [
                116 * f(y / y_n) - 16,
                500 * (f(x / x_n) - f(y / y_n)),
                200 * (f(y / y_n) - f(z / z_n))
            ],
            -1
        )
        lab = tf.concat([lab, alpha], -1)
        # normalize
        return (lab - self.lab_shift) / self.lab_scale

    def ulab2xyz(self, c):
        # denormalize
        lab = c * self.lab_scale + self.lab_shift
        c, alpha = tf.split(c, [3, 1], -1)
        
        # https://en.wikipedia.org/wiki/Lab_color_space
        def f(t):
            return tf.where(
                t > 6 / 29,
                t**3,
                3 * 6**2 / 29**2 * (t - 4 / 29)
            )
        l, a, b = (lab[:, :, :, i] for i in range(3))
        #l = tf.maximum(l, 0)
        x_n, y_n, z_n = 0.9505, 1.00, 1.089
        l2 = (l + 16) / 116
        xyz = tf.stack([
            x_n * f(l2 + a / 500),
            y_n * f(l2),
            z_n * f(l2 - b / 200)
        ], -1)
        return tf.concat([xyz, alpha], -1)
    
    def preprocess(self, images):
        #return self.srgb2xyz(images)
        return self.xyz2ulab(self.srgb2xyz(images))
        
    def postprocess(self, images):
        #return self.xyz2srgb(images)
        return self.xyz2srgb(self.ulab2xyz(images))
        
    def encode(self, x):
        with tf.variable_scope(
            'encode', reuse = tf.AUTO_REUSE
        ):
            x = tf.nn.selu(tf.layers.conv2d(
                x, 16,
                [3, 3], [1, 1], 'same', name = 'conv3x3_1'
            ))
            
            x = tf.nn.selu(tf.layers.conv2d(
                x, 32,
                [3, 3], [1, 1], 'same', name = 'conv3x3_2'
            ))
            
            x = tf.nn.selu(tf.layers.conv2d(
                x, 32,
                [3, 3], [1, 1], 'same', name = 'conv3x3_3'
            ))
            
            final = tf.layers.conv2d(
                x, 12 * 2,
                [4, 4], [2, 2], 'same', name = 'conv3x3_4'
            )
            
            return final[:, :, :, :12], tf.nn.softplus(final[:, :, :, 12:])
    
    def decode(self, x):
        with tf.variable_scope(
            'transform', reuse = tf.AUTO_REUSE
        ):
            x = tf.nn.selu(tf.layers.conv2d(
                x, 16,
                [3, 3], [1, 1], 'same', name = 'conv3x3_1'
            ))
            
            x = tf.nn.selu(tf.layers.conv2d(
                x, 32,
                [3, 3], [1, 1], 'same', name = 'conv3x3_2'
            ))
            
            x = tf.nn.selu(tf.layers.conv2d(
                x, 64,
                [3, 3], [1, 1], 'same', name = 'conv3x3_3'
            ))
            
            x = tf.nn.selu(tf.layers.conv2d(
                x, 128,
                [3, 3], [1, 1], 'same', name = 'conv3x3_4'
            ))
            
            #x = tf.nn.selu(tf.layers.conv2d(
            #    x, 128,
            #    [3, 3], [1, 1], 'same', name = 'conv3x3_5'
            #))
            
            x = tf.nn.selu(tf.layers.conv2d(
                x, 256,
                [3, 3], [1, 1], 'same', name = 'conv3x3_6'
            ))
            
            x = tf.layers.conv2d_transpose(
                x, 3, [4, 4], [2, 2], 'same', name = 'deconv4x4'
            )
            
            return x
            
    def scale(self, images, noise = None):
        with tf.variable_scope(
            'scale', reuse = tf.AUTO_REUSE
        ):
            if noise is None:
                noise = tf.zeros(tf.concat([tf.shape(images)[0:2], [12]], 0))
        
            x = tf.concat([images, noise], -1)
            
            x = self.decode(x)
            
            x = tf.pad(
                x,
                [[0, 0], [0, 0], [0, 0], [0, 1]], constant_values = 1.0
            )
            
            return x
 
    def train(self):
        step = self.session.run(self.global_step)
        
        while True:
            while True:
                try:
                    real, downscaled, scaled_distribution, scaled, \
                    reconstructed, \
                    g_loss, divergence_loss, summary = \
                        self.session.run([
                            self.real[:8, :, :, :],
                            self.downscaled[:8, :, :, :],
                            self.scaled_distribution[:8, :, :, :],
                            self.scaled[:8, :, :, :],
                            self.reconstructed[:8, :, :, :],
                            self.g_loss, self.divergence_loss,
                            self.summary
                        ])
                    break
                except tf.errors.InvalidArgumentError as e:
                    print(e.message)
                
            print(
                (
                    "#{}, g_loss: {:.4f}, divergence: {:.4f}"
                ).format(step, g_loss, divergence_loss)
            )
            
            #real[:, :self.size // 2, :self.size // 2, :] = downscaled

            i = np.concatenate(
                (
                    real[:4, :, :, :], 
                    scaled[:4, :, :, :],
                    scaled_distribution[:4, :, :, :],
                    reconstructed[:4, :, :, :],
                    
                    real[4:, :, :, :],
                    scaled[4:, :, :, :],
                    scaled_distribution[4:, :, :, :],
                    reconstructed[4:, :, :, :],
                ),
                axis = 2
            )
            i = np.concatenate(
                [np.squeeze(x, 0) for x in np.split(i, i.shape[0])]
            )

            scm.imsave("samples/{}.png".format(step) , i)
        
            self.summary_writer.add_summary(summary, step)
                
            for _ in tqdm(range(1000)):
                distance = 0
                #for _ in range(2):
                #while distance >= 0:
                #    _, distance = self.session.run(
                #        [self.d_optimizer, self.distance]
                #    )
                
                while True:
                    try:
                        _, step = self.session.run([
                            [
                                #self.d_optimizer, 
                                self.g_optimizer
                            ],
                            self.global_step
                        ])
                        break
                    except tf.errors.InvalidArgumentError as e:
                        print(e.message)
                
            if step % 4000 == 0:
                pass
                print("saving iteration " + str(step))
                self.saver.save(
                    self.session,
                    self.checkpoint_path + "/gansr",
                    global_step=step
                )

    def test(self):
        r, step = self.session.run(
            [self.random, self.global_step]
        )

        i = np.concatenate(
            [np.squeeze(x, 0) for x in np.split(r, r.shape[0])]
        )

        scm.imsave("test/{}.jpg".format(step) , i)
        
    
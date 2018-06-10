import numpy as np
import scipy.misc as scm
import tensorflow as tf
import os

from math import floor, sin, pi
from glob import glob
from tqdm import tqdm

def nice_power(x, n, s, e):
    sa = n * s**(n - 1)
    ea = n * e**(n - 1)
    
    sb = s**n - sa * s
    eb = e**n - ea * e
    
    return tf.where(
        x < s,
        sa * x + sb,
        tf.where(
            x > e,
            ea * x + eb,
            x**n
        )
    )

class GANSuperResolution:
    def __init__(
        self, session, continue_train = True, 
        learning_rate = 1e-4, batch_size = 1
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
        d = d.batch(self.batch_size).prefetch(16)
        
        
        iterator = d.make_one_shot_iterator()
        self.real = iterator.get_next()
        #self.tampered = tamper(self.real)
        
        self.moments = tf.nn.moments(
            self.preprocess(self.real), [0, 1, 2]
        )
        self.moments = tf.stack([
            self.moments[0], tf.sqrt(self.moments[1])
        ])
        
        moments, error1, error2 = self.session.run([
            self.moments,
            tf.reduce_mean(tf.abs(
                tf.to_float(self.real) -
                tf.to_float(self.xyz2srgb(self.srgb2xyz(self.real)))
            )),
            tf.reduce_mean(tf.abs(
                tf.to_float(self.srgb2xyz(self.real)) -
                tf.to_float(self.ulab2xyz(self.preprocess(self.real)))
            ))
        ])
        print(moments, "error:", error1, error2)
        #return
        
        real_xyz = self.srgb2xyz(self.real)
        real = self.xyz2ulab(real_xyz)
        #tampered = self.preprocess(self.tampered)
        downscaled_xyz = self.lanczos3_downscale(real_xyz)
        downscaled = self.xyz2ulab(downscaled_xyz)
        
        self.downscaled = self.postprocess(downscaled)
        
        encoded = self.encode(real)
        reconstructed = self.scale(downscaled, encoded)
        
        encoded_distribution = tf.random_normal(tf.shape(encoded))
        scaled_distribution = self.scale(
            downscaled, encoded_distribution
        )
        scaled = self.scale(
            downscaled, tf.zeros_like(encoded)
        )
        
        encoded_reconstruction = self.encode(scaled_distribution)
        
        #self.cleaned = self.postprocess(cleaned)
        self.scaled = self.postprocess(scaled)
        self.scaled_distribution = self.postprocess(scaled_distribution)
        self.reconstructed = self.postprocess(reconstructed)
        
        redownscaled = self.xyz2ulab(
            self.lanczos3_downscale(
                self.ulab2xyz(scaled_distribution), "VALID"
            )
        )
        redownscaled = self.xyz2ulab(
            self.lanczos3_downscale(self.ulab2xyz(scaled_distribution), "VALID")
        )
        self.redownscaled = self.postprocess(redownscaled)
        
        # losses
        fake_logits = self.discriminate(scaled_distribution, downscaled)
        real_logits = self.discriminate(real, downscaled)
        
        def norm_squared(x):
            return tf.reduce_sum(
                tf.maximum(tf.square(x), 1e-5), axis = -1
            )
        
        def norm(x):
            return tf.sqrt(norm_squared(x))
            
        def difference(real, fake):
            return tf.reduce_mean(
                tf.abs(real - fake) * 
                self.lab_scale / tf.reduce_mean(self.lab_scale)
            )
        
        self.distance = (
            tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
        )
        
        self.rescale_loss = difference(
            self.xyz2ulab(self.lanczos3_downscale(real_xyz, "VALID")), 
            redownscaled
        )
        
        median_loss = difference(scaled, real)
        
        self.g_loss = sum([
            #tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #    logits = fake_logits, labels = tf.ones_like(fake_logits)
            #)), # non-saturating GAN
            1e-2 * tf.reduce_mean(nice_power(fake_logits - 0.5, 2, -2, 2)),
            median_loss, # approximate median
            #self.rescale_loss,
            difference(real, reconstructed),
            #tf.reduce_mean(
            #    tf.abs(encoded_distribution - encoded_reconstruction)
            #),
        ])
        self.d_loss = sum([
            #tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #    logits = real_logits, labels = tf.ones_like(real_logits)
            #)),
            #tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #    logits = fake_logits, labels = tf.zeros_like(fake_logits)
            #)), 
            tf.reduce_mean(nice_power(real_logits - 0.5, 2, -2, 2)),
            tf.reduce_mean(nice_power(fake_logits + 0.5, 2, -2, 2)),
        ])
        
        variables = tf.trainable_variables()
        g_variables = [v for v in variables if 'discriminate' not in v.name]
        d_variables = [v for v in variables if 'discriminate' in v.name]
        
        learning_rate = tf.train.exponential_decay( 
            self.learning_rate, self.global_step, 100000, 0.5, True 
        )
        
        self.g_optimizer = tf.train.AdamOptimizer(
            learning_rate#, beta1 = 0.5#, beta2 = 0.9
        ).minimize(
            self.g_loss, self.global_step, var_list = g_variables
        )

        self.d_optimizer = tf.train.AdamOptimizer(
            learning_rate#, beta1 = 0.5#, beta2 = 0.9#, epsilon = 1e-1
        ).minimize(
            self.d_loss, var_list = d_variables
        )
        
        
        with tf.variable_scope('average'): 
            exponentialAverages = tf.train.ExponentialMovingAverage(0.995) 
            with tf.control_dependencies([self.g_optimizer]):
                self.g_optimizer = exponentialAverages.apply([ 
                    self.g_loss, self.d_loss, self.distance, median_loss
                ])
                
                self.g_loss = exponentialAverages.average(self.g_loss) 
                self.d_loss = exponentialAverages.average(self.d_loss) 
                self.distance = exponentialAverages.average(self.distance)
                median_loss = exponentialAverages.average(median_loss)
        

        self.saver = tf.train.Saver(max_to_keep = 2)
        
        tf.summary.scalar('generator loss', self.g_loss)
        tf.summary.scalar('discriminator loss', self.d_loss)
        tf.summary.scalar('distance', self.distance)
        tf.summary.scalar('median_loss', median_loss)
        tf.summary.histogram('fake score', fake_logits)
        tf.summary.histogram('real score', real_logits)
        
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
        #srgb = nice_power(linear, 1 / 2.4, 0.003131, 2) - 0.055
        srgb = srgb / tf.maximum(alpha, 1 / 255)
        srgb = tf.concat([srgb, alpha], -1)
        return tf.cast(tf.minimum(tf.maximum(
            srgb * 255, 0
        ), 255), tf.int32)
        
    def xyz2ulab(self, c):
        xyz, alpha = tf.split(c, [3, 1], -1)
        # https://en.wikipedia.org/wiki/Lab_color_space
        def f(t):
            #return t
            return nice_power(t, 1 / 3, 6**3 / 29**3, 1)
            return tf.where(
                t > 6**3 / 29**3, 
                t**(1 / 3), 
                t / (3 * 6**2 / 29**2) + 4 / 29
            )
        x, y, z = (xyz[:, :, :, i] for i in range(3))
        x_n, y_n, z_n = 0.9505, 1.00, 1.089
        lab = tf.stack(
            [
                116 * f(y / y_n) - 16,
                500 * (f(x / x_n) - f(y / y_n)),
                200 * (f(y / y_n) - f(z / z_n))
            ],
            -1
        )
        laba = tf.concat([lab, alpha], -1)
        # normalize
        return (laba - self.lab_shift) / self.lab_scale

    def ulab2xyz(self, c):
        # denormalize
        c = c * self.lab_scale + self.lab_shift
        lab, alpha = tf.split(c, [3, 1], -1)
        
        # https://en.wikipedia.org/wiki/Lab_color_space
        def f(t):
            #return t
            return nice_power(t, 3, 6 / 29, 1)
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
                x, 32,
                [4, 4], [2, 2], 'same', name = '1'
            ))
            
            x = tf.nn.selu(tf.layers.conv2d(
                x, 32,
                [3, 3], [1, 1], 'same', name = '2'
            ))
            
            x = tf.nn.selu(tf.layers.conv2d(
                x, 64,
                [3, 3], [1, 1], 'same', name = '3'
            ))
            
            x = tf.nn.selu(tf.layers.conv2d(
                x, 128,
                [3, 3], [1, 1], 'same', name = '4'
            ))
            
            #x = tf.nn.selu(tf.layers.conv2d(
            #    x, 128,
            #    [3, 3], [1, 1], 'same', name = '5'
            #))
            
            x = tf.nn.selu(tf.layers.conv2d(
                x, 256,
                [3, 3], [1, 1], 'same', name = '6'
            ))
            
            return tf.layers.conv2d(
                x, 9,
                [3, 3], [1, 1], 'same', name = 'final'
            )
    
    def decode(self, x):
        with tf.variable_scope(
            'transform', reuse = tf.AUTO_REUSE
        ):
            x = tf.nn.selu(tf.layers.conv2d(
                x, 32,
                [3, 3], [1, 1], 'same', name = '1'
            ))
            
            x = tf.nn.selu(tf.layers.conv2d(
                x, 32,
                [3, 3], [1, 1], 'same', name = '2'
            ))
            
            x = tf.nn.selu(tf.layers.conv2d(
                x, 64,
                [3, 3], [1, 1], 'same', name = '3'
            ))
            
            x = tf.nn.selu(tf.layers.conv2d(
                x, 128,
                [3, 3], [1, 1], 'same', name = '4'
            ))
            
            #x = tf.nn.selu(tf.layers.conv2d(
            #    x, 128,
            #    [3, 3], [1, 1], 'same', name = '5'
            #))
            
            x = tf.nn.selu(tf.layers.conv2d(
                x, 256,
                [3, 3], [1, 1], 'same', name = '6'
            ))
            
            x = tf.layers.conv2d_transpose(
                x, 3, [4, 4], [2, 2], 'same', name = 'final'
            )
            
            return x
            
    def scale(self, images, noise):
        with tf.variable_scope(
            'scale', reuse = tf.AUTO_REUSE
        ):
            x = tf.concat([images, noise], -1)
            
            x = self.decode(x)
            
            x = tf.pad(
                x,
                [[0, 0], [0, 0], [0, 0], [0, 1]], constant_values = 1.0
            )
            
            return x
            
    def discriminate(self, large_images, small_images):
        with tf.variable_scope(
            'discriminate', reuse = tf.AUTO_REUSE
        ):
            large_images *= self.lab_scale / tf.reduce_mean(self.lab_scale)
            small_images *= self.lab_scale / tf.reduce_mean(self.lab_scale)
            
            # cut away alpha
            small_images = small_images[:, :, :, :3]
            large_images = large_images[:, :, :, :3]
            
            #x = tf.concat([
            #    small_images, 
            #    tf.space_to_depth(large_images, 2)
            #], -1)
            x = x = tf.concat([
                large_images,
                tf.image.resize_nearest_neighbor(small_images, [self.size] * 2)
            ], -1)
            
            x = tf.nn.selu(tf.layers.conv2d(
                x, 32,
                [3, 3], [1, 1], 'valid', name = '1'
            ))
            
            x = tf.nn.selu(tf.layers.conv2d(
                x, 32,
                [3, 3], [1, 1], 'valid', name = '2'
            ))
            
            x = tf.nn.selu(tf.layers.conv2d(
                x, 64,
                [3, 3], [1, 1], 'valid', name = '3'
            ))
            
            x = tf.nn.selu(tf.layers.conv2d(
                x, 128,
                [3, 3], [1, 1], 'valid', name = '4'
            ))
            
            #x = tf.nn.selu(tf.layers.conv2d(
            #    x, 256,
            #    [3, 3], [1, 1], 'valid', name = '5'
            #))
            
            x = tf.layers.conv2d(
                x, 1,
                [3, 3], [1, 1], 'valid', name = '6'
            )
            
            return x
 
    def train(self):
        step = self.session.run(self.global_step)
        
        while True:
            while True:
                try:
                    real, scaled_distribution, scaled, reconstructed, \
                    g_loss, d_loss, distance, rescale_loss, summary = \
                        self.session.run([
                            self.real[:4, :, :, :],
                            self.scaled_distribution[:4, :, :, :],
                            self.scaled,
                            self.reconstructed[:4, :, :, :],
                            self.g_loss, self.d_loss, 
                            self.distance, self.rescale_loss,
                            self.summary
                        ])
                    break
                except tf.errors.InvalidArgumentError as e:
                    print(e.message)
                
            print(
                (
                    "#{}, g_loss: {:.4f}, rescale_loss: {:.4f}, " + 
                    "d_loss: {:.4f}, distance: {:.4f}"
                ).format(step, g_loss, rescale_loss, d_loss, distance)
            )
            
            #scaled_distribution[
            #    :, :redownscaled.shape[1], :redownscaled.shape[2], :
            #] = redownscaled

            i = np.concatenate(
                (
                    real[:, :, :, :],
                    scaled,
                    scaled_distribution[:, :, :, :],
                    reconstructed[:, :, :, :],
                ),
                axis = 2
            )
            i = np.concatenate(
                [np.squeeze(x, 0) for x in np.split(i, i.shape[0])]
            )

            scm.imsave("samples/{}.png".format(step) , i)
        
            self.summary_writer.add_summary(summary, step)
                
            for _ in tqdm(range(1000)):
                while True:
                    try:
                        _, step = self.session.run([
                            [
                                self.d_optimizer, 
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
        
    
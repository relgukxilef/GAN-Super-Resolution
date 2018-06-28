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
            tf.maximum(tf.minimum(x, e), s)**n
        )
    )

class GANSuperResolution:
    def __init__(
        self, session, continue_train = True, 
        learning_rate = 1e-4, # explosions at 5e-5
        batch_size = 4
    ):
        self.session = session
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.continue_train = continue_train
        self.filters = 64
        self.max_filters = 128
        self.dimensions = 32
        self.checkpoint_path = "checkpoints"
        self.size = 64
        
        self.global_step = tf.Variable(0, name = 'global_step')

        # build model
        print("lookup training data...")
        self.paths = glob("data/half/*.png")
                    
        def load(path):
            image = tf.image.decode_image(tf.read_file(path), 4)
            return tf.data.Dataset.from_tensor_slices([
                tf.random_crop(image, [self.size, self.size, 4])
                for _ in range(20)
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
        d = d.flat_map(load).shuffle(2000)
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
                tf.to_float(self.lab2xyz(self.preprocess(self.real)))
            ))
        ])
        print(moments, "error:", error1, error2)
        #return
        
        real = self.srgb2xyz(self.real)
        #tampered = self.preprocess(self.tampered)
        downscaled = self.lanczos3_downscale(real)
        
        self.downscaled = self.xyz2srgb(downscaled)
        
        scaled = self.scale(downscaled)
        
        #self.cleaned = self.xyz2srgb(cleaned)
        self.scaled = self.xyz2srgb(scaled)
        
        redownscaled = self.lanczos3_downscale(scaled, "VALID")
        self.redownscaled = self.xyz2srgb(redownscaled)
        
        # losses
        fake_logits = self.discriminate(
            scaled, downscaled
            #self.xyz2lab(scaled), self.xyz2lab(downscaled)
        )
        real_logits = self.discriminate(
            real, downscaled
            #self.xyz2lab(real), self.xyz2lab(downscaled)
        )
        
        def visual_difference(real, fake):
            return tf.reduce_mean((
                #self.xyz2lab(real) - self.xyz2lab(fake)
                nice_power(real, 1/3, 1e-2, 1) - nice_power(fake, 1/3, 1e-2, 1)
            )**2)
        
        self.distance = (
            tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
        )
        
        self.rescale_loss = tf.reduce_mean((
            self.lanczos3_downscale(real, "VALID") - 
            redownscaled
        )**2)
        
        median_loss = tf.reduce_mean((real - scaled)**2)
        #median_loss = visual_difference(real, scaled)
        
        self.g_loss = sum([
            1e-3 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits = fake_logits, labels = tf.ones_like(fake_logits)
            )), # non-saturating GAN
            #tf.minimum(1.0, tf.to_float(self.global_step) / 2e5) *
            #tf.reduce_mean((fake_logits - 0.5)**2),
            #2 * median_loss,
            #2 * self.rescale_loss,
            #2 * tf.reduce_mean(tf.abs(real - reconstructed)),
            #tf.reduce_mean((encoded_distribution - encoded_reconstruction)**2),
            tf.reduce_mean((real - scaled)**2)
        ])
        self.d_loss = sum([
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits = real_logits, labels = tf.ones_like(real_logits)
            )),
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits = fake_logits, labels = tf.zeros_like(fake_logits)
            )), 
            #tf.reduce_mean((real_logits - 0.5)**2),
            #tf.reduce_mean((fake_logits + 0.5)**2),
        ])
        
        variables = tf.trainable_variables()
        g_variables = [v for v in variables if 'discriminate' not in v.name]
        d_variables = [v for v in variables if 'discriminate' in v.name]
        
        learning_rate = tf.train.exponential_decay( 
            self.learning_rate, self.global_step, 100000, 0.5, True 
        )
        
        self.g_optimizer = tf.train.AdamOptimizer(
            learning_rate#, 0.5, 0.9#, use_nesterov = True#, beta1 = 0.5#, beta2 = 0.9
        ).minimize(
            self.g_loss, self.global_step, var_list = g_variables
        )

        self.d_optimizer = tf.train.AdamOptimizer(
            learning_rate#, 0.5, 0.9#, use_nesterov = True#, beta1 = 0.5#, beta2 = 0.9
            #epsilon = 1e-2
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
        
    def xyz2lab(self, c):
        xyz, alpha = tf.split(c, [3, 1], -1)
        # https://en.wikipedia.org/wiki/Lab_color_space
        def f(t):
            return nice_power(t, 1 / 3, 6**3 / 29**3, 1)
        x, y, z = (xyz[:, :, :, i] for i in range(3))
        x_n, y_n, z_n = 0.9505, 1.00, 1.089
        lab = tf.stack(
            [
                1.16 * f(y / y_n) - 0.16,
                5.00 * (f(x / x_n) - f(y / y_n)),
                2.00 * (f(y / y_n) - f(z / z_n))
            ],
            -1
        )
        return tf.concat([lab, alpha], -1)

    def lab2xyz(self, c):
        # denormalize
        lab, alpha = tf.split(c, [3, 1], -1)
        
        # https://en.wikipedia.org/wiki/Lab_color_space
        def f(t):
            return nice_power(t, 3, 6 / 29, 1)
        l, a, b = (lab[:, :, :, i] for i in range(3))
        #l = tf.maximum(l, 0)
        x_n, y_n, z_n = 0.9505, 1.00, 1.089
        l2 = (l + 0.16) / 1.16
        xyz = tf.stack([
            x_n * f(l2 + a / 5.00),
            y_n * f(l2),
            z_n * f(l2 - b / 2.00)
        ], -1)
        return tf.concat([xyz, alpha], -1)
    
    def preprocess(self, images):
        #return self.srgb2xyz(images)
        return self.xyz2lab(self.srgb2xyz(images))
        
    def postprocess(self, images):
        #return self.xyz2srgb(images)
        return self.xyz2srgb(self.lab2xyz(images))
    
    def decode(self, image):
        with tf.variable_scope(
            'transform', reuse = tf.AUTO_REUSE
        ):
            x = image
            
            for i in range(3):
                x = tf.nn.selu(tf.layers.conv2d(
                    x, 64,
                    [3, 3], [1, 1], 'same', name = 'conv3x3_0_' + str(i)
                ))
            
            x = tf.image.resize_nearest_neighbor(x, [self.size] * 2)
            #x = self.lanczos3_upscale(x)
        
            for i in range(1):
                x = tf.nn.selu(tf.layers.conv2d(
                    x, 64,
                    [3, 3], [1, 1], 'same', name = 'conv3x3_1_' + str(i)
                ))
                
            x = tf.layers.conv2d(
                x, 3,
                [3, 3], [1, 1], 'same', name = 'conv3x3_2_final'
            )
                
            #x = tf.depth_to_space(x, 2)
            
            return x
            
    def scale(self, images):
        with tf.variable_scope(
            'scale', reuse = tf.AUTO_REUSE
        ):
            x = self.decode(images * 2 - 1)
            
            x = tf.pad(
                x,
                [[0, 0], [0, 0], [0, 0], [0, 1]], constant_values = 1.0
            )
            
            return x * 0.5 + 0.5
            
    def discriminate(self, large_images, small_images):
        with tf.variable_scope(
            'discriminate', reuse = tf.AUTO_REUSE
        ):
            # cut away alpha
            small_images = small_images[:, :, :, :3] * 2 - 1
            large_images = large_images[:, :, :, :3] * 2 - 1
            
            result = []
            
            x = small_images
            
            for i in range(4):
                x = tf.nn.selu(tf.layers.conv2d(
                    x, 64, [3, 3], [1, 1], 'same', name = 'conv3x3_0_' + str(i)
                ))
            
            #x = tf.concat([
            #    small_images, 
            #    tf.space_to_depth(large_images, 2)
            #], -1)
            x = tf.concat([
                large_images,
                #self.lanczos3_upscale(small_images)[:, :, :, :3]
                tf.image.resize_nearest_neighbor(x, [self.size] * 2)
            ], -1)
            #x = tf.layers.conv2d_transpose(
            #    small_images, 3, [4, 4], [2, 2], 'same', name = '0'
            #)
            #x = tf.concat([x, large_images], -1)
            
            
            dilation = 1
            
            for i in range(2):
                x = tf.nn.selu(tf.layers.conv2d(
                    x, 128, [3, 3], [1, 1], 'same', 
                    name = 'conv3x3_1_' + str(i)
                ))
                
                result += [tf.layers.conv2d(
                    x, 1, [1, 1], [1, 1], 'same', 
                    name = 'dense_1_' + str(i)
                )]
            
            return tf.concat(result, -1)
 
    def train(self):
        step = self.session.run(self.global_step)
        
        while True:
            while True:
                try:
                    real, scaled, \
                    g_loss, d_loss, distance, rescale_loss, summary = \
                        self.session.run([
                            self.real,
                            self.scaled,
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
                    real,
                    scaled,
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
        
    
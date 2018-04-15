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
        learning_rate = 1e-4, batch_size = 32
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
        self.lab_shift = tf.constant([[[[58.190086, 22.409063, -4.994123]]]])
        self.lab_scale = tf.constant([[[[34.140823, 83.433235, 31.84137]]]])

        self.global_step = tf.Variable(0, name = 'global_step')

        # build model
        print("lookup training data...")
        self.paths = glob("data/half/*.png")
                    
        def load(path):
            image = tf.image.decode_image(tf.read_file(path), 3)
            return tf.data.Dataset.from_tensor_slices([
                tf.random_crop(image, [self.size, self.size, 3])
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
            0.5 * 3 * sin(pi * x) * sin(pi * x / 3) / pi**2 / x**2
            for x in np.linspace(-2.75, 2.75, 12)
        ]
        self.lanczos3_horizontal = tf.constant(
            [
                [[
                    [a if o == i else 0 for o in range(3)]
                    for i in range(3)
                ]] 
                for a in lanczos3
            ]
        )
        self.lanczos3_vertical = tf.constant(
            [[
                [
                    [a if o == i else 0 for o in range(3)]
                    for i in range(3)
                ]
                for a in lanczos3
            ]]
        )   
        
        d = tf.data.Dataset.from_tensor_slices(tf.constant(self.paths))
        d = d.shuffle(100000).repeat()
        d = d.flat_map(load).shuffle(2500)
        d = d.batch(self.batch_size).prefetch(20)
        
        
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
        
        #cleaned = self.denoise(tampered)
        scaled = self.scale(downscaled)
        
        #self.cleaned = self.postprocess(cleaned)
        self.scaled = self.postprocess(scaled)
        
        blurred = self.xyz2ulab(self.lanczos3_upscale(downscaled_xyz))
        
        # losses
        fake_logits = tf.concat(
            [
                #self.discriminate(cleaned), 
                self.discriminate(scaled)
                #self.discriminate(blurred)
            ], 0
        )
        #blur_logits = self.discriminate(blurred)
        real_logits = self.discriminate(real)
        
        def norm(x):
            return tf.sqrt(tf.reduce_sum(tf.square(x), axis = [1, 2, 3]) + 1e-8)
            
        def difference(real, fake):
            #return tf.reduce_mean(tf.norm(
            #    tf.abs(real - fake) * 
            #    self.lab_scale / tf.reduce_mean(self.lab_scale) + 1e-8, 
            #    axis = -1
            #))
            return tf.reduce_mean(
                tf.abs(real - fake) * 
                self.lab_scale / tf.reduce_mean(self.lab_scale)
            )
            
        def log(x):
            return tf.log(x + 1e-8)
            
        ratio = tf.random_uniform([self.batch_size, 1, 1, 1])
        # DRAGAN penalty
        #noise = real + tf.random_normal(
        #    [self.batch_size, self.size, self.size, 3], 
        #    0, tf.sqrt(tf.nn.moments(real, [0, 1, 2, 3])[1])
        #) * ratio
        # WGAN loss
        #noise = real * (1 - ratio) + scaled * ratio
        
        #penalty = tf.reduce_mean(
        #    (norm(tf.gradients(self.discriminate(noise), noise)) - 1e0) ** 2
        #)
        
        self.g_loss = (
            1e-2 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits = fake_logits, labels = tf.ones_like(fake_logits)
            )) + # non-saturating GAN
            #1e0 * difference(real, cleaned) +
            1e0 * tf.reduce_mean(tf.abs(real - scaled))
        )
        
        self.distance = (
            tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
        )
        #self.blur_distance = (
        #    tf.reduce_mean(blur_logits) - tf.reduce_mean(real_logits)
        #)
        
        self.d_loss = (
            1e0 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits = real_logits, labels = tf.ones_like(real_logits)
            )) + 
            1e0 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits = fake_logits, labels = tf.zeros_like(fake_logits)
            )) #+ 
            #1e0 * penalty
            #tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #    logits = blur_logits, labels = tf.zeros_like(blur_logits)
            #))
        )
        
        # Fisher-GAN loss
        #self.d_loss = (
        #    (tf.reduce_mean(real_logits) - tf.reduce_mean(fake_logits)) /
        #    (
        #        0.5 * tf.reduce_mean(real_logits**2) + 
        #        0.5 * tf.reduce_mean(fake_logits**2)
        #    )**0.5
        #)
        
        variables = tf.trainable_variables()
        g_variables = [v for v in variables if 'discriminate' not in v.name]
        d_variables = [v for v in variables if 'discriminate' in v.name]
        
        print(
            "g_variables:", len(g_variables), "d_variables:", len(d_variables)
        )
        print(g_variables)
        
        self.g_optimizer = tf.train.AdamOptimizer(
            self.learning_rate, beta1 = 0.5, beta2 = 0.9
        ).minimize(
            self.g_loss, self.global_step, var_list = g_variables
        )

        self.d_optimizer = tf.train.AdamOptimizer(
            self.learning_rate, beta1 = 0.5, beta2 = 0.9
        ).minimize(
            self.d_loss, var_list = d_variables
        )
        
        
        #padded_kernels = tf.concat([
        #    tf.pad([
        #        v for v in tf.trainable_variables() 
        #        if v.name == 'scale/conv3x3_1/kernel:0'
        #    ][0], [[1, 1], [1, 1], [0, 0], [0, 0]]),
        #    tf.pad([
        #        v for v in tf.trainable_variables() 
        #        if v.name == 'scale/conv3x3_2/kernel:0'
        #    ][0], [[1, 1], [1, 1], [0, 0], [0, 0]]),
        #    tf.pad([
        #        v for v in tf.trainable_variables() 
        #        if v.name == 'scale/conv3x3_3/kernel:0'
        #    ][0], [[1, 1], [1, 1], [0, 0], [0, 1]])
        #], 3)
        #
        #kernel_grid = tf.to_float(self.postprocess([
        #    tf.concat([
        #        tf.concat([
        #            padded_kernels[:, :, :3, x * 8 + y]
        #            for x in range(8)
        #        ], 1) for y in range(8)
        #    ], 0)
        #]))

        self.saver = tf.train.Saver(max_to_keep = 2)
        
        tf.summary.scalar('generator loss', self.g_loss)
        tf.summary.scalar('discriminator loss', self.d_loss)
        tf.summary.scalar('distance', self.distance)
        tf.summary.histogram('fake score', fake_logits)
        tf.summary.histogram('real score', real_logits)
        
        #tf.summary.image('kernels', kernel_grid)
        
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
        linear = tf.where(
            c <= 0.04045,
            c / 12.92,
            ((c + 0.055) / 1.055)**2.4
        )
        return (
            linear * [[[[0.4124, 0.2126, 0.0193]]]] +
            linear * [[[[0.3576, 0.7152, 0.1192]]]] +
            linear * [[[[0.1805, 0.0722, 0.9505]]]]
        )
        
    def xyz2srgb(self, c):
        linear = (
            c * [[[[3.2406, -0.9689, 0.0557]]]] +
            c * [[[[-1.5372, 1.8758, -0.204]]]] +
            c * [[[[-0.4986, 0.0415, 1.057]]]]
        )
        srgb = tf.where(
            linear <= 0.003131,
            12.92 * linear,
            1.055 * linear**(1 / 2.4) - 0.055
        )
        return tf.cast(tf.minimum(tf.maximum(
            srgb * 255, 0
        ), 255), tf.int32)
        
    def xyz2ulab(self, c):
        #c = tf.maximum(tf.minimum(c, 100), 0) # for stability
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
        
        # normalize
        return (lab - self.lab_shift) / self.lab_scale

    def ulab2xyz(self, c):
        # denormalize
        lab = c * self.lab_scale + self.lab_shift
        
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
        return tf.stack(
            [
                x_n * f(l2 + a / 500),
                y_n * f(l2),
                z_n * f(l2 - b / 200),
            ]
        , -1)
    
    def preprocess(self, images):
        #return self.srgb2xyz(images)
        return self.xyz2ulab(self.srgb2xyz(images))
        
    def postprocess(self, images):
        #return self.xyz2srgb(images)
        return self.xyz2srgb(self.ulab2xyz(images))
        
    def depth_wise_conv2d(
        self, input, filters, kernel_size, stride = [1, 1], 
        padding = 'VALID', name = 'depth_wise_conv2d'
    ):
        # TODO: this is like grayscaling before a convolution 
        # but more complicated
        with tf.variable_scope(
            name, reuse = tf.AUTO_REUSE
        ):
            kernel = tf.tile(tf.get_variable(
                'kernel', 
                [kernel_size[0], kernel_size[1], 1, filters]
            ), [1, 1, tf.shape(input)[-1], 1])
            bias = tf.get_variable(
                'bias',
                [1, 1, 1, filters]
            )
            return tf.nn.conv2d(
                input, kernel, [1] + stride + [1], padding
            ) + bias
        
    def symmetric_conv2d(
        self, input, filters, kernel_size, stride = [1, 1], 
        padding = 'VALID', name = 'symmetric_conv2d'
    ):
        with tf.variable_scope(
            name, reuse = tf.AUTO_REUSE
        ):
            kernel = tf.get_variable(
                'kernel', 
                [kernel_size[0], kernel_size[1], input.shape[3], filters]
            )
            symmetric_kernel = tf.concat([
                tf.reverse(tf.transpose(kernel, t), m)
                for m in [[], [0], [1], [0, 1]] 
                for t in [[0, 1, 2, 3], [1, 0, 2, 3]]
            ], -1)
            bias = tf.get_variable(
                'bias',
                [1, 1, 1, filters]
            )
            symmetric_bias = tf.tile(bias, [1, 1, 1, 8])
            return tf.nn.conv2d(
                input, kernel, [1] + stride + [1], padding
            ) + bias
            
    def symmetric(self, input, function):
        p = [
            (r, t) 
            for r in [[], [2], [1], [1, 2]] 
            for t in [[0, 1, 2, 3], [0, 2, 1, 3]]
        ]
        # permutations to batches
        x = tf.concat([tf.transpose(tf.reverse(input, r), t) for r, t in p], 0) 
        
        x = function(x)
        
        # sum up permutations
        x = tf.split(x, len(p))
        x = sum(
            [tf.reverse(tf.transpose(i, t), r) for i, (r, t) in zip(x, p)]
        ) / len(p)**0.5 # normalize
        return x
    
    def cross_conv2d(
        self, input, filters, kernel_size, stride = [1, 1], 
        padding = 'VALID', name = 'cross_conv2d'
    ):
        with tf.variable_scope(
            name, reuse = tf.AUTO_REUSE
        ):
            return (
                tf.nn.selu(tf.layers.conv2d(
                    input, filters,
                    [1, kernel_size[1]], stride, padding, name = 'horizontal'
                )) + 
                tf.nn.selu(tf.layers.conv2d(
                    input, filters,
                    [kernel_size[0], 1], stride, padding, name = 'vertical'
                ))
            ) * 0.5**0.5
    
    def unet(self, input, filters, depth, outputs, context = None):
        with tf.variable_scope(
            'unet', reuse = tf.AUTO_REUSE
        ):
            x = input
                        
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
            
            x = tf.nn.selu(tf.layers.conv2d(
                x, 128,
                [3, 3], [1, 1], 'same', name = 'conv3x3_5'
            ))
            
            x = tf.layers.conv2d(
                x, outputs,
                [3, 3], [1, 1], 'same', name = 'conv1x1'
            )
            
            return x
        
    def classify(self, x, filters, depth, outputs):
        with tf.variable_scope(
            'classify', reuse = tf.AUTO_REUSE
        ):            
            x = tf.nn.selu(tf.layers.conv2d(
                x, 16,
                [3, 3], [1, 1], 'valid', name = 'conv3x3_1'
            ))
            
            x = tf.nn.selu(tf.layers.conv2d(
                x, 32,
                [3, 3], [1, 1], 'valid', name = 'conv3x3_2'
            ))
            
            x = tf.nn.selu(tf.layers.conv2d(
                x, 32,
                [3, 3], [1, 1], 'valid', name = 'conv3x3_3'
            ))
            
            x = tf.reduce_mean(x, [1, 2], True)# * self.size
            
            # dense again to react to global properties
            x = tf.nn.selu(tf.layers.conv2d(
                x, 32, [1, 1], name = 'dense_1'
            ))
            x = tf.layers.conv2d(
                x, outputs, [1, 1], name = 'dense_2'
            )
            
            return x

    def denoise(self, images):
        with tf.variable_scope(
            'denoise', reuse = tf.AUTO_REUSE
        ):
            images = tf.pad(
                images, [[0, 0], [0, 0], [0, 0], [0, 1]], constant_values = 1.0
            )
            return self.unet(
                images, self.filters, 3, 3,
                None #self.classify(images, self.filters, 5)
            )
            
    def scale(self, images):
        with tf.variable_scope(
            'scale', reuse = tf.AUTO_REUSE
        ):
            #x = images * 2 - 1
            x = images
            
            x = tf.pad(
                x, [[0, 0], [0, 0], [0, 0], [0, 1]], constant_values = 1.0
            )
            
            #x = sum([
            #    atom(x, m, t) 
            #    for m in [[], [2], [1], [1, 2]] 
            #    for t in [[0, 1, 2, 3], [0, 2, 1, 3]]
            #])
            
            #x = self.symmetric(
            #    x,
            #    lambda x: 
            #    tf.depth_to_space(self.unet(x, self.filters, 2, 12), 2)
            #)
            
            x = tf.nn.selu(self.unet(x, self.filters, 2, 256))
            #x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, 3]]) # add fourth pixel
            #x = tf.layers.conv2d(
            #    x, 12,
            #    [1, 1], [1, 1], 'same', name = 'conv1x1_final'
            #)
            
            x = tf.layers.conv2d_transpose(
                x, 3, [4, 4], [2, 2], 'same', name = 'deconv4x4'
            )
            
            #x = tf.depth_to_space(x, 2)
            
            #x = tf.layers.conv2d_transpose(
            #    x, 3, 
            #    [2, 2], [2, 2], 'same', name = 'deconv2x2'
            #)
            
            #x = x * 0.5 + 0.5
            
            # force result to be a valid upscale
            #x += tf.stop_gradient(
            #    self.xyz2ulab(self.lanczos3_upscale(self.ulab2xyz(images))) -
            #    self.xyz2ulab(self.lanczos3_upscale(self.lanczos3_downscale(self.ulab2xyz(x))))
            #)
            
            return x
            
    def discriminate(self, images):
        with tf.variable_scope(
            'discriminate', reuse = tf.AUTO_REUSE
        ):
            x = images * self.lab_scale / tf.reduce_mean(self.lab_scale)

            #x = self.symmetric(
            #    x, lambda x: self.classify(x, self.filters, 2, 1)
            #)
            x = self.classify(x, self.filters, 2, 1)
            
            return x
 
    def train(self):
        step = self.session.run(self.global_step)
        
        while True:
            while True:
                try:
                    real, downscaled, scaled, g_loss, d_loss, distance, summary = \
                        self.session.run([
                            self.real[:8, :, :, :],
                            self.downscaled[:8, :, :, :],
                            #self.tampered[:4, :, :, :],
                            #self.cleaned[:4, :, :, :],
                            self.scaled[:8, :, :, :],
                            self.g_loss, self.d_loss, 
                            self.distance,
                            self.summary
                        ])
                    break
                except tf.errors.InvalidArgumentError as e:
                    print(e.message)
                
            print(
                (
                    "#{}, g_loss: {:.4f}, d_loss: {:.4f}, distance: {:.4f}"
                ).format(step, g_loss, d_loss, distance)
            )
            
            #real[:, :self.size // 2, :self.size // 2, :] = downscaled

            i = np.concatenate(
                (
                    real[:4, :, :, :], 
                    #tampered, 
                    #cleaned, 
                    scaled[:4, :, :, :],
                    real[4:, :, :, :],
                    scaled[4:, :, :, :]
                ),
                axis = 2
            )
            i = np.concatenate(
                [np.squeeze(x, 0) for x in np.split(i, i.shape[0])]
            )

            scm.imsave("samples/{}.png".format(step) , i)
        
            self.summary_writer.add_summary(summary, step)
                
            for _ in tqdm(range(250)):
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
                                self.d_optimizer, 
                                self.g_optimizer
                            ],
                            self.global_step
                        ])
                        break
                    except tf.errors.InvalidArgumentError as e:
                        print(e.message)
                
            if step % 1000 == 0:
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
        
    
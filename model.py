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
        learning_rate = 2e-4, batch_size = 16
    ):
        self.session = session
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.continue_train = continue_train
        self.filters = 16
        self.dimensions = 32
        self.checkpoint_path = "checkpoints"
        self.size = 256

        self.global_step = tf.Variable(0, name = 'global_step')

        # build model
        print("lookup training data...")
        self.paths = glob("data/images/*.png")
                    
        def load(path):
            return tf.data.Dataset.from_tensor_slices(
                tf.reshape(
                    tf.extract_image_patches(
                        [tf.image.decode_image(tf.read_file(path), 3)],
                        [1, self.size, self.size, 1], 
                        [1, self.size // 1, self.size // 1, 1], 
                        [1, 1, 1, 1], "VALID"
                    ),
                    [-1, self.size, self.size, 3]
                )
            )
            
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
        
        d = tf.data.Dataset.from_tensor_slices(tf.constant(self.paths))
        d = d.shuffle(100000).repeat()
        d = d.flat_map(load).shuffle(1000)
        d = d.batch(self.batch_size)
        
        
        iterator = d.make_one_shot_iterator()
        self.real = iterator.get_next()
        self.tampered = tamper(self.real)
        
        real = self.preprocess(self.real)
        tampered = self.preprocess(self.tampered)

        cleaned = self.denoise(tampered)
        scaled = self.scale(real)
        
        self.cleaned = self.postprocess(cleaned)
        self.scaled = self.postprocess(scaled)
        
        
        # losses
        fake_score = tf.reduce_mean(
            [self.discriminate(cleaned), self.discriminate(scaled)]
        )
        real_score = tf.reduce_mean(self.discriminate(real))
        
        penalty = 0.1 * tf.reduce_mean( # DRAGAN penalty
            (
                tf.norm(tf.gradients(
                    self.discriminate(
                        real + tf.random_normal(
                            [self.batch_size, self.size, self.size, 3], 
                            0, 10 / 127.5
                        )
                    ), real
                )) - 1
            ) ** 2
        )
        
        lanczos3 = [
            0.5 * 3 * sin(pi * x) * sin(pi * x / 3) / pi**2 / x**2
            for x in np.linspace(-2.75, 2.75, 12)
        ]
        lanczos3_kernel = tf.constant(
            [[[[a * b] for c in range(3)] for a in lanczos3] for b in lanczos3]
        )
        
        downscaled = tf.nn.depthwise_conv2d(
            scaled, lanczos3_kernel, [1, 2, 2, 1], "SAME"
        )
        
        self.downscaled = self.postprocess(downscaled)
        
        self.g_loss = (
            -fake_score + 
            0.001 * tf.reduce_mean(tf.abs(real - cleaned)) +
            10 * tf.reduce_mean(tf.abs(real - downscaled))
        )
        
        self.distance = fake_score - real_score
        
        self.d_loss = self.distance + penalty
        
        variables = tf.trainable_variables()
        g_variables = [v for v in variables if 'discriminate' not in v.name]
        d_variables = [v for v in variables if 'discriminate' in v.name]
        
        self.g_optimizer = tf.train.AdamOptimizer(
            self.learning_rate, beta1 = 0.0, beta2 = 0.99, epsilon = 1e-8
        ).minimize(
            self.g_loss, self.global_step, var_list = g_variables
        )

        self.d_optimizer = tf.train.AdamOptimizer(
            self.learning_rate, beta1 = 0.0, beta2 = 0.99, epsilon = 1e-8
        ).minimize(
            self.d_loss, var_list = d_variables
        )

        self.saver = tf.train.Saver()

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
            
    def preprocess(self, images):
        return tf.cast(images, tf.float32) / 127.5 - 1.0
        #return tf.cast(images, tf.float32) / 255.0
        
    def postprocess(self, images):
        return tf.cast(tf.minimum(tf.maximum(
            (images + 1.0) * 127.5, 0
        ), 255), tf.int32)
        
    def unet(self, x, filters, depth, context = None):
        with tf.variable_scope(
            'unet', reuse = tf.AUTO_REUSE
        ):
            y = tf.nn.elu(tf.layers.separable_conv2d(
                x, filters * 2, 
                [4, 4], [2, 2], 'same', name = 'conv4x4' + str(depth)
            ))
            
            y = tf.nn.elu(tf.layers.conv2d(
                y, filters * 2, 
                [1, 1], [1, 1], 'same', name = 'conv1x1' + str(depth)
            ))
            
            if depth > 0:
                y = self.unet(y, filters * 2, depth - 1)
            elif context != None:
                #pass # ignore for now
                y = tf.concat([y, context], -1)
            
            y = tf.nn.elu(tf.layers.conv2d(
                y, filters * 2, 
                [1, 1], [1, 1], 'same', name = 'deconv1x1' + str(depth)
            ))
                
            y = tf.nn.elu(tf.layers.conv2d_transpose(
                y, filters, 
                [4, 4], [2, 2], 'same', name = 'deconv4x4' + str(depth)
            ))
            
            return tf.concat([x, y], -1)
        
    def classify(self, x, dimensions, filters, depth):
        with tf.variable_scope(
            'classify', reuse = tf.AUTO_REUSE
        ):
            print("classify:", x.shape, dimensions, filters)
            def layer(x, filters):
                return tf.nn.elu(tf.layers.conv2d(
                    x, filters, 
                    [4, 4], [2, 2], 'same', name = 'conv4x4_' + str(depth)
                ))
                
            for i in range(depth):
                x = layer(x, filters)
                filters *= 2
                depth -= 1
                
            x = tf.layers.conv2d(
                x, filters, 
                [1, 1], [1, 1], 'same', name = 'conv1x1'
            )
            
            x = tf.nn.elu(tf.reduce_mean(x, [1, 2], True))
            
            return tf.layers.conv2d(
                x, dimensions, [1, 1], [1, 1], 'same', name = 'dense'
            )

    def denoise(self, images):
        with tf.variable_scope(
            'denoise', reuse = tf.AUTO_REUSE
        ):
            return tf.layers.conv2d(
                self.unet(
                    images, self.filters, 4, 
                    self.classify(images, self.dimensions, self.filters, 4)
                ), 3, [1, 1], [1, 1], 'same', name = 'conv1x1'
            )
            
    def scale(self, images):
        with tf.variable_scope(
            'scale', reuse = tf.AUTO_REUSE
        ):
            return tf.layers.conv2d_transpose(
                self.unet(
                    images, self.filters, 4, 
                    self.classify(images, self.dimensions, self.filters, 4)
                ), 3, 
                [4, 4], [2, 2], 'same', name = 'deconv4x4final'
            )
            
    def discriminate(self, images):
        with tf.variable_scope(
            'discriminate', reuse = tf.AUTO_REUSE
        ):
            return tf.sigmoid(self.classify(images, 1, self.filters, 4))
 
    def train(self):
        step = 0
        
        while True:
            if step % 100 == 0:
                real, tampered, cleaned, scaled, g_loss, d_loss, distance = \
                    self.session.run([
                        self.real[:4, :, :, :],
                        self.tampered[:4, :, :, :],
                        self.cleaned[:4, :, :, :],
                        self.scaled[:4, :self.size, :self.size, :],
                        self.g_loss, self.d_loss, self.distance
                    ])
                
                print(
                    "g_loss: {:.4f}, d_loss: {:.4f}, distance: {:.4f}"
                    .format(g_loss, d_loss, distance)
                )

                i = np.concatenate(
                    (
                        real, tampered, cleaned, scaled
                    ),
                    axis = 2
                )
                i = np.concatenate(
                    [np.squeeze(x, 0) for x in np.split(i, i.shape[0])]
                )

                scm.imsave("samples/{}.png".format(step) , i)
                
            distance = 0
            for _ in tqdm(range(100)):
                #for _ in range(4):
                #while distance >= 0:
                #    _, distance = self.session.run(
                #        [self.d_optimizer, self.distance]
                #    )
                    
                _, _, step = self.session.run(
                    [self.d_optimizer, self.g_optimizer, self.global_step]
                )
                
            if step % 500 == 0:
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
        
    
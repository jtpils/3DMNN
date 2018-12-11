import numpy as np
import time
import tensorflow as tf
import sys

sys.path.append("/home/viktorv/Projects/3DMNN/main/models/latent_space/src")

from classes.gan import GAN

from utils.utils import safe_log
from tflearn import is_training
from utils.io import create_dir, pickle_data, unpickle_data
from classes.gan import ConfigurationGAN as ConfGAN


class LatentGAN(GAN):
    def __init__(self, conf, gen_kwargs={}, disc_kwargs={}, graph=None):

        self.name = conf.name
        self.learning_rate = conf.learning_rate
        self.n_output = conf.n_output
        self.noise_dim = conf.n_z
        self.discriminator = conf.discriminator
        self.generator = conf.generator
        self.beta = conf.beta
        

        GAN.__init__(self, self.name, graph)

        with tf.variable_scope(self.name):

            self.noise = tf.placeholder(tf.float32, shape=[None, self.noise_dim])                  # Noise vector.
            self.gt_data = tf.placeholder(tf.float32, shape=[None] + self.n_output)           # Ground-truth.

            with tf.variable_scope('generator'):
                self.generator_out = self.generator(self.noise, self.n_output, **gen_kwargs)

            with tf.variable_scope('discriminator') as scope:
                self.real_prob, self.real_logit = self.discriminator(self.gt_data, scope=scope, **disc_kwargs)
                self.synthetic_prob, self.synthetic_logit = self.discriminator(self.generator_out, reuse=True, scope=scope, **disc_kwargs)

            self.loss_d = tf.reduce_mean(-tf.log(self.real_prob) - tf.log(1 - self.synthetic_prob))
            self.loss_g = tf.reduce_mean(-tf.log(self.synthetic_prob))

            #try safe_log

            train_vars = tf.trainable_variables()

            d_params = [v for v in train_vars if v.name.startswith(self.name + '/discriminator/')]
            g_params = [v for v in train_vars if v.name.startswith(self.name + '/generator/')]

            self.opt_d = self.optimizer(self.learning_rate, self.beta, self.loss_d, d_params)
            self.opt_g = self.optimizer(self.learning_rate, self.beta, self.loss_g, g_params)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
            self.init = tf.global_variables_initializer()

            # Launch the session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

    def generator_noise_distribution(self, n_samples, ndims, mu, sigma):
        return np.random.normal(mu, sigma, (n_samples, ndims))

    def _single_epoch_train(self, train_data):

        batch_size = self.conf.batch_size
        noise_params = self.conf.noise_params

        '''
        see: http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
             http://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/
        '''

        n_examples = train_data.num_examples
        epoch_loss_d = 0.
        epoch_loss_g = 0.
        n_batches = int(n_examples / batch_size)
        start_time = time.time()

        is_training(True, session=self.sess)
        try:
            # Loop over all batches
            for _ in range(n_batches):
                feed, _, _ = train_data.next_batch(batch_size)

                # Update discriminator.
                z = self.generator_noise_distribution(batch_size, self.noise_dim, **noise_params)

                # Feed dict = Ground truth data, Latent noise vector.
                feed_dict = { self.gt_data: feed, self.noise: z }
                loss_d, _ = self.sess.run([self.loss_d, self.opt_d], feed_dict=feed_dict)
                loss_g, _ = self.sess.run([self.loss_g, self.opt_g], feed_dict=feed_dict)

                # Compute average loss
                epoch_loss_d += loss_d
                epoch_loss_g += loss_g

            is_training(False, session=self.sess)
        except Exception:
            raise
        finally:
            is_training(False, session=self.sess)

        epoch_loss_d /= n_batches
        epoch_loss_g /= n_batches
        duration = time.time() - start_time
        return (epoch_loss_d, epoch_loss_g), duration


    def train(self, train_data, log_file=None, held_out_data=None):
        
        stats = []

        if self.conf.saver_step is not None:
            create_dir(self.conf.train_dir)

        for _ in range(self.conf.training_epochs):
            loss, duration = self._single_epoch_train(train_data)
            epoch = int(self.sess.run(self.epoch.assign_add(tf.constant(1.0))))
            stats.append((epoch, loss, duration))

            if epoch % self.conf.loss_display_step == 0:
                
                print("Epoch:", '%04d' % (epoch), 'training time (minutes)=', "{:.4f}".format(duration / 60.0), "loss=", "{:.9f}".format(loss))
                
                if log_file is not None:
                    log_file.write('%04d\t%.9f\t%.4f\n' % (epoch, loss, duration / 60.0))

            # Save the models checkpoint periodically.
            if self.conf.saver_step is not None and (epoch % self.conf.saver_step == 0 or epoch - 1 == 0):
                
                checkpoint_path = osp.join(self.conf.train_dir, model_saver_id)
                self.saver.save(self.sess, checkpoint_path, global_step=self.epoch)

            if self.conf.exists_and_is_not_none('summary_step') and (epoch % self.conf.summary_step == 0 or epoch - 1 == 0):
                
                summary = self.sess.run(self.merged_summaries)
                self.train_writer.add_summary(summary, epoch)
        
        return stats

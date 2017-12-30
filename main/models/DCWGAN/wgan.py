class WGAN:
    def __init__(self, batch_size = 128, learning_rate = 0.0002, alpha = 0.2, LAMBDA = 10):
        self.X = tf.placeholder(tf.float32, (None, 100))
        self.Y = tf.placeholder(tf.float32, (None, 64, 64, 3))
        
        g_model = generator(self.X, 'generator', alpha = alpha)
        self.g_out = generator(self.X, 'generator', reuse = True, training = False)
        d_logits_real = discriminator(self.Y, 'discriminator', alpha = alpha)
        d_logits_fake = discriminator(g_model, 'discriminator', reuse = True, alpha = alpha)
        
        self.g_loss = -tf.reduce_mean(d_logits_fake)
        self.d_loss = tf.reduce_mean(d_logits_fake) - tf.reduce_mean(d_logits_real)
        
        alpha = tf.random_uniform(shape = [batch_size, 1], minval = 0., maxval = 1.)
        differences = g_model - self.Y
        interpolates = self.Y + (alpha * differences)
        gradients = tf.gradients(discriminator(interpolates, 'discriminator', reuse = True, alpha = alpha), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices = [1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        self.d_loss += LAMBDA * gradient_penalty
        
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]

        self.d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5, beta2 = 0.999).minimize(self.d_loss, var_list = d_vars)
        self.g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5, beta2 = 0.999).minimize(self.g_loss, var_list = g_vars)

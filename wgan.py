\"\"\"
WGAN-GP for TimeGAN — prevents discriminator collapse.

Adds to the existing TimeGAN architecture:
  1. Wasserstein loss (replaces BCE for discriminator/critic)
  2. Gradient penalty (enforces Lipschitz constraint)
  3. Critic trained 5x per generator step
  4. Linear output (no sigmoid) on critic

Reference: Gulrajani et al., 'Improved Training of Wasserstein GANs', NeurIPS 2017

Usage:
    from wgan import WGANLosses
    d_loss, g_loss = WGANLosses.get_losses(real_score, fake_score, real_data, fake_data, critic)
\"\"\"

import tensorflow as tf

class WGANLosses:
    \"\"\"Wasserstein GAN with Gradient Penalty loss functions.\"\"\"
    
    @staticmethod
    def critic_loss(real_score, fake_score):
        \"\"\"Wasserstein critic loss: maximize E[critic(real)] - E[critic(fake)].
        
        Args:
            real_score: critic output for real windows (batch, 1)
            fake_score: critic output for generated windows (batch, 1)
        
        Returns:
            critic_loss: Wasserstein distance (scalar)
        \"\"\"
        return tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)
    
    @staticmethod
    def gradient_penalty(critic, real_data, fake_data, batch_size):
        \"\"\"Gradient penalty for Lipschitz constraint.
        
        Args:
            critic: the critic model (must be callable)
            real_data: real market windows (batch, seq_len, features)
            fake_data: generated windows (batch, seq_len, features)
            batch_size: current batch size
        
        Returns:
            gp: gradient penalty term (scalar)
        \"\"\"
        eps = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        interpolated = eps * real_data + (1.0 - eps) * fake_data
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            score = critic(interpolated)
        
        grads = tape.gradient(score, interpolated)
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        return tf.reduce_mean((grad_norm - 1.0) ** 2) * 10.0  # lambda=10
    
    @staticmethod
    def generator_loss(fake_score):
        \"\"\"Generator loss: maximize E[critic(fake)].
        Equivalent to minimizing -E[critic(fake)].
        \"\"\"
        return -tf.reduce_mean(fake_score)

    @classmethod
    def get_losses(cls, critic, real_data, fake_data, real_score, fake_score, batch_size):
        \"\"\"Compute all WGAN-GP losses in one call.
        
        Returns:
            (d_loss, g_loss, gp) — critic loss, generator loss, gradient penalty
        \"\"\"
        d_loss = cls.critic_loss(real_score, fake_score)
        gp = cls.gradient_penalty(critic, real_data, fake_data, batch_size)
        g_loss = cls.generator_loss(fake_score)
        return d_loss, g_loss, gp


class WGANCritic(tf.keras.Model):
    \"\"\"WGAN critic — same architecture as discriminator but with linear output.
    
    For TimeGAN: The original discriminator uses sigmoid for binary classification.
    The WGAN critic uses NO activation (linear output) for Wasserstein distance.
    \"\"\"
    def __init__(self, hidden_dim=24, num_layers=3, n_features=15):
        super().__init__()
        self.gru_layers = [
            tf.keras.layers.GRU(hidden_dim, return_sequences=(i < num_layers - 1))
            for i in range(num_layers)
        ]
        self.fc = tf.keras.layers.Dense(1)  # Linear — no activation!
        self._n_features = n_features
        self._seq_len = 60
    
    def call(self, x):
        for gru in self.gru_layers:
            x = gru(x)
        return self.fc(x)


def train_wgan_step(critic, generator, real_data, c_optimizer, g_optimizer, batch_size, gp_lambda=10.0):
    \"\"\"Single WGAN-GP training step.
    
    Args:
        critic: WGAN critic model
        generator: Generator model
        real_data: batch of real market windows
        c_optimizer: critic optimizer
        g_optimizer: generator optimizer
        batch_size: current batch size
        gp_lambda: gradient penalty weight (default 10)
    
    Returns:
        (c_loss, g_loss, gp) — critic loss, generator loss, gradient penalty
    \"\"\"
    actual_bs = tf.shape(real_data)[0]
    
    # Critic trained N_CRITIC times per generator step
    N_CRITIC = 5
    c_loss_avg = 0.0
    gp_avg = 0.0
    
    for _ in range(N_CRITIC):
        z = tf.random.normal((actual_bs, generator._seq_len, 24))
        with tf.GradientTape() as tape:
            fake_data = generator(z)
            real_score = critic(real_data)
            fake_score = critic(fake_data)
            c_loss = WGANLosses.critic_loss(real_score, fake_score)
            gp = WGANLosses.gradient_penalty(critic, real_data, fake_data, actual_bs)
            c_total = c_loss + gp_lambda * gp
        
        grads = tape.gradient(c_total, critic.trainable_variables)
        c_optimizer.apply_gradients(zip(grads, critic.trainable_variables))
        c_loss_avg += c_loss
        gp_avg += gp
    
    c_loss_avg /= N_CRITIC
    gp_avg /= N_CRITIC
    
    # Generator step
    z = tf.random.normal((actual_bs, generator._seq_len, 24))
    with tf.GradientTape() as tape:
        fake_data = generator(z)
        fake_score = critic(fake_data)
        g_loss = WGANLosses.generator_loss(fake_score)
    
    grads = tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
    
    return float(c_loss_avg), float(g_loss), float(gp_avg)

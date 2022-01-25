import time
import tensorflow as tf

from model import evaluate
from model import sxesrgan

from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay


class Trainer:
    def __init__(self,
                 model,
                 loss,
                 learning_rate,
                 checkpoint_dir='./ckpt/edsr'):

        self.now = None
        self.loss = loss
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate),
                                              model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)

        self.restore()

    @property
    def model(self):
        return self.checkpoint.model

    def train(self, train_dataset, valid_dataset, steps, evaluate_every=1000, save_best_only=False):
        loss_mean = Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        self.now = time.perf_counter()

        pre_writer = tf.summary.create_file_writer('logs/sxesr_pretrain')
        with pre_writer.as_default():
            for lr, hr in train_dataset.take(steps - ckpt.step.numpy()):
                ckpt.step.assign_add(1)
                step = ckpt.step.numpy()

                loss = self.train_step(lr, hr)
                loss_mean(loss)

                if step % evaluate_every == 0:
                    loss_value = loss_mean.result()
                    loss_mean.reset_states()

                    # Compute PSNR on validation dataset
                    psnr_value = self.evaluate(valid_dataset)
                    tf.summary.scalar('MSE Loss', loss_value, step=tf.cast(step, tf.int64))
                    tf.summary.scalar('PSNR', psnr_value, step=tf.cast(step, tf.int64))
                    duration = time.perf_counter() - self.now
                    pre_writer.flush()
                    print(
                        f'{step}/{steps}: loss = {loss_value.numpy():.3f}, PSNR = {psnr_value.numpy():3f} ({duration:.2f}s)')

                    if save_best_only and psnr_value <= ckpt.psnr:
                        self.now = time.perf_counter()
                        # skip saving checkpoint, no PSNR improvement
                        continue

                    ckpt.psnr = psnr_value
                    ckpt_mgr.save()

                    self.now = time.perf_counter()
            

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.checkpoint.model(lr, training=True)
            loss_value = self.loss(hr, sr)

        gradients = tape.gradient(
            loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(
            zip(gradients, self.checkpoint.model.trainable_variables))

        return loss_value

    def evaluate(self, dataset):
        return evaluate(self.checkpoint.model, dataset)

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(
                f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')


class EdsrTrainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5])):
        super().__init__(model, loss=MeanAbsoluteError(),
                         learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class WdsrTrainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-3, 5e-4])):
        super().__init__(model, loss=MeanAbsoluteError(),
                         learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class SXESrganGeneratorTrainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=1e-4):
        super().__init__(model, loss=MeanSquaredError(),
                         learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def train(self, train_dataset, valid_dataset, steps=1000000, evaluate_every=10, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class SXESrganTrainer:
    #
    # TODO: model and optimizer checkpoints
    #
    def __init__(self,
                 generator,
                 discriminator,
                 content_loss='VGG54',
                 learning_rate=PiecewiseConstantDecay(boundaries=[1000], values=[1e-4, 1e-5]),
                 checkpoint_dir = './ckpt/sxesrgan'):

        self.now = None

        if content_loss == 'VGG22':
            self.vgg = sxesrgan.vgg_22()
        elif content_loss == 'VGG54':
            self.vgg = sxesrgan.vgg_54()
        else:
            raise ValueError("content_loss must be either 'VGG22' or 'VGG54'")

        self.content_loss = content_loss
        
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              gen_loss=tf.Variable(-1.0),
                                              disc_loss=tf.Variable(-1.0),
                                              generator_optimizer = Adam(learning_rate=learning_rate),
                                              discriminator_optimizer = Adam(learning_rate=learning_rate),
                                              generator = generator,
                                              discriminator = discriminator)

        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)
        
        self.restore()

        # self.generator = generator
        # self.discriminator = discriminator
        # self.generator_optimizer = Adam(learning_rate=learning_rate)
        # self.discriminator_optimizer = Adam(learning_rate=learning_rate)

        self.gen_weight = 0.005
        self.l1_weight =  0.01

        self.binary_cross_entropy = BinaryCrossentropy(from_logits=True)
        self.mean_squared_error = MeanSquaredError()
        self.mean_absolute_error = MeanAbsoluteError()

    @property
    def generator(self):
        return self.checkpoint.generator
    
    @property
    def discriminator(self):
        return self.checkpoint.discriminator

    def train(self, train_dataset, evaluate_every=20, steps=200000):
        pls_metric = Mean()
        dls_metric = Mean()
        
        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        self.now = time.perf_counter()

        gan_writer = tf.summary.create_file_writer('logs/sxesrgan')

        with gan_writer.as_default():
            for lr, hr in train_dataset.take(steps-ckpt.step.numpy()):
                ckpt.step.assign_add(1)
                step = ckpt.step.numpy()

                pl, dl, gl, cl = self.train_step(lr, hr)
                pls_metric(pl)
                dls_metric(dl)

                if step % evaluate_every == 0:
                    tf.summary.scalar('Perceptual Loss', pl, step=tf.cast(step, tf.int64))
                    tf.summary.scalar('Discriminator Loss', dl, step=tf.cast(step, tf.int64))
                    tf.summary.scalar('Generator Loss', gl, step=tf.cast(step, tf.int64))
                    tf.summary.scalar('Content Loss', cl, step=tf.cast(step, tf.int64))
                    gan_writer.flush()
                    print(
                        f'{step}/{steps}, perceptual loss = {pls_metric.result():.4f}, discriminator loss = {dls_metric.result():.4f}')
                    pls_metric.reset_states()
                    dls_metric.reset_states()

                    ckpt.gen_loss = gl
                    ckpt.disc_loss = dl

                    ckpt_mgr.save()


    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.checkpoint.generator(lr, training=True)

            hr_output = self.checkpoint.discriminator(hr, training=True)
            sr_output = self.checkpoint.discriminator(sr, training=True)

            pixel_loss = self._pixelwise_loss(hr, sr)
            con_loss = self._content_loss(hr, sr)
            gen_loss = self._generator_loss(hr_output, sr_output)
            
            # Calculate LPIPS Similarity
            # loss_fn_alex = lpips.LPIPS(net='alex')
            # lpips_loss = loss_fn_alex.forward(
            #     hr, sr)  # Calculate LPIPS Similarity
            
            perc_loss = con_loss + self.gen_weight * gen_loss + self.l1_weight * pixel_loss

            disc_loss = self._discriminator_loss(hr_output, sr_output)
            
            # disc_loss = self._discriminator_loss_ragan(hr_output, sr_output)

        gradients_of_generator = gen_tape.gradient(
            perc_loss, self.checkpoint.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.checkpoint.discriminator.trainable_variables)

        self.checkpoint.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.checkpoint.generator.trainable_variables))
        self.checkpoint.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.checkpoint.discriminator.trainable_variables))

        return perc_loss, disc_loss, gen_loss, con_loss

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')
        else:
            print('Training Model from Scratch')


    @tf.function
    def _content_loss(self, hr, sr):
        sr = preprocess_input(sr)
        hr = preprocess_input(hr)
        sr_features = self.vgg(sr) / 12.75
        hr_features = self.vgg(hr) / 12.75
        return self.mean_squared_error(hr_features, sr_features)
        # return self.mean_absolute_error(hr_features, sr_features)

    def _pixelwise_loss(self, hr, sr):
        return self.mean_absolute_error(hr, sr)

    def _generator_loss(self, hr_out, sr_out):
        sigma = tf.sigmoid
        hr_loss = sigma(hr_out - tf.reduce_mean(sr_out))
        sr_loss = sigma(sr_out - tf.reduce_mean(hr_out))
        return 0.5 * (
            self.binary_cross_entropy(tf.ones_like(sr_loss), sr_loss) +
            self.binary_cross_entropy(tf.zeros_like(hr_loss), hr_loss))
        # return self.binary_cross_entropy(tf.ones_like(sr_out), sr_out)

    def _discriminator_loss(self, hr_out, sr_out):
        sigma = tf.sigmoid
        hr_loss = sigma(hr_out - tf.reduce_mean(sr_out))
        sr_loss = sigma(sr_out - tf.reduce_mean(hr_out))

        return 0.5 * (self.binary_cross_entropy(tf.ones_like(hr_loss), hr_loss) + self.binary_cross_entropy(tf.zeros_like(sr_loss), sr_loss))
        

    # def _discriminator_loss_ragan(self, real_discriminator_logits, fake_discriminator_logits):
    #     sigma = tf.sigmoid
    #     real_logits = sigma(real_discriminator_logits -
    #                         tf.reduce_mean(fake_discriminator_logits))
    #     fake_logits = sigma(fake_discriminator_logits -
    #                         tf.reduce_mean(real_discriminator_logits))
    #     return 0.5 * (
    #         self.binary_cross_entropy(tf.ones_like(real_logits), real_logits) +
    #         self.binary_cross_entropy(tf.zeros_like(fake_logits), fake_logits))

# -*- coding: utf-8 -*-
"""Copy of major_project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EMXbM5HdeNNj9wc6y_1KwHYfA7QTlhqN

## Mounting GDrive
"""

# from google.colab import drive
# drive.mount('/content/gdrive')

# cd /content/gdrive/MyDrive/xsrgan/

"""## Imports"""

# Commented out IPython magic to ensure Python compatibility.

# %matplotlib inline

# """### Importing SRGAN Model"""

# from model.srgan import generator as srgen
# from model.srgan import discriminator as srdisc
# from srgantrain import SrganTrainer
# from srgantrain import SrganGeneratorTrainer

# """### Importing XSRGAN Model"""

# from model.xsrgan import generator as xsrgen
# from model.xsrgan import discriminator as xsrdisc
# from xsrgantrain import XSrganTrainer
# from xsrgantrain import XSrganGeneratorTrainer

# """### Importing ESRGAN Model"""

# from model.esrgan import generator as esrgen
# from model.esrgan import discriminator as esrdisc
# from esrgantrain import ESrganTrainer
# from esrgantrain import ESrganGeneratorTrainer

"""### Importing XESRGAN Model"""


"""## Checking Number of Parameters"""

# srgen().summary()

# xsrgen().summary()

# esrgen().summary()

# import torch
# import lpips
import time
from xesrgantrain import XESrganGeneratorTrainer
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from model import resolve_single
from data import DIV2K
from utils import load_image
from PIL import Image
import numpy as np
from model.xesrgan import generator as xesrgen
from model.xesrgan import discriminator as xesrdisc
from xesrgantrain import XESrganTrainer
# xesrgen().summary()

tf.compat.v1.enable_eager_execution()


"""## Loading Datasets"""

div2k_train = DIV2K(scale=4, subset='train', downgrade='bicubic', images_dir='/content/gdrive/My Drive/super-resolution/.div2k/images',
                    caches_dir='/content/gdrive/My Drive/super-resolution/.div2k/caches')
div2k_valid = DIV2K(scale=4, subset='valid', downgrade='bicubic', images_dir='/content/gdrive/My Drive/super-resolution/.div2k/images',
                    caches_dir='/content/gdrive/My Drive/super-resolution/.div2k/caches')

train_ds = div2k_train.dataset(batch_size=1, random_transform=True)
valid_ds = div2k_valid.dataset(
    batch_size=1, random_transform=True, repeat_count=1)

"""## Setting Weights Directory"""

# weights_dir = 'weights/xsrgan'
weights_dir = 'weights/xesrgan'
def weights_file(filename): return os.path.join(weights_dir, filename)


os.makedirs(weights_dir, exist_ok=True)

"""## Training XSRGAN

### Pre-Trainer
"""

# pre_trainer = XSrganGeneratorTrainer(model=xsrgen(), checkpoint_dir=f'.ckpt_xesr/pre_generator')

# pre_trainer.train(train_ds,
#                   valid_ds,
#                   steps=15000,
#                   evaluate_every=20)

# pre_trainer.model.save_weights(weights_file('pre_generator.h5'))

"""### GAN Trainer"""

# gan_generator = xsrgen()
# gan_discriminator = xsrdisc()
# gan_generator.load_weights(weights_file('pre_generator.h5'))

gan_generator = xesrgen()
gan_discriminator = xesrdisc()
# gan_generator.load_weights(weights_file('xesr_pre_generator.h5'))

# gan_generator.vgg()

gan_trainer = XESrganTrainer(
    generator=gan_generator, discriminator=gan_discriminator, checkpoint_dir='./ckpt/xesrgan')
gan_trainer.train(train_ds, evaluate_every=20, steps=2500)

# gan_trainer = XSrganTrainer(generator=gan_generator, discriminator=gan_discriminator, checkpoint_dir = './ckpt/xsrgan')
# gan_trainer.train(train_ds, evaluate_every=20, steps=2500)

# gan_trainer.generator.save_weights(weights_file('gan_generator.h5'))
# gan_trainer.discriminator.save_weights(weights_file('gan_discriminator.h5'))

# """## Demo

# #### Loading Orignal SRGAN
# """

# # Location of SRGAN model weights (needed for demo)
# weights_dir = 'orignal_weights/srgan'
# def weights_file(filename): return os.path.join(weights_dir, filename)


# os.makedirs(weights_dir, exist_ok=True)

# srgan_generator = srgen()
# srgan_generator.load_weights(weights_file('gan_generator.h5'))

# """#### Loading New XSRGAN"""

# weights_dir = 'weights/xsrgan'
# def weights_file(filename): return os.path.join(weights_dir, filename)


# os.makedirs(weights_dir, exist_ok=True)

# xsrgan_generator = xsrgen()
# xsrgan_generator.load_weights(weights_file('gan_generator.h5'))

# """#### Defining Helper Functions"""


# def preprocess_image(image_path):
#     """ Loads image from path and preprocesses to make it model ready
#         Args:
#           image_path: Path to the image file
#     """
#     hr_image = tf.image.decode_image(tf.io.read_file(image_path))
#     if hr_image.shape[-1] == 4:
#         hr_image = hr_image[..., :-1]
#     hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4)
#     hr_image = tf.image.crop_to_bounding_box(
#         hr_image, hr_size[0], hr_size[1], 96, 96)
#     hr_image = tf.cast(hr_image, tf.float32)
#     return tf.expand_dims(hr_image, 0)


# def downscale_image(image):
#     """
#         Scales down images using bicubic downsampling.
#         Args:
#             image: 3D or 4D tensor of preprocessed image
#     """
#     image_size = []
#     if len(image.shape) == 3:
#         image_size = [image.shape[1], image.shape[0]]
#     else:
#         raise ValueError("Dimension mismatch. Can work only on single image.")

#     image = tf.squeeze(
#         tf.cast(
#             tf.clip_by_value(image, 0, 255), tf.uint8))

#     lr_image = np.asarray(
#         Image.fromarray(image.numpy())
#         .resize([image_size[0] // 4, image_size[1] // 4],
#                 Image.BICUBIC))
#     lr_image = tf.expand_dims(lr_image, 0)
#     lr_image = tf.cast(lr_image, tf.float32)
#     return lr_image


# !pip install lpips
# loss_fn_alex = lpips.LPIPS(net='alex')


# def evaluate_LPIPS(lr, hr, sr):
#     sum_LPIPS, num_images = 0, 0
#     hr = tf.image.resize(
#         hr, (sr.shape[0], sr.shape[1]), method=tf.image.ResizeMethod.BICUBIC)
#     sr, hr = tf.expand_dims(tf.transpose(sr, [2, 0, 1]), axis=0), tf.expand_dims(
#         tf.transpose(hr, [2, 0, 1]), axis=0)

#     # Calculate LPIPS Similarity
#     sum_LPIPS += loss_fn_alex.forward(torch.Tensor(hr.numpy()),
#                                       torch.Tensor(sr.numpy()))
#     num_images += 1
#     return sum_LPIPS / num_images


# def super_resolution(path):

#     hr_image = tf.squeeze(preprocess_image(path))
#     lr_image = tf.squeeze(downscale_image(tf.squeeze(hr_image)))

#     start = time.time()
#     bicubic_image = tf.image.resize(lr_image, [
#                                     hr_image.shape[0], hr_image.shape[1]], method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False)
#     time_bicubic = time.time() - start

#     start = time.time()
#     srgan_sr = resolve_single(srgan_generator, lr_image)
#     time_srgan = time.time() - start

#     start = time.time()
#     xsrgan_sr = resolve_single(xsrgan_generator, lr_image)
#     time_xsrgan = time.time() - start

#     print('LPIP for SRGAN: {}'.format(evaluate_LPIPS(
#         lr_image, hr_image, srgan_sr).data.cpu().numpy().reshape(1, )))
#     print('LPIP for BICUBIC: {}'.format(evaluate_LPIPS(
#         lr_image, hr_image, bicubic_image).data.cpu().numpy().reshape(1,)))
#     print('LPIP for XSRGAN: {}'.format(evaluate_LPIPS(
#         lr_image, hr_image, xsrgan_sr).data.cpu().numpy().reshape((1,))))

#     print('Time taken by Bicubic Interpolation %f' % time_bicubic)
#     print('Time taken by SRGAN %f' % time_srgan)
#     print('Time taken by XSRGAN %f' % time_xsrgan)

#     plt.figure(figsize=(40, 40))

#     srgan_sr = tf.image.resize_with_pad(
#         srgan_sr, hr_image.shape[0], hr_image.shape[1])
#     xsrgan_sr = tf.image.resize_with_pad(
#         xsrgan_sr, hr_image.shape[0], hr_image.shape[1])

#     images = [lr_image, bicubic_image, srgan_sr, xsrgan_sr]
#     titles = ['LR', 'BICUBIC', 'SRGAN', 'XSRGAN (Ours)']
#     positions = [1, 2, 3, 4]

#     for i, (img, title, pos) in enumerate(zip(images, titles, positions)):
#         img = np.asarray(img)
#         img = tf.clip_by_value(img, 0, 255)
#         label = ''
#         if(title != 'Orignal' and title != 'LR'):
#             psnr = tf.image.psnr(img, tf.clip_by_value(
#                 hr_image, 0, 255), max_val=255)
#             ssim = tf.image.ssim(img, tf.clip_by_value(
#                 hr_image, 0, 255), max_val=255)
#             label = '({0:.3f}/'.format(psnr) + '{0:.3f})'.format(ssim)
#         else:
#             label = '(PSNR/SSIM)'
#         shape = img.shape
#         img = Image.fromarray(tf.cast(img, tf.uint8).numpy())
#         plt.subplot(1, 4, pos)
#         plt.imshow(img)
#         plt.title(title, fontsize=20)
#         plt.xticks([])
#         plt.yticks([])
#         plt.xlabel(label, fontsize=20)


# def plot_sr_from_lr(path):
#     image = tf.image.decode_image(tf.io.read_file(path))
#     if image.shape[-1] == 4:
#         image = image[..., :-1]
#     image = tf.squeeze(
#         tf.cast(
#             tf.clip_by_value(image, 0, 255), tf.uint8))
#     lr_image = tf.cast(image, tf.float32)
#     lr_image = tf.image.resize(lr_image, [lr_image.shape[0]//2, lr_image.shape[1]//2],
#                                method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False)
#     srgan_sr = resolve_single(srgan_generator, tf.squeeze(lr_image))
#     bicubic_image = tf.image.resize(lr_image, [
#                                     lr_image.shape[0]*4, lr_image.shape[1]*4], method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False)
#     xsrgan_sr = resolve_single(xsrgan_generator, lr_image)
#     plt.figure(figsize=(40, 40))

#     images = [lr_image, bicubic_image, srgan_sr, xsrgan_sr]
#     titles = ['LR', 'BICUBIC', 'SRGAN', 'XSRGAN (Ours)']
#     positions = [1, 2, 3, 4]
#     for i, (img, title, pos) in enumerate(zip(images, titles, positions)):
#         img = np.asarray(img)
#         img = tf.clip_by_value(img, 0, 255)
#         img = Image.fromarray(tf.cast(img, tf.uint8).numpy())
#         plt.subplot(1, 4, pos)
#         plt.imshow(img)
#         plt.title(title, fontsize=20)
#         plt.xticks([])
#         plt.yticks([])


# """### Examples"""

# IMAGE_PATH = ".div2k/images/DIV2K_valid_HR/0849.png"
# super_resolution(IMAGE_PATH)

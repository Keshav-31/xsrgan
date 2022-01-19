from tensorflow.python.keras.applications import mobilenet_v2
from tensorflow.python.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda, SeparableConv2D, Concatenate
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow as tf
from model.common import pixel_shuffle, normalize_01, normalize_m11, denormalize_m11

LR_SIZE = 24
HR_SIZE = 96


# def upsample(x_in, num_filters):
#     x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
#     x = Lambda(pixel_shuffle(scale=2))(x)
#     return PReLU(shared_axes=[1, 2])(x)


def sep_bn(x, filters, kernel_size=3, strides=1, batch_norm=False):

    x = SeparableConv2D(filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding='same',
                        use_bias=False)(x)
    if(batch_norm):
        x = BatchNormalization()(x)
    return x


def res_block(x_in, num_filters, momentum=0.8):
    # x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    # x = BatchNormalization(momentum=momentum)(x)
    x = sep_bn(x=x_in, filters=num_filters, kernel_size=3, batch_norm=True)
    x = PReLU(shared_axes=[1, 2])(x)
    # x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    # x = BatchNormalization(momentum=momentum)(x)
    x = sep_bn(x=x_in, filters=num_filters, kernel_size=3, batch_norm=True)

    x = Add()([x_in, x])
    return x


def sr_resnet(num_filters=64, num_res_blocks=16):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize_01)(x_in)

    x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
    x = x_1 = PReLU(shared_axes=[1, 2])(x)

    for _ in range(num_res_blocks):
        x = res_block(x, num_filters)

    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x_1, x])

    x = upsample(x, num_filters * 4)
    x = upsample(x, num_filters * 4)

    x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
    x = Lambda(denormalize_m11)(x)

    return Model(x_in, x)


def residual_dense_block_orignal(input, filters):
    x1 = Conv2D(filters=filters, kernel_size=3,
                strides=1, padding='same')(input)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Concatenate()([input, x1])

    x2 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x1)
    x2 = LeakyReLU(0.2)(x2)
    x2 = Concatenate()([input, x1, x2])

    x3 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x2)
    x3 = LeakyReLU(0.2)(x3)
    x3 = Concatenate()([input, x1, x2, x3])

    x4 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x3)
    x4 = LeakyReLU(0.2)(x4)
    x4 = Concatenate()([input, x1, x2, x3, x4])

    x5 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x4)
    x5 = Lambda(lambda x: x * 0.2)(x5)
    x = Add()([x5, input])

    return x

def residual_dense_block(input, filters):
    x1 = sep_bn(x=input, filters=filters, kernel_size=3, strides=1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Concatenate()([input, x1])

    x2 = sep_bn(x=x1, filters=filters, kernel_size=3, strides=1)
    x2 = LeakyReLU(0.2)(x2)
    x2 = Concatenate()([input, x1, x2])

    x3 = sep_bn(x=x2, filters=filters, kernel_size=3, strides=1)
    x3 = LeakyReLU(0.2)(x3)
    x3 = Concatenate()([input, x1, x2, x3])

    x4 = sep_bn(x=x3, filters=filters, kernel_size=3, strides=1)
    x4 = LeakyReLU(0.2)(x4)
    x4 = Concatenate()([input, x1, x2, x3, x4])

    x5 = sep_bn(x=x4, filters=filters, kernel_size=3, strides=1)
    x5 = Lambda(lambda x: x * 0.2)(x5)
    x = Add()([x5, input])

    return x


def rrdb(input, filters):
    x = residual_dense_block(input, filters)
    x = residual_dense_block(x, filters)
    x = residual_dense_block(x, filters)
    x = Lambda(lambda x: x * 0.2)(x)
    out = Add()([x, input])
    return out


def sub_pixel_conv2d(scale_factor=2, **kwargs):
    return Lambda(lambda x: tf.nn.depth_to_space(x, scale_factor), **kwargs)


def upsample(input_tensor, filters, scale_factor=2):
    x = Conv2D(filters=filters*4, kernel_size=3,
               strides=1, padding='same')(input_tensor)
    x = sub_pixel_conv2d(scale_factor=scale_factor)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    return x


def rrdb_net(input_shape=(None, None, 3), filters=64, scale_factor=4, name='RRDB_model'):
    lr_image = Input(shape=input_shape, name='input')

    # Pre-residual
    x_start = Conv2D(filters, kernel_size=3, strides=1,
                     padding='same')(lr_image)
    x_start = LeakyReLU(0.2)(x_start)

    # Residual block
    x = rrdb(x_start, filters)

    # Post Residual block
    x = Conv2D(filters,  kernel_size=3, strides=1, padding='same')(x)
    x = Lambda(lambda x: x * 0.2)(x)
    x = Add()([x, x_start])

    # Upsampling
    x = upsample(x, filters, scale_factor)

    x = Conv2D(filters, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    out = Conv2D(filters=3, kernel_size=3, strides=1, padding='same')(x)

    return Model(inputs=lr_image, outputs=out, name=name)


# generator = sr_resnet
generator = rrdb_net


def discriminator_block(x_in, num_filters, strides=1, batchnorm=True, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3,
               strides=strides, padding='same')(x_in)
    if batchnorm:
        x = BatchNormalization(momentum=momentum)(x)
    return LeakyReLU(alpha=0.2)(x)


def discriminator(num_filters=64):
    x_in = Input(shape=(HR_SIZE, HR_SIZE, 3))
    x = Lambda(normalize_m11)(x_in)

    x = discriminator_block(x, num_filters, batchnorm=False)
    x = discriminator_block(x, num_filters, strides=2)

    x = discriminator_block(x, num_filters * 2)
    x = discriminator_block(x, num_filters * 2, strides=2)

    x = discriminator_block(x, num_filters * 4)
    x = discriminator_block(x, num_filters * 4, strides=2)

    x = discriminator_block(x, num_filters * 8)
    x = discriminator_block(x, num_filters * 8, strides=2)

    x = Flatten()(x)

    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(x_in, x)


# def discriminator(num_filters=64):
#     mobileNetV2 = MobileNetV2(input_shape=(
#         HR_SIZE, HR_SIZE, 3), include_top=False, weights='imagenet')
#     for layer in mobileNetV2.layers:
#         layer.trainable = False

#     x = mobileNetV2.output
#     x = MaxPool2D(pool_size=(2, 2))(x)
#     x = Flatten()(x)
#     x = Dense(1024)(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Dense(256)(x)
#     x = Dropout(0.3)(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Dense(1, activation='sigmoid')(x)

#     model = Model(mobileNetV2.input, x)
#     return model


def vgg_22():
    return _vgg(5)


def vgg_54():
    return _vgg(20)


def _vgg(output_layer):
    vgg = VGG19(input_shape=(None, None, 3),
                include_top=False, weights='imagenet')
    return Model(vgg.input, vgg.layers[output_layer].output)


# print(discriminator().summary())

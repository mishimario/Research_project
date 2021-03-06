'''
UNet
'''

# built-in
import pdb

# external
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model

# customs
from . import components


class UNet(Layer):
    '''U-Net'''
    def __init__(
        self,
        filters_first,
        n_downsample,
        rate,
        kernel_size,
        conv_stride,
        bn=False,
        trainable=True,
        padding='valid',
        activation='relu',
        kernel_regularizer=None,
        **kargs,
    ):
        super().__init__(**kargs)
        self.configs = dict(
            filters_first=filters_first,
            n_downsample=n_downsample,
            rate=rate,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            bn=bn,
            trainable=trainable,
            padding=padding,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            **kargs,
        )
        self.encoder = components.Encoder(
            filters_first=filters_first,
            n_downsample=n_downsample,
            rate=rate,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            bn=bn,
            padding=padding,
            activation=activation,
            trainable=trainable,
            kernel_regularizer=kernel_regularizer,
        )
        self.latent = components.Latent(
            filters_first=filters_first,
            n_downsample=n_downsample,
            rate=rate,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            bn=bn,
            padding=padding,
            activation=activation,
            trainable=trainable,
            kernel_regularizer=kernel_regularizer,
        )
        self.decoder = components.Decoder(
            rate=rate,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            bn=bn,
            padding=padding,
            activation=activation,
            trainable=trainable,
            kernel_regularizer=kernel_regularizer,
        )
        return

    def get_config(self):
        config = super().get_config()
        config.update(self.configs)
        return config

    def build(self, input_shape):
        self.encoder_output_shape, self.ref_shapes = self.encoder.build(input_shape)
        self.latent_output_shape = self.latent.build(self.encoder_output_shape)
        decoder_out = self.decoder.build(self.latent_output_shape, self.ref_shapes)
        self.built = True
        return decoder_out

    @tf.function
    def call(self, inputs, training=False):
        res_list, downsampled = self.encoder(inputs=inputs, training=training)
        latent_output = self.latent(inputs=downsampled, training=training)
        output = self.decoder(inputs=latent_output, res_list=res_list, training=training)
        return output


class UNetAnnotator(keras.Model):
    def __init__(
        self,
        n_filters_first,
        n_downsample,
        rate,
        kernel_size,
        conv_stride,
        bn=False,
        padding='valid',
        activation='relu',
        kernel_regularizer=None,
        color_num=7,
        **kargs,
    ):
        '''
        A class that represents a part of model which will produce annotation or segmentation
        Args:
            input_: input tensor
            n_filters_first: the num of filters in the first block
            n_downsample: the num of downsample
            rate: the rate of downsample and upsample
            kernel_size: kernel_size of every Conv
            conv_stride: stride in conv
            bn (bool): whether or not BN is applied
            training (bool): whether the model is being trained
                this can be None as long as bn=False
            padding: padding method used in internal components
            trainable (bool): whether or not this block is trainable
            kernel_regularizer: kernel regularizer
                can be either None, string, dict
        '''
        super().__init__(**kargs)
        self.configs = dict(
            n_filters_first=n_filters_first,
            n_downsample=n_downsample,
            rate=rate,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            bn=bn,
            padding=padding,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            **kargs,
        )
        self.kargs = kargs
        unet = self.construct_internal_model()
        last_conv = layers.Conv2D(
            filters=color_num, kernel_size=(1,1), activation='softmax', padding=padding,
            kernel_regularizer=kernel_regularizer, **kargs,
        )
        self.unet = unet
        self.padding = padding
        self.last_conv = last_conv
        return

    def construct_internal_model(self):
        model = UNet(
            filters_first=self.configs['n_filters_first'],
            n_downsample=self.configs['n_downsample'],
            rate=self.configs['rate'],
            kernel_size=self.configs['kernel_size'],
            conv_stride=self.configs['conv_stride'],
            bn=self.configs['bn'],
            padding=self.configs['padding'],
            activation=components.solve_activation(self.configs['activation']),
            kernel_regularizer=self.configs['kernel_regularizer'],
            **self.kargs,
        )
        return model

    def get_config(self):
        return self.configs

    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        return instance

    def build(self, input_shape):
        unet_out = self.unet.build(input_shape)
        self.last_conv.build(unet_out)
        self.built = True
        return

    def call(self, x, training=False):
        unet_out = self.unet(x, training=training)
        output = self.last_conv(unet_out, training=training)
        return output

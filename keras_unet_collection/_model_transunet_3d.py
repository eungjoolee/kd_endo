from __future__ import absolute_import
import sys
# sys.path.append('D:/code/VIT/keras-unet-collection-main/keras-unet-collection-main/examples/')

# from keras_unet_collection.layer_utils import *
from keras_unet_collection.activations import GELU, Snake
from keras_unet_collection._model_unet_3d import UNET_left_3D, UNET_right_3D

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, MultiHeadAttention, LayerNormalization, Dense, Embedding

from tensorflow.keras.layers import MaxPooling3D, AveragePooling3D, UpSampling3D, Conv3DTranspose, \
    GlobalAveragePooling3D
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import BatchNormalization, Activation, concatenate, multiply, add
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, ELU, Softmax


# import tensorflow.extract_volume_patches as extract_patches


def freeze_model(model, freeze_batch_norm=False):
    '''
    freeze a keras model

    Input
    ----------
        model: a keras model
        freeze_batch_norm: False for not freezing batch notmalization layers
    '''
    if freeze_batch_norm:
        for layer in model.layers:
            layer.trainable = False
    else:
        from tensorflow.keras.layers import BatchNormalization
        for layer in model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
    return model


import warnings

layer_cadidates = {
    'VGG16': ('block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3'),
    'VGG19': ('block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4', 'block5_conv4'),
    'ResNet50': ('conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out'),
    'ResNet101': ('conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block23_out', 'conv5_block3_out'),
    'ResNet152': ('conv1_relu', 'conv2_block3_out', 'conv3_block8_out', 'conv4_block36_out', 'conv5_block3_out'),
    'ResNet50V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block6_1_relu', 'post_relu'),
    'ResNet101V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block23_1_relu', 'post_relu'),
    'ResNet152V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block8_1_relu', 'conv4_block36_1_relu', 'post_relu'),
    'DenseNet121': ('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
    'DenseNet169': ('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
    'DenseNet201': ('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
    'EfficientNetB0': (
        'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation',
        'block6a_expand_activation',
        'top_activation'),
    'EfficientNetB1': (
        'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation',
        'block6a_expand_activation',
        'top_activation'),
    'EfficientNetB2': (
        'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation',
        'block6a_expand_activation',
        'top_activation'),
    'EfficientNetB3': (
        'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation',
        'block6a_expand_activation',
        'top_activation'),
    'EfficientNetB4': (
        'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation',
        'block6a_expand_activation',
        'top_activation'),
    'EfficientNetB5': (
        'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation',
        'block6a_expand_activation',
        'top_activation'),
    'EfficientNetB6': (
        'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation',
        'block6a_expand_activation',
        'top_activation'),
    'EfficientNetB7': (
        'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation',
        'block6a_expand_activation',
        'top_activation'), }


def bach_norm_checker(backbone_name, batch_norm):
    '''batch norm checker'''
    if 'VGG' in backbone_name:
        batch_norm_backbone = False
    else:
        batch_norm_backbone = True

    if batch_norm_backbone != batch_norm:
        if batch_norm_backbone:
            param_mismatch = "\n\nBackbone {} uses batch norm, but other layers received batch_norm={}".format(
                backbone_name, batch_norm)
        else:
            param_mismatch = "\n\nBackbone {} does not use batch norm, but other layers received batch_norm={}".format(
                backbone_name, batch_norm)

        warnings.warn(param_mismatch);


def backbone_zoo(backbone_name, weights, input_tensor, depth, freeze_backbone, freeze_batch_norm):
    '''
    Configuring a user specified encoder model based on the `tensorflow.keras.applications`

    Input
    ----------
        backbone_name: the bakcbone model name. Expected as one of the `tensorflow.keras.applications` class.
                       Currently supported backbones are:
                       (1) VGG16, VGG19
                       (2) ResNet50, ResNet101, ResNet152
                       (3) ResNet50V2, ResNet101V2, ResNet152V2
                       (4) DenseNet121, DenseNet169, DenseNet201
                       (5) EfficientNetB[0,7]

        weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet),
                 or the path to the weights file to be loaded.
        input_tensor: the input tensor
        depth: number of encoded feature maps.
               If four dwonsampling levels are needed, then depth=4.

        freeze_backbone: True for a frozen backbone
        freeze_batch_norm: False for not freezing batch normalization layers.

    Output
    ----------
        model: a keras backbone model.

    '''

    cadidate = layer_cadidates[backbone_name]

    # ----- #
    # depth checking
    depth_max = len(cadidate)
    if depth > depth_max:
        depth = depth_max
    # ----- #

    backbone_func = eval(backbone_name)
    backbone_ = backbone_func(include_top=False, weights=weights, input_tensor=input_tensor, pooling=None, )

    X_skip = []

    for i in range(depth):
        X_skip.append(backbone_.get_layer(cadidate[i]).output)

    model = Model(inputs=[input_tensor, ], outputs=X_skip, name='{}_backbone'.format(backbone_name))

    if freeze_backbone:
        model = freeze_model(model, freeze_batch_norm=freeze_batch_norm)

    return model


def CONV_output_3D(X, n_labels, kernel_size=1, activation='Softmax', name='conv_output'):
    '''
    Convolutional layer with output activation.

    CONV_output(X, n_labels, kernel_size=1, activation='Softmax', name='conv_output')

    Input
    ----------
        X: input tensor.
        n_labels: number of classification label(s).
        kernel_size: size of 2-d convolution kernels. Default is 1-by-1.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interface or 'Sigmoid'.
                    Default option is 'Softmax'.
                    if None is received, then linear activation is applied.
        name: prefix of the created keras layers.

    Output
    ----------
        X: output tensor.

    '''

    X = Conv3D(n_labels, kernel_size, padding='same', use_bias=True, name=name)(X)

    if activation:

        if activation == 'Sigmoid':
            X = Activation('sigmoid', name='{}_activation'.format(name))(X)

        else:
            activation_func = eval(activation)
            X = activation_func(name='{}_activation'.format(name))(X)

    return X


def decode_layer_3D(X, channel, pool_size, unpool, kernel_size=3,
                    activation='ReLU', batch_norm=False, name='decode'):
    '''
    An overall decode layer, based on either upsampling or trans conv.

    decode_layer(X, channel, pool_size, unpool, kernel_size=3,
                 activation='ReLU', batch_norm=False, name='decode')

    Input
    ----------
        X: input tensor.
        pool_size: the decoding factor.
        channel: (for trans conv only) number of convolution filters.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv3DTranspose + batch norm + activation.
        kernel_size: size of convolution kernels.
                     If kernel_size='auto', then it equals to the `pool_size`.
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.

    Output
    ----------
        X: output tensor.

    * The defaut: `kernel_size=3`, is suitable for `pool_size=2`.

    '''
    # parsers
    if unpool is False:
        # trans conv configurations
        bias_flag = not batch_norm

    elif unpool == 'nearest':
        # upsample2d configurations
        unpool = True
        interp = 'nearest'

    elif (unpool is True) or (unpool == 'bilinear'):
        # upsample2d configurations
        unpool = True
        interp = 'bilinear'

    else:
        raise ValueError('Invalid unpool keyword')

    if unpool:
        # X = UpSampling3D(size=(pool_size, pool_size, pool_size), interpolation=interp, name='{}_unpool'.format(name))(X)
        X = UpSampling3D(size=(pool_size, pool_size, pool_size), name='{}_unpool'.format(name))(X)
    else:
        if kernel_size == 'auto':
            kernel_size = pool_size

        X = Conv3DTranspose(channel, kernel_size, strides=(pool_size, pool_size, pool_size),
                            padding='same', name='{}_trans_conv'.format(name))(X)

        # batch normalization
        if batch_norm:
            X = BatchNormalization(axis=4, name='{}_bn'.format(name))(X)

        # activation
        if activation is not None:
            activation_func = eval(activation)
            X = activation_func(name='{}_activation'.format(name))(X)

    return X


def encode_layer_3D(X, channel, pool_size, pool, kernel_size='auto',
                    activation='ReLU', batch_norm=False, name='encode'):
    '''
    An overall encode layer, based on one of the:
    (1) max-pooling, (2) average-pooling, (3) strided conv3d.

    encode_layer(X, channel, pool_size, pool, kernel_size='auto',
                 activation='ReLU', batch_norm=False, name='encode')

    Input
    ----------
        X: input tensor.
        pool_size: the encoding factor.
        channel: (for strided conv only) number of convolution filters.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        kernel_size: size of convolution kernels.
                     If kernel_size='auto', then it equals to the `pool_size`.
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.

    Output
    ----------
        X: output tensor.

    '''
    # parsers
    if (pool in [False, True, 'max', 'ave']) is not True:
        raise ValueError('Invalid pool keyword')

    # maxpooling3d as default
    if pool is True:
        pool = 'max'

    elif pool is False:
        # stride conv configurations
        bias_flag = not batch_norm

    if pool == 'max':
        X = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size), name='{}_maxpool'.format(name))(X)

    elif pool == 'ave':
        X = AveragePooling3D(pool_size=(pool_size, pool_size, pool_size), name='{}_avepool'.format(name))(X)

    else:
        if kernel_size == 'auto':
            kernel_size = pool_size

        # linear convolution with strides
        X = Conv3D(channel, kernel_size, strides=(pool_size, pool_size, pool_size),
                   padding='valid', use_bias=bias_flag, name='{}_stride_conv'.format(name))(X)

        # batch normalization
        if batch_norm:
            X = BatchNormalization(axis=4, name='{}_bn'.format(name))(X)

        # activation
        if activation is not None:
            activation_func = eval(activation)
            X = activation_func(name='{}_activation'.format(name))(X)

    return X


def CONV_stack_3D(X, channel, kernel_size=3, stack_num=2,
                  dilation_rate=1, activation='ReLU',
                  batch_norm=False, name='conv_stack'):
    '''
    Stacked convolutional layers:
    (Convolutional layer --> batch normalization --> Activation)*stack_num

    CONV_stack(X, channel, kernel_size=3, stack_num=2, dilation_rate=1, activation='ReLU',
               batch_norm=False, name='conv_stack')


    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of stacked Conv3D-BN-Activation layers.
        dilation_rate: optional dilated convolution kernel.
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.

    Output
    ----------
        X: output tensor

    '''

    bias_flag = not batch_norm

    # stacking Convolutional layers
    for i in range(stack_num):

        activation_func = eval(activation)

        # linear convolution
        X = Conv3D(channel, kernel_size, padding='same', use_bias=bias_flag,
                   dilation_rate=dilation_rate, name='{}_{}'.format(name, i))(X)

        # batch normalization
        if batch_norm:
            X = BatchNormalization(axis=4, name='{}_{}_bn'.format(name, i))(X)

        # activation
        activation_func = eval(activation)
        X = activation_func(name='{}_{}_activation'.format(name, i))(X)

    return X


def UNET_left_3D(X, channel, kernel_size=3, stack_num=2, activation='ReLU',
                 pool=True, batch_norm=False, name='left0'):
    '''
    The encoder block of U-net.

    UNET_left(X, channel, kernel_size=3, stack_num=2, activation='ReLU',
              pool=True, batch_norm=False, name='left0')

    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        kernel_size: size of 3-d convolution kernels.
        stack_num: number of convolutional layers.
        activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
        pool: True or 'max' for MaxPooling3D.
              'ave' for AveragePoolin3D.
              False for strided conv + batch norm + activation.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.

    Output
    ----------
        X: output tensor.

    '''
    pool_size = 2

    X = encode_layer_3D(X, channel, pool_size, pool, activation=activation,
                        batch_norm=batch_norm, name='{}_encode'.format(name))

    X = CONV_stack_3D(X, channel, kernel_size, stack_num=stack_num, activation=activation,
                      batch_norm=batch_norm, name='{}_conv'.format(name))

    return X


def UNET_right_3D(X, X_list, channel, kernel_size=3,
                  stack_num=2, activation='ReLU',
                  unpool=True, batch_norm=False, concat=True, name='right0'):
    '''
    The decoder block of U-net.

    Input
    ----------
        X: input tensor.
        X_list: a list of other tensors that connected to the input tensor.
        channel: number of convolution filters.
        kernel_size: size of 3-d convolution kernels.
        stack_num: number of convolutional layers.
        activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
        unpool: True or 'bilinear' for Upsampling3D with bilinear interpolation.
                'nearest' for Upsampling3D with nearest interpolation.
                False for Conv3DTranspose + batch norm + activation.
        batch_norm: True for batch normalization, False otherwise.
        concat: True for concatenating the corresponded X_list elements.
        name: prefix of the created keras layers.

    Output
    ----------
        X: output tensor.

    '''

    pool_size = 2

    X = decode_layer_3D(X, channel, pool_size, unpool,
                        activation=activation, batch_norm=batch_norm, name='{}_decode'.format(name))

    # linear convolutional layers before concatenation
    X = CONV_stack_3D(X, channel, kernel_size, stack_num=1, activation=activation,
                      batch_norm=batch_norm, name='{}_conv_before_concat'.format(name))
    if concat:
        # <--- *stacked convolutional can be applied here
        X = concatenate([X, ] + X_list, axis=4, name=name + '_concat')

    # Stacked convolutions after concatenation
    X = CONV_stack_3D(X, channel, kernel_size, stack_num=stack_num, activation=activation,
                      batch_norm=batch_norm, name=name + '_conv_after_concat')

    return X


class patch_extract(Layer):
    '''
    Extract patches from the input feature map.

    patches = patch_extract(patch_size)(feature_map)

    ----------
    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner,
    T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020.
    An image is worth 16x16 words: Transformers for image recognition at scale.
    arXiv preprint arXiv:2010.11929.

    Input
    ----------
        feature_map: a four-dimensional tensor of (num_sample, width, height, channel)
        patch_size: size of split patches (width=height)

    Output
    ----------
        patches: a two-dimensional tensor of (num_sample*num_patch, patch_size*patch_size)
                 where `num_patch = (width // patch_size) * (height // patch_size)`

    For further information see: https://www.tensorflow.org/api_docs/python/tf/image/extract_patches

    '''

    def __init__(self, patch_size, patch_stride):
        super(patch_extract, self).__init__()
        self.patch_size = patch_size
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[1]
        self.patch_size_z = patch_size[2]
        self.patch_stride = patch_stride

    def call(self, images):
        batch_size = tf.shape(images)[0]

        patches = tf.extract_volume_patches(input=images,
                                            ksizes=(1, self.patch_size_x, self.patch_size_y, self.patch_size_z, 1),
                                            strides=(1, self.patch_stride, self.patch_stride, self.patch_stride, 1),
                                            padding='VALID', )
        # patches.shape = (num_sample, patch_num, patch_num, patch_size*channel)

#         patch_dim = patches.shape[-1]*self.patch_size_x*self.patch_size_y*self.patch_size_z
        patch_dim = patches.shape[-1]
        patch_num_x = patches.shape[1]
        patch_num_y = patches.shape[2]
        patch_num_z = patches.shape[3]
        patches = tf.reshape(patches, (batch_size, patch_num_x * patch_num_y * patch_num_z, patch_dim))
        # patches.shape = (num_sample, patch_num*patch_num, patch_size*channel)

        return patches

    def get_config(self):
        config = super().get_config().copy()
        config.update({'patch_size': self.patch_size, })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class patch_embedding(Layer):
    '''
    Embed patches to tokens.

    patches_embed = patch_embedding(num_patch, embed_dim)(pathes)

    ----------
    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner,
    T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020.
    An image is worth 16x16 words: Transformers for image recognition at scale.
    arXiv preprint arXiv:2010.11929.

    Input
    ----------
        num_patch: number of patches to be embedded.
        embed_dim: number of embedded dimensions.

    Output
    ----------
        embed: Embedded patches.

    For further information see: https://keras.io/api/layers/core_layers/embedding/

    '''

    def __init__(self, num_patch, embed_dim):
        super(patch_embedding, self).__init__()
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.proj = Dense(embed_dim)
        self.pos_embed = Embedding(input_dim=num_patch, output_dim=embed_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patch': self.num_patch,
            'embed_dim': self.embed_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, patch):
#         print(self.num_patch)
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
#         print(self.proj(patch).shape, self.pos_embed(pos).shape)
        embed = self.proj(patch) + self.pos_embed(pos)

        return embed


def ViT_MLP(X, filter_num, activation='GELU', name='MLP'):
    '''
    The MLP block of ViT.

    ----------
    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner,
    T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020.
    An image is worth 16x16 words: Transformers for image recognition at scale.
    arXiv preprint arXiv:2010.11929.

    Input
    ----------
        X: the input tensor of MLP, i.e., after MSA and skip connections
        filter_num: a list that defines the number of nodes for each MLP layer.
                        For the last MLP layer, its number of node must equal to the dimension of key.
        activation: activation of MLP nodes.
        name: prefix of the created keras layers.

    Output
    ----------
        V: output tensor.

    '''
    activation_func = eval(activation)

    for i, f in enumerate(filter_num):
        X = Dense(f, name='{}_dense_{}'.format(name, i))(X)
        X = activation_func(name='{}_activation_{}'.format(name, i))(X)

    return X


def ViT_block(V, num_heads, key_dim, filter_num_MLP, activation='GELU', name='ViT'):
    '''

    Vision transformer (ViT) block.

    ViT_block(V, num_heads, key_dim, filter_num_MLP, activation='GELU', name='ViT')

    ----------
    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner,
    T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020.
    An image is worth 16x16 words: Transformers for image recognition at scale.
    arXiv preprint arXiv:2010.11929.

    Input
    ----------
        V: embedded input features.
        num_heads: number of attention heads.
        key_dim: dimension of the attention key (equals to the embeded dimensions).
        filter_num_MLP: a list that defines the number of nodes for each MLP layer.
                        For the last MLP layer, its number of node must equal to the dimension of key.
        activation: activation of MLP nodes.
        name: prefix of the created keras layers.

    Output
    ----------
        V: output tensor.

    '''
    # Multiheaded self-attention (MSA)
    V_atten = V  # <--- skip
    V_atten = LayerNormalization(name='{}_layer_norm_1'.format(name))(V_atten)
    V_atten = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,
                                 name='{}_atten'.format(name))(V_atten, V_atten)
    # Skip connection
    V_add = add([V_atten, V], name='{}_skip_1'.format(name))  # <--- skip

    # MLP
    V_MLP = V_add  # <--- skip
    V_MLP = LayerNormalization(name='{}_layer_norm_2'.format(name))(V_MLP)
    V_MLP = ViT_MLP(V_MLP, filter_num_MLP, activation, name='{}_mlp'.format(name))
    # Skip connection
    V_out = add([V_MLP, V_add], name='{}_skip_2'.format(name))  # <--- skip

    return V_out


def transunet_3d_base(input_tensor, filter_num, patch_size=3, patch_stride=2, stack_num_down=2, stack_num_up=2,
                      embed_dim=768, num_mlp=3072, num_heads=12, num_transformer=12,
                      activation='ReLU', mlp_activation='GELU', batch_norm=False, pool=True, unpool=True,
                      backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True,
                      name='transunet'):
    '''
    The base of transUNET with an optional ImageNet-trained backbone.

    ----------
    Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., Lu, L., Yuille, A.L. and Zhou, Y., 2021.
    Transunet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306.

    Input
    ----------
        input_tensor: the input tensor of the base, e.g., `keras.layers.Inpyt((None, None, 3))`.
        filter_num: a list that defines the number of filters for each \
                    down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                    The depth is expected as `len(filter_num)`.
        stack_num_down: number of convolutional layers per downsampling level/block.
        stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.
        name: prefix of the created keras model and its layers.

        ---------- (keywords of ViT) ----------
        embed_dim: number of embedded dimensions.
        num_mlp: number of MLP nodes.
        num_heads: number of attention heads.
        num_transformer: number of stacked ViTs.
        mlp_activation: activation of MLP nodes.

        ---------- (keywords of backbone options) ----------
        backbone_name: the bakcbone model name. Should be one of the `tensorflow.keras.applications` class.
                       None (default) means no backbone.
                       Currently supported backbones are:
                       (1) VGG16, VGG19
                       (2) ResNet50, ResNet101, ResNet152
                       (3) ResNet50V2, ResNet101V2, ResNet152V2
                       (4) DenseNet121, DenseNet169, DenseNet201
                       (5) EfficientNetB[0-7]
        weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet),
                 or the path to the weights file to be loaded.
        freeze_backbone: True for a frozen backbone.
        freeze_batch_norm: False for not freezing batch normalization layers.

    Output
    ----------
        X: output tensor.

    '''
    activation_func = eval(activation)

    X_skip = []
    depth_ = len(filter_num)

    # ----- internal parameters ----- #

    # patch size (fixed to 1-by-1)
    patch_size_x = patch_size
    patch_size_y = patch_size
    patch_size_z = patch_size

    # input tensor size
    input_size_x = input_tensor.shape[1]
    input_size_y = input_tensor.shape[2]
    input_size_z = input_tensor.shape[3]

    # encoded feature map size
    encode_size_x = input_size_x // 2 ** (depth_ - 1) 
    encode_size_y = input_size_y // 2 ** (depth_ - 1) 
    encode_size_z = input_size_z // 2 ** (depth_ - 1)

    # number of size-1 patches
    num_patches_old = encode_size_x * encode_size_y * encode_size_z

    # dimension of the attention key (= dimension of embedings)
    key_dim = embed_dim

    # number of MLP nodes
    filter_num_MLP = [num_mlp, embed_dim]

    # ----- UNet-like downsampling ----- #

    # no backbone cases
    if backbone is None:

        X = input_tensor

        # stacked conv3d before downsampling
        X = CONV_stack_3D(X, filter_num[0], stack_num=stack_num_down, activation=activation,
                          batch_norm=batch_norm, name='{}_down0'.format(name))
        X_skip.append(X)

        # downsampling blocks
        for i, f in enumerate(filter_num[1:]):
            X = UNET_left_3D(X, f, stack_num=stack_num_down, activation=activation, pool=pool,
                             batch_norm=batch_norm, name='{}_down{}'.format(name, i + 1))
            X_skip.append(X)

    # backbone cases
    else:
        # handling VGG16 and VGG19 separately
        if 'VGG' in backbone:
            backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_skip = backbone_([input_tensor, ])
            depth_encode = len(X_skip)

        # for other backbones
        else:
            backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_ - 1, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_skip = backbone_([input_tensor, ])
            depth_encode = len(X_skip) + 1

        # extra conv2d blocks are applied
        # if downsampling levels of a backbone < user-specified downsampling levels
        if depth_encode < depth_:

            # begins at the deepest available tensor
            X = X_skip[-1]

            # extra downsamplings
            for i in range(depth_ - depth_encode):
                i_real = i + depth_encode

                X = UNET_left_3D(X, filter_num[i_real], stack_num=stack_num_down, activation=activation, pool=pool,
                                 batch_norm=batch_norm, name='{}_down{}'.format(name, i_real + 1))
                X_skip.append(X)

    # subtrack the last tensor (will be replaced by the ViT output)
    X = X_skip[-1]
    X_skip = X_skip[:-1]

    #     print(tf.shape(X))
    # 1-by-1 linear transformation before entering ViT blocks
    X = Conv3D(filter_num[-1], 1, padding='valid', use_bias=False, name='{}_conv_trans_before'.format(name))(X)
    # print(tf.shape(X))
    X = patch_extract((patch_size_x, patch_size_y, patch_size_z), patch_stride)(X)
    num_patches = X.shape[-2]
    X = patch_embedding(num_patches, embed_dim)(X)
#     print(X.shape)

    # stacked ViTs
    for i in range(num_transformer):
        X = ViT_block(X, num_heads, key_dim, filter_num_MLP, activation=mlp_activation,
                      name='{}_ViT_{}'.format(name, i))

#         print(tf.shape(X))
    

    # reshape patches to feature maps
    X = Dense(encode_size_x*encode_size_y*encode_size_z*embed_dim/num_patches)(X)
    X = tf.reshape(X, (-1, encode_size_x, encode_size_y, encode_size_z, embed_dim))
#     print(X.shape)
    # 1-by-1 linear transformation to adjust the number of channels
    X = Conv3D(filter_num[-1], 1, padding='valid', use_bias=False, name='{}_conv_trans_after'.format(name))(X)
#     print(X.shape)
    X_skip.append(X)

    # ----- UNet-like upsampling ----- #

    # reverse indexing encoded feature maps
    X_skip = X_skip[::-1]
    # upsampling begins at the deepest available tensor
    X = X_skip[0]
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]
#     print(X.shape)
    depth_decode = len(X_decode)

    # reverse indexing filter numbers
    filter_num_decode = filter_num[:-1][::-1]

    # upsampling with concatenation
    for i in range(depth_decode):
        X = UNET_right_3D(X, [X_decode[i], ], filter_num_decode[i], stack_num=stack_num_up, activation=activation,
                          unpool=unpool, batch_norm=batch_norm, name='{}_up{}'.format(name, i))
#         print(X.shape)

    # if tensors for concatenation is not enough
    # then use upsampling without concatenation
    if depth_decode < depth_ - 1:
        for i in range(depth_ - depth_decode - 1):
            i_real = i + depth_decode
            X = UNET_right_3D(X, None, filter_num_decode[i_real], stack_num=stack_num_up, activation=activation,
                              unpool=unpool, batch_norm=batch_norm, concat=False, name='{}_up{}'.format(name, i_real))
#             print(X.shape)

    return X


def transunet_3d(input_size, filter_num, n_labels, patch_size=3, patch_stride=2, stack_num_down=2, stack_num_up=2,
                 embed_dim=768, num_mlp=3072, num_heads=12, num_transformer=12,
                 activation='ReLU', mlp_activation='GELU', output_activation='Softmax', batch_norm=False, pool=True,
                 unpool=True,
                 backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='transunet'):
    '''
    TransUNET with an optional ImageNet-trained bakcbone.


    ----------
    Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., Lu, L., Yuille, A.L. and Zhou, Y., 2021.
    Transunet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306.

    Input
    ----------
        input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
        filter_num: a list that defines the number of filters for each \
                    down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                    The depth is expected as `len(filter_num)`.
        n_labels: number of output labels.
        stack_num_down: number of convolutional layers per downsampling level/block.
        stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        output_activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interface or 'Sigmoid'.
                           Default option is 'Softmax'.
                           if None is received, then linear activation is applied.
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.
        name: prefix of the created keras model and its layers.

        ---------- (keywords of ViT) ----------
        embed_dim: number of embedded dimensions.
        num_mlp: number of MLP nodes.
        num_heads: number of attention heads.
        num_transformer: number of stacked ViTs.
        mlp_activation: activation of MLP nodes.

        ---------- (keywords of backbone options) ----------
        backbone_name: the bakcbone model name. Should be one of the `tensorflow.keras.applications` class.
                       None (default) means no backbone.
                       Currently supported backbones are:
                       (1) VGG16, VGG19
                       (2) ResNet50, ResNet101, ResNet152
                       (3) ResNet50V2, ResNet101V2, ResNet152V2
                       (4) DenseNet121, DenseNet169, DenseNet201
                       (5) EfficientNetB[0-7]
        weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet),
                 or the path to the weights file to be loaded.
        freeze_backbone: True for a frozen backbone.
        freeze_batch_norm: False for not freezing batch normalization layers.

    Output
    ----------
        model: a keras model.

    '''

    activation_func = eval(activation)

    IN = Input(input_size)

    # base
    X = transunet_3d_base(IN, filter_num, patch_size=patch_size, patch_stride=patch_stride,
                          stack_num_down=stack_num_down, stack_num_up=stack_num_up,
                          embed_dim=embed_dim, num_mlp=num_mlp, num_heads=num_heads, num_transformer=num_transformer,
                          activation=activation, mlp_activation=mlp_activation, batch_norm=batch_norm, pool=pool,
                          unpool=unpool,
                          backbone=backbone, weights=weights, freeze_backbone=freeze_backbone,
                          freeze_batch_norm=freeze_batch_norm, name=name)

    # output layer
    OUT = CONV_output_3D(X, n_labels, kernel_size=1, activation=output_activation, name='{}_output'.format(name))

    # functional API model
    model = Model(inputs=[IN, ], outputs=[OUT, ], name='{}_model'.format(name))

    return model


if __name__ == "__main__":
    model = transunet_3d((128, 128, 32, 1), filter_num=[20, 32, 64, 96], patch_size=3, patch_stride=2, n_labels=2,
                         stack_num_down=2,
                         stack_num_up=2,
                         embed_dim=768, num_mlp=3072, num_heads=12, num_transformer=12,
                         activation='ReLU', mlp_activation='GELU', output_activation='Softmax',
                         batch_norm=True, pool=True, unpool='bilinear', name='transunet')

    model.summary(line_length=150)
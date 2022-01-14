
from __future__ import absolute_import

# from keras_unet_collection.layer_utils import *
from keras_unet_collection.activations import GELU, Snake
# from _backbone_zoo import backbone_zoo, bach_norm_checker

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow import expand_dims
# from tensorflow.compat.v1 import image
# from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, UpSampling2D, Conv2DTranspose, GlobalAveragePooling2D
# from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Lambda
from tensorflow.keras.layers import BatchNormalization, Activation, concatenate, multiply, add
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, ELU, Softmax


from tensorflow.keras.layers import MaxPooling3D, AveragePooling3D, UpSampling3D, Conv3DTranspose, GlobalAveragePooling3D
from tensorflow.keras.layers import Conv3D, Lambda


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
    'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation',
    'top_activation'),
    'EfficientNetB1': (
    'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation',
    'top_activation'),
    'EfficientNetB2': (
    'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation',
    'top_activation'),
    'EfficientNetB3': (
    'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation',
    'top_activation'),
    'EfficientNetB4': (
    'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation',
    'top_activation'),
    'EfficientNetB5': (
    'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation',
    'top_activation'),
    'EfficientNetB6': (
    'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation',
    'top_activation'),
    'EfficientNetB7': (
    'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation',
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
        X = UpSampling3D(size=(pool_size, pool_size, pool_size),  name='{}_unpool'.format(name))(X)
    else:
        if kernel_size == 'auto':
            kernel_size = pool_size

        X = Conv3DTranspose(channel, kernel_size, strides=(pool_size, pool_size,pool_size),
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
        X = concatenate([X,]+X_list, axis=4, name=name+'_concat')
    
    # Stacked convolutions after concatenation 
    X = CONV_stack_3D(X, channel, kernel_size, stack_num=stack_num, activation=activation,
                   batch_norm=batch_norm, name=name+'_conv_after_concat')
    
    return X

def unet_3d_base(input_tensor, filter_num, stack_num_down=2, stack_num_up=2,
                 activation='ReLU', batch_norm=False, pool=True, unpool=True, 
                 backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet'):
    
    '''
    The base of U-net with an optional ImageNet-trained backbone.
    
    unet_2d_base(input_tensor, filter_num, stack_num_down=2, stack_num_up=2, 
                 activation='ReLU', batch_norm=False, pool=True, unpool=True, 
                 backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet')
    
    ----------
    Ronneberger, O., Fischer, P. and Brox, T., 2015, October. U-net: Convolutional networks for biomedical image segmentation. 
    In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
    
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
        unpool: True or 'bilinear' for Upsampling3D with bilinear interpolation.
                'nearest' for Upsamplin3D with nearest interpolation.
                False for Conv3DTranspose + batch norm + activation.
        name: prefix of the created keras model and its layers.
        
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
                          batch_norm=batch_norm, name='{}_down{}'.format(name, i+1))        
            X_skip.append(X)

    # backbone cases
    else:
        # handling VGG16 and VGG19 separately
        if 'VGG' in backbone:
            backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_skip = backbone_([input_tensor,])
            depth_encode = len(X_skip)
            
        # for other backbones
        else:
            backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_-1, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_skip = backbone_([input_tensor,])
            depth_encode = len(X_skip) + 1


        # extra conv3d blocks are applied
        # if downsampling levels of a backbone < user-specified downsampling levels
        if depth_encode < depth_:

            # begins at the deepest available tensor  
            X = X_skip[-1]

            # extra downsamplings
            for i in range(depth_-depth_encode):
                i_real = i + depth_encode

                X = UNET_left_3D(X, filter_num[i_real], stack_num=stack_num_down, activation=activation, pool=pool,
                              batch_norm=batch_norm, name='{}_down{}'.format(name, i_real+1))
                X_skip.append(X)

    # reverse indexing encoded feature maps
    X_skip = X_skip[::-1]
    # upsampling begins at the deepest available tensor
    X = X_skip[0]
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]
    depth_decode = len(X_decode)

    # reverse indexing filter numbers
    filter_num_decode = filter_num[:-1][::-1]

    # upsampling with concatenation
    for i in range(depth_decode):
        X = UNET_right_3D(X, [X_decode[i],], filter_num_decode[i], stack_num=stack_num_up, activation=activation,
                       unpool=unpool, batch_norm=batch_norm, name='{}_up{}'.format(name, i))

    # if tensors for concatenation is not enough
    # then use upsampling without concatenation 
    if depth_decode < depth_-1:
        for i in range(depth_-depth_decode-1):
            i_real = i + depth_decode
            X = UNET_right_3D(X, None, filter_num_decode[i_real], stack_num=stack_num_up, activation=activation,
                       unpool=unpool, batch_norm=batch_norm, concat=False, name='{}_up{}'.format(name, i_real))   
    return X

def unet_3d(input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2,
            activation='ReLU', output_activation='Softmax', batch_norm=False, pool=True, unpool=True, 
            backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet'):
    '''
    U-net with an optional ImageNet-trained bakcbone.
    
    unet_2d(input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2,
            activation='ReLU', output_activation='Softmax', batch_norm=False, pool=True, unpool=True, 
            backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet')
    
    ----------
    Ronneberger, O., Fischer, P. and Brox, T., 2015, October. U-net: Convolutional networks for biomedical image segmentation. 
    In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
    
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
        pool: True or 'max' for MaxPooling3D.
              'ave' for AveragePooling3D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling3D with bilinear interpolation.
                'nearest' for Upsampling3D with nearest interpolation.
                False for Conv3DTranspose + batch norm + activation.
        name: prefix of the created keras model and its layers.
        
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
    
    if backbone is not None:
        bach_norm_checker(backbone, batch_norm)
        
    IN = Input(input_size)
    
    # base    
    X = unet_3d_base(IN, filter_num, stack_num_down=stack_num_down, stack_num_up=stack_num_up,
                     activation=activation, batch_norm=batch_norm, pool=pool, unpool=unpool, 
                     backbone=backbone, weights=weights, freeze_backbone=freeze_backbone, 
                     freeze_batch_norm=freeze_backbone, name=name)
    
    # output layer
    OUT = CONV_output_3D(X, n_labels, kernel_size=1, activation=output_activation, name='{}_output'.format(name))
    
    # functional API model
    model = Model(inputs=[IN,], outputs=[OUT,], name='{}_model'.format(name))
    
    return model

if __name__ =="__main__":

    model = unet_3d((128, 128, 32, 1), [20, 30, 48, 64, 72, 96], n_labels=2,
                           stack_num_down=2, stack_num_up=1,
                           activation='GELU', output_activation='Softmax',
                           batch_norm=True, pool='max', unpool='nearest', name='unet')

    model.summary(line_length=150)

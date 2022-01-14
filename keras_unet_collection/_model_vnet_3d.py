

from __future__ import absolute_import
import sys
sys.path.append('D:/code/VIT/keras-unet-collection-main/keras-unet-collection-main/examples/')

# from keras_unet_collection.layer_utils import *
from keras_unet_collection.activations import GELU, Snake

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from tensorflow.keras.layers import BatchNormalization, Activation, concatenate, multiply, add
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, ELU, Softmax


from tensorflow.keras.layers import MaxPooling3D, AveragePooling3D, UpSampling3D, Conv3DTranspose, GlobalAveragePooling3D
from tensorflow.keras.layers import Conv3D, Lambda



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


def Res_CONV_stack_3D(X, X_skip, channel, res_num, activation='ReLU', batch_norm=False, name='res_conv'):
    '''
    Stacked convolutional layers with residual path.

    Res_CONV_stack(X, X_skip, channel, res_num, activation='ReLU', batch_norm=False, name='res_conv')

    Input
    ----------
        X: input tensor.
        X_skip: the tensor that does go into the residual path
                can be a copy of X (e.g., the identity block of ResNet).
        channel: number of convolution filters.
        res_num: number of convolutional layers within the residual path.
        activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.

    Output
    ----------
        X: output tensor.

    '''
    X = CONV_stack_3D(X, channel, kernel_size=3, stack_num=res_num, dilation_rate=1,
                   activation=activation, batch_norm=batch_norm, name=name)

    X = add([X_skip, X], name='{}_add'.format(name))

    activation_func = eval(activation)
    X = activation_func(name='{}_add_activation'.format(name))(X)

    return X

def vnet_left_3D(X, channel, res_num, activation='ReLU', pool=True, batch_norm=False, name='left'):
    '''
    The encoder block of 2-d V-net.
    
    vnet_left(X, channel, res_num, activation='ReLU', pool=True, batch_norm=False, name='left')
    
    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        res_num: number of convolutional layers within the residual path.
        activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        batch_norm: True for batch normalization, False otherwise.
        name: name of the created keras layers.
        
    Output
    ----------
        X: output tensor.
        
    '''
    
    pool_size = 2

    X = encode_layer_3D(X, channel, pool_size, pool, activation=activation,
                     batch_norm=batch_norm, name='{}_encode'.format(name))
    
    if pool is not False:
        X = CONV_stack_3D(X, channel, kernel_size=3, stack_num=1, dilation_rate=1,
                       activation=activation, batch_norm=batch_norm, name='{}_pre_conv'.format(name))

    X = Res_CONV_stack_3D(X, X, channel, res_num=res_num, activation=activation,
                       batch_norm=batch_norm, name='{}_res_conv'.format(name))
    return X

def vnet_right_3D(X, X_list, channel, res_num, activation='ReLU', unpool=True, batch_norm=False, name='right'):
    '''
    The decoder block of 2-d V-net.
    
    vnet_right(X, X_list, channel, res_num, activation='ReLU', unpool=True, batch_norm=False, name='right')
    
    Input
    ----------
        X: input tensor.
        X_list: a list of other tensors that connected to the input tensor.
        channel: number of convolution filters.
        stack_num: number of convolutional layers.
        res_num: number of convolutional layers within the residual path.
        activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.
        batch_norm: True for batch normalization, False otherwise.
        name: name of the created keras layers.
        
    Output
    ----------
        X: output tensor.
    
    '''
    pool_size = 2
    
    X = decode_layer_3D(X, channel, pool_size, unpool,
                     activation=activation, batch_norm=batch_norm, name='{}_decode'.format(name))
    
    X_skip = X
    
    X = concatenate([X,]+X_list, axis=-1, name='{}_concat'.format(name))
    
    X = Res_CONV_stack_3D(X, X_skip, channel, res_num, activation=activation,
                       batch_norm=batch_norm, name='{}_res_conv'.format(name))
    
    return X

def vnet_3d_base(input_tensor, filter_num, res_num_ini=1, res_num_max=3,
                 activation='ReLU', batch_norm=False, pool=True, unpool=True, name='vnet'):
    '''
    The base of 2-d V-net.
    
    vnet_2d_base(input_tensor, filter_num, res_num_ini=1, res_num_max=3, 
                 activation='ReLU', batch_norm=False, pool=True, unpool=True, name='vnet')
    
    Milletari, F., Navab, N. and Ahmadi, S.A., 2016, October. V-net: Fully convolutional neural 
    networks for volumetric medical image segmentation. In 2016 fourth international conference 
    on 3D vision (3DV) (pp. 565-571). IEEE.
    
    The Two-dimensional version is inspired by:
    https://github.com/FENGShuanglang/2D-Vnet-Keras
    
    Input
    ----------
        input_tensor: the input tensor of the base, e.g., `keras.layers.Inpyt((None, None, 3))`.
        filter_num: a list that defines the number of filters for each \
                    down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                    The depth is expected as `len(filter_num)`.
        res_num_ini: number of convolutional layers of the first first residual block (before downsampling).
        res_num_max: the max number of convolutional layers within a residual block.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.                 
        name: prefix of the created keras layers.
            
    Output
    ----------
        X: output tensor.
    
    * This is a modified version of V-net for 2-d inputw.
    * The original work supports `pool=False` only. 
      If pool is True, 'max', or 'ave', an additional conv2d layer will be applied. 
    * All the 5-by-5 convolutional kernels are changed (and fixed) to 3-by-3.
    
    '''

    depth_ = len(filter_num)

    # determine the number of res conv layers in each down- and upsampling level
    res_num_list = []
    for i in range(depth_):
        temp_num = res_num_ini + i
        if temp_num > res_num_max:
            temp_num = res_num_max
        res_num_list.append(temp_num)

    X_skip = []

    X = input_tensor
    # ini conv layer
    X = CONV_stack_3D(X, filter_num[0], kernel_size=3, stack_num=1, dilation_rate=1,
                   activation=activation, batch_norm=batch_norm, name='{}_input_conv'.format(name))

    X = Res_CONV_stack_3D(X, X, filter_num[0], res_num=res_num_list[0], activation=activation,
                 batch_norm=batch_norm, name='{}_down_0'.format(name))
    X_skip.append(X)

    # downsampling levels
    for i, f in enumerate(filter_num[1:]):
        X = vnet_left_3D(X, f, res_num=res_num_list[i+1], activation=activation, pool=pool,
                      batch_norm=batch_norm, name='{}_down_{}'.format(name, i+1))

        X_skip.append(X)

    X_skip = X_skip[:-1][::-1]
    filter_num = filter_num[:-1][::-1]
    res_num_list = res_num_list[:-1][::-1]

    # upsampling levels
    for i, f in enumerate(filter_num):
        X = vnet_right_3D(X, [X_skip[i],], f, res_num=res_num_list[i],
                       activation=activation, unpool=unpool, batch_norm=batch_norm, name='{}_up_{}'.format(name, i))

    return X


def vnet_3d(input_size, filter_num, n_labels,
            res_num_ini=1, res_num_max=3, 
            activation='ReLU', output_activation='Softmax', 
            batch_norm=False, pool=True, unpool=True, name='vnet'):
    '''
    vnet 2d
    
    vnet_2d(input_size, filter_num, n_labels,
            res_num_ini=1, res_num_max=3, 
            activation='ReLU', output_activation='Softmax', 
            batch_norm=False, pool=True, unpool=True, name='vnet')
    
    Milletari, F., Navab, N. and Ahmadi, S.A., 2016, October. V-net: Fully convolutional neural 
    networks for volumetric medical image segmentation. In 2016 fourth international conference 
    on 3D vision (3DV) (pp. 565-571). IEEE.
    
    The Two-dimensional version is inspired by:
    https://github.com/FENGShuanglang/2D-Vnet-Keras
    
    Input
    ----------
        input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
        filter_num: a list that defines the number of filters for each \
                    down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                    The depth is expected as `len(filter_num)`.
        n_labels: number of output labels.
        res_num_ini: number of convolutional layers of the first first residual block (before downsampling).
        res_num_max: the max number of convolutional layers within a residual block.
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
        name: prefix of the created keras layers.
            
    Output
    ----------
        model: a keras model. 
    
    * This is a modified version of V-net for 2-d inputw.
    * The original work supports `pool=False` only. 
      If pool is True, 'max', or 'ave', an additional conv2d layer will be applied. 
    * All the 5-by-5 convolutional kernels are changed (and fixed) to 3-by-3.
    '''
    
    IN = Input(input_size)
    X = IN
    # base
    X = vnet_3d_base(X, filter_num, res_num_ini=res_num_ini, res_num_max= res_num_max,
                     activation=activation, batch_norm=batch_norm, pool=pool, unpool=unpool, name=name)
    # output layer
    OUT = CONV_output_3D(X, n_labels, kernel_size=1, activation=output_activation, name='{}_output'.format(name))
    
    # functional API model
    model = Model(inputs=[IN,], outputs=[OUT,], name='{}_model'.format(name))
    
    return model

if __name__ =="__main__":

    # model = vnet_2d((128, 128, 1), [20, 30, 48, 64, 72, 96], n_labels=2,
    #                 res_num_ini=1, res_num_max=3,
    #                 activation='ReLU', output_activation='Softmax',
    #                 batch_norm=False, pool=True, unpool=True, name='vnet')

    model = vnet_3d((128, 128,32, 1), filter_num=[4, 8, 12, 24], n_labels=1,
                           res_num_ini=1, res_num_max=3,
                           activation='PReLU', output_activation='Sigmoid',
                           batch_norm=True, pool=False, unpool=False, name='vnet_3d')


    model.summary(line_length=150)
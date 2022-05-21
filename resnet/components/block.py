import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LayerNormalization, ReLU, Add, DepthwiseConv2D

def downsampleblock(input, filter_num, stage_idx=-1, block_idx=-1):
    '''A complete `Downsample` of ResNet

    Args:
      filter_num: the number of filters in the convolution
      stage_idx: index of current stage
      block_idx: index of current block in stage
    '''

    # Downsample
    down = input
    if block_idx > 0:
        down = Conv2D(filters=filter_num,
                    kernel_size=2,
                    strides=2,
                    padding='same',
                    kernel_initializer='he_normal',
                    name='conv{}_block{}_downsample_conv'.format(stage_idx, block_idx))(input)

    return down

def microblock(input, filter_num, stage_idx=-1, block_idx=-1):
    '''Large Kernel use stack of 2 layers: Depthwise_Layer and Pointwise_Layer

    Args:
      filter_num: the number of filters in the convolution
      stage_idx: index of current stage
      block_idx: index of current block in stage
    '''

    # Depthwise_Layer
    depthwise = DepthwiseConv2D(
        kernel_size=7, strides=1, padding='same')(input)

    nn1 = LayerNormalization(name='conv{}_block{}_1_nn'.format(
        stage_idx, block_idx))(depthwise)

    # Pointwise_Layer
    conv1 = Conv2D(filters=4*filter_num,
                   kernel_size=1,
                   strides=1,
                   padding='same',
                   kernel_initializer='he_normal',
                   name='conv{}_block{}_1_conv'.format(stage_idx, block_idx))(nn1)
    gelu = tf.nn.gelu(conv1)

    # Pointwise_Layer
    conv2 = Conv2D(filters=filter_num,
                   kernel_size=1,
                   strides=1,
                   padding='same',
                   kernel_initializer='he_normal',
                   name='conv{}_block{}_2_conv'.format(stage_idx, block_idx))(gelu)

    return conv2

def resblock(input, filter_num, stage_idx=-1, block_idx=-1):
    '''A complete `Residual Unit` of ResNet

    Args:
      filter_num: the number of filters in the convolution
      stage_idx: index of current stage
      block_idx: index of current block in stage
    '''

    residual = microblock(input, filter_num, stage_idx, block_idx)

    output = Add(name='conv{}_block{}_add'.format(
        stage_idx, block_idx))([input, residual])

    return output

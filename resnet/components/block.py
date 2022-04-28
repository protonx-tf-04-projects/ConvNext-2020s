import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LayerNormalization, ReLU, Add, DepthwiseConv2D
from tensorflow.keras import Model

def micro_block(input, filter_num, stride=1, stage_idx=-1, block_idx=-1):
    '''Large Kernel use stack of 2 layers: Depthwise_Layer and Pointwise_Layer

    Args:
      filter_num: the number of filters in the convolution
      stride: the number of strides in the convolution. stride = 1 if you want
              output shape is same as input shape
      stage_idx: index of current stage
      block_idx: index of current block in stage
    '''
    # Depthwise_Layer
    depthwise = DepthwiseConv2D(
        kernel_size=7, strides=stride, padding='same')(input)
    nn1 = LayerNormalization(name='conv{}_block{}_1_nn'.format(
        stage_idx, block_idx))(depthwise)

    # Pointwise_Layer
    conv1 = Conv2D(filters=filter_num,
                   kernel_size=1,
                   strides=1,
                   padding='same',
                   kernel_initializer='he_normal',
                   name='conv{}_block{}_1_conv'.format(stage_idx, block_idx))(nn1)
    gelu = tf.nn.gelu(conv1)

    # Pointwise_Layer
    conv2 = Conv2D(filters=4*filter_num,
                   kernel_size=1,
                   strides=1,
                   padding='same',
                   kernel_initializer='he_normal',
                   name='conv{}_block{}_2_conv'.format(stage_idx, block_idx))(gelu)

    return conv2

def resblock(input, filter_num, stride=1, stage_idx=-1, block_idx=-1):
    '''A complete `Residual Unit` of ResNet

    Args:
      filter_num: the number of filters in the convolution
      stride: the number of strides in the convolution. stride = 1 if you want
              output shape is same as input shape
      stage_idx: index of current stage
      block_idx: index of current block in stage
    '''
    
    residual = micro_block(input, filter_num, stride, stage_idx, block_idx)
    expansion = 4

    shortcut = input
    # use projection short cut when dimensions increase
    if stride > 1 or input.shape[3] != residual.shape[3]:
        shortcut = Conv2D(expansion*filter_num,
                          kernel_size=1,
                          strides=stride,
                          padding='valid',
                          kernel_initializer='he_normal',
                          name='conv{}_block{}_projection-shortcut_conv'.format(stage_idx, block_idx))(input)

        shortcut = BatchNormalization(
            name='conv{}_block{}_projection-shortcut_bn'.format(stage_idx, block_idx))(shortcut)

    output = Add(name='conv{}_block{}_add'.format(
        stage_idx, block_idx))([residual, shortcut])

    return ReLU(name='conv{}_block{}_relu'.format(stage_idx, block_idx))(output)

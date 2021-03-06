from .components import stage, downsample
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, LayerNormalization
from tensorflow.keras import Model


def build_convnext(input_shape, num_classes, layers, model_name='base'):
    '''A complete `stage` of ConvNeXt
    '''
    input = Input(input_shape, name='input')

    filter_cnn = 96
    filters = [96, 192, 384, 768]

    if model_name == 'base':
        filter_cnn = 128
        filters = [128, 256, 512, 1024]
    elif model_name == 'large':
        filter_cnn = 192
        filters = [192, 384, 768, 1536]
    elif model_name == 'xlarge':
        filter_cnn = 256
        filters = [256, 512, 1024, 2048]
        
    # conv1
    # change replace the ResNet-style stem cell
    # with a patchify layer implemented using a 4×4, stride 4 convolutional layer
    net = Conv2D(filters=filter_cnn,
                 kernel_size=4,
                 strides=4,
                 padding='same',
                 kernel_initializer='he_normal',
                 name='conv1_conv')(input)           

    # conv2_x, conv3_x, conv4_x, conv5_x
    for i in range(len(filters)):
        net = downsample(input=net,
                    filter_num=filters[i],
                    block_idx=i,
                    stage_idx=i+2)
        net = stage(input=net,
                    filter_num=filters[i],
                    num_block=layers[i],
                    stage_idx=i+2)

    net = GlobalAveragePooling2D(name='avg_pool')(net)
    net = LayerNormalization(name='norm')(net)
    output = Dense(num_classes, activation='softmax', name='predictions')(net)
    model = Model(input, output)

    return model
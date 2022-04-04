from .components import stage
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D,ReLU, Dense
from tensorflow.keras import Model

def build_patchify(input_shape,num_classes,layers,use_bottleneck=False):
  '''A complete `stage` of ResNet
  '''
  input=Input(input_shape,name='input')
  # conv1
  # change replace the ResNet-style stem cell
  # with a patchify layer implemented using a 4×4, stride 4 convolutional layer
  net=Conv2D(filters=64,
             kernel_size=4, 
             strides=4,
             padding='same',
             kernel_initializer='he_normal',
             name='conv1_conv')(input)
  net=BatchNormalization(name='conv1_bn')(net)
  net=ReLU(name='conv1_relu')(net)
  net=MaxPooling2D(pool_size=3,
                   strides=2,
                   padding='same',
                   name='conv1_max_pool')(net)

  # conv2_x, conv3_x, conv4_x, conv5_x
  filters=[64,128,256,512]
  for i in range(len(filters)):
    net=stage(input = net,
              filter_num = filters[i],
              num_block = layers[i],
              use_downsample = i!=0,
              use_bottleneck = use_bottleneck,
              stage_idx = i+2)

  net=GlobalAveragePooling2D(name='avg_pool')(net)
  output=Dense(num_classes,activation='softmax',name='predictions')(net)
  model=Model(input,output)

  return model

def build(input_shape,num_classes,layers,use_bottleneck=False):
  '''A complete `stage` of ResNet
  '''
  input=Input(input_shape,name='input')
  # conv1
  net=Conv2D(filters=64,
             kernel_size=7, 
             strides=2,
             padding='same',
             kernel_initializer='he_normal',
             name='conv1_conv')(input)
  net=BatchNormalization(name='conv1_bn')(net)
  net=ReLU(name='conv1_relu')(net)
  net=MaxPooling2D(pool_size=3,
                   strides=2,
                   padding='same',
                   name='conv1_max_pool')(net)

  # conv2_x, conv3_x, conv4_x, conv5_x
  filters=[64,128,256,512]
  for i in range(len(filters)):
    net=stage(input = net,
              filter_num = filters[i],
              num_block = layers[i],
              use_downsample = i!=0,
              use_bottleneck = use_bottleneck,
              stage_idx = i+2)

  net=GlobalAveragePooling2D(name='avg_pool')(net)
  output=Dense(num_classes,activation='softmax',name='predictions')(net)
  model=Model(input,output)

  return model

def Resnet18(input_shape=(224,224,3), num_classes=1000):
  return build(input_shape, num_classes, [2,2,2,2],use_bottleneck=False)

def Resnet34(input_shape=(224,224,3), num_classes=1000):
  return build(input_shape, num_classes,[3,4,6,3],use_bottleneck=False)

def Resnet50(input_shape=(224,224,3), num_classes=1000):
  return build(input_shape, num_classes,[3,4,6,3],use_bottleneck=True)

def Resnet101(input_shape=(224,224,3), num_classes=1000):
  return build(input_shape, num_classes, [3,4,23,3],use_bottleneck=True)

def Resnet152(input_shape=(224,224,3), num_classes=1000):
  return build(input_shape, num_classes, [3,8,36,3],use_bottleneck=True)

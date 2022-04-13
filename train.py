from convnext.model import ConvNeXt, ConvNeXtTiny, ConvNeXtSmall, PlayThroughResNeXt, PlayThroughMacro, PlayThroughInvertedBottleneck, PlayThroughLargeKernel
from resnet.model import Resnet50
from tensorflow import keras
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.data import Dataset
from tensorflow.keras.optimizers import Adam
import numpy as np

import os
import tensorflow as tf
from argparse import ArgumentParser
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Arguments users used when running command lines
    parser.add_argument('--model', default='macro', type=str,
                        help='Type of ConvNeXt model, valid option: resnet50, macro, resnext')
    parser.add_argument('--lr', default=0.001,
                        type=float, help='Learning rate')
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument('--num-classes', default=10,
                        type=int, help='Number of classes')
    parser.add_argument('--image-size', default=224,
                        type=int, help='Size of input image')
    parser.add_argument('--image-channels', default=3,
                        type=int, help='Number channel of input image')
    parser.add_argument('--train-folder', default='', type=str,
                        help='Where training data is located')
    parser.add_argument('--valid-folder', default='', type=str,
                        help='Where validation data is located')
    parser.add_argument('--class-mode', default='categorical', type=str, help='Class mode to compile')
    parser.add_argument('--model-folder', default='output/',
                        type=str, help='Folder to save trained model')  

    args = parser.parse_args()

    # Project Description

    print('---------------------Welcome to ConvNext-2020s paper Implementation-------------------')
    print('Github: https://github.com/protonx-tf-04-projects')
    print('Email: thinguyenkhtn@gmail.com, nguyenthanhlinh58@gmail.com')
    print('---------------------------------------------------------------------')
    print('Training ConvNext2020s model with hyper-params:') 
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')

    # Assign arguments to variables to avoid repetition 
    train_folder = args.train_folder
    valid_folder = args.valid_folder
    batch_size =  args.batch_size
    image_size = args.image_size
    image_channels = args.image_channels
    num_classes = args.num_classes
    epoch = args.epochs
    class_mode = args.class_mode
    lr = args.lr

    """
    # Data Augmentation is used to expand the training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2)

    val_datagen = ImageDataGenerator(rescale=1./255)
    """

    # Data loader
    if args.train_folder != '' and args.valid_folder != '':
        """
        # Load train images from folder
        train_ds =  train_datagen.flow_from_directory(
            train_folder,
            seed=43,
            target_size=(image_size, image_size),
            shuffle=True,
            batch_size=batch_size,
            class_mode=class_mode
        )
        # Load validation images from folder
        val_ds = val_datagen.flow_from_directory(
            valid_folder,
            seed=43,
            target_size=(image_size, image_size),
            shuffle=True,
            batch_size=batch_size,
            class_mode=class_mode
        )
        """
        # Load train images from folder
        train_ds = image_dataset_from_directory(
            args.train_folder,
            seed=123,
            image_size=(image_size, image_size),
            shuffle=True,
            batch_size=batch_size,
        )
        val_ds = image_dataset_from_directory(
            args.valid_folder,
            seed=123,
            image_size=(image_size, image_size),
            shuffle=True,
            batch_size=args.batch_size,
        )
    else:
        # If you do not have your own data, you can use the CIFAR-10 dataset
        # The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
        # There are 50000 training images and 10000 test images.
        print("Data folder is not set. Use CIFAR 10 dataset")

        num_classes = 10

        # We should pass the image size (--image-size) argument to running command is 32x32x3 while using Cifar10
        # For example: python .\train.py  --num-classes 2 --batch-size 10 --image-size 32  --epochs 200 
        (x_train, y_train), (x_val, y_val) = keras.datasets.cifar10.load_data()
        
        # Modify the image size if you do not want to pass the default value (32)
        x_train = (x_train.reshape(-1, image_size, image_size,
                                   image_channels)).astype(np.float32)
        x_val = (x_val.reshape(-1, image_size, image_size,
                               image_channels)).astype(np.float32)
   
        # create dataset
        train_ds = Dataset.from_tensor_slices((x_train, y_train))
        train_ds = train_ds.batch(batch_size)

        val_ds = Dataset.from_tensor_slices((x_val, y_val))
        val_ds = val_ds.batch(batch_size)

    # ConvNeXt
    if args.model == 'tiny':
        model = ConvNeXtTiny()
    elif args.model == 'small':
        model = ConvNeXtSmall()

    # PlayThrough model --->>>
    elif args.model == 'resnet50':
        model = Resnet50(input_shape=(image_size,
                             image_size, image_channels),
                             num_classes = num_classes)
    elif args.model == 'macro':
        model = PlayThroughMacro()
    elif args.model == 'resnext':
        model =PlayThroughResNeXt();
    elif args.model == 'invertedbottleneck':
        model = PlayThroughInvertedBottleneck();
    elif args.model == 'largekernel':
        model = PlayThroughLargeKernel();
    else:
        model = ConvNeXt(
            num_classes = num_classes,
            image_size = image_size
        )
    # PlayThrough model --->>>

    model.build(input_shape=(None, image_size,
                             image_size, image_channels))

    optimizer = Adam(learning_rate=args.lr)

    loss = SparseCategoricalCrossentropy()
    model.compile(optimizer, loss=loss,
                  metrics=['accuracy'])

    # Traning
    model.fit(train_ds,
              epochs=args.epochs,
              batch_size=batch_size,
              validation_data=val_ds)

    # Save model
    model.save(args.model_folder)

import os
import tensorflow as tf
from argparse import ArgumentParser
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.data import Dataset
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Arguments users used when running command lines
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
                        
    args = parser.parse_args()

    # Project Description

    print('---------------------Welcome to ${name}-------------------')
    print('Github: ${accout}')
    print('Email: ${email}')
    print('---------------------------------------------------------------------')
    print('Training ConvNext2020s model with hyper-params:') 
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')

    # Data loader
    if args.train_folder != '' and args.valid_folder != '':
        # Load train images from folder
        train_ds = image_dataset_from_directory(
            args.train_folder,
            seed=43,
            image_size=(args.image_size, args.image_size),
            shuffle=True,
            batch_size=args.batch_size,
        )
        # Load validation images from folder
        val_ds = image_dataset_from_directory(
            args.valid_folder,
            seed=43,
            image_size=(args.image_size, args.image_size),
            shuffle=True,
            batch_size=args.batch_size,
        )
    else:
        # The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
        # There are 50000 training images and 10000 test images.
        print("Data folder is not set. Use CIFAR 10 dataset")

        args.image_channels = 3
        args.num_classes = 10

        # We need to pass the image size (--image-size) argument to running command is 32x32x3 while using Cifar10
        # For example: python .\train.py  --num-classes 2 --batch-size 10 --image-size 32  --epochs 200 
        (x_train, y_train), (x_val, y_val) = keras.datasets.cifar10.load_data()
        x_train = (x_train.reshape(-1, args.image_size, args.image_size,
                                   args.image_channels)).astype(np.float32)
        x_val = (x_val.reshape(-1, args.image_size, args.image_size,
                               args.image_channels)).astype(np.float32)

        # create dataset
        train_ds = Dataset.from_tensor_slices((x_train, y_train))
        train_ds = train_ds.batch(args.batch_size)

        val_ds = Dataset.from_tensor_slices((x_val, y_val))
        val_ds = val_ds.batch(args.batch_size)
    



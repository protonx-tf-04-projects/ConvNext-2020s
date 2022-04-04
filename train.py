from convnext.model import ConvNeXt, ConvNeXtMacro
from resnet.model import Resnet50
from tensorflow import keras
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.data import Dataset
from tensorflow.keras.optimizers import Adam
import numpy as np

import os
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Arguments users used when running command lines
    parser.add_argument('--model', default='base', type=str,
                        help='Type of ConvNeXt model, valid option: base')
    parser.add_argument('--num-classes', default=10,
                        type=int, help='Number of classes')
    parser.add_argument('--lr', default=0.001,
                        type=float, help='Learning rate')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size')
    parser.add_argument("--epochs", default=1000, 
                        type=int)
    parser.add_argument('--image-size', default=224,
                        type=int, help='Size of input image')
    parser.add_argument('--image-channels', default=3,
                        type=int, help='Number channel of input image')
    parser.add_argument('--train-folder', default='', type=str,
                        help='Where training data is located')
    parser.add_argument('--valid-folder', default='', type=str,
                        help='Where validation data is located')
    parser.add_argument('--model-folder', default='output/',
                        type=str, help='Folder to save trained model')

    home_dir = os.getcwd()
    args = parser.parse_args()

    # Project Description
    args = parser.parse_args()

    print('---------------------Welcome to ConvNeXt-------------------')
    print('Github: thinguyenkhtn')
    print('Email: thinguyenkhtn@gmail.com')
    print('---------------------------------------------------------------------')
    print('Training CONVNEXT-2020s model with hyper-params:') 
    print('===========================')
    
    # Do Prediction
    if args.train_folder != '' and args.valid_folder != '':
        # Load train images from folder
        train_ds = image_dataset_from_directory(
            args.train_folder,
            seed=123,
            image_size=(args.image_size, args.image_size),
            shuffle=True,
            batch_size=args.batch_size,
        )
        val_ds = image_dataset_from_directory(
            args.valid_folder,
            seed=123,
            image_size=(args.image_size, args.image_size),
            shuffle=True,
            batch_size=args.batch_size,
        )
    else:
        print("Data folder is not set. Use CIFAR 10 dataset")

        args.image_channels = 3
        args.num_classes = 10

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
    if args.model == 'resnet50':
        model = model = Resnet50(input_shape=(args.image_size,
                             args.image_size, args.image_channels),
                             num_classes = args.num_classes)
    elif args.model == 'macro':
        model = ConvNeXtMacro()
    else:
        model = ConvNeXt(
            num_classes = args.num_classes,
            image_size = args.image_size
        )

    model.build(input_shape=(None, args.image_size,
                             args.image_size, args.image_channels))

    optimizer = Adam(learning_rate=args.lr)

    loss = SparseCategoricalCrossentropy()
    model.compile(optimizer, loss=loss,
                  metrics=['accuracy'])

    # Traning
    model.fit(train_ds,
              epochs=args.epochs,
              batch_size=args.batch_size,
              validation_data=val_ds)

    # Save model
    model.save(args.model_folder)



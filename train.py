from convnext.model import ConvNeXt, ConvNeXtBase, ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge, ConvNeXtMXLarge
from tensorflow import keras
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.data import Dataset
from tensorflow.keras.optimizers import Adam
import numpy as np
from argparse import ArgumentParser
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_addons as tfa

if __name__ == "__main__":
    parser = ArgumentParser()

    # Arguments users used when running command lines
    parser.add_argument('--model', default='tiny', type=str,
                        help='Type of ConvNeXt model, valid option: tiny, small, base, large, xlarge')
    parser.add_argument('--lr', default=0.001,
                        type=float, help='Learning rate')
    parser.add_argument('--weight-decay', default=1e-4,
                        type=float, help='Weight decay')
    parser.add_argument("--batch-size", default=32, type=int)
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
    parser.add_argument('--class-mode', default='categorical',
                        type=str, help='Class mode to compile')
    parser.add_argument('--model-folder', default='output/',
                        type=str, help='Folder to save trained model')

    args = parser.parse_args()

    # Project Description

    print('---------------------Welcome to ConvNext-2020s paper Implementation-------------------')
    print('Github: thinguyenkhtn')
    print('Email: thinguyenkhtn@gmail.com')
    print('---------------------------------------------------------------------')
    print('Training ConvNext2020s model with hyper-params:')
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')

    # Assign arguments to variables to avoid repetition
    train_folder = args.train_folder
    valid_folder = args.valid_folder
    batch_size = args.batch_size
    image_size = args.image_size
    image_channels = args.image_channels
    num_classes = args.num_classes
    epoch = args.epochs
    class_mode = args.class_mode
    lr = args.lr
    weight_decay = args.weight_decay

    # Data loader
    if train_folder != '' and valid_folder != '':
        # Load train images from folder
        train_ds = image_dataset_from_directory(
            train_folder,
            seed=123,
            image_size=(image_size, image_size),
            shuffle=True,
            batch_size=batch_size,
        )
        val_ds = image_dataset_from_directory(
            valid_folder,
            seed=123,
            image_size=(image_size, image_size),
            shuffle=True,
            batch_size=batch_size,
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
    elif args.model == 'base':
        model = ConvNeXtBase()
    elif args.model == 'large':
        model = ConvNeXtLarge()
    elif args.model == 'xlarge':
        model = ConvNeXtMXLarge()
    else:
        model = ConvNeXt(
            num_classes=num_classes,
            image_size=image_size
        )

    model.build(input_shape=(None, image_size,
                             image_size, image_channels))

    optimizer = Adam(learning_rate=lr)

    loss = SparseCategoricalCrossentropy()

    model.compile(optimizer, loss=loss,
                  metrics=['accuracy'])

    best_model = ModelCheckpoint(args.model_folder,
                                 save_weights_only=False,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 mode='max',
                                 save_best_only=True)

    # Traning
    model.fit(train_ds,
              epochs=epoch,
              batch_size=batch_size,
              validation_data=val_ds,
              callbacks=[best_model])

    # Save model
    model.save(args.model_folder)

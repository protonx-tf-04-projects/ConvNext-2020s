from convnext.model import ConvNeXt, ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge, ConvNeXtMXLarge
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from argparse import ArgumentParser
from tensorflow.keras.callbacks import  ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator
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
    parser.add_argument('--class-mode', default='sparse',
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
    training_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = training_datagen.flow_from_directory(train_folder, target_size=(image_size, image_size), batch_size= batch_size, class_mode = class_mode )
    val_generator = val_datagen.flow_from_directory(valid_folder, target_size=(image_size, image_size), batch_size= batch_size, class_mode = class_mode)

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

    optimizer = tfa.optimizers.AdamW(
        learning_rate=lr, weight_decay=weight_decay)

    model.compile(optimizer=optimizer, 
                loss=SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
    
    best_model = ModelCheckpoint(args.model_folder,
                                 save_weights_only=False,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 mode='max',
                                 save_best_only=True)
    # Traning
    model.fit(
        train_generator,
        epochs=args.epochs,
        verbose=1,
        validation_data=val_generator,
        callbacks=[best_model])

    # Save model
    model.save(args.model_folder)

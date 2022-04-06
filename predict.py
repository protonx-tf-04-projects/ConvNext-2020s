from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import tensorflow as tf
from argparse import ArgumentParser
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--test-image", default='./test.png', type=str, required=True)
    parser.add_argument(
        "--model-folder", default='output/', type=str)

    args = parser.parse_args()

    # Project Description

    print('---------------------Welcome to ConvNeXt-------------------')
    print('Github: thinguyenkhtn')
    print('Email: thinguyenkhtn@gmail.com')
    print('---------------------------------------------------------------------')
    print('Training ConvNeXt-2020s model with hyper-params:') 
    print('===========================')

    # Loading Model
    model = load_model(args.model_folder)

    # Load test image
    image = preprocessing.image.load_img(args.test_image, target_size=(150, 150))
    input_arr = preprocessing.image.img_to_array(image)
    x = np.array([input_arr])

    predictions = model.predict(x)
    print('Result: {}'.format(np.argmax(predictions), axis=1))


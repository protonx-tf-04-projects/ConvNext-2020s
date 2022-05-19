# A ConvNet for the 2020s

Our implementation of paper: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545), using [tensorflow 2](https://www.tensorflow.org/)

Library Logo <--- **FIXME**

<p align="center">
    <img src='figures/ConvNeXt.png' width=400 class="center">
    <img src='figures/ConvNeXtModel.png' width=400 class="center">
</p>

Description about your project. Why do you choose to build this?  <--- **FIXME**

Slide about your project (if it's available) <--- **FIXME**

Architecture Image <--- **FIXME**


Authors:
- Github: thinguyenkhtn
- Email: thinguyenkhtn@gmail.com

Advisors:
- Github: https://github.com/bangoc123
- Email: protonxai@gmail.com

## I.  Set up environment
- Step 1: Make sure you have installed Miniconda. If not yet, see the setup document [here](https://conda.io/en/latest/user-guide/install/index.html#regular-installation).

- Step 1: Clone this repository: `git clone https://github.com/protonx-tf-04-projects/ConvNext-2020s`

## II.  Set up your dataset

<!-- - Guide user how to download your data and set the data pipeline  -->
1. Download the data:
- Download dataset [here](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip)
2. Extract file and put folder ```train``` and ```validation``` to ```./data``` 
- train folder was used for the training process
- validation folder was used for validating training result after each epoch

This library use ImageDataGenerator API from Tensorflow 2.0 to load images. Make sure you have some understanding of how it works via [its document](https://keras.io/api/preprocessing/image/)
Structure of these folders in ```./data```

```
train/
...cats/
......cat.0.jpg
......cat.1.jpg
...dogs/
......dog.0.jpg
......dog.1.jpg
```

```
validation/
...cats/
......cat.2000.jpg
......cat.2001.jpg
...dogs/
......dog.2000.jpg
......dog.2001.jpg
```

<!-- - References: [NLP](https://github.com/bangoc123/transformer) and [CV](https://github.com/bangoc123/mlp-mixer) -->

## III. Training Process

Training script:

```python

python train.py --epochs ${epochs} --input-lang en --target-lang vi --input-path ${path_to_en_text_file} --target-path ${path_to_vi_text_file}

```

Example:

```python

!python train.py --train-folder ${train_folder} --valid-folder ${valid_folder} --num-classes 2 --image-size 224 --lr 0.0001 --model tiny --epochs 30

``` 

There are some important arguments for the script you should consider when running it:

- `train-folder`: The folder of training data
- `valid-folder`: The folder of validation data
- `model-folder`: Where the model after training saved
- `num-classes`: The number of your problem classes.
- `batch-size`: The batch size of the dataset
- `image-size`: The image size of the dataset
- `lr`: The learning rate
- `model`: Type of ConvNeXt model, valid option: tiny, small, base, large, xlarge

## IV. Predict Process

```bash
python predict.py --test-data ${link_to_test_data}
```

## V. Result and Comparision

**FIXME**

Your implementation
```
Epoch 7/10
782/782 [==============================] - 261s 334ms/step - loss: 0.8315 - acc: 0.8565 - val_loss: 0.8357 - val_acc: 0.7978
Epoch 8/10
782/782 [==============================] - 261s 334ms/step - loss: 0.3182 - acc: 0.8930 - val_loss: 0.6161 - val_acc: 0.8047
Epoch 9/10
782/782 [==============================] - 261s 333ms/step - loss: 1.1965 - acc: 0.8946 - val_loss: 3.9842 - val_acc: 0.7855
Epoch 10/10
782/782 [==============================] - 261s 333ms/step - loss: 0.4717 - acc: 0.8878 - val_loss: 0.4894 - val_acc: 0.8262

```

**FIXME**

Other architecture

```
Epoch 6/10
391/391 [==============================] - 115s 292ms/step - loss: 0.1999 - acc: 0.9277 - val_loss: 0.4719 - val_acc: 0.8130
Epoch 7/10
391/391 [==============================] - 114s 291ms/step - loss: 0.1526 - acc: 0.9494 - val_loss: 0.5224 - val_acc: 0.8318
Epoch 8/10
391/391 [==============================] - 115s 293ms/step - loss: 0.1441 - acc: 0.9513 - val_loss: 0.5811 - val_acc: 0.7875
```

Your comments about these results <--- **FIXME**


## VI. Running Test

When you want to modify the model, you need to run the test to make sure your change does not affect the whole system.

In the `./folder-name` **(FIXME)** folder please run:

```bash
pytest
```



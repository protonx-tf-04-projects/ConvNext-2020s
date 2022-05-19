from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Resizing, RandomFlip, RandomRotation, RandomZoom, Rescaling
from resnet import build_convnext


class ConvNeXt(Model):
    def __init__(self, num_classes=10, image_size=224, layer=[3, 3, 9, 3], model='tiny'):
        """
            ConvNeXt Model
            Parameters
            ----------
            num_classes:
                number of classes
            image_size: int,
                size of a image (H or W)
        """
        super(ConvNeXt, self).__init__()
        # Data augmentation
        self.data_augmentation = Sequential([
            Rescaling(scale=1./255),
            Resizing(image_size, image_size),
            RandomFlip("horizontal"),
            RandomRotation(factor=0.02),
            RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ])

        # Compute ratio
        input_shape = (image_size, image_size, 3)

        self.ratio = build_convnext(
            input_shape, num_classes, layer, model_name=model)

    def call(self, inputs):
        # ratio
        # output shape: (..., num_classes)
        # Create augmented data
        # augmented shape: (..., image_size, image_size, c)
        augmented = self.data_augmentation(inputs)

        output = self.ratio(augmented)

        return output


class ConvNeXtTiny(ConvNeXt):
    def __init__(self, num_classes=10, image_size=224):
        super().__init__(num_classes=num_classes,
                         image_size=image_size, layer=[3, 3, 9, 3], model='tiny')


class ConvNeXtSmall(ConvNeXt):
    def __init__(self, num_classes=10, image_size=224):
        super().__init__(num_classes=num_classes,
                         image_size=image_size, layer=[3, 3, 27, 3], model='small')


class ConvNeXtBase(ConvNeXt):
    def __init__(self, num_classes=10, image_size=224):
        super().__init__(num_classes=num_classes,
                         image_size=image_size, layer=[3, 3, 27, 3], model='base')


class ConvNeXtLarge(ConvNeXt):
    def __init__(self, num_classes=10, image_size=224):
        super().__init__(num_classes=num_classes,
                         image_size=image_size, layer=[3, 3, 27, 3], model='large')


class ConvNeXtMXLarge(ConvNeXt):
    def __init__(self, num_classes=10, image_size=224):
        super().__init__(num_classes=num_classes, image_size=image_size,
                         layer=[3, 3, 27, 3], model='xlarge')
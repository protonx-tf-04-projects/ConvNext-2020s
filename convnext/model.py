from cgitb import reset
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers.experimental.preprocessing import Resizing, RandomFlip, RandomRotation, RandomZoom, Rescaling
from tensorflow.keras import Sequential
from resnet import build_patchify

class ConvNeXt(Model):
    def __init__(self, num_classes=10, image_size=224):
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

        # Compute ratio
        input_shape=(image_size, image_size, 3)
        self.ratio = build_patchify(input_shape,num_classes,[3,3,9,3],use_bottleneck=True)

    def call(self, inputs):       
        # ratio
        # output shape: (..., num_classes)

        output = self.ratio(inputs)

        return output

class ConvNeXtMacro(ConvNeXt):
    def __init__(self, num_classes=10, image_size=224):
        super().__init__(num_classes=num_classes, image_size=image_size)
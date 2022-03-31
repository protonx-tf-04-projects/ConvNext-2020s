from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers.experimental.preprocessing import Resizing, RandomFlip, RandomRotation, RandomZoom, Rescaling
from tensorflow.keras import Sequential

class ConvNeXt(Model):
    def __init__(self, num_classes=10):
        """
            ConvNeXt Model
            Parameters
            ----------
            num_classes:
                number of classes
        """
        super(ConvNeXt, self).__init__()

        # Deep Conv2D
        self.conv2d = Sequential([
            Conv2D(16, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])

    def call(self, inputs):       
        # conv2d
        # output shape: (..., num_classes)

        output = self.conv2d(inputs)

        return output

class ConvNeXtBase(ConvNeXt):
    def __init__(self, num_classes=10):
        super().__init__(num_classes=num_classes)
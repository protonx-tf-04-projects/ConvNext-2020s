from tensorflow.keras.models import Model
from resnet import build_patchify, build_convnext

class ConvNeXt(Model):
    def __init__(self, num_classes=10, image_size=224, layer=[3,3,9,3], model='tiny'):
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

        self.ratio = build_convnext(input_shape,num_classes,layer,model_name=model)       

    def call(self, inputs):       
        # ratio
        # output shape: (..., num_classes)

        output = self.ratio(inputs)

        return output

class ConvNeXtTiny(ConvNeXt):
    def __init__(self, num_classes=10, image_size=224):
        super().__init__(num_classes=num_classes, image_size=image_size, layer=[3,3,9,3], model='tiny')

class ConvNeXtSmall(ConvNeXt):
    def __init__(self, num_classes=10, image_size=224):
        super().__init__(num_classes=num_classes, image_size=image_size, layer=[3,3,27,3], model='small')

class ConvNeXtBig(ConvNeXt):
    def __init__(self, num_classes=10, image_size=224):
        super().__init__(num_classes=num_classes, image_size=image_size, layer=[3,3,27,3], model='big')

class ConvNeXtLarge(ConvNeXt):
    def __init__(self, num_classes=10, image_size=224):
        super().__init__(num_classes=num_classes, image_size=image_size, layer=[3,3,27,3], model='large')

class ConvNeXtMXLarge(ConvNeXt):
    def __init__(self, num_classes=10, image_size=224):
        super().__init__(num_classes=num_classes, image_size=image_size, layer=[3,3,27,3], model='xlarge')

class PlayThrough(Model):
    def __init__(self, num_classes=10, image_size=224, layer=[3,3,9,3], model='marco'):
        """
            PlayThrough Model
            Parameters
            ----------
            num_classes:
                number of classes
            image_size: int,
                size of a image (H or W)
        """
        super(PlayThrough, self).__init__()

        # Compute ratio
        input_shape=(image_size, image_size, 3)

        self.ratio = build_patchify(input_shape,num_classes,layer,model_name=model)

    def call(self, inputs):       
        # ratio
        # output shape: (..., num_classes)

        output = self.ratio(inputs)

        return output

class PlayThroughMacro(PlayThrough):
    def __init__(self, num_classes=10, image_size=224, model='marco'):
        super().__init__(num_classes=num_classes, image_size=image_size, model=model)

class PlayThroughResNeXt(PlayThrough):
    def __init__(self, num_classes=10, image_size=224, model='resnext'):
        super().__init__(num_classes=num_classes, image_size=image_size, model=model)

class PlayThroughInvertedBottleneck(PlayThrough):
    def __init__(self, num_classes=10, image_size=224, model='invertedbottleneck'):
        super().__init__(num_classes=num_classes, image_size=image_size, model=model)

class PlayThroughLargeKernel(PlayThrough):
    def __init__(self, num_classes=10, image_size=224, model='largekernel'):
        super().__init__(num_classes=num_classes, image_size=image_size, model=model)


__author__ = 'Jie'
"""
data augmentation based on existing image.
"""

from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler

## data augmentation
datagen=ImageDataGenerator(rotation_range=40,
                           width_shift_range=0.2,
                           height_shift_range=0.2,
                           shear_range=0.2,
                           zoom_range=0.2,
                           horizontal_flip=True,
                           fill_mode="nearest",
                           channel_shift_range=0,
                           vertical_flip=True)

# augmentation
gen=datagen.flow_from_directory("D:/python-ml/deepLearning-cases/keras_handwrinting/dataAugmentation/",
                                target_size=(64,64),
                                batch_size=5,
                                save_to_dir="D:/python-ml/deepLearning-cases/keras_handwrinting/dataAugmentation/",
                                save_prefix="me_",
                                save_format="jpg")
# generate figures based on loop (it is a generator).
for i in range(5):
    gen.next()
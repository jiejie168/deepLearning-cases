__author__ = 'Jie'
"""
the test accuracy is high for all the downloaded images from internet, except my own image. :)
It turns out the accuracy is low for prediction of bitter face.
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input,Dense,Activation,Conv2D,MaxPooling2D,ZeroPadding2D,BatchNormalization,Flatten
from keras.models import Model
# from smileDetect_keras_cnn import happyModel

def happyModel(input_shape):
    """
    Implementation of the HappyModel.
    Arguments:
    input_shape -- shape of the images of the dataset
        (height, width, channels) as a tuple.
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train',
        then you can provide the input_shape using
        X_train.shape[1:]
    Returns:
    model -- a Model() instance in Keras
    """
    X_input=Input(input_shape) # create placeholders
    X=ZeroPadding2D((3,3))(X_input)
    X=Conv2D(32,(7,7),strides=(1,1),name='conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)  # normalize the batch
    X=Activation('relu')(X)
    X=MaxPooling2D((2,2),name='max_pool')(X)
    X=Flatten()(X)
    X=Dense(1,activation='sigmoid',name='fc')(X)

    model =Model(inputs=X_input,outputs=X,name='happyModel') # check the document?
    return model

def check_image(model,dir="D:/python-ml/deepLearning-cases/keras_handwrinting/keras_smileDetecting/",imageName='me.jpg',fileName='happyModel_me.h5'):
    # Load the model's saved weights.
    model.load_weights(str(dir)+str(fileName))

    image_path=str(dir)+str("man_bitter2.jpg")
    img=image.load_img(image_path,target_size=(64,64)) # check documents?



    plt.imshow(img)
    X=image.img_to_array(img)
    X=np.expand_dims(X,axis=0)  # check documents
    X=preprocess_input(X)   # check documents, remember to normalize the data !!
    pred=np.squeeze(model.predict(X/255))
    print (pred)
    plt.show()

happymodel =happyModel((64,64,3))
check_image(happymodel)
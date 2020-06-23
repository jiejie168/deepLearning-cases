__author__ = 'Jie'

"""
this case is to detect whether or not the face is smiling based on the binary classification cnn neural network.
the orginal data set is train_happy.h5;
Emotion tracking.
train accuracy:1
test accuracy :0.967 after 60 epochs.
"""
import numpy as np
from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D
from keras.layers import AveragePooling2D,MaxPooling2D,Dropout
from keras.models import Sequential, Model
from keras.preprocessing import image
from keras.utils import layer_utils
from kt_utils import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

########################################################################################################################
# load data
X_train_orig,Y_train,X_test_orig,Y_test,classes=load_dataset()
X_train=X_train_orig/255  # shape(600,64,64,3)
X_test=X_test_orig/255  #shape(150,64,64,3)

Y_train=Y_train.T    # shape(60,1)
Y_test=Y_test.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


########################################################################################################################
# construct nn model based on the keras.
# two ways can be used.
# way 1
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

def happyModel_2(input_shape):

    #########################
    # way 2
    model=Sequential()
    model.add(Conv2D(32,(7,7),strides=(1,1),input_shape=(64,64,3),name='conv0'))
    model.add(ZeroPadding2D((3,3)))
    model.add(BatchNormalization(axis = 3, name = 'bn0'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2,2),name='max_pool'))
    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid',name='fc'))
    return model

happyModel_me=happyModel(X_train.shape[1:])  # noted the input shape.
# print (X_train.shape[1:])

########################################################################################################################
# compile the model
happyModel_me.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
########################################################################################################################
happyModel_me.fit(X_train,
                  Y_train,
                  epochs=60,
                  batch_size=32)
########################################################################################################################
preds = happyModel_me.evaluate(X_test,Y_test)
print("##############################################################################################################")
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

########################################################################################################################
# save model
happyModel_me.save_weights("happyModel_me.h5")
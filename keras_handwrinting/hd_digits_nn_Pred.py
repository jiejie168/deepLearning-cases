__author__ = 'Jie'
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import mnist

def reload(dir="D:/python-ml/deepLearning-cases/keras_handwrinting/",fileName="hd_digits_3nn_pred.h5"):
    # Build the model.
    # the same nn framework should be used
    # the layers of NN should be the same with the training model
    model = Sequential([
      Dense(64, activation='relu', input_shape=(784,)),
      Dense(64, activation='relu'),
      Dense(10, activation='softmax'),
    ])
    # Load the model's saved weights.
    model.load_weights(str(dir)+str(fileName))

    # predict the test data
    # load data
    test_X_orig=mnist.test_images()  #shape(10000,28,28)
    test_y=mnist.test_labels()

    #flatten the image data into 2D. (60000,784)
    test_X=test_X_orig.reshape((test_X_orig.shape[0],-1))

    # normalize data by 255 into [-0.5,0.5]
    test_X=test_X/255-0.5
    predictions = model.predict(test_X[:5])
    predictions=pd.DataFrame(predictions)
    predictions_me=predictions.apply(np.argmax,axis=1)
    predictions_test=pd.DataFrame(test_y[:5])
    predictions=pd.concat([predictions_me,predictions_test],axis=1)
    print (predictions.head())


def imageShow(dir="D:/python-ml/deepLearning-cases/keras_handwrinting/",fileName="hd_digits_3nn_pred.h5",imageName="two.jpg"):
    import scipy
    from scipy import ndimage
    import matplotlib.pyplot as plt

    model = Sequential([
      Dense(64, activation='relu', input_shape=(784,)),
      Dense(64, activation='relu'),
      Dense(10, activation='softmax'),
    ])
    # Load the model's saved weights.
    model.load_weights(str(dir)+str(fileName))

    my_image = str(dir)+str(imageName) # image directory
    #transform image to data
    image = np.array(ndimage.imread(my_image, flatten=True)) # read a greyscale image
    image = image/255-0.5
    # print (image.shape)
    my_image = scipy.misc.imresize(image, size=(28,28)).reshape((1, 28*28)) # shape(1, 28*28)
    my_image_prediction = model.predict(my_image)
    print (my_image_prediction)
    final_pred=np.argmax(my_image_prediction)
    # print (my_image.shape)

    plt.imshow(image)
    print("Your algorithm predicts: y = " + str(np.squeeze(final_pred)))
    plt.show()

model=reload()
# imageShow()

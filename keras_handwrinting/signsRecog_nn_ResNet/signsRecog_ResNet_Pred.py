__author__ = 'Jie'

from keras.models import load_model
import numpy as np
from resnets_utils import *
from keras.preprocessing import image
from keras.utils import plot_model
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import scipy
# import pydot  # data visulization, output framework files.

# load the data
_,_,X_test_orig, Y_test_orig, classes = load_dataset()
X_test = X_test_orig/255.
Y_test = convert_to_one_hot(Y_test_orig, 6).T

print ("number of test examples = " + str(X_test.shape[0]))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

model=load_model("signsRec_ResNet.h5")
# model.summary()
#visualize your ResNet50
# plot_model(model, to_file='model_1.png')

preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

def check_image(model,dir="D:/python-ml/deepLearning-cases/keras_handwrinting/signsRecog_nn_ResNet/",imageName='me_0.jpg'):
    image_path=str(dir)+str(imageName)
    img=image.load_img(image_path,target_size=(64,64)) #return a PIL Image instance

    X=image.img_to_array(img) # Converts a PIL Image instance to a Numpy array
    # print (X.shape)
    X=np.expand_dims(X,axis=0)  # expand the array dimension; (2,)-->(1,2)
    X=X/255
    print('Input image shape:', X.shape) # (64,64,3)--> (1,64,64,3)
    pred=np.squeeze(model.predict(X))
    pred_class=np.argmax(pred,axis=-1) # return the array of indices into the array.
    print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
    print(pred)
    print (pred_class)

    my_image = scipy.misc.imread(image_path)
    plt.imshow(my_image)
    plt.show()

# check_image(model)
check_image(model,imageName="me_1.jpg")
# check_image(model,imageName="me_2.jpg")
# check_image(model,imageName="me_4.jpg")
# check_image(model,imageName="me_5.jpg")
# check_image(model,imageName="me_5_2.jpg")
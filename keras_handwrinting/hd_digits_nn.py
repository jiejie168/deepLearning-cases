__author__ = 'Jie'

"""
a code to predict the handwriting of digits based on 3 layers nn.
the dataset is from an exsiting dataset library mnist
"""
import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np
import pandas as pd

# load data
trains_X_orig=mnist.train_images() #  shape(60000,28,28)
trains_y=mnist.train_labels()
test_X_orig=mnist.test_images()  #shape(10000,28,28)
test_y=mnist.test_labels()

#flatten the image data into 2D. (60000,784)
trains_X=trains_X_orig.reshape((trains_X_orig.shape[0],-1)) # input shape (60000,28*28)
test_X=test_X_orig.reshape((test_X_orig.shape[0],-1))

# normalize data by 255 into [-0.5,0.5]
trains_X=trains_X/255-0.5
test_X=test_X/255-0.5

# create model, 2 hidden layers+ 1 output layer= 3 layers nn.
model=Sequential([Dense(64,activation='relu',input_shape=(784,)),
                  Dense(64,activation='relu'),
                  Dense(10,activation='softmax')])

# comile the model,
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# fit model
model.fit(trains_X,to_categorical(trains_y),batch_size=32,epochs=10)

# evaluate the fitted model
evals=model.evaluate(test_X,to_categorical(test_y))
print (evals)

# save model
model. save_weights("D:/python-ml/deepLearning-cases/keras_handwrinting/hd_digits_3nn_pred.h5")

# prediction and print
predictions = model.predict(test_X[:5])
predictions=pd.DataFrame(predictions)
predictions_me=predictions.apply(np.argmax,axis=1)
predictions_test=pd.DataFrame(test_y[:5])
predictions=pd.concat([predictions_me,predictions_test],axis=1)
print (predictions.head())









__author__ = 'Jie'

"""
the case is from kaggle competition for the prediction of digits number from 0-9
Dataset is original from mnist. However, the dataset for this case is rearranged to csv files.
Only a normal three layers NN is used for prediction
The test accuracy obtained from kaggle submission is 96.842% (ranking ~2400)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model,Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D, Dense,MaxPooling2D,Flatten,Dropout

#######################################################################################################################
# load data
trains=pd.read_csv("D:/python-ml/deepLearning-cases/keras_handwrinting/hd_dataset/train.csv")
tests=pd.read_csv("D:/python-ml/deepLearning-cases/keras_handwrinting/hd_dataset/test.csv")

### check data
trains.describe().T
trains.info()
train_X=trains.iloc[:,1:]
train_y=trains["label"]
test_X=tests

########################################################################################################################
# Show image, the shape of image --> (28,28)
index=10
image=train_X.iloc[index].values
image=image.reshape((28,28))
plt.imshow(image,cmap=plt.cm.get_cmap(), interpolation="nearest")
plt.show()

########################################################################################################################
train_X=train_X/255-0.5  # normalize by 255, and scale it to -0.5-0.5
test_X=test_X/255-0.5
train_X=train_X.values  # dataFrame--> numpy array. the conv2D/nn is operate on ndarray
test_X=test_X.values
train_y=train_y.values
train_y=to_categorical(train_y) # one-hot transformation. --> (42000,10)

########################################################################################################################
# only 3 normal layers are used.
model=Sequential ()
model.add(Dense(64,input_shape=(784,),activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',
             loss="categorical_crossentropy",
             metrics=['accuracy'])

model.fit(train_X,
         train_y,
         epochs=30,
         batch_size=32)

########################################################################################################################
# use the model for prediction
predictions_hw=model.predict(test_X)
predictions_hw=pd.DataFrame(predictions_hw)
predictions_hw=predictions_hw.apply(np.argmax,axis=1)  # obtain the index of the maximum prob after softmax
predictions_hw=pd.DataFrame(predictions_hw)


predictions_hw.reset_index(level=0,inplace=True)  #return DataFrame with the new index None if inplace=True.
predictions_hw=pd.DataFrame(predictions_hw)
predictions_hw.columns=["ImageId","Label"]
predictions_hw['ImageId']=predictions_hw['ImageId']+1  # index +1 to satisfy the output requirment of kaggle.
predictions_hw.to_csv("pred_me_hw.csv",index=False)
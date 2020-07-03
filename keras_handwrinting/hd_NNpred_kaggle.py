__author__ = 'Jie'

"""
the case is from kaggle competition for the prediction of digits number from 0-9
Dataset is original from mnist. However, the dataset for this case is rearranged to csv files.
a LeNet-5 cnn strategy is used for this model prediction.
The test accuracy obtained from kaggle submission is 99.128% (ranking 979)
It is obvious that the cnn and multiple layers strategy has dramatically improved the prediction model.
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

########################################################################################################################
#reshape the train_X for cnn use
train_X=np.reshape(train_X,(42000,28,28,-1)) # --> (42000,28,28,1)-->(nums,height,width,depth)
test_X=np.reshape(test_X,(28000,28,28,-1))
train_y=to_categorical(train_y) # one-hot transformation. --> (42000,10)

########################################################################################################################
# build the cnn framework
model=Sequential()
model.add(Conv2D(32,(5,5),strides=(1,1),activation="relu",input_shape=(28,28,1),name='conv0')) # shape=(height,width,depth)  -->24*24*32
model.add(MaxPooling2D((2,2),name="max_pool_1")) # --> 12*12*32
model.add(Conv2D(32,(5,5),strides=(1,1),activation="relu",padding="same",name='conv1'))  #--> 12*12*64
model.add(MaxPooling2D((2,2),name="max_pool_2")) #--> 6*6*64
model.add(Flatten())
model.add(Dense(128,activation='relu',name='fc_1'))
model.add(Dense(64,activation='relu',name='fc_2'))
model.add(Dense(10,activation='softmax',name="fc_3"))

model.compile(optimizer='adam',
             loss="categorical_crossentropy",
             metrics=['accuracy'])
model.fit(train_X,
         train_y,
         epochs=80,
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
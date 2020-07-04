__author__ = 'Jie'

"""
the case is from kaggle competition for the prediction of digits number from 0-9
Dataset is original from mnist. However, the dataset for this case is rearranged to csv files.
a updated cnn strategy is used for this model prediction.
data augmentation strategy is used. 5 loop of CNN is used for training together.

The test accuracy obtained from kaggle submission is 99.585% (ranking 348)
the problem is the use of learningRateScheduler cannot be used.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from keras.models import Model,Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D, Dense,MaxPooling2D,Flatten,Dropout,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler


#######################################################################################################################
# load data
trains=pd.read_csv("D:/python-ml/deepLearning-cases/keras_handwrinting/hd_dataset/train.csv")
tests=pd.read_csv("D:/python-ml/deepLearning-cases/keras_handwrinting/hd_dataset/test.csv")

### check data
trains.describe().T
trains.info()
#######################################################################################################################
train_y=trains["label"]
train_X=trains.drop("label",axis=1)
# train_X=trains.iloc[:,1:]  # an alternatively method
test_X=tests
print ("the shape of training data is:", train_X.shape)
print ("the shape of target data is:", train_y.shape)

#######################################################################################################################
# data exploration
sns.countplot(train_y)

# present the counts of each digits.
counts=train_y.value_counts()
print (type(counts))
print (counts)

mean_val=np.mean(counts)
std_val=np.std(counts,ddof=1)
print ("the mean value of counts is:",mean_val)
print ("the std value of counts is:",std_val)

# 68% - 95% - 99% rule, For an approximately normal data set,
#the values within one standard deviation of the mean account for about 68% of the set;
#while within two standard deviations account for about 95%;
# https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule

# if the data is skewed then we won't be able to use accurace as its results will be misleading and we may use F-beta score instead.
print (mean_val* (0.6827 / 2))
if std_val > mean_val* (0.6827 / 2):
    print("The standard deviation is high")

#######################################################################################################################
# check null data
def check_missData(df):
    # check the missing data of dataFrame.
    # output them in the form of dataFrame
    # sum(): count the number of missing data in each column
    # isnull(): return a dataFrame with True, and False

    miss_tot=df.isnull().sum().sort_values(ascending=False)
    counts_all=df.isnull().count() # count all the elements, including the missing elements
    miss_per=((df.isnull().sum())*100/counts_all).sort_values(ascending=False)
    miss_all=pd.concat([miss_tot,miss_per],axis=1,keys=['TotalNum','TotalPerc'])
    return miss_all
check_missData(train_X)

#######################################################################################################################
# convert train dataset to (num_images, img_rows, img_cols) format in order to plot it
numtrain=train_X.shape[0]
dim=28
xtrain_vis = train_X.values.reshape(numtrain, dim, dim)

# subplot(2,3,3) = subplot(233)
# a grid of 3x3 is created, then plots are inserted in some of these slots
for i in range(0,9): # how many imgs will show from the 3x3 grid
    plt.subplot(330 + (i+1)) # open next subplot
    plt.imshow(xtrain_vis[i], cmap=plt.get_cmap('gray'))
    plt.title(train_y[i])
plt.show()

#######################################################################################################################
# normalize data
train_X=train_X/255-0.5
test_X=test_X/255-0.5
train_X=train_X.values
test_X=test_X.values
train_y=train_y.values

train_X=np.reshape(train_X,(42000,28,28,-1))  # or train_X.values.reshape(-1,28,28,1) # -1 indicate not care  of the number of axis=0 location
test_X=np.reshape(test_X,(28000,28,28,-1))

print ("the shape of train_X is reshaped: ", train_X.shape)
print ("the shape of test_X is reshaped:",test_X.shape)

train_y=to_categorical(train_y)  # one-hot
print (train_y.shape)

#######################################################################################################################
# split training data
from sklearn.model_selection import train_test_split
# fix random seed for reproducibility
seed = 2
np.random.seed(seed)

# percentage of xtrain which will be x_validation
split_pct = 0.1

# Split the train and the validation set
x_train, x_val, y_train, y_val = train_test_split(train_X,
                                              train_y,
                                              test_size=split_pct,
                                              random_state=seed,
                                              shuffle=True,
                                              stratify=train_y)
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
#######################################################################################################################
# construct nn framework for further use
# five nets are used for accuracy prediction
nets=5  # five loops in the following training.
model=[0]*nets # define a zero list with length of nets

for i in range(nets):
    model[i]=Sequential()
    model[i].add(Conv2D(32,(5,5),strides=(1,1),activation="relu",input_shape=(28,28,1),name='conv0')) # shape=(height,width,depth)  -->24*24*32
    model[i].add(BatchNormalization())
    model[i].add(MaxPooling2D((2,2),name="max_pool_1")) # --> 12*12*32
    model[i].add(Conv2D(32,(5,5),strides=(1,1),activation="relu",padding="same",name='conv1'))  #--> 12*12*64
    model[i].add(BatchNormalization())
    model[i].add(Dropout(0.4))

    model[i].add(Conv2D(64,(3,3),strides=(1,1),activation='relu',name='conv2')) #-->10*10*128
    model[i].add(BatchNormalization())
    model[i].add(MaxPooling2D((2,2),name="max_pool_2")) #--> 5*5*128
    model[i].add(Conv2D(64,(3,3),strides=(1,1),activation="relu",padding="same",name='conv4')) #-->5*5*192
    model[i].add(BatchNormalization())
    model[i].add(Dropout(0.4))

    model[i].add(Conv2D(128,(4,4),strides=(1,1),activation='relu',name='conv5'))  #-->2*2*320
    model[i].add(BatchNormalization())
    model[i].add(Flatten())
    model[i].add(Dropout(0.4))
    model[i].add(Dense(128,activation='relu',name='fc_1'))
    model[i].add(Dense(64,activation='relu',name='fc_2'))
    model[i].add(Dense(10,activation='softmax',name="fc_3"))
    model[i].compile(optimizer='adam',loss="categorical_crossentropy",metrics=["acc"])  # be careful, either "acc" or "accuracy" in metrics due to the new version of keras 2.3.0


#######################################################################################################################
# data augmentation; first build a framework for image generator
datagen = ImageDataGenerator(
          featurewise_center=False,            # set input mean to 0 over the dataset
          samplewise_center=False,             # set each sample mean to 0
          featurewise_std_normalization=False, # divide inputs by std of the dataset
          samplewise_std_normalization=False,  # divide each input by its std
          zca_whitening=False,                 # apply ZCA whitening
          rotation_range=10,                   # randomly rotate images in the range (degrees, 0 to 180)
          zoom_range = 0.1,                    # Randomly zoom image
          width_shift_range=0.1,               # randomly shift images horizontally (fraction of total width)
          height_shift_range=0.1,              # randomly shift images vertically (fraction of total height)
          horizontal_flip=False,               # randomly flip images
          vertical_flip=False)                 # randomly flip images

#######################################################################################################################
# training the five models with cnn and different augmented data.
# annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x) # check ?
# callbacks=[annealer],
# train the network
history = [0] * nets
epochs = 50

for j in range(nets):
    print("CNN ",j+1)
    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(x_train, y_train, test_size = 0.1)
    history[j] = model[j].fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64),
                                        epochs = epochs,
                                        steps_per_epoch = X_train2.shape[0]//64,
                                        validation_data = (X_val2,Y_val2),
                                        verbose=1)
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        j+1,epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))

#######################################################################################################################
#plot the loss fuction
fig, ax = plt.subplots(2,1) # creat a 2 rows, 1 column plot zones.

ax[0].plot(history[3].history['loss'],color='b',label="Trainning loss")
ax[0].plot(history[4].history['val_loss'], color='r', label="Validation loss",axes =ax[0])

ax[0].grid(color='black', linestyle='-', linewidth=0.25)
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history[4].history['acc'], color='b', label="Training accuracy")
ax[1].plot(history[4].history['val_acc'], color='r',label="Validation accuracy")
ax[1].grid(color='black', linestyle='-', linewidth=0.25)
legend = ax[1].legend(loc='best', shadow=True)

#######################################################################################################################
# emsembling results and save data.
results = np.zeros( (test_X.shape[0],10) )  #--> shape (28000,10)

for j in range(nets):
    # obtain the maximum probality by adding five cnn. (28000,10)
    results = results + model[j].predict(test_X)  # return a numpy ndarray.

results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("predictions_again.csv",index=False)
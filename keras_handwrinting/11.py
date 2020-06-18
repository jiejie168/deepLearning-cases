__author__ = 'Jie'
from keras.utils import to_categorical

y=[0,1,2,3,4]
y1=to_categorical(y,num_classes=5)
print (y1.shape)
print (y1)
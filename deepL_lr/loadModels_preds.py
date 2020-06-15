__author__ = 'Jie'
from sklearn.externals import joblib
from lr_nn_oneLayer_catDetect import LR_oneL_cat
from deepNN_application import Llayer_model

#######################################################################################################################
# reload the lr_nn_onelayer_catDetect.py for prediction
#######################################################################################################################
clf=joblib.load("D:/python-ml/deepLearning-cases/lr_catPred.pkl")
print (set(clf))
w=clf['w']
b=clf['b']
num_px=64
dir="D:/python-ml/deepLearning-cases/"
image="noncat2.jpg"
lr_onecat=LR_oneL_cat()
lr_onecat.testOwnImage(w,b,num_px,dir,image)
print ("+--------------------------------------------------------+ \n")

#######################################################################################################################
# reload the four layers deepNN_application.py for prediction
#######################################################################################################################
print ("the prediction from the 4 layers neural network starts\n")
clf1=joblib.load("D:/python-ml/deepLearning-cases/lr_catNNPred.pkl")
print ("the clf1 model have parameters: {}".format(set(clf1)))
llayer_model=Llayer_model()
parameters=clf1['parameters']
pred_test=clf1["Y_prediction_test"]
pred_train=clf1["Y_prediction_train"]

X_flatten=llayer_model.imageToData(num_px,dir,image)
Y_label=[0]
p=llayer_model.testOwnImage(X_flatten,parameters,num_px, Y_label,dir,image)
# llayer_model.plotCost(clf1['costs'],clf1["learning_rate"])
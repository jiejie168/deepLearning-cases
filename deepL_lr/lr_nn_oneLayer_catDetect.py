__author__ = 'Jie'
# -*- coding: utf-8 -*-

"""
This is a simple lr case for the prediction of whether or  not an input image is  a cat.
the original source is from the assignment of WEEK 3 in the course of "Deep learning -NG"
This coding is a full repeating of this assignment, and clarifying every detail.

noted: this is actually a normal LR, no hidden layer; the objective is to use the idea of NN.
you can play it with your own image. The accuracy is not very high for this code.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import misc
import h5py
# from PIL import Image

class LR_oneL_cat():

    # def __init__(self):
    #     pass
    def load_dataset(self):
        """
        load the dataset for model training.
        data.h5 incudes:
        - a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
        - a test set of m_test images labeled as cat or non-cat
        - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB).
        Thus, each image is square (height = num_px) and (width = num_px).
        :return:
        """
        train_dataset=h5py.File('D:/python-ml/deepLearning-NG/datasets/train_catvnoncat.h5',"r")
        train_set_x_orig=np.array(train_dataset["train_set_x"][:])
        train_set_y_orig=np.array(train_dataset["train_set_y"][:])

        test_dataset=h5py.File("D:/python-ml/deepLearning-NG/datasets/test_catvnoncat.h5", "r")
        test_set_x_orig=np.array(test_dataset['test_set_x'][:])
        test_set_y_orig=np.array(test_dataset['test_set_y'][:])

        classes=np.array(test_dataset["list_classes"][:])

        train_set_y_orig=train_set_y_orig.reshape((1,train_set_y_orig.shape[0])) # reshape (1,m)
        test_set_y_orig=test_set_y_orig.reshape((1,test_set_y_orig.shape[0]))   # reshape(1,m1)
        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

    def imageShow(self,datasetX,datasetY,classes,index):
        """
        show the image.
        :param datasetX: shape:[index, num_px,num_px,3]-> RGB
        :param datasetY: shape:[index,num_px,num_px,3]-> RGB
        :param classes: [b'non-cat' b'cat']
        :param index:  the index of image
        """
        plt.imshow(datasetX[index])
        print ("y="+str(datasetY[:,index])+",it's a'"+classes[np.squeeze(datasetY[:,index])].decode('utf-8')+"'picture.")
        plt.show()

    def sigmoid(self,Z):
        """
        compute the sigmoid of Z
        :param Z: a scalar or numpy array of any size
        :return: xiegama -> sigmoid(z)
        """
        xiegama=1/(1+np.exp(-Z))
        return xiegama

    def initialize_with_zeros(self,dim):
        """
        this function creates a vector of zeros of shape(dim,1) for w and initialize b to 0
        :param dim: size of the w vector we want( or number of parameters)
        :return: w,b.
        """
        w=np.zeros((dim,1))  # shape convention (n_l,n_(l-1))
        b=0
        assert w.shape==(dim,1)
        assert isinstance(b,float) or isinstance(b,int)
        return w,b

    def propagate(self,w,b,X,Y):
        """
        implement the cost function and its gradient for the propagations
        :param w:weight, a numpy array of size (number of features, 1)-> (num_px*num_px*3,1)
        :param b: bias, a float scalar.
        :param X: training data; (num_px*num_px*3,number of samples)
        :param Y: target data; (1,number of samples)
        :return:
        cost:negative log-likelihood cost for lr
        dw:gradient of the loss with respect to w, the same shape with w
        db:gradient of the loss with respect to b.
        """
        m=X.shape[1]
        #Forward propagation; vectorization
        Z=np.dot(w.T,X)+b  # matrix multiply. -> [1,m]
        A=self.sigmoid(Z)  #-> [1,m]
        cost= -np.sum(np.multiply(Y,np.log(A))+np.multiply((1-Y),np.log(1-A)))/m # element wise

        #BP; vectorization
        dw=np.dot(X,(A-Y).T)/m  # dL/dw ; [n,m]X[m,1] -> [n,1];
        db=np.sum(A-Y)/m  # dL/db=dL/dZ *dZ/db=dL/dZ ;  -> [n,1]

        assert dw.shape==w.shape
        assert db.dtype==float
        cost=np.squeeze(cost) # remove the one-dimension of cost
        assert cost.shape==()
        grads={"dw":dw,
              "db":db}
        return grads,cost

    def optimize(self,w,b,X,Y,num_iter,learning_rate,print_cost=False):
        """
        optimize the w,b by running a gradient descent algorithm
        :param w: weights, a numpy array of size (number of features, 1)-> (num_px*num_px*3,1)
        :param b: bias, a float scalar.
        :param X: training data; (num_px*num_px*3,number of samples)
        :param Y: target data; (1,number of samples)
        :param num_iter: number of iterations
        :param learning_rate: learning rate of GD
        :param print_cost: printing switch
        :return: the optimized w,b,cost
        params: dictionary including the final weights w and bias b
        gards: dictionary including the gradients of the final weights w and the bias b
        costs: lists of all the costs computed during the optimization
        """
        costs=[]
        for i in range(num_iter):
            grads,cost=self.propagate(w,b,X,Y)
            dw=grads['dw']
            db=grads['db']
            w=w-learning_rate*dw # element_wise
            b=b-learning_rate*db

            # print cost every 100 iteration
            if i % 100 ==0:
                costs.append(cost)
            if print_cost and i% 100==0:
                print ("cost after iteration {} is : {}".format(i,cost))

        params={"w":w,
                "b":b}
        grads={"dw":dw,
               "db":db}
        return params, grads,costs

    def predict(self,w,b,X):
        """
        compute the prediction value using the optimized w, and b
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        :return:
        """
        m=X.shape[1] # number of samples
        Y_pred=np.zeros((1,m))
        w=w.reshape(X.shape[0],1) # the number of features

        Z=np.dot(w.T,X)+b # (1,m)
        A=self.sigmoid(Z)

        for i in range(A.shape[1]):
            if A[0,i]<=0.5:
                Y_pred[0,i]=0
            else:
                Y_pred[0,i]=1
        assert  Y_pred.shape==(1,m)
        return Y_pred

    def model(self,X_train,y_train,X_test,y_test,num_iter=3000,learning_rate=0.1,print_cost=True):
        """

        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :param num_iter:
        :param learning_rate:
        :param print_cost:
        :return:
        """
        w,b=self.initialize_with_zeros(X_train.shape[0]) # w->[num_px*num_px*3,1]
        params, grads,costs=self.optimize(w,b,X_train,y_train,num_iter,learning_rate,print_cost)
        w=params["w"]
        b=params["b"]

        # predict
        Y_pred_train=self.predict(w,b,X_train)# (1,m)
        Y_pred_test=self.predict(w,b,X_test)

        # print prediction errors
        train_accuracy=100-np.mean(np.abs(Y_pred_train-y_train))*100
        test_accuracy=100-np.mean(np.abs(Y_pred_test-y_test))*100

        print ("train accuracy : {}%".format(train_accuracy))
        print ("test accuracy : {}%".format(test_accuracy))

        d={"costs":costs,
           "Y_prediction_test":Y_pred_test,
           "Y_prediction_train":Y_pred_train,
           "w":w,
           "b":b,
           "learning_rate":learning_rate,
           "num_iter":num_iter}
        return d

    def testOwnImage(self,w,b,num_px,dir='',image=""):

        my_image=str(dir)+str(image)

        # the image should be processed first
        image1=np.array(ndimage.imread(my_image,flatten=False))#  turn an image to a readable format （xdim,ydin,3or4）
        my_image=misc.imresize(image1,size=(num_px,num_px))  # resize image into a new size
        my_image=my_image.reshape((1,num_px*num_px*3)).T
        my_predict=self.predict(w,b,my_image)
        plt.imshow(image1)
        plt.show()
        print ("y = " + str(np.squeeze(my_predict)))

    def leaningRateTune(self,train_set_x, train_set_y, test_set_x, test_set_y,rates=[]):
        models={}
        for i in rates:
            print ("learning rate is: {}".format(i))
            models[str(i)]=self.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 3000, learning_rate = i, print_cost = False)
            print ('\n' + "-------------------------------------------------------" + '\n')
        for i in rates:
            plt.plot(np.squeeze(models[str(i)])['cost'],label= str(models[str(i)]["learning_rate"]))

        plt.ylabel('cost')
        plt.xlabel('iterations (hundreds)')

        legend=plt.legend(loc='upper_center',shodow=True)
        frame=legend.get_frame()
        frame.set_facecolor('0.90')
        plt.show()

    def saveModel(self,model,dir="D:/python-ml/deepLearning-cases/lr_catPred.pkl"):
        from sklearn.externals import joblib
        lr_catPred=joblib.dump(model,dir)
        return lr_catPred

def main():
    lr_oneL_cat=LR_oneL_cat()
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y,classes=lr_oneL_cat.load_dataset()

    # lr_oneL_cat.imageShow(train_set_x_orig,train_set_y,classes,6)
    # m_train=train_set_x_orig.shape[0]  # noted in nn, the convention is different with ML. [features,m_train]
    # m_test=test_set_x_orig.shape[0]
    # num_px=train_set_x_orig.shape[1]
    # print ("Number of training examples: m_train = " + str(m_train))
    # print ("Number of testing examples: m_test = " + str(m_test))
    # print ("Height/Width of each image: num_px = " + str(num_px))
    # print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    # print ("train_set_x shape: " + str(train_set_x_orig.shape))
    # print ("train_set_y shape: " + str(train_set_y.shape))
    # print ("test_set_x shape: " + str(test_set_x_orig.shape))
    # print ("test_set_y shape: " + str(test_set_y.shape))

    # preprocess data
    train_set_x_flatten=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T # shape(num_px*num_px*3,m_train)
    test_set_x_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T # shape(num_px*num_px*3,m_test)
    print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))
    print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

    # normalization
    train_set_x_flatten=train_set_x_flatten/255
    test_set_x_flatten=test_set_x_flatten/255

    d=lr_oneL_cat.model(train_set_x_flatten,train_set_y,test_set_x_flatten,test_set_y,10000,0.001,True)
    lr_catPred_oneLayer=lr_oneL_cat.saveModel(d) #save the fitted model

    # learning rate tune
    # it is not meaningful here.
if __name__ == '__main__':
    main()

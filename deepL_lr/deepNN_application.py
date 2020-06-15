__author__ = 'Jie'
# -*- coding: utf-8 -*-

"""
a L layers nn is created for the prediction of cat or noncat pictures.LR classification is used for the final loss function calcuation.
A four layers nn is trained based on the codes, with a train accuracy of 99%, and a test accuracy of 82%.
It is obviously a overfitting here due to the relative small number of samples.
15/06/2015
"""

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import  Image
from scipy import  ndimage
from dnn_app_utils_v3 import *

# plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'
# np.random.seed(1)

########################################################################################################################
# L-layer NN
########################################################################################################################
# GRADED FUNCTION: L_layer_model

class Llayer_model():

    def two_layer_model(self,X,Y,layers_dims,learning_rate=0.0075,num_iterations=3000,print_cost=False):
        """
        Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
        X -- input data, of shape (n_x, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- dimensions of the layers (n_x, n_h, n_y)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- If set to True, this will print the cost every 100 iterations
        :return:
         parameters -- a dictionary containing W1, W2, b1, and b2
        """
        np.random.seed(1)
        grads={}
        costs=[]
        m=X.shape[1]
        (n_x,n_h,n_y)=layers_dims

        parameters=initialize_parameters(n_x,n_h,n_y)
        W1=parameters["W1"]
        b1=parameters["b1"]
        W2=parameters["W2"]
        b2=parameters["b2"]

        # loop (gradient descent)
        for i in range(0,num_iterations):
            A1,cache1=linear_activation_forward(X,W1,b1,activation="relu")
            A2,cache2=linear_activation_forward(A1,W2,b2,activation="sigmoid")

            cost =compute_cost(A2,Y)

            dA2=-(np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
            dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid")
            dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation="relu")

            grads['dW1'] = dW1
            grads['db1'] = db1
            grads['dW2'] = dW2
            grads['db2'] = db2

            parameters = update_parameters(parameters, grads, learning_rate)

            # Retrieve W1, b1, W2, b2 from parameters
            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]

            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if print_cost and i % 100 == 0:
                costs.append(cost)

        # plot the cost

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        return parameters

    def L_layer_model(self,X_train, Y_train, X_test,Y_test,layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        X_train -- data, numpy array of shape (number of examples, num_px * num_px * 3)
        Y_train -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """
        np.random.seed(1)
        costs = []                         # keep track of cost
        # Parameters initialization.
        parameters = initialize_parameters_deep(layers_dims)
        # Loop (gradient descent)
        for i in range(0, num_iterations):
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = L_model_forward(X_train, parameters)
            # Compute cost.
            cost = compute_cost(AL, Y_train)

            # Backward propagation.
            grads = L_model_backward(AL, Y_train, caches)
            # Update parameters.
            parameters = update_parameters(parameters, grads, learning_rate)

            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)
        # predict
        Y_pred_train=predict(X_train, Y_train, parameters)
        Y_pred_test=predict(X_test, Y_test, parameters)

        d={"costs":costs,
           "Y_prediction_test":Y_pred_test,
           "Y_prediction_train":Y_pred_train,
           "learning_rate":learning_rate,
           "num_iter":num_iterations,
           "parameters":parameters}
        return d

    def plotCost(self,costs,learning_rate):
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    def imageToData(self,num_px,dir="D:/python-ml/deepLearning-cases/",image="mycat.jpg"):
        """

        :param num_px:
        :param dir:
        :param image:
        :return:
        """
        from scipy import misc
        my_image=str(dir)+str(image)
        image1=np.array(ndimage.imread(my_image,flatten=False)) # turn an image to a readable format （xdim,ydin,3or4）
        imageData=misc.imresize(image1,size=(num_px,num_px))  # resize image into a new size
        imageData=imageData.reshape((1,num_px*num_px*3)).T

        # Standardize data to have feature values between 0 and 1.
        X_flatten = imageData/255.
        print ("my image's shape: " + str(X_flatten.shape))
        plt.imshow(image1)
        plt.show()
        return X_flatten

    def saveModel(self,model,dir="D:/python-ml/deepLearning-cases/lr_catPred.pkl"):
        from sklearn.externals import joblib
        # save the fitted model for the application
        lr_catPred=joblib.dump(model,dir)
        return lr_catPred

    def testOwnImage(self,X,parameters,num_px, Y_label=[],dir="D:/python-ml/deepLearning-cases/",image="mycat.jpg"):
        """
        test the fitted model with my own image.
        :param X: the input of image.  (number of features, 1)
        :param Y: the label you input for the image:  (1--> cat, 0 noncat)
        :param parameters: parameters from the fitted model
        :param num_px: original num_px the same with the training photo. here is 64
        :param dir: image directory
        :param image: image name
        :return:
        p: predicted value: 1 for a cat, 0 for noncat.
        """
        # my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
        imageData=self.imageToData(num_px,dir,image)
        my_predict = predict(X, Y_label, parameters)
        print ("y = " + str(np.squeeze(my_predict)))

########################################################################################################################
########################################################################################################################
def main():
    ###################################################################################################################
    # load data for training.
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]
    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_x_orig shape: " + str(train_x_orig.shape))
    print ("train_y shape: " + str(train_y.shape))
    print ("test_x_orig shape: " + str(test_x_orig.shape))
    print ("test_y shape: " + str(test_y.shape))

    ###################################################################################################################
    # Reshape the training and test examples
    # The "-1" makes reshape flatten the remaining dimensions
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.
    print ("train_x's shape: " + str(train_x.shape))
    print ("test_x's shape: " + str(test_x.shape))

    ####################################################################################################################
    # define any layers neural network.
    ####################################################################################################################
    llayer_model=Llayer_model()
    layers_dims = [12288, 20, 7, 5, 1] #  4-layer model
    model = llayer_model.L_layer_model(train_x, train_y, test_x,test_y,layers_dims,print_cost=True)

    #save model and test model with my own image
    parameters=model['parameters']
    pred_test=model["Y_prediction_test"]
    pred_train=model["Y_prediction_train"]
    llayer_model.saveModel(model,dir="D:/python-ml/deepLearning-cases/lr_catNNPred.pkl")
    X_flatten=llayer_model.imageToData(num_px,dir="D:/python-ml/deepLearning-cases/",image="mycat.jpg")
    Y_label=[1]
    p=llayer_model.testOwnImage(X_flatten,parameters,num_px, Y_label,dir="D:/python-ml/deepLearning-cases/",image="mycat.jpg")

    llayer_model.plotCost(model['costs'],model["learning_rate"])
    ####################################################################################################################
    # architecture of your model
    #####2-layer NN
    ####################################################################################################################
    ## constants defining the model #
    # n_x=12288
    # n_h=7
    # n_y=1
    # layers_dims=(n_x,n_h,n_y)  # layers : len(layers_dims)-1--> 2 layers nn.
    # parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
    # predictions_train = predict(train_x, train_y, parameters)
    # predictions_test = predict(test_x, test_y, parameters)
    ####################################################################################################################

if __name__ == '__main__':
    main()











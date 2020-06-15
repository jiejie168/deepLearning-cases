__author__ = 'Jie'
# -*- coding: utf-8 -*-
import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v4a import *
from dnn_utils_v2 import sigmoid, sigmoid_backward,relu,relu_backward

'''
in this one, you will implement all the functions required to build a deep neural network
and use these functions to build a deep neural network for image classification

The codes here inlcude all the necessary functions for a L layers neural network, including initialize weights, bias,
FP, BF. Only logistic regression loss function is used for the final prediction. 15/0/2020
'''
plt.rcParams['figure.figsize']=(5.0,4.0) # set default size of plots
plt.rcParams['image.interpolation']='nearest'
plt.rcParams['image.cmap']='gray'

np.random.seed(1)

########################################################################################################################
#initialization
## 3.1: 2-layer NN
########################################################################################################################

def initialize_parameters(n_x,n_h,n_y):
    """
    :param n_x: size of the input layer; feature numbers
    :param n_h: size of the hidden layer ;
    :param n_y: size of the output layer ;
    :return:
    parameters- python dictionary containing your parameters
    W1-weight matrix of shape (n_h,n_x)  # general rule: for l layer: (n_l,n_{l-1}) ;  no transportation needed when following use
    b1-bias vector of shape (n_h,1)
    W2-weight matrix of shape (n_y,n_h)
    b2-bias vector of shape (n_y,1)
    """
    np.random.seed(1)
    W1=np.random.randn(n_h,n_x)*0.01  # no zeros randoms allowed!
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))

    parameters={"W1":W1,
                "b1":b1,
                "W2":W2,
                "b2":b2}
    return parameters

# parameters=initialize_parameters(3,2,1)
# print ("W1="+str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

########################################################################################################################
# L-layer NN
########################################################################################################################

def initialize_parameters_deep(layer_dims=[]):
    """
    :param layer_dims: python array (list) containing the dimensions of each layer in our network
    :return:
    parameters-python dictionary containing your parameters : W1,b1,..WL,bL

    """
    np.random.seed(3)
    parameters={}
    L=len(layer_dims) # the length includes the input layer, which in convention is not counted in the layers.

    for l in range(1,L):
        parameters['W'+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])*0.01  # layer_dims[0] is input layer
        parameters['b'+str(l)]=np.zeros((layer_dims[l],1))

        assert (parameters['W'+str(l)].shape==(layer_dims[l],layer_dims[l-1]))
        assert (parameters['b'+str(l)].shape==(layer_dims[l],1))

    return parameters

# parameters = initialize_parameters_deep([5,4,3])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


########################################################################################################################
# 4 Forward propagation module
########################################################################################################################
def linear_forward(A,W,b):
    """
    implement the linear part of a layer's forward propagation.
    The linear part only includes:  Z=WA+b
    :param A: activations from previous layer (or input data): (size of previous layer, number of example)
    :param W: weights matrix: numpy array of shape (sie of current layer, size of previous layer)
    :param b:bias vector, numpy array of shape (size of the current layer, 1)
    :return:
    Z-the input of the activation function,also called pre-activation parameter ; (size of the current layer, 1)
    cache-a python dictionary containing A, W, and b, stored for computing the backward pass efficiently
    """
    Z=np.dot(W,A)+b  # matrix multiplying;  Z=WA+b  ; no needed transpose under this circumstance.
    cache=(A,W,b)
    return Z,cache

# A,W,b=linear_forward_test_case()
# Z,linear_cache=linear_forward(A,W,b)
# print (Z)
# print (linear_cache)

########################################################################################################################
### 4.2 linear activation forward
########################################################################################################################

def linear_activation_forward(A_prev,W,b,activation):
    """
    implement the forward propagaton for the LINEAR-ACTIVATION layer
    # it contains: Z=WA+b; A=g(Z)
    :param A_prev: activations from previous layer (or input data): (size of previous layer, number of examples)
    :param W:weights matrix:numpy array of shape (size of current layer,size of previous layer)
    :param b:bias vector, numpy array of shape (size of the current layer,1)
    :param activation: the activation to be used in this layer, stored as a text string: sigmoid or relu

    :return:
    A- the output of the activation function, also called the post-activation value
    cache- a python dictionary containing "linear_cache" and "activation_cache", stored for computing the BP
    """
    if activation=="sigmoid":
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=sigmoid(Z)

    elif activation=="relu":
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=relu(Z)

    assert (A.shape==(W.shape[0],A_prev.shape[1]))
    cache=(linear_cache,activation_cache)

    return A, cache

# A_prev, W, b = linear_activation_forward_test_case()
#
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
# print("With sigmoid: A = " + str(A))
#
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
# print("With ReLU: A = " + str(A))

########################################################################################################################
# L-Layer model
########################################################################################################################
def L_model_forward(X,parameters):
    """
    implement FP for the [LINEAR-RELU]*(L-1)-LINEAR-SIGMOID computation
    Noted: input_data--> (L-1) layers of hidden layer(LINEAR-RELU)--> output layer(LINEAR-SIGMOID) ;  L layers
    :param X:data,numpy array of shape (feature numbers,number of examples)
    :param parameters:output of initialize_parameters_deep()
    :return:
    AL-last post-activation value;
    caches-list of caches containing: every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 from L-1)
    """
    caches=[]
    A=X
    L=len(parameters)//2  # parameters includes both W_l, and b_l.
    for l in range(1,L):
        A_prev=A
        A,cache=linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],activation="relu")
        caches.append(cache)

    # the last layer is linear & sigmoid
    AL,cache=linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],activation="sigmoid")
    caches.append(cache)
    # assert AL.shape==(1,X.shape[l]) # this assertion has a problem
    return AL, caches

X, parameters = L_model_forward_test_case_2hidden()
AL, caches = L_model_forward(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))


########################################################################################################################
# cost function
########################################################################################################################

def compute_cost(AL,Y):
    """
    implement the cost function
    :param AL: probability vector corresponding to your label predictions, shape (1,number of examples)
    :param Y:true label vector (for example: containing 0 if non-cat, 1 if cat), shape (1,number of examples)
    :return:
    cost-cross-entropy cost
    """
    m=Y.shape[1] # the number of samples
    cost=-np.sum(np.multiply(Y,np.log(AL))+np.multiply((1-Y),np.log(1-AL)))/m  # element wise.
    cost=np.squeeze(cost)
    assert (cost.shape==())

    return cost

# Y, AL = compute_cost_test_case()
# print("cost = " + str(compute_cost(AL, Y)))


########################################################################################################################
# backward propagation module
########################################################################################################################
# linear backward
def linear_backward(dZ,cache):
    """
    implement the linear portion of backward propagation for a single layer (layer l), (noted: l is not the first layer)
    :param dZ: gradient of the cost with respect to the linear output ( current layer l)
    :param cache: tuple of values (A_prev,W,b) coming from the forward propagation in the current layer
    :return:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev,W,b=cache
    m=A_prev.shape[1]

    dW=np.dot(dZ,A_prev.T)/m             # (1,m) * ( size of previous layer, m).T = (n_l,n_{l-1})
    db=np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev=np.dot(W.T,dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


# dZ, linear_cache = linear_backward_test_case()
# dA_prev, dW, db = linear_backward(dZ, linear_cache)
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))

########################################################################################################################
# linear-activation backward
########################################################################################################################
def linear_activation_backward(dA,cache,activation):
    """
    implement the backward propagation for the linear-activation layer
    :param dA:post-activation gradient for current layer l
    :param cache: tuple of vaules (linear_cache,activation_cache)
    :param activation: the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    :return:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache,activation_cache=cache

    if activation=="relu":
        dZ=relu_backward(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache)
    elif activation=="sigmoid":
        dZ=sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache)

    return dA_prev,dW,db

# dAL, linear_activation_cache = linear_activation_backward_test_case()
# dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "sigmoid")
# print ("sigmoid:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db) + "\n")
#
# dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "relu")
# print ("relu:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))

########################################################################################################################
# L-model backward
########################################################################################################################

def L_model_backward(AL,Y,caches):
    """
    implement the BP for the [linear-relu]*(L-1)<-linear <-sigmoid group  ; L layers together
    :param AL: probability vector, output of the FP (L_model_forward())
    :param Y: ture label vector
    :param caches: list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    :return:
        grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads={}
    L=len(caches)
    m=AL.shape[1] # number of samples
    Y=Y.reshape(AL.shape)

    dAL=-np.divide(Y,AL)+np.divide((1-Y),1-AL)  # the derivative of ACTIVATION in layer L.--> -Y/A+(1-Y)/(1-A); only suitable for LR.

    current_cache=caches[L-1]
    grads["dA"+str(L-1)],grads["dW",str(L)],grads["db"+str(L)]=linear_activation_backward(dAL, current_cache, activation="sigmoid")

    # loop from l=L-2 to l=0
    # reversed: 0...L-2
    for l in reversed (range(L-1)):
        current_cache=caches[l]
        dA_prev_temp,dW_temp,db_temp=linear_activation_backward(grads["dA"+str(l+1)], current_cache, activation="relu")
        grads["dA"+str(l)]=dA_prev_temp
        grads["dW"+str(l+1)]=dW_temp
        grads["db"+str(l+1)]=db_temp

    return grads

# AL, Y_assess, caches = L_model_backward_test_case()
# grads = L_model_backward(AL, Y_assess, caches)
# print_grads(grads)

########################################################################################################################
# 6.4 update parameters
########################################################################################################################
def update_parameters(parameters, grads, learning_rate):
    """
    update parameters using gradient descent
    :param parameters: python dictionary containing your parameters
    :param grads: python dictionary containing your gradients, output of L_model_backward
    :param learning_rate:
    :return:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """
    L=len(parameters)//2
    # apply GD
    for l in range(L):
        parameters["W"+str(l+1)]=parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]
        parameters["b" + str(l+1)] = parameters["b"+str(l+1)]-learning_rate*grads["db"+str(l+1)]
    return  parameters

parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)
print ("W1 = "+ str(parameters["W1"]))
print ("b1 = "+ str(parameters["b1"]))
print ("W2 = "+ str(parameters["W2"]))
print ("b2 = "+ str(parameters["b2"]))




































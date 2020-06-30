__author__ = 'Jie'
"""
This is a very interesting case using rnn. A train set in terms of a series of dinosaur names is used for training.
Then, the training model will be used for making new names for dinosaurs. This case is originally from the assignment 2 of week 1
in the course sequence model
By complete this case:
1) how to store text data for processing using  an RNN
2) how to synthesize data, by sampling predictions at each time step and passing it to the next RNN-cell units
3) how to build a character-level text generation RNN
4) why clipping the gradients is important
"""

import numpy as np
from utils import *
import random

########################################################################################################################
# dataset and preprocessing

data=open("dinos.txt").read() # read all the text as a string, including "\n"
data1=open('dinos.txt').readlines()
data=data.lower()
chars=list(set(data))
data_size,vocab_size=len(data),len(chars) # data_size: the length of chars
data1_size=len(data1)
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))
print ("there are {} training samples".format(data1_size))

# make dictionary for further use
# "\n" is a token, indicating the end of dinosaur name
chars=sorted(chars)
char_to_ix={ch:i for i, ch in enumerate(chars)}
ix_to_char={i:ch for i,ch in enumerate(chars)}

########################################################################################################################
# define clipping function, which will be used for the clipping of gradients.
def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.

    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue

    Returns:
    gradients -- a dictionary with the clipped gradients.
    '''
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']

    for gradient in [dWaa,dWax,dWya,db,dby]:
        np.clip(gradient,-maxValue,maxValue,out=gradient)
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    return gradients

########################################################################################################################
# assume the model is already trained, a sample function is needed to generate new text based on your inputs x^<1> in the
# first time step.

def sample(parameters, char_to_ix, seed):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN
    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b.
    char_to_ix -- python dictionary mapping each character to an index.
    seed -- used for grading purposes.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """
    # Retrieve parameters and relevant shapes from "parameters" dictionary
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    # Step 1: Create the a zero vector x that can be used as the one-hot vector
    # representing the first character (initializing the sequence generation).
    x = np.zeros((vocab_size,1))  # 27 in rows, 1 in column
    # Step 1': Initialize a_prev as zeros
    a_prev = np.zeros((n_a,1))

    indices = []

    # idx is the index of the one-hot vector x that is set to 1
    # All other positions in x are zero.
    # We will initialize idx to -1
    idx = -1

    # Loop over time-steps t. At each time-step:
    # sample a character from a probability distribution
    # and append its index (`idx`) to the list "indices".
    # We'll stop if we reach 50 characters
    # (which should be very unlikely with a well trained model).
    # Setting the maximum number of characters helps with debugging and prevents infinite loops.
    counter = 0
    newline_character = char_to_ix['\n']

    while (idx != newline_character and counter != 50):

        # Step 2: Forward propagate x using the equations (1), (2) and (3)
        a = np.tanh(np.dot(Waa,a_prev)+np.dot(Wax,x)+b)
        z = np.dot(Wya,a)+by
        y = softmax(z)  # obtain the probability distribution of y

        # for grading purposes
        np.random.seed(counter+seed)

        # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
        # (see additional hints above)
        idx = np.random.choice(list(range(vocab_size)), p = y.ravel())

        # Append the index to "indices"
        indices.append(idx)

        # Step 4: Overwrite the input x with one that corresponds to the sampled index `idx`.
        # (see additional hints above)
        x = np.zeros((vocab_size,1))
        x[idx] = 1

        # Update "a_prev" to be "a"
        a_prev = a

        # for grading purposes
        seed += 1
        counter +=1

    if (counter == 50):
        indices.append(char_to_ix['\n'])
    return indices

########################################################################################################################
# build the character-level language model to generate text. the common procedure of the CNN loop:
# 1)Forward propagate through the RNN to compute the loss
# 2) Backward propagate through time to compute the gradients of the loss with respect to the parameters
# 3) Clip the gradients
# 4) Update the parameters using gradient descent

def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    """
    Execute one step of the optimization to train the model.
    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.

    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """
    # Forward propagate through time
    loss, cache = rnn_forward(X, Y, a_prev, parameters)
    # Backpropagate through time
    gradients, a = rnn_backward(X, Y, parameters, cache)
    # Clip your gradients between -5 (min) and 5 (max)
    gradients = clip(gradients, 5)

    # Update parameters
    parameters = update_parameters(parameters, gradients, learning_rate)
    return loss, gradients, a[len(X)-1]

def model(data, ix_to_char, char_to_ix, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27):
    """
    Trains the model and generates dinosaur names.
    Arguments:
    data -- text corpus
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration.
    vocab_size -- number of unique characters found in the text (size of the vocabulary)

    Returns:
    parameters -- learned parameters
    """
    # Retrieve n_x and n_y from vocab_size
    n_x, n_y = vocab_size, vocab_size

    # Initialize parameters
    parameters = initialize_parameters(n_a, n_x, n_y)

    # Initialize loss (this is required because we want to smooth our loss)
    loss = get_initial_loss(vocab_size, dino_names)

    # Build list of all dinosaur names (training examples).
    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    # Shuffle list of all dinosaur names
    np.random.seed(0)
    np.random.shuffle(examples)

    # Initialize the hidden state of your LSTM
    a_prev = np.zeros((n_a, 1))

    # Optimization loop
    for j in range(num_iterations):

        # Set the index `idx` (see instructions above)
        # choose only one example in each time.
        idx =j%len(examples)  # the instruction is a bit confused. I think it is just to make sure the keep feeding of data
        # when the number of iterations is larger than the length of examples.

        # Set the input X (see instructions above)
        single_example = examples[idx]
        single_example_chars = [c for c in single_example]
        single_example_ix =[char_to_ix[c] for c in single_example_chars ]
        X = [None]+single_example_ix  # the training X. e.g., [1,3,5,6,8]

        # Set the labels Y (see instructions above)
        ix_newline = char_to_ix['\n']
        Y = X[1:]+[ix_newline]  # one time step ahead of the chara in input X.  the end of name token should be added

        # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
        # Choose a learning rate of 0.01
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, 0.01)

        # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
        loss = smooth(loss, curr_loss)

        # Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
        if j % 2000 == 0:
            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
            # The number of dinosaur names to print
            seed = 0
            for name in range(dino_names):
                # Sample indices and print them
                sampled_indices = sample(parameters, char_to_ix, seed) # the parameters are trained.
                print_sample(sampled_indices, ix_to_char)
                seed += 1  # To get the same result,increment the seed by one.
            print('\n')
    return parameters

parameters = model(data, ix_to_char, char_to_ix)
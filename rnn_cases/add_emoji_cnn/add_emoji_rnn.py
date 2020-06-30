__author__ = 'Jie'
"""
Build an Emojifier via training so that it can be used to add emoji after your input sentence.
RNN algorithm (LSTM) is used.
The origincal case is from the second assignment of week 2 in course sequence model.
"""
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, LSTM, Input,Dropout,Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
np.random.seed(1)
from emo_utils import *
import emoji

def sentences_to_indices(X,word_to_index,max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding().
    :param X: array of sentences (strings), shape (m,1)
    :param word_to_index: a dictionary
    :param max_len: maximum number of words in a sentence, here set to 10
    :return:
    X_indices: array of indices corresponding to words in the sentences from X, shape(m,max_len)
    """
    m=X.shape[0]
    X_indices=np.zeros((m,max_len)) # initialize

    for i in range(m):
        sentence_words=list(X[i].lower().split())
        j=0

        for w in sentence_words:
            X_indices[i,j]=word_to_index[w]
            j+=1
    return X_indices


def pretrained_embedding_layer(word_to_vec_map,word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    vocab_len=len(word_to_index)+1 # adding 1 to fit Keras embedding (requirement)
    emb_dim=word_to_vec_map['cucumber'].shape[0]  # define dimensionality of GloVe word vector (50,here)
    emb_matrix=np.zeros((vocab_len,emb_dim))

    for word,idx in word_to_index.items():
        emb_matrix[idx,:]=word_to_vec_map[word]
    embedding_layer=Embedding(input_dim=vocab_len,output_dim=emb_dim)
    embedding_layer.build((None,)) # Build the embedding layer, it is required before setting the weights of the embedding layer.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def Emojify_V2(input_shape,word_to_vec_map,word_to_index):
    """
    to create the emojify model graph
    :param input_shape: (max_len,)
    :param word_to_vec_map: dictionary
    :param word_to_index:
    :return:
    model- a model instance in keras
    """
    sentence_indices=Input(shape=input_shape,dtype="int32")
    ## for embedding layer, we can image it as one-hot. (vocab_len,emb_dim)--> (40001,50)
    embedding_layer=pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings=embedding_layer(sentence_indices)
    X=LSTM(128,return_sequences=True)(embeddings)
    X=Dropout(0.5)(X)
    X=LSTM(128,return_sequences=False)(X)
    X=Dropout(0.5)(X)
    X=Dense(5)(X)
    X=Activation("softmax")(X)

    model=Model(inputs=sentence_indices,outputs=X)

    return model

def misLable(model):
    C=5
    # y_test_oh=np.eye(C)[Y_test.reshape(-1)]
    X_test_indices=sentences_to_indices(X_test,word_to_index,maxLen)
    pred=model.predict(X_test_indices)

    for i in range(len(X_test)):
        x=X_test_indices
        num=np.argmax(pred[i])
        if (num != Y_test[i]):
            print ('Expected emoji:'+label_to_emoji(Y_test[i])+ ' prediction: '+ X_test[i] + label_to_emoji(num).strip())

def pred_own(model,maxLen,sentence="not feeling happy"):
    x_test=np.array([sentence])
    x_test_indices=sentences_to_indices(x_test,word_to_index,maxLen)
    print (x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))



########################################################################################################################
# load data, and word embedding vector
X_train, Y_train = read_csv('train_emoji.csv')  #(132,)
X_test, Y_test = read_csv('tesss.csv')   #(56,)
print ("The training data (X_train) is ndarray with a shape of {}".format(X_train.shape))
print ("The test data(X_test) is ndarray with a shape of {}".format(X_test.shape))

#e.g.,
print ("####################################################################################")
for idx in range(3):
    print (X_train[idx])
# print (label_to_emoji(Y_train[idx]))

Y_oh_train = convert_to_one_hot(Y_train, C = 5)  # (132,5) , 5 outputs classes(5 emojis).
#The (1,5) probability vector is passed to an argmax layer, which extracts the index of the emoji with the highest probability.
Y_oh_test = convert_to_one_hot(Y_test, C = 5)
# load word vector representation; 400,001 words together
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')


def main():
    maxLen=10  # here is hard-coding to 10.
    model=Emojify_V2((maxLen,),word_to_vec_map,word_to_index)
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    X_train_indices=sentences_to_indices(X_train,word_to_index,maxLen)
    Y_train_oh=convert_to_one_hot(Y_train,C=5)

    model.fit(X_train_indices,Y_train_oh,epochs=50,batch_size=32,shuffle=True)

    # evaluate model
    X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
    Y_test_oh = convert_to_one_hot(Y_test, C = 5)
    loss, acc = model.evaluate(X_test_indices, Y_test_oh)
    print()
    print("Test accuracy = ", acc)

    # prediction my own sentence
    pred_own(model,maxLen,sentence="not feeling happy")

if __name__ == '__main__':
    main()

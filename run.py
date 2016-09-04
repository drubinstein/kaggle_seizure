#!/bin/python
#Main for running the kaggle training and test data
#Write in a list of files that you want to use for training and testing
#TODO: Use cmdline arguments for specifying training and test data
#Files will be in .mat format

import numpy as np
import scipy.io as sio
import re
import os
from random import randint

import tensorflow as tf

printing = False

def mlp(x, weights, biases, dropout):
    """ Setup the neural net """
    #Hidden layer with RELU activation
    #layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    #layer_1 = tf.nn.relu(layer_1)
    #layer_1 = tf.nn.dropout(layer_1, dropout)
    #if printing: layer_1 = tf.Print(layer_1, [layer_1], 'layer 1: ')

    #Hidden layer with RELU activation
    #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    #layer_2 = tf.nn.relu(layer_2)
    #layer_2 = tf.nn.dropout(layer_2, dropout)
    #if printing: layer_2 = tf.Print(layer_2, [layer_2], 'layer 2: ')

    #Output layer:
    #Here we combine the 3D tensors
    out_layer = tf.add(tf.matmul(x, weights['out']), biases['out'])
    if printing:  out_layer = tf.Print(out_layer, [out_layer], 'out layer: ')

    return out_layer



def main():
    dataPath = '/media/david/linux_media/kaggle/eeg/'
    """
    data = sio.loadmat(dataPath + 'train_1/1_1000_0.mat')
    print '1_1000_0'
    print 'Samples per Segment: {0}'.format(data['dataStruct']['nSamplesSegment'])
    print 'iEEG sampling rate: {0}'.format(data['dataStruct']['iEEGsamplingRate'])
    print 'channelIndices: {0}'.format(data['dataStruct']['channelIndices'])
    print 'sequence: {0}'.format(data['dataStruct']['sequence'])
    print 'shape: {0}'.format(data['dataStruct']['data'][0][0].shape)
    """

    print('Initializing neural net')
    #data_shape = data['dataStruct']['data'][0][0].shape
    learning_rate = .001
    dropout = .75
    n_samps = 240000*16#data_shape[0]
    n_nodes = 16#data_shape[1]
    n_hidden_1 = 16
    n_hidden_2 = 8
    n_out = 2 # -1 for pre, +1 for post
    X = tf.placeholder(tf.float32 )# shape = (n_samps, n_nodes)
    Y = tf.placeholder(tf.float32, [None, n_out])
    # Store layers weight & bias
    weights = {
        #'h1': tf.Variable(tf.random_normal([n_samps, n_hidden_1])),
        #'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_samps, n_out]))
    }
    biases = {
        #'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        #'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_out]))
    }

    #create multilayer perceptron
    pred = mlp(X,weights, biases, dropout)

    #define cost function
    cost = tf.pow(pred-Y,2)
    #if printing: cost = tf.Print(cost,[cost],'Sq.Err.: ')
    cost = tf.reduce_mean(cost)
    if printing: cost = tf.Print(cost,[cost],'MSE: ')
    #cost = tf.sqrt(cost)
    #if printing: cost = tf.Print(cost,[cost],'RMSE: ')
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    #open output file
    submission = open("submission.csv", "w")
    submission.write("File,Class\n")


    #4 because range only goes from n to m-1
    for dc in xrange(1,4):
        print "Running dataset: {0}".format(dc)
        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)

            print "Training "
            folder = '{0}train_{1}/'.format(dataPath, dc)
            for filename in os.listdir(folder):
                try:
                    data = sio.loadmat('{0}{1}'.format(folder, filename))
                    metadata = re.split(r'[_.]+',filename)
                    prepost = int(metadata[2]) # the class is the 3rd number
                    y = [[1,-1]] if prepost == 0 else [[0,1]]
                    x = np.reshape(data['dataStruct']['data'][0][0],(1,n_samps))
                    sess.run(optimizer, feed_dict={X: x, Y: y})
                except ValueError:
                    print '{0} is an invalid file skipping training'.format(filename)

            print "Testing"
            folder = '{0}test_{1}/'.format(dataPath, dc)
            for filename in os.listdir(folder):
                data = sio.loadmat('{0}{1}'.format(folder, filename))
                metadata = re.split(r'[_.]+',filename)
                x = np.reshape(data['dataStruct']['data'][0][0],(1,n_samps))
                p = sess.run(pred, feed_dict={X: x})
                p = 0 if p[0][0] > p[0][1] else 1

                submission.write("{0},{1}\n".format(filename, p))

    submission.close()


if __name__ == '__main__':
    main()


#!/bin/python
#Main for running the kaggle training and test data
#Write in a list of files that you want to use for training and testing
#TODO: Use cmdline arguments for specifying training and test data
#Files will be in .mat format

#This file is specifically made for RNN as the dataset is so big that we problably need a way to work off the sequence
#(think AR filter/predictive filter modelling)


import numpy as np
import scipy.io as sio
import re
import os
from random import randint

import tensorflow as tf

printing = False

# Network Parameters
n_samps = 240000*16
samps_per_step = 240
n_input = samps_per_step*16 # MNIST data input (img shape: 16 channels, 240 samples per channel)
n_steps = n_samps/n_input # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 1 # Total # classes (pre and postictal)


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)

    # Get lstm cell output
    outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

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

    # tf Graph input
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    #create multilayer perceptron
    pred = RNN(x,weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    if printing: cost = tf.Print(cost,[cost],'MSE: ')

    #open output file
    submission = open("submission.csv", "w")
    submission.write("File,Class\n")


    #4 because range only goes from n to m-1
    for dc in xrange(1,4):
        print "Running dataset: {0}".format(dc)
        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)

            print "Training"
            folder = '{0}train_{1}/'.format(dataPath, dc)

            for filename in os.listdir(folder):
                if filename=='1_45_1.mat':
                    continue
                data = sio.loadmat('{0}{1}'.format(folder, filename))
                metadata = re.split(r'[_.]+',filename)
                prepost = int(metadata[2]) # the class is the 3rd number
                #now perform the batch processing
                for step in xrange(0,n_steps):
                    batch_x = data['dataStruct']['data'][0][0][:][step*samps_per_step:step*(samps_per_step+1)-1]
                    print batch_x.shape
                    batch_x = np.reshape(batch_x, (1,n_input))
                    sess.run(optimizer, feed_dict={x: batch_x, y: prepost})

            print "Testing"
            folder = '{0}test_{1}/'.format(dataPath, dc)
            for filename in os.listdir(folder):
                data = sio.loadmat('{0}{1}'.format(folder, filename))
                metadata = re.split(r'[_.]+',filename)
                p = 0
                for step in xrange(0,n_steps):
                    batch_x = data['dataStruct']['data'][0][0][:][step*samps_per_step:step*(samps_per_step+1)-1]
                    batch_x = np.reshape(batch_x, (1,n_input))
                    p += sess.run(pred, feed_dict={x: batch_x})

                #now average p to figure out where it belongs
                p = 0 if p/n_steps < .5 else 1
                submission.write("{0},{1}\n".format(filename, p/n_steps))

    submission.close()


if __name__ == '__main__':
    main()


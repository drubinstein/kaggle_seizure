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

#Parameters
learning_rate = .001
batch_size = 128
display_step = 10

# Network Parameters
n_channels = 16
n_samps = 240000*n_channels
ch_samps_per_step = 240 #240 samples from one channel per step
samps_per_step = ch_samps_per_step*n_channels
n_input = samps_per_step
n_steps = n_samps/n_input # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 2 # Total # classes (pre and postictal)


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
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

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
    pred = RNN(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    if printing: cost = tf.Print(cost,[cost],'Cost: ')

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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
            num_files = len(os.listdir(folder))
            fidx = 0

            for filename in os.listdir(folder):
                if filename=='1_45_1.mat':
                    continue
                data = sio.loadmat('{0}{1}'.format(folder, filename))
                metadata = re.split(r'[_.]+',filename)
                prepost = int(metadata[2]) # the class is the 3rd number
                #now perform the batch processing
                #Reshape the data to get n_steps sequences of samps_per_step elements

                batch_x = np.empty([n_steps, samps_per_step])
                batch_y = [1,0] if prepost == 0 else [0,1]
                for s in xrange(n_steps):
                    batch_x[s] = np.reshape( \
                        data['dataStruct']['data'][0][0][ch_samps_per_step*s:ch_samps_per_step*(s+1)][:] \
                        , (1,samps_per_step) \
                        , order = 'F') #because this was originally matlab data....
                sess.run(optimizer, feed_dict={x: [batch_x], y: [prepost]})

                if fidx % display_step == 0:
                    #Calculate batch accuracy
                    acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                    #Calculate batch loss
                    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                    print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                                                        "{:.5f}".format(acc))

                fidx+=1
                sys.stdout.write("\r{0}/{1}".format(fidx, num_files))
                sys.stdout.flush()

            print "\nTesting"
            folder = '{0}test_{1}/'.format(dataPath, dc)
            num_files = len(os.listdir(folder))
            fidx = 0

            for filename in os.listdir(folder):
                data = sio.loadmat('{0}{1}'.format(folder, filename))
                metadata = re.split(r'[_.]+',filename)
                batch_x = np.empty([n_steps, samps_per_step])
                for s in xrange(n_steps):
                    batch_x[s] = np.reshape( \
                        data['dataStruct']['data'][0][0][ch_samps_per_step*s:ch_samps_per_step*(s+1)][:] \
                        , (1,samps_per_step) \
                        , order = 'F') #because this was originally matlab data....
                p = sess.run(pred, feed_dict={x: [batch_x]})

                submission.write("{0},{1}\n".format(filename, 0 if p[0] > p[1] else 1))
                fidx+=1
                sys.stdout.write("\r{0}/{1}".format(fidx, num_files))
                sys.stdout.flush()

            print ""

    submission.close()


if __name__ == '__main__':
    main()


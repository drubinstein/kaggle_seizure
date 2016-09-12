#!/bin/python
#Main for running the kaggle training and test data
#Write in a list of files that you want to use for training and testing
#TODO: Use cmdline arguments for specifying training and test data
#Files will be in .mat format

#This file is specifically made for RNN as the dataset is so big that we problably need a way to work off the sequence
#(think AR filter/predictive filter modelling)


import numpy as np
import scipy.io as sio
import re, os, sys
from random import randint

import tensorflow as tf
from sklearn.cross_validation import train_test_split

printing = False
logs_path = '/tmp/tensorflow_logs/example'

#Parameters
learning_rate = .001
batch_size = 128
display_step = 50

# Network Parameters
n_channels = 16
n_samps = 240000*n_channels
ch_samps_per_step = 240 #240 samples from one channel per step
samps_per_step = ch_samps_per_step*n_channels
n_input = samps_per_step
n_steps = n_samps/n_input # timesteps
n_hidden = n_channels*10 # hidden layer num of features
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
    cell = tf.nn.rnn_cell.GRUCell(n_hidden)
    outputs, states = tf.nn.rnn(cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

def BiRNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    fwd_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
    bwd_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
    outputs, _, _ = tf.nn.bidirectional_rnn(fwd_cell, bwd_cell, x, dtype=tf.float32)

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
            'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
            #'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
            }
    biases = {
            'out': tf.Variable(tf.random_normal([n_classes]))
            }

    #create multilayer perceptron
    with tf.name_scope('Model'):
        pred = BiRNN(x, weights, biases)

    # Define loss and optimizer
    with tf.name_scope('Loss'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

    with tf.name_scope('SGD'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    if printing: cost = tf.Print(cost,[cost],'Cost: ')

    # Evaluate model
    with tf.name_scope('Accuracy'):
        accuracy = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

    #disable gpu
    #config = tf.ConfigProto(device_count = {'GPU' : 0})

    #4 because range only goes from n to m-1
    init = tf.initialize_all_variables()
    print("Variables initialized")

    #setup for cross validation
    dc = 1
    folder = '{0}train_{1}/'.format(dataPath, dc)
    num_files = len(os.listdir(folder))
    #split the data into training and validation sets
    x_train, x_test = train_test_split(os.listdir(folder))

    #create monitors
    tf.scalar_summary("loss", cost)
    tf.scalar_summary("accuracy", accuracy)
    merged_summary_op = tf.merge_all_summaries()


    with tf.Session() as sess:
        #with tf.Session(config = config) as sess:
        print("Running")
        sess.run(init)

        summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

        print "Training"
        fidx = 0
        num_files = len(x_train)
        for filename in x_train:
            data = sio.loadmat('{0}{1}'.format(folder, filename))
            metadata = re.split(r'[_.]+',filename)
            prepost = int(metadata[2]) # the class is the 3rd number
            #now perform the batch processing
            #Reshape the data to get n_steps sequences of samps_per_step elements

            batch_x = np.empty([n_steps, samps_per_step])
            batch_y = [1.,0.] if prepost == 0 else [0.,1.]
            for s in xrange(n_steps):
                batch_x[s] = np.reshape( \
                        data['dataStruct']['data'][0][0][ch_samps_per_step*s:ch_samps_per_step*(s+1)][:] \
                        , (1,samps_per_step) \
                        , order = 'F') #because this was originally matlab data....
            _, c, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={x: [batch_x], y: [batch_y]})

            summary_writer.add_summary(summary, fidx)

            if fidx % display_step == 0:
                #Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: [batch_x], y: [batch_y]})
                #Calculate batch loss
                loss = sess.run(cost, feed_dict={x: [batch_x], y: [batch_y]})
                print("\nIter " + str(fidx) + ", Minibatch Loss= " + \
                        "{:.6f}".format(loss) + ", Training Accuracy= " + \
                        "{:.5f}".format(acc))

            fidx += 1
            sys.stdout.write("\r{0}/{1}".format(fidx, num_files))
            sys.stdout.flush()

        print ""
        print "Testing"
        fidx = 0
        num_files = len(x_test)
        cum_acc = 0
        for filename in x_test:
            data = sio.loadmat('{0}{1}'.format(folder, filename))
            metadata = re.split(r'[_.]+',filename)
            prepost = int(metadata[2]) # the class is the 3rd number
            #now perform the batch processing
            #Reshape the data to get n_steps sequences of samps_per_step elements

            batch_x = np.empty([n_steps, samps_per_step])
            batch_y = [1.,0.] if prepost == 0 else [0.,1.]

            #Calculate batch accuracy
            p = sess.run(pred, feed_dict={x: [batch_x]})

            p = 0 if p[0][0] > p[0][1] else 1
            if p == prepost:
                cum_acc += 1

            fidx+=1
            sys.stdout.write("\r{0}/{1}".format(fidx, num_files))
            sys.stdout.flush()

        print ""
        print "Average accuracy: {0}".format(float(cum_acc)/num_files)

        print("Run the command line:\n" \
                "--> tensorboard --logdir=/tmp/tensorflow_logs " \
                "\nThen open http://0.0.0.0:6006/ into your web browser")




if __name__ == '__main__':
    main()


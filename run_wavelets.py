#!/bin/python
#Main for running the kaggle training and test data
#Write in a list of files that you want to use for training and testing
#TODO: Use cmdline arguments for specifying training and test data
#Files will be in .mat format

import numpy as np
import scipy.io as sio
import re, os, sys
import matplotlib.pyplot as plt
from random import randint

from sklearn.linear_model import SGDClassifier
import sklearn
import pywt


printing = False


def main():
    dataPath = '/media/david/linux_media/kaggle/eeg/'
    """
    data = sio.loadmat(dataPath + 'train_1/1_2_0.mat')
    print '1_1000_0'
    print 'Samples per Segment: {0}'.format(data['dataStruct']['nSamplesSegment'])
    print 'iEEG sampling rate: {0}'.format(data['dataStruct']['iEEGsamplingRate'])
    print 'channelIndices: {0}'.format(data['dataStruct']['channelIndices'])
    print 'sequence: {0}'.format(data['dataStruct']['sequence'])
    print 'shape: {0}'.format(data['dataStruct']['data'][0][0].shape)
    x = np.transpose(data['dataStruct']['data'][0][0])
    coeffs = pywt.wavedec(x[0], 'bior3.1', level=3)
    ca, cd3, cd2, cd1 = coeffs
    f, ax = plt.subplots(4, sharex=True)
    ax[0].plot(ca)
    ax[1].plot(cd3)
    ax[2].plot(cd2)
    ax[3].plot(cd1)
    plt.show()
    #arma_mod = sm.tsa.ARIMA(x[c], (10,2,5)).fit(disp=0, start_params=)
    #print 'Channel {0} {1}'.format(c, arma_mod.summary())
    """

    #open output file
    submission = open("submission.csv", "w")
    submission.write("File,Class\n")
    n_level = 3 #3-level wavelet decomposition
    n_channels = 16
    n_clfs = n_channels*(n_level+1)


    #4 because range only goes from n to m-1
    for dc in xrange(1,4):
        print "Running dataset: {0}".format(dc)
        clfs = []
        for _ in xrange(n_clfs):
            clfs.append(SGDClassifier(loss='log'))

        print "Training "
        folder = '{0}train_{1}/'.format(dataPath, dc)
        num_files = len(os.listdir(folder))
        fidx = 0
        for filename in os.listdir(folder):
            if filename=='1_45_1.mat':
                continue
            data = sio.loadmat('{0}{1}'.format(folder, filename))
            metadata = re.split(r'[_.]+',filename)
            cl = np.array([int(metadata[2])]) # the class is the 3rd number
            x = np.transpose(data['dataStruct']['data'][0][0])
            for chan in xrange(n_channels):
                coeffs = pywt.wavedec(x[chan], 'bior3.1', level=n_level)
                clfs[chan].partial_fit(np.concatenate(coeffs).reshape(1,-1), cl, classes=[0,1])
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
            x = np.transpose(data['dataStruct']['data'][0][0])
            likelihood = 0
            for chan in xrange(n_channels):
                #TODO: Figure out way to make this arbitrary
                coeffs = pywt.wavedec(x[chan], 'bior3.1', level=n_level)
                likelihood += clfs[chan].predict(np.concatenate(coeffs).reshape(1,-1))

            p = 0 if float(likelihood)/n_channels < .5 else 1
            submission.write("{0},{1}\n".format(filename, p))

            fidx+=1
            sys.stdout.write("\r{0}/{1}".format(fidx, num_files))
            sys.stdout.flush()

        print ""

    submission.close()

if __name__ == '__main__':
    main()


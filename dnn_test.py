#import gnumpy as gpu
import numpy as np
#from dnn import nn_layer
import cPickle, gzip, numpy

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

print np.shape(train_set)
print np.shape(valid_set)
print train_set[0][0]
print np.shape(test_set)
#import gnumpy as gpu
import numpy as np
#from dnn import nn_layer
import cPickle, gzip, numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

print np.shape(train_set)
print np.shape(valid_set)
print np.shape(test_set)

print train_set[1][264]
imgplot = plt.imshow(train_set[0][264].reshape(28,28))
plt.show()
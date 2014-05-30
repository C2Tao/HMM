import gnumpy as gpu
import numpy as np
from dnn import nn_layer
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
#plt.show()

q=100

layer_one   = nn_layer(28*28,200,3,q)
layer_two   = nn_layer(200,200,3,q)
layer_three = nn_layer(200, 10,1,q)

layer_one.outward(layer_three)
#layer_two.outward(layer_three)


for k in range(100):
    for i in range(50000/q):
        layer_one.load_input(gpu.garray(train_set[0][i*q:(i+1)*q]))
        layer_one.forward()
        layer_two.forward()
        layer_three.forward()
        

        Y = np.zeros((q,10))
        #print np.shape(layer_three.s)
        #print np.shape(train_set[1][i*q:(i+1)*q])
        for j in range(q):
            Y[j][train_set[1][i*q+j]] = 1.0 
        #print Y
        error =  (layer_three.s- Y)
        #print layer_one.s.shape
        #print layer_one.r
        #print layer_one.w[:10][:10]
        #print layer_one.x[:10][:10]
        layer_three.load_output(error)

        pred = np.argmax(layer_three.s,1).reshape(q)
        answ = train_set[1][i*q:(i+1)*q].reshape(q)
        #print pred
        #print layer_one.s
        #print layer_one.w
        print float(sum(pred == answ))/q
        #print layer_one 
        #print layer_two
        #print layer_three 
        print i
        #print layer_one.w


        layer_three.backward()
        layer_two.backward()
        layer_one.backward()

        layer_one.update()
        layer_two.update()
        layer_three.update()




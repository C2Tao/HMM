import gnumpy as gpu
import numpy as np
from dnn import nn_layer
from dnn import nn_network
import cPickle, gzip, numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

def parse_corpus(tset,name):
    Y = np.zeros((len(tset[0]),10))
    for j in range(len(tset[0])):
        Y[j][tset[1][j]] = 1.0 
    corpus = {'data': tset[0], 'label': Y, 'name': name}
    return corpus


corpus_train = parse_corpus(train_set,'train')
corpus_test  = parse_corpus(test_set,'test')
corpus_valid = parse_corpus(valid_set,'valid')



nn = nn_network([28*28,10],100)
print nn.layer[0]
#print nn.layer[1]

for i in range(1000):
    nn.train(corpus_train)
    if i%10==0:
        #print nn.layer[0].w[:5][:5]
        nn.test(corpus_valid)
        nn.test(corpus_train)



'''
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

        #print layer_one.s
        #print layer_one.w
     
        #print layer_one 
        #print layer_two
        #print layer_three
        if i==1:
            pred = np.argmax(layer_three.s,1).reshape(q)
            answ = train_set[1][i*q:(i+1)*q].reshape(q)
            print float(sum(pred == answ))/q

        #print layer_one.w


        layer_three.backward()
        layer_two.backward()
        layer_one.backward()
        layer_three.update()
        layer_two.update()
        layer_one.update()
        
        

    #print i
'''


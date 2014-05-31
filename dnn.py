import gnumpy as gpu
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class nn_corpus:
    def __init__(self, train_set, valid_set, test_set):
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set  = test_set


class nn_network:
    def __init__(self, layers, batch):
        #self.input_size  = layers[0]
        #self.output_size = layers[-1]
        self.shape = layers

        self.layer = []
        for i in range(len(layers)-1):
            if i==(len(layers)-2):                
                self.layer += final_layer(layers[i],layers[i+1],batch,name=str(i)),
                #print self.layer[0]
                if i==0:
                    self.layer[0].f = lambda z: z
                    self.layer[0].g = lambda z: 1.0
            elif i==0:
                self.layer += first_layer(layers[i],layers[i+1],batch,name=str(i)),                
            else:
                self.layer += nn_layer(layers[i],layers[i+1],batch,name=str(i)),

        
        #for i in range(len(self.layer)-1):
        #    self.layer[i].add_next(self.layer[i+1])

        #self.errfun = lambda x,y: gpu.sqrt((x-y)*(x-y))
        #self.derrfun = lambda x,y: (x-y)/gpu.sqrt((x-y)*(x-y)+10**-7)
        #self.derrfun = lambda x,y: x-y
        self.batch = batch
        self.dropbatch = 0.05

    def train(self, corpus):
        blist = np.random.permutation(len(corpus['label'])/self.batch)
        self.setdrop(0.5)
        for i in blist:
            if np.random.uniform(0,1) < self.dropbatch: continue
            self.forward(corpus['data'][i*self.batch:(i+1)*self.batch])
            self.backward(self.layer[-1].s - corpus['label'][i*self.batch:(i+1)*self.batch])
            self.update()

    def error(x,y):
        return x - y

    def test(self, corpus):
        correct = 0.0
        self.setdrop(0.0)
        for i in range(len(corpus['label'])/self.batch):
            data = corpus['data'][i*self.batch:(i+1)*self.batch]
            self.forward(data)
            correct += np.sum([np.argmax(self.layer[-1].s,1) == np.argmax(corpus['label'][i*self.batch:(i+1)*self.batch],1)])
        print corpus['name']+" accuracy:",str(correct/len(corpus['label']))

    def forward(self, x):
        self.layer[0].load_input(x)
        for i in range(len(self.layer)):
            self.layer[i].forward()
            if i==len(self.layer)-1: break
            self.layer[i+1].x = self.layer[i].s
    def backward(self, e):
        self.layer[-1].load_output(e)
        for i in range(len(self.layer)-1,-1,-1):
            self.layer[i].backward()
            if i==0: break
            self.layer[i-1].d = self.layer[i].e
    def update(self):
        for i in range(len(self.layer)):
            self.layer[i].update()
    def setdrop(self, dropout):
        for i in range(len(self.layer)):
            self.layer[i].dropout = dropout

    def sdainit_layer(self,corpus):
        lcorpus = {'data': corpus['data'],'label': corpus['data']}
        for layer in self.layer:
            tempnn = sda_network(layer)
            
            #print layer.w[0]
            for i in range(1000):
                tempnn.train(lcorpus)
            
            '''
            print lcorpus['data']
            print tempnn.layer[-1].s[0].reshape(28,28)
            imgplot = plt.imshow(tempnn.layer[-1].s[0].reshape(28,28).as_numpy_array())
            plt.colorbar()
            plt.show()
            '''
            #print layer.w[0]

            lcorpus = tempnn.test(lcorpus)
            
        
class sda_network(nn_network):
    def __init__(self,some_layer):
        self.layer = [[],[]]
        self.layer[0] = some_layer
        self.layer[1] = nn_layer(some_layer.n,some_layer.m,some_layer.q)
        self.layer[1].w = self.layer[0].w.transpose(1,0)
        self.batch = some_layer.q
        self.learn = some_layer.learn
        self.dropbatch = 0.1

    def test(self,corpus):
        next_corpus= {'data': np.zeros([len(corpus['label']),self.layer[0].n])}    
        for i in range(len(corpus['label'])/self.batch):
            self.forward(corpus['data'][i*self.batch:(i+1)*self.batch])
            #print next_corpus['data'][i*self.batch:(i+1)*self.batch].shape
            next_corpus['data'][i*self.batch:(i+1)*self.batch] = self.layer[0].s.as_numpy_array()
        return {'data': next_corpus['data'],'label': next_corpus['data']}
class nn_layer:
    def __init__(self, m, n, q = 100, name=""):
        # name of layer
        self.name = name
        self.m = m # input layer size
        self.n = n # output layer size
        #self.p = p # piece group
        self.q = q # batch size

        self.dropout = 0.1 # dropout rate
        self.learn = 10**-4
        self.l2reg = (1.0-10**-9)

        # activation function
        #self.f = lambda z: 1.0/(gpu.exp(-z)+1.0)
        self.f = lambda z: z
        # deriviative of activation function
        #self.g = lambda z: self.f(z) *(1.0-self.f(z))
        self.g = lambda z: 1.0

        d = 10**-5
        # weight matrix
        self.w = gpu.garray(np.random.uniform(low=-d, high=d, size=(m, n)).astype(np.float32))
        # bias vector
        self.b = gpu.garray(np.random.uniform(low=-d, high=d, size=(n)).astype(np.float32))
        
        # input of forward propagation
        self.x = gpu.garray(np.random.uniform(low=-d, high=d, size=(q, m)).astype(np.float32))
        # output of forward propagation
        self.s = gpu.garray(np.random.uniform(low=-d, high=d, size=(q, n)).astype(np.float32))
        # input of back propagation
        self.d = gpu.garray(np.random.uniform(low=-d, high=d, size=(q, n)).astype(np.float32))
        # output of back propagation
        self.e = gpu.garray(np.random.uniform(low=-d, high=d, size=(q, m)).astype(np.float32))
        # temporary array for error
        #self.u = gpu.garray(np.random.uniform(low=-d, high=d, size=(q, n, m)).astype(np.float32))


        # novelty key ****-> set self.t.size to (n, 1, p, 1)  ---> group max        
        # mask for dropout
        self.r = gpu.garray(np.random.uniform(low=0., high=1., size=(self.m)).astype(np.float32)>self.dropout)/(1.0-self.dropout)
        #print self.r       
        # mask for piece group
        #self.t = gpu.garray(np.random.randint(low=0,  high=2,  size=(1, n, q)).astype(np.float32)) 

        # outward connections
        self.next = []
        # inward connections
        self.prev = []

    #def __repr__(self):
    #    return "{}:({},{})x{}".format(self.name,str(self.m),str(self.n),str(self.q))
        #return self.name+':'+str(self.m)+','+str(self.n)+'x'+
    
    def add_next(self,mm_layer):
        self.next += mm_layer,
        mm_layer.prev += self,

    def add_prev(self,mm_layer):
        self.prev += mm_layer,
        mm_layer.prev += self,

    def forward(self):
        if self.dropout > 0:
            self.r = gpu.garray(np.random.uniform(low=0., high=1., size=(self.m)).astype(np.float32)>self.dropout)/(1.0-self.dropout)
            self.s = gpu.dot(self.f(self.x) * self.r, self.w) + self.b
        else:
            self.s = gpu.dot(self.f(self.x), self.w) + self.b

    def backward(self):
        if self.dropout > 0:
            self.e = gpu.dot(self.d,self.w.T) * self.g(self.x) * self.r
        else:
            self.e = gpu.dot(self.d,self.w.T) * self.g(self.x) 

    def update(self):
        self.w *= self.l2reg
        if self.dropout > 0:
            self.w -= gpu.dot((self.f(self.x)* self.r).T, self.d) * self.learn / self.q
        else:
            self.w -= gpu.dot(self.f(self.x).T, self.d) * self.learn / self.q
        self.b *= self.l2reg
        self.b -= gpu.sum(self.d, 0) * self.learn
    
    def load_input(self,x):
        self.x = gpu.garray(x.reshape(self.q, self.m))
    def load_output(self,x):
        self.d = gpu.garray(x.reshape(self.q, self.n))


class final_layer(nn_layer):
    def forward(self):
        self.s = gpu.exp(gpu.dot(self.f(self.x),self.w) + self.b)
        self.s /= gpu.sum(self.s,1).reshape(self.q, 1)
        
class first_layer(nn_layer):
    def __init__(self, m, n, q , name=""):
        nn_layer.__init__(self, m, n, q, name="")
        self.f = lambda z: z
        self.g = lambda z: 1.0
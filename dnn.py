import gnumpy as gpu
import numpy as np

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
        self.dropbatch = 0.01

    def train(self, corpus):
        blist = np.random.permutation(len(corpus['label'])/self.batch)
        for i in blist:
            if np.random.uniform(0,1) < self.dropbatch: continue
            self.forward(corpus['data'][i*self.batch:(i+1)*self.batch])
            self.backward(self.layer[-1].s - corpus['label'][i*self.batch:(i+1)*self.batch])
            self.update()

    def error(x,y):
        return x - y

    def test(self, corpus):
        correct = 0.0
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
        for i in range(len(self.layer)):
            self.layer[len(self.layer)-1 - i].backward()
            if i==0: break
            self.layer[i-1].d = self.layer[i].e
    def update(self):
        for i in range(len(self.layer)):
            self.layer[i].update()



class nn_layer:
    def __init__(self, m, n, q = 100, name=""):
        # name of layer
        self.name = name
        self.m = m # input layer size
        self.n = n # output layer size
        #self.p = p # piece group
        self.q = q # batch size

        self.drop = 0.0 # dropout rate
        self.learn = 10**-2
        self.l2reg = (1.0-10**-7)

        # activation function
        self.f = lambda z: 1.0/(gpu.exp(-z)+1.0)
        #self.f = lambda z: z
        # deriviative of activation function
        self.g = lambda z: self.f(z) *(1.0-self.f(z))
        #self.g = lambda z: 1.0

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
        #self.r = gpu.garray(np.random.uniform(low=0., high=1., size=(m, 1, 1)).astype(np.float32)>self.drop)        
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
        #self.x = self.x)
        #self.s = gpu.sum(self.w * self.x + self.b,0).reshape(1, self.n, self.q)
        #print gpu.dot(self.x,self.w).shape
        #print self.b.shape
        
        #self.w /= gpu.sum(self.w*self.w,0)
        self.s = gpu.dot(self.f(self.x),self.w) + self.b
        #self.s = gpu.dot(self.x,self.w) + self.b

        #print self.s.shape
        #self.t = gpu.sum((self.w * self.x) + self.b,0).reshape(1, self.n, self.q)
        #self.s = gpu.max(self.t, 2).reshape(1, self.n, 1, self.q)
        #self.t = (self.s == self.t)

    def backward(self):
        #self.u = gpu.sum(self.w * self.t,2).reshape(self.m, self.n, 1, self.q)
        self.e = gpu.dot(self.d,self.w.T) * self.g(self.x) 
        #self.e = gpu.dot(self.d,self.w.T) 

    def update(self):
        #self.w /= gpu.sqrt(gpu.sum(self.w*self.w,0))
        self.w *= self.l2reg
        self.w -= gpu.dot(self.f(self.x).T, self.d)  / self.q * self.learn
        #self.w -= gpu.dot(self.x.T, self.d)  / self.q * self.learn
        self.b *= self.l2reg
        self.b -= gpu.sum(self.d, 0) /self.q * self.learn
    
    def load_input(self,x):
        self.x = gpu.garray(x.reshape(self.q, self.m))
    def load_output(self,x):
        self.d = gpu.garray(x.reshape(self.q, self.n))

class final_layer(nn_layer):
    def forward(self):
        self.s = gpu.exp(gpu.dot(self.x,self.w) + self.b)
        self.s /= gpu.sum(self.s,1).reshape(self.q, 1)
        
class first_layer(nn_layer):
    def __init__(self, m, n, q = 100, name=""):
        nn_layer.__init__(self, m, n, q = 100, name="")
        self.f = lambda z: z
        self.g = lambda z: 1.0
import gnumpy as gpu
import numpy as np



class nn_layer:
    def __init__(self, m, n, p = 1, q = 100, name=""):
        # name of layer
        self.name = name
        self.m = m # input layer size
        self.n = n # output layer size
        self.p = p # piece group
        self.q = q # batch size

        self.drop = 0.0 # dropout rate
        self.learn = 0.01
        self.l2reg = 1.0

        # activation function
        self.f = lambda z: gpu.logistic(z)
        #self.f = lambda z: z
        # deriviative of activation function
        self.g = lambda z: (1.0-z.logistic())*z.logistic()
        #self.g = lambda z: 1.0

        d = 0.0001
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
        self.r = gpu.garray(np.random.uniform(low=0., high=1., size=(m, 1, 1)).astype(np.float32)>self.drop)        
        # mask for piece group
        self.t = gpu.garray(np.random.randint(low=0,  high=2,  size=(1, n, q)).astype(np.float32)) 

        # outward connections
        self.o = []
        # inward connections
        self.i = []
        
    def __repr__(self):
        #print "layer shape:",np.shape(self.w)
        if self.o:
            #print "previous layer: "
            for parent in self.o:
                print parent
        return "{}:({},{})x({},{})".format(self.name,str(self.m),str(self.n),str(self.p),str(self.q))
        #return self.name+':'+str(self.m)+','+str(self.n)+'x'+
    def outward(self,mm_layer):
        self.o += mm_layer,
        mm_layer.i += self,

    def inward(self,mm_layer):
        self.i += mm_layer,
        mm_layer.o += self,

    def forward(self):
        #self.x = self.x)
        #self.s = gpu.sum(self.w * self.x + self.b,0).reshape(1, self.n, self.q)
        #print gpu.dot(self.x,self.w).shape
        #print self.b.shape
        self.s = gpu.dot(self.f(self.x),self.w) + self.b

        #print self.s.shape
        #self.t = gpu.sum((self.w * self.x) + self.b,0).reshape(1, self.n, self.q)
        #self.s = gpu.max(self.t, 2).reshape(1, self.n, 1, self.q)
        #self.t = (self.s == self.t)
        for next_layer in self.o:
            if not self.o:break
            if self == next_layer.i[0]:
                #print self.s.shape
                next_layer.x = self.s
                #print self.s.shape
            else:
                break 
                #next_layer.x += self.s.transpose(1,0,2,3)

    def backward(self):
        #self.u = gpu.sum(self.w * self.t,2).reshape(self.m, self.n, 1, self.q)
        self.e = gpu.dot(self.d,self.w.T) * self.g(self.x) 
        for prev_layer in self.i:
            if not self.i:break
            if self == prev_layer.o[0]:
                prev_layer.d = self.e
            else:
                break
                #prev_layer.d += self.e.transpose(1,0,2,3)

    def update(self):
        self.w *= self.l2reg
        self.w -= gpu.dot(self.f(self.x).T, self.d)  / self.q * self.learn
        self.b *= self.l2reg
        self.b -= gpu.sum(self.d, 0) /self.q * self.learn
    
    def load_input(self,x):
        self.x = gpu.garray(x.reshape(self.q, self.m))
    def load_output(self,x):
        self.d = gpu.garray(x.reshape(self.q, self.n))


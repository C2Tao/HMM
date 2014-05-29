import gnumpy as gpu
import numpy as np



class nn_layer:
    def __init__(self, m, n, p = 1, q = 100, name=""):
        self.name = name
        self.m = m # input layer size
        self.n = n # output layer size
        self.p = p # piece group
        self.q = q # batch size
        
        self.w = [gpu.garray(np.random.uniform(low=0., high=1., size=(n, m)).astype(np.float32)) for _ in range(p)]
        self.b = [gpu.garray(np.random.uniform(low=0., high=1., size=(n, q)).astype(np.float32)) for _ in range(p)]
        self.x = gpu.garray(np.random.uniform(low=0., high=1., size=(m, q)).astype(np.float32))
        self.s = gpu.garray(np.random.uniform(low=0., high=1., size=(n, q)).astype(np.float32))
        
        self.d = 

        self.f = lambda g: g

        self.i = []
        self.o = []
    def __repr__(self):
        #print "layer shape:",np.shape(self.w)
        if self.o:
            #print "previous layer: "
            for parent in self.o:
                print parent
        return "{}:({},{})x{}".format(self.name,str(self.m),str(self.n),str(self.p))
        #return self.name+':'+str(self.m)+','+str(self.n)+'x'+
    def outward(self,mm_layer):
        self.o += mm_layer,
        mm_layer.i += self,

    def inward(self,mm_layer):
        self.i += mm_layer,
        mm_layer.o += self,
        
    def forward(self):
        self.s = self.f(max([(self.w[i]).dot(self.x) + self.b[i] for i in range(self.p)]))
        for next_layer in self.o:
            next_layer.x = gpu.garray(self.s,copy=False)
    
    def backward(self):
        self.d = self.f(max([(self.w[i]).dot(self.x) + self.b[i] for i in range(self.p)]))
    
    def update(self):
        self.d


a = nn_layer(2,3,2,5,name = '1')
b = nn_layer(9,7,2,5,name = '2')
c = nn_layer(5,6,2,5,name = '3')

a.outward(b)
b.outward(c)
print a,np.shape(b.x),c
a.forward()
print a,np.shape(b.x),c
#print ga.as_numpy_array(dtype=np.float32) - a
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

        self.drop = 0.1 # dropout rate
        
        # activation function
        self.f = lambda z: z
        # deriviative of activation function
        self.g = lambda z: 1

        # weight matrix
        self.w = gpu.garray(np.random.uniform(low=0., high=1., size=(m, n, p, 1)).astype(np.float32))
        # bias vector
        self.b = gpu.garray(np.random.uniform(low=0., high=1., size=(1, n, p, 1)).astype(np.float32))
        
        # input of forward propagation
        self.x = gpu.garray(np.random.uniform(low=0., high=1., size=(m, 1, 1, q)).astype(np.float32))
        # output of forward propagation
        self.s = gpu.garray(np.random.uniform(low=0., high=1., size=(1, n, 1, q)).astype(np.float32))
        # input of back propagation
        self.d = gpu.garray(np.random.uniform(low=0., high=1., size=(1, n, 1, q)).astype(np.float32))
        # output of back propagation
        self.e = gpu.garray(np.random.uniform(low=0., high=1., size=(m, 1, 1, q)).astype(np.float32))
        # temporary array for error
        self.u = gpu.garray(np.random.uniform(low=0., high=1., size=(m, n, p, 1)).astype(np.float32))

        
        # novelty key ****-> set self.t.size to (n, 1, p, 1)  ---> group max
        # mask for piece group
        self.t = gpu.garray(np.random.randint(low=0,  high=2,  size=(1, n, p, q)).astype(np.float32)) 
        # mask for dropout
        self.r = gpu.garray(np.random.uniform(low=0., high=1., size=(m, 1, 1, 1)).astype(np.float32)>self.drop)

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
        self.x = self.f(self.x)
        self.t = gpu.sum((self.w * self.r * self.x)/(1.0-self.drop) + self.b,0).reshape(1, self.n, self.p, self.q)
        self.s = gpu.max(self.t, 2).reshape(1, self.n, 1, self.q)
        self.t = (self.s == self.t)
        for next_layer in self.o:
            if not self.o:break
            if self == next_layer.i[0]:
                next_layer.x = self.s.reshape(self.n, 1, 1, self.q)
            else: 
                next_layer.x += self.s.reshape(self.n, 1, 1, self.q)

    def backward(self):
        self.e = gpu.sum(self.w * self.d * self.t , 1).reshape(self.m, 1, self.p, self.q) 
        #self.z = self.w * self.d * self.g(self.x) * self.t * self.r
        #self.e = gpu.sum(self.z, 1).reshape(self.m, 1, self.p, self.q)
        self.e = gpu.sum(self.e ,2).reshape(self.m, 1, 1, self.q) * self.r * self.g(self.x) 
        for prev_layer in self.i:
            if not self.i:break
            if self == prev_layer.o[0]:
                prev_layer.d = self.e.reshape(1, self.m, 1, self.q)
            else:
                prev_layer.d += self.e.reshape(1, self.m, 1, self.q)
    
    def update(self):
        self.w += gpu.sum(self.d * self.t * self.f(self.x),3).reshape(self.m, self.n, self.p, 1)  * self.r 
        self.b += gpu.sum(self.d * self.t,3).reshape(1, self.n, self.p, 1) 
        

a = nn_layer(100,100,3,10,name = '1')
b = nn_layer(100,100,3,10,name = '2')
c = nn_layer(100,100,2,5,name = '3')

a.outward(b)
b.outward(c)
print a,np.shape(b.x),c

for i in range(10):
    print i
    b.forward()
    b.backward()
    b.update()

print a,np.shape(b.x),c


A =  np.random.randint(0,5,(2,1,3))
print A
print A.reshape(1,2,3)
#print ga.as_numpy_array(dtype=np.float32) - a


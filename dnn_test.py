import gnumpy as gpu
import numpy as np


n = 2
m = 3


a = np.random.uniform(low=0., high=1., size=(n, m)).astype(np.float32)
b = np.random.uniform(low=0., high=1., size=(n, m)).astype(np.float32)
c = np.random.uniform(low=0., high=1., size=(n, m)).astype(np.float32)

ga = gpu.garray(a)
gb = gpu.garray(b)
gc = gpu.garray(c)

print ga.T.dot(gb)
print a.dot(b.T)

print ga
print gb
print ga*gb
print gc
print ga-gb


print max(ga,gb,gc)
#print ga.as_numpy_array(dtype=np.float32) - a
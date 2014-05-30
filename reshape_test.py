import gnumpy as gpu
import numpy as np


A = gpu.garray(np.random.randint(0,1000,[2,3,2]))

print A
print A.shape

B = A.reshape(3,2,2)
D = A.transpose(1,0,2)
C = A.reshape(2,1,1,3,1,1,2)
E = A.reshape(1,1,2,3,1,1,2)
i,j,k = 1,2,1
print A[i][j][k],B[j][i][k]
assert(A[i][j][k]==B[j][i][k])
assert(A[i][j][k]==D[j][i][k])
assert(A[i][j][k]==C[i][0][0][j][0][0][k])
assert(A[i][j][k]==E[0][0][i][j][0][0][k])


A = np.random.randint(0,1909,[2,10])
b = np.random.randint(0,1909,[10])
A+=b


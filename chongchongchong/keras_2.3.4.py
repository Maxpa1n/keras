import numpy as np

x = np.array([[0., 1.], [2., 3.], [4., 5.]])
print(x)
print(x.shape)

y = x.reshape((6, 1))
print(y)

z = x.reshape((2, 3))
print(z)

tran_x = np.transpose(x)
print(tran_x)
print(tran_x.shape)

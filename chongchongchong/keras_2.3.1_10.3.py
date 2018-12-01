import numpy as np


def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x


def naive_add(x,y):
    assert len(x.shape) == 2
    assert x.shape == y.shape

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x


def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert len(x.shape[1]) ==len(y.shape[0])

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x


z1 = np.array([[16, 17, 123], [18, 19, 12]])
z = np.array([[12, 13, 22], [14, 15, 123]])
print(z.shape)
print(z.ndim)

output_re = naive_relu(z)
print(output_re)

output_add = naive_add(z, z1)
print(output_add)

print(z.shape[0])

import numpy as np

x = np.array(
    [[[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]], [[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]]])

y = x.mean(axis=(0, 2, 3), keepdims=True)

y_ = x.mean(axis=(0, 2, 3))

print(y)
print(y_)

y__=x.mean()

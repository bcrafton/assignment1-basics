
import numpy as np

x = np.array([1, 2])
w = np.array([
               [1, 2], 
               [3, 4]
             ])
y = w @ x
print (y)

yy = np.sum(w * x, axis=1)
assert np.all(np.isclose(y, yy))

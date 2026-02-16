
import numpy as np

#####################

# fw = np.require(np.memmap('test.dat', dtype=np.uint16, mode='w+', shape=100), requirements=['O'])
fw = np.memmap('test.dat', dtype=np.uint16, mode='w+', shape=100)

for i in range(50):
  fw[i] = i

fw.resize(50)
del fw

#####################

fr = np.memmap('test.dat', dtype=np.uint16, mode='r+', shape=100)

for i in range(100):
  print (fr[i])

#####################

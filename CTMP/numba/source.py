from numba import jit
import numpy as np
import time


#@jit
#@jit(nopython=True, parallel=True)
@jit(nogil=True)
def f():
    # A somewhat trivial example
    A_arr = np.random.randn(10 ** 3)
    B_arr = np.random.randn(10 ** 3)
    for i in range(10000000):
        res = np.dot(A_arr, B_arr)
    return res

s = time.time()
z = f()
print("sec", time.time() - s)
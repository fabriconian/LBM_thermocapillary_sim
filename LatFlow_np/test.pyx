import cython
cimport cython

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def sum(np.ndarray[DTYPE_t, ndim=2] a, np.ndarray[DTYPE_t, ndim=2] b):

    cdef unsigned int rowa = a.shape[0]
    cdef unsigned int cola = a.shape[1]

    cdef unsigned int ia, ja,
    cdef np.ndarray[DTYPE_t, ndim=2] out = np.empty((rowa, cola))

    for ia in range(rowa):
        for ja in range(cola):
                out[ia, ja] += a[ia, ja] + b[ia, ja]

    return out

import cython
cimport cython

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def convolve(np.ndarray[DTYPE_t, ndim=4] x, np.ndarray[DTYPE_t, ndim=4] k):

    # cdef unsigned int h_x = x.shape[1]
    # cdef unsigned int w_x = x.shape[2]

    cdef unsigned int h_filter = k.shape[0]
    cdef unsigned int w_filter= k.shape[1]
    cdef unsigned int c = k.shape[3]
    cdef unsigned int c_out = k.shape[3]
    cdef unsigned int h_out = x.shape[1] - h_filter + 1
    cdef unsigned int w_out = x.shape[2] - w_filter + 1

    cdef int i_x, j_x, i_f, j_f, c_x, c_f
    cdef np.ndarray[DTYPE_t, ndim=4] out = np.empty((x.shape[0], h_out, w_out, c_out))
    cdef DTYPE_t temp = 0.0

    for i_x in range(h_out):
        for j_x in range(w_out):
            for c_f in range(c_out):
                for c_x in range(c):
                    for i_f in range(-(int(h_filter/2)),int(h_filter / 2)+1):
                        for j_f in range(-(int(w_filter/2)),int(w_filter / 2)+1):
                            temp+= x[0,i_x+i_f,j_x+j_f,c_x]*k[i_f + int(h_filter / 2), j_f + int(w_filter / 2),c_x,c_f]

                out[0,i_x, j_x, c_f] = temp
                temp = 0.0

    return out

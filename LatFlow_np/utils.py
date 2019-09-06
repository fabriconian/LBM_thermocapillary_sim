import numpy as np
import LatFlow_np.cutils as cu
# import LatFlow_np.cutils_fix as cu

def pad_mobius(f):
  f_mobius = f
  f_mobius = np.concatenate([f_mobius[:,-1:],   f_mobius, f_mobius[:,0:1]],axis=1)
  f_mobius = np.concatenate([f_mobius[:,:,-1:], f_mobius, f_mobius[:,:,0:1]],axis=2)
  return f_mobius


def simple_conv(x, k, pad=0):

    if pad!= 0:
        y = pad_mobius(x)
    else:
        y = x
    res = cu.convolve(y, k)
    return res



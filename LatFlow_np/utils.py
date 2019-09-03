import numpy as np
import LatFlow_np.cutils as cu
# import LatFlow_np.cutils_fix as cu

def pad_mobius(f):
  f_mobius = f
  f_mobius = np.concatenate([f_mobius[:,-1:],   f_mobius, f_mobius[:,0:1]],axis=1)
  f_mobius = np.concatenate([f_mobius[:,:,-1:], f_mobius, f_mobius[:,:,0:1]],axis=2)
  return f_mobius


def simple_conv(x, k,pad=0):

    if pad!= 0:
        y = pad_mobius(x)
    else:
        y = x
    res = cu.convolve(y, k)
    return res


def conv_forward(X, W, b=0, stride=1, padding=0):
    # cache = W, b, stride, padding
    h_filter, w_filter, d_filters, n_filters = W.shape
    n_x, h_x, w_x, d_x, = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)

    out = np.dot(W_col,X_col) + b
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 1, 2, 0)

    # cache = (X, W, b, stride, padding, X_col)

    return out


def get_im2col_indices(x_shape, field_height, field_width, padding=0, stride=1):
  # First figure out what the size of the output should be
  N, H, W, C = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_height) % stride == 0
  out_height = int((H + 2 * padding - field_height) / stride + 1)
  out_width = int((W + 2 * padding - field_width) / stride + 1)

  i0 = np.repeat(np.arange(field_height), field_width)
  i0 = np.tile(i0, C)
  i1 = stride * np.repeat(np.arange(int(out_height)), int(out_width))
  j0 = np.tile(np.arange(field_width), field_height * C)
  j1 = stride * np.tile(np.arange(int(out_width)), int(out_height))
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)

  k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

  return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
  """ An implementation of im2col based on some fancy indexing """
  # Zero-pad the input
  p = padding
  x_padded = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), mode='edge')

  k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                               stride)

  cols = x_padded[:, i, j, k]
  C = x.shape[3]
  cols = cols.transpose(2, 1, 0).reshape(field_height * field_width * C, -1)
  return cols



import numpy as np


def simple_conv(x, k,pad=0):
  """A simplified 2D or 3D convolution operation"""
  if   len(x.shape) == 4:
    y =conv_forward(x.transpose(0, 3, 1, 2), k.transpose(3, 2, 0, 1),    padding=pad)
  return y.transpose(0,2,3,1)


def conv_forward(X, W, b=0, stride=1, padding=1):
  # cache = W, b, stride, padding
  n_filters, d_filter, h_filter, w_filter = W.shape
  n_x, d_x, h_x, w_x = X.shape
  h_out = (h_x - h_filter + 2 * padding) / stride + 1
  w_out = (w_x - w_filter + 2 * padding) / stride + 1

  if not h_out.is_integer() or not w_out.is_integer():
    raise Exception('Invalid output dimension!')

  h_out, w_out = int(h_out), int(w_out)

  X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
  W_col = W.reshape(n_filters, -1)

  out = W_col @ X_col
  out = out.reshape(n_filters, h_out, w_out, n_x)
  out = out.transpose(3, 0, 1, 2)


  # cache = (X, W, b, stride, padding, X_col)

  return out


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_height) % stride == 0
  out_height = int((H + 2 * padding - field_height) / stride + 1)
  out_width =int( (W + 2 * padding - field_width) / stride + 1)

  i0 = np.repeat(np.arange(field_height), field_width)
  i0 = np.tile(i0, C)
  i1 = stride * np.repeat(np.arange(out_height), out_width)
  j0 = np.tile(np.arange(field_width), field_height * C)
  j1 = stride * np.tile(np.arange(out_width), out_height)
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)

  k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

  return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
  """ An implementation of im2col based on some fancy indexing """
  # Zero-pad the input
  p = padding
  x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='edge')

  k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                               stride)

  cols = x_padded[:, k, i, j]
  C = x.shape[1]
  cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
  return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
  """ An implementation of col2im based on fancy indexing and np.add.at """
  N, C, H, W = x_shape
  H_padded, W_padded = H + 2 * padding, W + 2 * padding
  x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
  k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                               stride)
  cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
  cols_reshaped = cols_reshaped.transpose(2, 0, 1)
  np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
  if padding == 0:
    return x_padded
  return x_padded[:, :, padding:-padding, padding:-padding]


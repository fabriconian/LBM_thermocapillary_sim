import numpy as np
import tensorflow as tf
from skimage import feature

def simple_conv(x, k):
  """A simplified 2D or 3D convolution operation"""
  if   len(x.get_shape()) == 4:
    y = tf.nn.conv2d(x, k, [1, 1, 1, 1],    padding='VALID')
  elif len(x.get_shape()) == 5:
    print(x.get_shape())
    y = tf.nn.conv3d(x, k, [1, 1, 1, 1, 1], padding='VALID')
    print(y.get_shape())
    #y = y[:,:-1,0:-1,0:-1,:]
    #y = y[:,:-1,0:-1,0:-1,:]
  return y

def pad_mobius(f):
  f_mobius = f
  f_mobius = tf.concat(axis=1, values=[f_mobius[:,-1:],   f_mobius, f_mobius[:,0:1]]) 
  f_mobius = tf.concat(axis=2, values=[f_mobius[:,:,-1:], f_mobius, f_mobius[:,:,0:1]])
  if len(f.get_shape()) == 5:
    f_mobius = tf.concat(axis=3, values=[f_mobius[:,:,:,-1:], f_mobius, f_mobius[:,:,:,0:1]])
  return f_mobius


def grad(field):
  '''

  :param field: 2D tensor from which we want to calculate gradient
  :return:3D array -  gradient of the field
  '''
  fxp = tf.concat(axis=1, values=[field[:, 1:], field[:, -1:]])
  fxm = tf.concat(axis=1, values=[field[:, 0:1], -1 * field[:, 0:-1]])
  fx = tf.expand_dims(tf.concat(axis=1,
                  values=[fxp[:, 0:1] - fxm[:, 0:1],
                          (fxp[:, 1:-1] + fxm[:, 1:-1]) / 2, fxp[:, -1:] + fxm[:, -1:]]),-1)
  fyp = tf.concat(axis=0, values=[field[1:,:], field[ -1:,:]])
  fym = tf.concat(axis=0, values=[field[0:1,:], -1 * field[ 0:-1,:]])
  fy = tf.expand_dims(tf.concat(axis=0,
                 values=[fyp[0:1,:] - fym[0:1,:],
                         (fyp[1:-1,:] + fym[1:-1,:]) / 2, fyp[-1:,:] + fym[ -1:,:]]),-1)
  return tf.concat(axis=2,values=[fy,fx])

def create_polygon(shape, vertices):
    """
    Creates np.array with dimensions defined by shape
    Fills polygon defined by vertices with ones, all other values zero"""
    base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros

    fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill

    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k-1], vertices[k], base_array)], axis=0)

    base_array[fill] = 1.0

    return base_array


def create_polygon_vek(shape, vertices):
  """
  Creates np.array with dimensions defined by shape
  Fills polygon defined by vertices with ones, all other values zero"""
  base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros


  p2 = vertices.copy()
  p1 = np.empty_like(vertices)
  p1[1:] = vertices[0:-1]
  p1[0] = vertices[-1]

  fill = np.all( check_vector(p1, p2, base_array),axis=0)
  base_array[fill] = 1.0

  return base_array

def check(p1, p2, base_array):
    """
    Uses the line defined by p1 and p2 to check array of
    input indices against interpolated value

    Returns boolean array, with True inside and False outside of shape
    """

    idxs = np.indices(base_array.shape)  # Create 3D array of indices

    p1 = p1.astype(float)
    p2 = p2.astype(float)

    # Calculate max column idx for each row idx based on interpolated line between two points
    max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]
    sign = np.sign(p2[0] - p1[0])
    return idxs[1] * sign <= max_col_idx * sign

def check_vector_tf(p1, p2, base_array):
    """
    Uses the line defined by p1 and p2 to check array of
    input indices against interpolated value

    Returns boolean array, with True inside and False outside of shape
    """

    idxs = np.indices(base_array.shape)  # Create 3D array of indices
    idxs = np.concatenate([p1.shape[0] * [idxs]], axis=0)
    p1x = (np.ones_like(idxs[:, 0]).transpose(2, 1, 0) * p1[:, 0]).transpose(2, 1, 0)
    p1y = (np.ones_like(idxs[:, 0]).transpose(2, 1, 0) * p1[:, 1]).transpose(2, 1, 0)
    p2x = (np.ones_like(idxs[:, 0]).transpose(2, 1, 0) * p2[:, 0]).transpose(2, 1, 0)
    p2y = (np.ones_like(idxs[:, 0]).transpose(2, 1, 0) * p2[:, 1]).transpose(2, 1, 0)

    # Calculate max column idx for each row idx based on interpolated line between two points
    max_col_idx = (idxs[:, 0] - p1x) / (p2x - p1x) * (p2y - p1y) + p1y
    sign = np.sign(p2x - p1x)
    return idxs[:, 1] * sign <= max_col_idx * sign

def check_vector(p1, p2, base_array):
    """
    Uses the line defined by p1 and p2 to check array of
    input indices against interpolated value

    Returns boolean array, with True inside and False outside of shape
    """

    idxs = np.indices(base_array.shape)  # Create 3D array of indices
    idxs = np.concatenate([p1.shape[0] * [idxs]], axis=0)
    p1x = (np.ones_like(idxs[:, 0]).transpose(2, 1, 0) * p1[:, 0]).transpose(2, 1, 0)
    p1y = (np.ones_like(idxs[:, 0]).transpose(2, 1, 0) * p1[:, 1]).transpose(2, 1, 0)
    p2x = (np.ones_like(idxs[:, 0]).transpose(2, 1, 0) * p2[:, 0]).transpose(2, 1, 0)
    p2y = (np.ones_like(idxs[:, 0]).transpose(2, 1, 0) * p2[:, 1]).transpose(2, 1, 0)

    # Calculate max column idx for each row idx based on interpolated line between two points
    max_col_idx = (idxs[:, 0] - p1x) / (p2x - p1x) * (p2y - p1y) + p1y
    sign = np.sign(p2x - p1x)
    return idxs[:, 1] * sign <= max_col_idx * sign

def create_polygon_tf(shape, vertices):
  """
  Creates np.array with dimensions defined by shape
  Fills polygon defined by vertices with ones, all other values zero"""
  base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros

  p2 = vertices
  p1 = tf.concatenate(p2[:-1],p2[1:-1])

  fill = tf.reduce_all(check_vector_tf(p1, p2, shape),axis=0)
  res = tf.dtype.cast(fill,tf.float32)

  return res


def get_edge(field,sigma = 5):
    return feature.canny(field,sigma)
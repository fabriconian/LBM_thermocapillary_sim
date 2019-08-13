
import tensorflow as tf

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
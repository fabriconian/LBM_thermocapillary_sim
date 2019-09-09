
import time

import numpy as np
import tensorflow as tf
import math
import cv2

from matplotlib import pyplot as plt

import LatFlow.DomainTherm as dom
from   LatFlow.utils  import *

# video init
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter()

shape = [40, 40]
success = video.open('some_videos/rb4_T3.mov', fourcc, 30, (shape[1], shape[0]), True)

FLAGS = tf.app.flags.FLAGS


def make_lid_boundaryt(shape):
  boundary = np.zeros((1, shape[0], shape[1], 1), dtype=np.float32)
  boundary[:,:,0,:] = 1.0
  boundary[:,shape[0]-1,:,:] = 1.0
  boundary[:,:,shape[1]-1,:] = 1.0
  boundary[:, 0,:, :] = 1.0
  return boundary

def make_lid_boundary_T(shape, Tup=0.4, Tdown=0.6):

  #boundaryy upp
  boundary = np.zeros((1, shape[0], shape[1], 1), dtype=np.float32)
  boundary[:,0,:,:] = 1.0
  bup = dom.BoundaryT(boundary=boundary,value=Tup,type='CT',n=np.array([-1,0,0]))

  # boundaryy down
  boundary = np.zeros((1, shape[0], shape[1], 1), dtype=np.float32)
  boundary[:, -1, :, :] = 1.0
  bdown = dom.BoundaryT(boundary=boundary, value=Tdown, type='CT',n=np.array([1,0,0]))

  # boundaryy LEFT
  boundary = np.zeros((1, shape[0], shape[1], 1), dtype=np.float32)
  boundary[:, :, 0, :] = 1.0
  bl = dom.BoundaryT(boundary=boundary,  type='ZF',n=np.array([0,0,0]))

  # boundaryy LEFT
  boundary = np.zeros((1, shape[0], shape[1], 1), dtype=np.float32)
  boundary[:, :, -1, :] = 1.0
  br = dom.BoundaryT(boundary=boundary, type='ZF')

  return [bup,br,bdown,bl]


def lid_init_step(domain, value=0.08):
  vel_dir = tf.zeros_like(domain.Vel[0][:,:,:,0:1])
  vel = tf.concat([vel_dir, vel_dir, vel_dir], axis=3)
  vel_dot_vel = tf.expand_dims(tf.reduce_sum(vel * vel, axis=3), axis=3)
  vel_dot_c = tf.reduce_sum(tf.expand_dims(vel, axis=3) * tf.reshape(domain.C, [1,1,1,domain.Nneigh,3]), axis=4)
  feq = tf.reshape(domain.W, [1,1,1,domain.Nneigh]) * (1.0 + 3.0*vel_dot_c/domain.Cs + 4.5*vel_dot_c*vel_dot_c/(domain.Cs*domain.Cs) - 1.5*vel_dot_vel/(domain.Cs*domain.Cs))

  vel = vel * (1.0 - domain.boundary)
  rho = (1.0 - domain.boundary)
  force = tf.zeros_like(vel)


  f_step = domain.F[0].assign(feq)
  rho_step = domain.Rho[0].assign(rho)
  vel_step = domain.Vel[0].assign(vel)
  force_step = domain.BForce[0].assign(force)

  initialize_step = tf.group(*[f_step, rho_step, vel_step, force_step])
  return initialize_step

def lid_init_step_T(domain, value=0.5):
  vel_dir = tf.zeros_like(domain.Vel[0][:, :, :, 0:1])
  vel = tf.concat([vel_dir, vel_dir, vel_dir], axis=3)
  vel_dot_vel = tf.expand_dims(tf.reduce_sum(vel * vel, axis=3), axis=3)
  vel_dot_c = tf.reduce_sum(tf.expand_dims(vel, axis=3) * tf.reshape(domain.C, [1, 1, 1, domain.Nneigh, 3]), axis=4)

  geq = tf.reshape(domain.W, [1,1,1,domain.Nneigh])*value * (1.0 + 3.0*vel_dot_c/domain.Cs + 4.5*vel_dot_c*vel_dot_c/(domain.Cs*domain.Cs) - 1.5*vel_dot_vel/(domain.Cs*domain.Cs))

  T = tf.ones_like(1.0-domain.boundary)*value

  stream_stepg = domain.g[0].assign(geq)
  stream_stepg_temp = domain.gtemp[0].assign(geq)
  T_step = domain.T[0].assign(T)
  step = tf.group(*[stream_stepg,stream_stepg_temp, T_step])
  return step

def lid_save_T(domain, sess):
  frame = sess.run(domain.T[0])
  frame = np.sqrt(np.square(frame[0, :, :, 0]))

  # print('\n', np.max(frame), '\t', np.min(frame))
  frame = np.uint8(255 * (frame-np.min(frame)) / (np.max(frame)-np.min(frame)))
  frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT, 2)
  video.write(frame)

def lid_save_vel(domain, sess):
  frame = sess.run(domain.Vel[0])
  frame = np.sqrt(np.square(frame[0,:,:,0])+ np.square(frame[0,:,:,1]) + np.square(frame[0,:,:,2]))
  # print('\n',np.max(frame),'\t',np.min(frame))


  frame = np.uint8(255 * frame/np.max(frame))
  frame = cv2.applyColorMap(frame, cv2.COLORMAP_RAINBOW, 2)

  video.write(frame)

def ForceUpdate(domain,Tref):
    g = 9.8 * domain.dt_real * domain.dt_real / domain.dx_real
    # Tref = domain.Tref
    force = tf.concat(values=[tf.zeros_like(domain.T[0]), ( -domain.beta * g * (domain.T[0] -  Tref)),
                              tf.zeros_like(domain.T[0])], axis=3)
    update = domain.BForce[0].assign(force)
    return update

def Create_objects(shape=None,n=10,r=None,R0=np.array([10,10]),V0=np.array([0,0])):
    if r is None:
        r = shape[0]/10
    phi = np.linspace(0.,(2*np.pi*(n-1)/float(n)),num=n)
    vertx = r * np.cos(phi)
    verty = - r * np.sin(phi)
    vertices = np.ones([verty.shape[0],2])
    vertices[:,0] = vertx
    vertices[:,1] = verty
    objects = [dom.Object(
        vertices=vertices,
        rc=R0,
        vc=V0

    )]
    return objects

def run():
  # constants
  input_vel = 0.001
  nu = 1.004E-4
  K = 0.143E-5
  dx = 7.0E-4
  dt = 1E-4
  beta= 3.00
  Tf=2.0
  Tref = 1.1
  Ndim = shape
  boundary_T = make_lid_boundary_T(shape=Ndim,Tup=0.2, Tdown=0.4)
  boundaryt2 = make_lid_boundaryt(shape=Ndim)
  objects = Create_objects(Ndim,n=80,r=10)

  # domain
  domain = dom.Domain(method="D2Q9",
                      Ndim=Ndim,
                      tauf= 0.53,
                      taug=0.9,
                      objects=objects,
                      boundary=boundaryt2,
                      dt=dt,
                      dx=dx,
                      boundary_T= boundary_T,
                      les=False)

  # make lattice state, boundary and input velocity

  initialize_step = lambda domain:lid_init_step(domain, value=0.08)
  initialize_step_T = lambda domain: lid_init_step_T(domain, value=Tref)
  force_update =  lambda domain:ForceUpdate(domain,Tref=Tref)

  # init things
  init = tf.global_variables_initializer()
  # start sess
  sess = tf.Session()

  # init variables
  sess.run(init)

  # run steps
  domain.Solve(sess, Tf, initialize_step, initialize_step_T, force_update, lid_save_T, 30)

def main(argv=None):  # pylint: disable=unused-argument
  run()

if __name__ == '__main__':
  tf.app.run()




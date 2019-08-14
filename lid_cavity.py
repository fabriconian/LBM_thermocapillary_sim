
import time

import numpy as np
import tensorflow as tf
import math 
import cv2

import LatFlow.DomainTerm as dom
from   LatFlow.utils  import *

# video init
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter()

shape = [100, 100]
success = video.open('lid_flow.mov', fourcc, 30, (shape[1], shape[0]), True)

FLAGS = tf.app.flags.FLAGS

def make_lid_boundary(shape):
  boundary = np.zeros((1, shape[0], shape[1], 1), dtype=np.float32)
  boundary[:,:,0,:] = 1.0
  boundary[:,shape[0]-1,:,:] = 1.0
  boundary[:,:,shape[1]-1,:] = 1.0
  return boundary

def make_lid_boundary_T(shape):
  boundary = np.zeros((1, shape[0], shape[1], 1), dtype=np.float32)
  boundary[:,:,0,:] = 1.0
  boundary[:,shape[0]-1,:,:] = 1.0
  boundary[:,:,shape[1]-1,:] = 1.0
  boundary[:, 0, :,:] = 1.0
  return boundary


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

def lid_init_step_T(domain, value=1):
  vel_dir = tf.zeros_like(domain.Vel[0][:, :, :, 0:1])
  vel = tf.concat([vel_dir, vel_dir, vel_dir], axis=3)
  vel_dot_vel = tf.expand_dims(tf.reduce_sum(vel * vel, axis=3), axis=3)
  vel_dot_c = tf.reduce_sum(tf.expand_dims(vel, axis=3) * tf.reshape(domain.C, [1, 1, 1, domain.Nneigh, 3]), axis=4)

  geq = tf.reshape(domain.W, [1,1,1,domain.Nneigh])*value * (1.0 + 3.0*vel_dot_c/domain.Cs + 4.5*vel_dot_c*vel_dot_c/(domain.Cs*domain.Cs) - 1.5*vel_dot_vel/(domain.Cs*domain.Cs))


  T = (1.0 - domain.boundaryT)*value

  stream_stepg = domain.g[0].assign(geq)
  T_step = domain.T[0].assign(T)
  step = tf.group(*[stream_stepg, T_step])
  return step


def lid_setup_step(domain, value=0.001):
  # inputing top velocity 
  vel = domain.Vel[0]
  vel_out  = vel[:,1:]
  vel_edge = vel[:,:1]
  vel_edge = tf.split(vel_edge, 3, axis=3)
  vel_edge[0] = vel_edge[0]+value
  vel_edge = tf.concat(vel_edge, axis=3)
  vel = tf.concat([vel_edge,vel_out],axis=1)

  # make steps
  vel_step = domain.Vel[0].assign(vel)
  return vel_step


def lid_save_T(domain, sess):
  frame = sess.run(domain.T[0])
  frame = np.sqrt(np.square(frame[0,:,:,0]) )
  # print(np.max(frame))
  # print(np.min(frame))
  frame = np.uint8(255 * frame/np.max(frame))
  frame = cv2.applyColorMap(frame, 2)
  video.write(frame)


def ForceUpdate(self):
    force = tf.concat(values=[(-0.0001 * (self.T[0] - tf.ones_like(self.T[0]) * self.Tref))
      , tf.zeros_like(self.T[0]), tf.zeros_like(self.T[0])], axis=3)
    update = self.BForce[0].assign(force)
    return update

def lid_save_vel(domain, sess):
  frame = sess.run(domain.Vel[0])
  frame = np.sqrt(np.square(frame[0,:,:,0])+ np.square(frame[0,:,:,1]) + np.square(frame[0,:,:,2]))
  # print(np.max(frame))
  # print(np.min(frame))
  frame = np.uint8(255 * frame/np.max(frame))
  frame = cv2.applyColorMap(frame, 2)
  video.write(frame)

def run():
  # constants
  input_vel = 0.001
  nu = input_vel*(10.0)
  Ndim = shape
  boundary = make_lid_boundary(shape=Ndim)
  boundary_T = make_lid_boundary_T(shape=Ndim
  )

  # domain
  domain = dom.Domain("D2Q9", nu, Ndim, boundary,boundaryT=boundary_T,les=False)

  # make lattice state, boundary and input velocity
  initialize_step = lid_init_step(domain, value=0.08)
  initialize_step_T = lid_init_step(domain, value=1)
  setup_step = lid_setup_step(domain, value=input_vel)


  # init things
  init = tf.global_variables_initializer()
  a = tf.constant([10, 20])
  b = tf.constant([1.0, 2.0])
  # start sess
  sess = tf.Session()

  # init variables
  sess.run(init)

  # run steps
  domain.Solve(sess, 10000, initialize_step,initialize_step_T ,setup_step, lid_save_vel, 60)

def main(argv=None):  # pylint: disable=unused-argument
  run()

if __name__ == '__main__':
  tf.app.run()





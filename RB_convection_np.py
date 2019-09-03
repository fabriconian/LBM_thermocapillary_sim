
import time

import numpy as np
import math
import cv2

from matplotlib import pyplot as plt

import LatFlow_np.DomainTherm as dom
from   LatFlow_np.utils  import *

# video init
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter()

shape = [2, 2]
success = video.open('some_videos/lidtst_np64.mov', fourcc, 30, (shape[1], shape[0]), True)


def make_lid_boundaryt2(shape):
  boundary = np.zeros((1, shape[0], shape[1], 1), dtype=np.float64)
  boundary[:,:,0,:] = 1.0
  boundary[:,shape[0]-1,:,:] = 1.0
  boundary[:,:,shape[1]-1,:] = 1.0
  boundary[:, 0,:, :] = 1.0
  return boundary

def make_lid_boundary(shape):
  boundary = np.zeros((1, shape[0], shape[1], 1), dtype=np.float64)
  boundary[:,:,0,:] = 1.0
  boundary[:,shape[0]-1,:,:] = 1.0
  boundary[:,:,shape[1]-1,:] = 1.0
  return boundary

def make_lid_boundary_T(shape, Tup=0.4, Tdown=0.6):

  #boundaryy upp
  boundary = np.zeros((1, shape[0], shape[1], 1), dtype=np.float64)
  boundary[:,0,:,:] = 1.0
  bup = dom.BoundaryT(boundary=boundary,value=Tup,type='CT',n=np.array([-1,0,0]))

  # boundaryy down
  boundary = np.zeros((1, shape[0], shape[1], 1), dtype=np.float64)
  boundary[:, -1, :, :] = 1.0
  bdown = dom.BoundaryT(boundary=boundary, value=Tdown, type='CT',n=np.array([1,0,0]))

  # boundaryy LEFT
  boundary = np.zeros((1, shape[0], shape[1], 1), dtype=np.float64)
  boundary[:, :, 0, :] = 1.0
  bl = dom.BoundaryT(boundary=boundary,  type='ZF',n=np.array([0,0,0]))

  # boundaryy LEFT
  boundary = np.zeros((1, shape[0], shape[1], 1), dtype=np.float64)
  boundary[:, :, -1, :] = 1.0
  br = dom.BoundaryT(boundary=boundary, type='ZF')

  return [bup,br,bdown,bl]


def lid_init_step(domain, value=0.08):
  vel = np.zeros_like(domain.Vel[0])
  vel_dot_vel = np.expand_dims(np.sum(vel * vel, axis=3), axis=3)
  vel_dot_c = np.sum(np.expand_dims(vel, axis=3) * np.reshape(domain.C, [1,1,1,domain.Nneigh,3]), axis=4)
  feq = np.reshape(domain.W, [1,1,1,domain.Nneigh]) * (1.0 + 3.0*vel_dot_c/domain.Cs + 4.5*vel_dot_c*vel_dot_c/
                                                       (domain.Cs*domain.Cs) - 1.5*vel_dot_vel/(domain.Cs*domain.Cs))

  vel = vel * (1.0 - domain.boundary)
  rho = (1.0 - domain.boundary)
  force = np.zeros_like(vel)


  domain.F[0] = feq
  domain.Rho[0] = rho
  domain.Vel[0] = vel
  domain.BForce[0] = force

  return [domain.F[0], domain.Rho[0], domain.Vel[0], domain.BForce[0]]

def lid_init_step_T(domain, value=0.5):
  vel = np.zeros_like(domain.Vel[0])
  vel_dot_vel = np.expand_dims(np.sum(vel * vel, axis=3), axis=3)
  vel_dot_c = np.sum(np.expand_dims(vel, axis=3) * np.reshape(domain.C, [1, 1, 1, domain.Nneigh, 3]), axis=4)

  geq = np.reshape(domain.W, [1,1,1,domain.Nneigh])*value * (1.0 + 3.0*vel_dot_c/domain.Cs + 4.5*vel_dot_c*vel_dot_c/
                                                        (domain.Cs*domain.Cs) - 1.5*vel_dot_vel/(domain.Cs*domain.Cs))
  T = np.ones_like(1.0-domain.boundary)*value

  domain.g[0] = geq
  domain.gtemp[0] = geq
  domain.T[0] = T

  return [domain.g[0],domain.gtemp[0], domain.T[0]]

def lid_setup_step(domain, value=0):
  # inputing top velocity
  vel = domain.Vel[0]
  vel_out  = vel[:,1:]
  vel_edge = vel[:,:1]
  vel_edge[:,:,:,0] = vel_edge[:,:,:,0]+value
  vel = np.concatenate([vel_edge,vel_out],axis=1)

  # make steps
  domain.Vel[0] = vel
  return domain.Vel[0]



def lid_save_T(domain):
  frame = domain.T[0]
  frame = np.sqrt(np.square(frame[0, :, :, 0]))

  # print('\n', np.max(frame), '\t', np.min(frame))
  frame = np.uint8(255 * (frame-np.min(frame)) / (np.max(frame)-np.min(frame)))
  frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT, 2)
  video.write(frame)



def lid_save_vel(domain):
  frame = domain.Vel[0]
  frame = np.sqrt(np.square(frame[0,:,:,0])+ np.square(frame[0,:,:,1]) + np.square(frame[0,:,:,2]))
  print('\n',np.max(frame),'\t',np.min(frame))
  frame = np.uint8(255 * frame/np.max(frame))
  frame = cv2.applyColorMap(frame, cv2.COLORMAP_RAINBOW, 2)

  video.write(frame)

def ForceUpdate(domain,Tref,beta):
    g = 9.8 * domain.dt_real * domain.dt_real / domain.dx_real
    # Tref = domain.Tref
    force = np.zeros_like(domain.Vel[0])
    force[0,:,:,1] = -beta * g * (domain.T[0][:,:,:,0] -  Tref)
    domain.BForce[0] = force
    return domain.BForce[0]


def run():
  # with tf.device('/device:GPU:0'):
  # constants
  input_vel = 0.001
  nu = 1.004E-4
  K = 0.143E-5
  dx = 7.0E-4
  dt = 1E-4
  beta= 0.0
  Tf=1.0
  Tref = 1.1
  Ndim = shape
  boundary = make_lid_boundary(shape=Ndim)
  boundary_T = make_lid_boundary_T(shape=Ndim,Tup=0.2, Tdown=0.4)
  boundaryt2 = make_lid_boundaryt2(shape=Ndim)

  # domain
  domain = dom.Domain(method="D2Q9",
                      Ndim=Ndim,
                      tauf= 0.53,
                      taug=0.9,
                      boundary=boundaryt2,
                      dt=dt,
                      dx=dx,
                      boundary_T= boundary_T,
                      les=False)

  # make lattice state, boundary and input velocity
  initialize_step = lambda domain: lid_init_step(domain, value=0.08)
  initialize_step_T = lambda domain: lid_init_step_T(domain, value=Tref)
  force_update = lambda domain: ForceUpdate(domain,Tref=Tref,beta=beta)
  setup_step = lambda domain: lid_setup_step(domain, value=input_vel)
  # run steps
  domain.Solve(Tf, initialize_step, initialize_step_T,  force_update, lid_save_T, setup_step, 1)

# def main(argv=None):  # pylint: disable=unused-argument
#   run()

if __name__ == '__main__':
  run()




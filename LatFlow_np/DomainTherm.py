
import cv2
from LatFlow_np.utils_v2 import *
import time
from tqdm import *
from matplotlib import pyplot as plt
import LatFlow_np   .D2Q9 as D2Q9

class Object():
    def __init__(self,
                 vertices,
                 rc=np.array([0,0]),
                 vc=np.array([0,0]),
                ):
        self.rc = rc
        self.vc = vc
        self.vertices = vertices

    def Updait(self,dt):
        self.rc +=self.vc*dt

class BoundaryT():
    def __init__(self,
                 boundary,
                 type='CT',
                 n=np.array([1,0,0]),
                 value=None):
        self.boundary = boundary
        self.type = type
        self.value = None
        self.n = n
        if self.type == 'CT':
            self.value = value

        if self.type=='ZF':
            self.value = 0

#domain
class Domain():
    def __init__(self,
                 Ndim,
                 method,
                 boundary,
                 boundary_T,
                 objects=None,
                 nu=None,
                 K=None,
                 tauf=None,
                 taug=None,
                 dx=1.0,
                 dt=1.0,
                 les=True,
                 train_les=False):

        self.Ndim = Ndim

        if method == "D2Q9":
            self.Nneigh = 9
            self.Dim = 2
            self.W = np.reshape(D2Q9.WEIGHTS, (self.Dim + 1) * [1] + [self.Nneigh])
            self.C = np.reshape(D2Q9.LVELOC, self.Dim * [1] + [self.Nneigh, 3])
            self.Cten = np.expand_dims(np.concatenate(([[[self.C[0, 0]] * self.Ndim[1]] * self.Ndim[0]]),axis=0),0)
            self.Op = np.reshape(D2Q9.BOUNCE, self.Dim * [1] + [self.Nneigh, self.Nneigh])
            self.St = D2Q9.STREAM

        if nu is not list:
            nu = [nu]
            K = [K]

        self.les = les
        self.time = 0.0
        self.dt = 1.0  # starred unit (dimensionalised)
        self.dx = 1.0  # starred unit (dimensionalised)
        self.Cs = 1.0 / 1.0  # starred unit (dimensionalised)
        self.dt_real = dt
        self.dx_real = dx
        self.rho_real = 1000
        self.Step = 1
        self.Sc = 0.17

        # self.Ncells = np.prod(np.array(Ndim))
        self.boundary = boundary
        self.boundaryT2 = boundary_T
        self.objects  = objects
        self.Nl = len(nu)
        self.tau = []
        self.taug = []
        self.G = []
        self.Gs = []
        self.Rhoref = []
        self.Psi = []
        self.Gmix = 0.0

        self.F = []
        self.Ftemp = []
        self.g = []
        self.gtemp = []
        self.Vel = []
        self.T = []
        self.BForce = []
        self.QSource = []
        self.Rho = []
        self.IsSolid = []
        self.Tref = 0.5
        self.step_count = 0

        for i in range(len(nu)):
            if tauf is None:
                self.tau.append(3.0 * nu[i] * self.dt_real / (self.dx_real * self.dx_real) + 0.5)
            else:
                self.tau.append(tauf)

            if taug is None:
                self.taug.append(3.0 * K[i] * self.dt_real / (self.dx_real * self.dx_real) + 0.5)
            else:
                self.taug.append(taug)
            print(self.tau, self.taug)
            self.G.append(0.0)
            self.Gs.append(0.0)
            self.Rhoref.append(200.0)

            self.Psi.append(4.0)

            self.F.append(np.zeros([1] + Ndim + [self.Nneigh], dtype=np.float64))
            self.Ftemp.append(np.zeros([1] + Ndim + [self.Nneigh], dtype=np.float64))
            self.g.append(np.zeros([1] + Ndim + [self.Nneigh], dtype=np.float64))
            self.gtemp.append(np.zeros([1] + Ndim + [self.Nneigh], dtype=np.float64))

            self.Vel.append(np.zeros([1] + Ndim + [3], dtype=np.float64))
            self.T.append(np.zeros([1] + Ndim + [1], dtype=np.float64))

            self.BForce.append(np.zeros([1] + Ndim + [3], dtype=np.float64))
            self.QSource.append(np.zeros([1] + Ndim + [1], dtype=np.float64))
            self.Rho.append(np.zeros([1] + Ndim + [1], dtype=np.float64))
            self.IsSolid.append(np.zeros([1] + Ndim + [1], dtype=np.float64))

        self.EEk = np.zeros(self.Dim * [1] + [self.Nneigh])
        for n in range(3):
            for m in range(3):
                if self.Dim == 2:
                    self.EEk = self.EEk + np.abs(self.C[:, :, :, n] * self.C[:, :, :, m])
                elif self.Dim == 3:
                    self.EEk = self.EEk + np.abs(self.C[:, :, :, :, n] * self.C[:, :, :, :, m])

    def CollideSC(self, graph_unroll=False):
        # boundary bounce piece
        f_boundary = np.multiply(self.F[0], self.boundary)
        f_boundary = simple_conv(f_boundary, self.Op)
         # to stop dividing by zero

        # make vel bforce and rho
        vel = self.Vel[0]
        f = self.F[0]

        force = self.BForce[0]
        rho = self.Rho[0]  # to stop dividing by zero

        # rho = self.Rho[0] + 1e-12 # to stop dividing by zero

        # calc v dots

        vel_dot_vel = np.expand_dims(np.sum(vel * vel, axis=self.Dim + 1), axis=self.Dim + 1)
        if self.Dim == 2:
            vel_dot_c = simple_conv(vel, np.transpose(self.C, [0, 1, 3, 2]))
        else:
            vel_dot_c = simple_conv(vel, np.transpose(self.C, [0, 1, 2, 4, 3]))

        f_dot_c = simple_conv(force, np.transpose(self.C, [0, 1, 3, 2]))

        uten = np.reshape(np.concatenate([[self.Vel[0]] * int(self.Nneigh)],axis=0), newshape=self.Cten.shape)
        ften = np.reshape(np.concatenate([[force] * int(self.Nneigh)],axis=0), newshape=self.Cten.shape)
        ften = 3.0 * self.W * np.sum(ften * (self.Cten - uten), axis=-1) / self.Cs ** 2

        # calc Feq
        Feq = self.W * rho * (1.0 + 3.0 * vel_dot_c / self.Cs ** 2 + 4.5 * vel_dot_c * vel_dot_c / (
                    self.Cs * self.Cs) - 1.5 * vel_dot_vel / (self.Cs * self.Cs))
        Fi = 9.0 * self.W * f_dot_c * vel_dot_c / self.Cs ** 4 + ften
        # Fi = 3.0 * self.W * f_dot_c
        # collision calc
        NonEq = f - Feq
        if self.les:
            Q = np.expand_dims(np.sum(NonEq * NonEq * self.EEk, axis=self.Dim + 1), axis=self.Dim + 1)
            Q = np.sqrt(2.0 * Q)
            tau = 0.5 * (self.tau[0] + np.sqrt(self.tau[0] * self.tau[0] + 6.0 * Q * self.Sc / rho))
        else:
            tau = self.tau[0]
        f = f - NonEq / tau + Fi * (1 - 1 / 2 / tau)

        # combine boundary and no boundary values
        f_no_boundary = np.multiply(f, (1.0 - self.boundary))
        f = f_no_boundary + f_boundary

        if not graph_unroll:
            # make step
            self.Ftemp[0] = f
            return self.Ftemp[0]
        else:
            # put computation back in graph
            self.Ftemp[0] = f

    def Collide_T(self, graph_unroll=False):

        # make vel bforce and rho
        # g_boundary = tf.multiply(self.g[0], self.boundary)
        g = self.g[0]
        vel = self.Vel[0]
        # rho = self.Rho[0] + 1e-12 # to stop dividing by zero
        T = self.T[0]

        # calc v dots
        vel_dot_vel = np.expand_dims(np.sum(vel * vel, axis=self.Dim + 1), axis=self.Dim + 1)
        if self.Dim == 2:
            vel_dot_c = simple_conv(vel, np.transpose(self.C, [0, 1, 3, 2]))
        else:
            vel_dot_c = simple_conv(vel, np.transpose(self.C, [0, 1, 2, 4, 3]))

        # calc Feq
        geq = self.W * T * (1.0 + 3.0 * vel_dot_c / self.Cs ** 2 + 4.5 * vel_dot_c * vel_dot_c / (
                    self.Cs * self.Cs) - 1.5 * vel_dot_vel / (self.Cs * self.Cs))

        # collision calc
        NonEq = g - geq

        tau = self.taug[0]
        g = g - NonEq / tau

        if not graph_unroll:
            # make step
            self.gtemp[0] = g
            return self.gtemp[0]
        else:
            # put computation back in graph
            self.gtemp[0] = g

        # applying Inumaro BC for the thermal population

    def ApplyBC(self):
        # upper boundary
        g = self.g[0]
        # g = self.gtemp[0]
        # update upper wall
        if self.boundaryT2[0].type == 'CT':
            gup = g[:, :1, :, :]
            gwall = (np.ones_like(gup[:, :, :, 0:1]) * self.boundaryT2[0].value \
                     - gup[:, :, :, 0:1] + gup[:, :, :, 1:2] + gup[:, :, :, 3:4] + gup[:, :, :, 4:5] + gup[:, :, :,7:8]\
                     + gup[:, :,:, 8:]) \
                    / (self.W[:, :, :, 2:3] + self.W[:, :, :, 5:6] + self.W[:, :, :, 6:7])
            g[:, :1, :, :] = np.concatenate([gup[:, :, :, 0:2], gwall * self.W[:, :, :, 2:3], \
                                     gup[:, :, :, 3:5], gwall * self.W[:, :, :, 5:6], \
                                     gwall * self.W[:, :, :, 6:7], gup[:, :, :, 7:]], axis=-1)
        # update down wall
        if self.boundaryT2[2].type == 'CT':
            gup = g[:, -2:-1, :, :]
            gwall = (np.ones_like(gup[:, :, :, 0:1]) * self.boundaryT2[2].value \
                     - gup[:, :, :, 0:1] + gup[:, :, :, 1:2] + gup[:, :, :, 2:3] + gup[:, :, :, 3:4] + gup[:, :, :,5:6]\
                     + gup[:, :, :, 6:7]) \
                    / (self.W[:, :, :, 4:5] + self.W[:, :, :, 7:8] + self.W[:, :, :, 8:])
            g[:, -2:-1, :, :] = np.concatenate([gup[:, :, :, 0:4], gwall * self.W[:, :, :, 4:5], \
                                      gup[:, :, :, 5:7], gwall * self.W[:, :, :, 7:8], \
                                      gwall * self.W[:, :, :, 8:]], axis=-1)

        # update right wall
        if self.boundaryT2[1].type == 'ZF':
            gup = g[:, :, -2:-1, :]
            gwall = (gup[:, :, :, 1:2] + gup[:, :, :, 5:6] + gup[:, :, :, 8:]) \
                    / (self.W[:, :, :, 3:4] + self.W[:, :, :, 6:7] + self.W[:, :, :, 7:8])
            g[:, :, -2:-1, :] = np.concatenate([gup[:, :, :, 0:3], gwall * self.W[:, :, :, 3:4], \
                                       gup[:, :, :, 4:6], gwall * self.W[:, :, :, 6:7], \
                                       gwall * self.W[:, :, :, 7:8], gup[:, :, :, 8:]], axis=-1)

        # update left wall
        if self.boundaryT2[3].type == 'ZF':
            gup = g[:, :, :1, :]
            gwall = (gup[:, :, :, 3:4] + gup[:, :, :, 6:7] + gup[:, :, :, 7:8]) \
                    / (self.W[:, :, :, 1:2] + self.W[:, :, :, 5:6] + self.W[:, :, :, 8:])
            g[:, :, :1, :] = np.concatenate([gup[:, :, :, 0:1], gwall * self.W[:, :, :, 1:2], \
                                      gup[:, :, :, 2:5], gwall * self.W[:, :, :, 5:6], \
                                       gup[:, :, :, 6:8],gwall * self.W[:, :, :, 8:]], axis=-1)

        self.g[0] = g
        return self.g[0]



    def MomentsUpdate_T(self, graph_unroll=False):
        g_pad = self.g[0]
        T = np.expand_dims(np.sum(g_pad, self.Dim + 1), self.Dim + 1)
        if not graph_unroll:
            # create steps
            self.T[0] = T
            return self.T[0]
        else:
            self.T[0] = T

    def MomentsUpdate(self, graph_unroll=False):
        f_pad = self.F[0]
        Force = self.BForce[0]
        Rho = np.expand_dims(np.sum(f_pad, self.Dim + 1), self.Dim + 1)
        Vel = simple_conv(f_pad, self.C)
        Vel = Vel / (self.Cs * Rho) + Force / 2.0 / Rho
        if not graph_unroll:
            # create steps
            self.Rho[0] = Rho
            self.Vel[0] = Vel
            # force_step = self.BForce[0].assign(Force)
            step = [Rho, Vel]
            return step
        else:
            self.Rho_step[0] = Rho
            self.Vel_step[0] = Vel

    def StreamSC(self, graph_unroll=False):
        # f_pad = pad_mobius(self.Ftemp[0])
        # f_pad = simple_conv(f_pad, self.St)
        f_pad = simple_conv(self.Ftemp[0], self.St,1)
        if not graph_unroll:
            self.F[0] = f_pad
            return self.F[0]
        else:
            self.F[0] = f_pad

    def Stream_T(self, graph_unroll=False):
        # stream f
        # g_pad = pad_mobius(self.gtemp[0])
        # g_pad = simple_conv(g_pad, self.St)
        g_pad = simple_conv(self.gtemp[0], self.St, 1)
        # calc new velocity and density
        if not graph_unroll:
            # create steps
            self.g[0] = g_pad
            return self.g[0]
        else:
            self.g[0] = g_pad


    def Initialize(self, graph_unroll=False):
        f_zero = np.zeros([1] + self.Ndim + [self.Nneigh], dtype=np.float64)
        f_zero = f_zero + self.W
        if not graph_unroll:
            self.F[0] = f_zero
            self.Ftemp[0] = f_zero
            return [self.F[0], self.Ftemp[0]]
        else:
            self.F[0] = f_zero
            self.Ftemp[0] = f_zero

    def Initialize_T(self, graph_unroll=False):
        g_zero = np.zeros([1] + self.Ndim + [self.Nneigh], dtype=np.float64)
        g_zero = g_zero + self.W * self.Tref
        if not graph_unroll:
            self.g[0] = g_zero
            self.gtemp[0] = g_zero
            return [self.g[0], self.gtemp[0]]
        else:
            self.g[0] = g_zero
            self.gtemp[0] = g_zero

    def Solve(self,
              Tf,  # final time
              initialize_step,
              initialize_step_T,
              force_update,
              save_step,
              setup_step=None,
              save_interval=25):

        # make steps


        # run solver
        self.Initialize()
        self.Initialize_T()

        initialize_step(self)
        initialize_step_T(self)

        self.StreamSC()
        self.Stream_T()

        self.MomentsUpdate()
        self.MomentsUpdate_T()

        num_steps = int(Tf / self.dt_real)
        # save_step(self, sess)
        # plt.show()
        # the status bar initializer
        for i in tqdm(range(num_steps)):
            if int(self.step_count % save_interval) == 0:
                save_step(self)
            setup_step(self)
            force_update(self)

            self.CollideSC()
            self.Collide_T()

            self.StreamSC()
            self.Stream_T()
            self.ApplyBC()

            self.MomentsUpdate()
            self.MomentsUpdate_T()

            self.time += self.dt_real
            self.step_count += 1

    def Unroll(self, start_f, num_steps, setup_computation):
        # run solver
        self.F[0] = start_f
        F_return_state = []
        for i in range(num_steps):
            setup_computation(self)
            self.CollideSC(graph_unroll=True)
            self.StreamSC(graph_unroll=True)
            F_return_state.append(self.F[0])
        return F_return_state

    def Unroll_les_train(self, start_f, num_steps, setup_computation):
        # run solver
        self.F[0] = start_f
        F_return_state = []
        for i in range(num_steps):
            setup_computation(self)
            self.CollideSC(graph_unroll=True)
            self.StreamSC(graph_unroll=True)
            F_return_state.append(self.F[0])
        return F_return_state






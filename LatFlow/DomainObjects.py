
import cv2
from LatFlow.utils import *
import time
from tqdm import *
from matplotlib import pyplot as plt
import LatFlow.D2Q9 as D2Q9

class Object_np():
    def __init__(self,
                 vertices,
                 rc=np.array([0,0]),
                 vc=np.array([0,0])

                ):
        self.rc = rc
        self.vc = vc
        self.vertices = vertices

    def Updait(self,dt):
        self.rc +=self.vc*dt

class Object():
    def __init__(self,
                 vertices,
                 rc=np.array([0,0]),
                 vc=np.array([0,0]),
                ):
        self.rc = tf.Variable(rc,dtype=tf.float32)
        self.vc = tf.Variable(vc,dtype=tf.float32)
        self.vertices = tf.Variable(vertices,dtype=tf.float32)


    def Updait(self,dt):
        self.rc.assign(self.vc*dt+self.rc)

    def create_polygon(self, idxs):
        """
        Creates np.array with dimensions defined by shape
        Fills polygon defined by vertices with ones, all other values zero"""

        p2 = self.vertices
        p1 = tf.concat([p2[-1:], p2[:-1]], 0)

        fill = tf.reduce_all(self.check_vector_tf(p1, p2, idxs), axis=0)
        res = tf.dtypes.cast(fill, tf.float32)

        return res

    def check_vector_tf(self, p1, p2, idxs):
        """
        Uses the line defined by p1 and p2 to check array of
        input indices against interpolated value

        Returns boolean array, with True inside and False outside of shape
        """

        idxs = tf.dtypes.cast(tf.concat([p1.shape[0] * [idxs]], axis=0),tf.float32)
        shape = idxs[:,0].shape

        p1x = tf.transpose(tf.ones([shape[2], shape[1], shape[0]]) * p1[:, 0], (2,1,0))
        p1y = tf.transpose(tf.ones([shape[2], shape[1],shape[0]]) * p1[:, 1],(2,1,0))
        p2x = tf.transpose(tf.ones([shape[2], shape[1],shape[0]]) * p2[:, 0],(2,1,0))
        p2y = tf.transpose(tf.ones([shape[2], shape[1],shape[0]]) * p2[:, 1],(2,1,0))

        # Calculate max column idx for each row idx based on interpolated line between two points
        max_col_idx = (idxs[:, 0] - p1x) / (p2x - p1x) * (p2y - p1y) + p1y
        sign = tf.sign(p2x - p1x)
        res = idxs[:, 1] * sign <= max_col_idx * sign
        return res

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
            self.W = tf.reshape(D2Q9.WEIGHTS, (self.Dim + 1) * [1] + [self.Nneigh])
            self.C = tf.reshape(D2Q9.LVELOC, self.Dim * [1] + [self.Nneigh, 3])
            self.Cten = tf.expand_dims(tf.concat(axis=0, values=[[[self.C[0, 0]] * self.Ndim[1]] * self.Ndim[0]]), 0)
            self.Op = tf.reshape(D2Q9.BOUNCE, self.Dim * [1] + [self.Nneigh, self.Nneigh])
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

        self.Ncells = np.prod(np.array(Ndim))
        self.boundary = tf.constant(boundary)
        self.boundaryT2 = boundary_T
        self.objects  = objects
        self.idxs = tf.concat([tf.expand_dims(tf.tile(tf.expand_dims(tf.range(0,Ndim[0]),1),[1,Ndim[1]]),0),
                                   tf.expand_dims(tf.tile(tf.expand_dims(tf.range(0, Ndim[1]), 0), [Ndim[0], 1]),0)],0)
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

            self.F.append(tf.Variable(np.zeros([1] + Ndim + [self.Nneigh], dtype=np.float32)))
            self.Ftemp.append(tf.Variable(np.zeros([1] + Ndim + [self.Nneigh], dtype=np.float32)))
            self.g.append(tf.Variable(np.zeros([1] + Ndim + [self.Nneigh], dtype=np.float32)))
            self.gtemp.append(tf.Variable(np.zeros([1] + Ndim + [self.Nneigh], dtype=np.float32)))

            self.Vel.append(tf.Variable(np.zeros([1] + Ndim + [3], dtype=np.float32)))
            self.T.append(tf.Variable(np.zeros([1] + Ndim + [1], dtype=np.float32)))

            self.BForce.append(tf.Variable(np.zeros([1] + Ndim + [3], dtype=np.float32)))
            self.QSource.append(tf.Variable(np.zeros([1] + Ndim + [1], dtype=np.float32)))
            self.Rho.append(tf.Variable(np.zeros([1] + Ndim + [1], dtype=np.float32)))
            self.IsSolid.append(tf.Variable(np.zeros([1] + Ndim + [1], dtype=np.float32)))

        self.EEk = tf.zeros(self.Dim * [1] + [self.Nneigh])
        for n in range(3):
            for m in range(3):
                if self.Dim == 2:
                    self.EEk = self.EEk + tf.abs(self.C[:, :, :, n] * self.C[:, :, :, m])
                elif self.Dim == 3:
                    self.EEk = self.EEk + tf.abs(self.C[:, :, :, :, n] * self.C[:, :, :, :, m])

    def CollideSC(self, graph_unroll=False):
        # boundary bounce piece
        f_boundary = tf.multiply(self.F[0], self.boundary)
        f_boundary = simple_conv(f_boundary, self.Op)
         # to stop dividing by zero

        # make vel bforce and rho
        vel = self.Vel[0]
        f = self.F[0]

        force = self.BForce[0]
        rho = self.Rho[0]  # to stop dividing by zero

        # rho = self.Rho[0] + 1e-12 # to stop dividing by zero

        # calc v dots

        vel_dot_vel = tf.expand_dims(tf.reduce_sum(vel * vel, axis=self.Dim + 1), axis=self.Dim + 1)
        if self.Dim == 2:
            vel_dot_c = simple_conv(vel, tf.transpose(self.C, [0, 1, 3, 2]))
        else:
            vel_dot_c = simple_conv(vel, tf.transpose(self.C, [0, 1, 2, 4, 3]))

        f_dot_c = simple_conv(force, tf.transpose(self.C, [0, 1, 3, 2]))

        uten = tf.reshape(tf.concat(axis=0, values=[[self.Vel[0]] * int(self.Nneigh)]), shape=tf.shape(self.Cten))
        ften = tf.reshape(tf.concat(axis=0, values=[[force] * int(self.Nneigh)]), shape=tf.shape(self.Cten))
        ften = 3.0 * self.W * tf.reduce_sum(ften * (self.Cten - uten), axis=-1) / self.Cs ** 2

        # calc Feq
        Feq = self.W * rho * (1.0 + 3.0 * vel_dot_c / self.Cs ** 2 + 4.5 * vel_dot_c * vel_dot_c / (
                    self.Cs * self.Cs) - 1.5 * vel_dot_vel / (self.Cs * self.Cs))
        Fi = 9.0 * self.W * f_dot_c * vel_dot_c / self.Cs ** 4 + ften
        # Fi = 3.0 * self.W * f_dot_c
        # collision calc
        NonEq = f - Feq
        if self.les:
            Q = tf.expand_dims(tf.reduce_sum(NonEq * NonEq * self.EEk, axis=self.Dim + 1), axis=self.Dim + 1)
            Q = tf.sqrt(2.0 * Q)
            tau = 0.5 * (self.tau[0] + tf.sqrt(self.tau[0] * self.tau[0] + 6.0 * Q * self.Sc / rho))
        else:
            tau = self.tau[0]
        f = f - NonEq / tau + Fi * (1 - 1 / 2 / tau)

        # combine boundary and no boundary values
        f_no_boundary = tf.multiply(f, (1.0 - self.boundary))
        f = f_no_boundary + f_boundary

        if not graph_unroll:
            # make step
            collid_step = self.Ftemp[0].assign(f)
            return collid_step
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
        vel_dot_vel = tf.expand_dims(tf.reduce_sum(vel * vel, axis=self.Dim + 1), axis=self.Dim + 1)
        if self.Dim == 2:
            vel_dot_c = simple_conv(vel, tf.transpose(self.C, [0, 1, 3, 2]))
        else:
            vel_dot_c = simple_conv(vel, tf.transpose(self.C, [0, 1, 2, 4, 3]))

        # calc Feq
        geq = self.W * T * (1.0 + 3.0 * vel_dot_c / self.Cs ** 2 + 4.5 * vel_dot_c * vel_dot_c / (
                    self.Cs * self.Cs) - 1.5 * vel_dot_vel / (self.Cs * self.Cs))

        # collision calc
        NonEq = g - geq

        tau = self.taug[0]
        g = g - NonEq / tau

        # combine boundary and no boundary values
        # g_no_boundary = tf.multiply(g, (1.0-self.boundary))
        # g = g_no_boundary + g_boundary

        if not graph_unroll:
            # make step
            collid_step = self.gtemp[0].assign(g)
            return collid_step
        else:
            # put computation back in graph
            self.gtemp[0] = g

        # applying Inumaro BC for the thermal population

    def ApplyBC(self):
        # upper boundary
        g = self.g[0]
        g_inn = g[:, 1:-1, 1:-1, :]
        g = self.gtemp[0]
        # update upper wall
        if self.boundaryT2[0].type == 'CT':
            gup = g[:, :1, :, :]
            gwall = (tf.ones_like(gup[:, :, :, 0:1]) * self.boundaryT2[0].value \
                     - gup[:, :, :, 0:1] + gup[:, :, :, 1:2] + gup[:, :, :, 3:4] + gup[:, :, :, 4:5] + gup[:, :, :,7:8]\
                     + gup[:, :,:, 8:]) \
                    / (self.W[:, :, :, 2:3] + self.W[:, :, :, 5:6] + self.W[:, :, :, 6:7])
            guup = tf.concat(values=[gup[:, :, :, 0:2], gwall * self.W[:, :, :, 2:3], \
                                     gup[:, :, :, 3:5], gwall * self.W[:, :, :, 5:6], \
                                     gwall * self.W[:, :, :, 6:7], gup[:, :, :, 7:]], axis=-1)
        # update down wall
        if self.boundaryT2[2].type == 'CT':
            gup = g[:, -2:-1, :, :]
            gwall = (tf.ones_like(gup[:, :, :, 0:1]) * self.boundaryT2[2].value \
                     - gup[:, :, :, 0:1] + gup[:, :, :, 1:2] + gup[:, :, :, 2:3] + gup[:, :, :, 3:4] + gup[:, :, :,5:6]\
                     + gup[:, :, :, 6:7]) \
                    / (self.W[:, :, :, 4:5] + self.W[:, :, :, 7:8] + self.W[:, :, :, 8:])
            gdown = tf.concat(values=[gup[:, :, :, 0:4], gwall * self.W[:, :, :, 4:5], \
                                      gup[:, :, :, 5:7], gwall * self.W[:, :, :, 7:8], \
                                      gwall * self.W[:, :, :, 8:]], axis=-1)

        # update right wall
        if self.boundaryT2[1].type == 'ZF':
            gup = g[:, :, -2:-1, :]
            gwall = (gup[:, :, :, 1:2] + gup[:, :, :, 5:6] + gup[:, :, :, 8:]) \
                    / (self.W[:, :, :, 3:4] + self.W[:, :, :, 6:7] + self.W[:, :, :, 7:8])
            gright = tf.concat(values=[gup[:, :, :, 0:3], gwall * self.W[:, :, :, 3:4], \
                                       gup[:, :, :, 4:6], gwall * self.W[:, :, :, 6:7], \
                                       gwall * self.W[:, :, :, 7:8], gup[:, :, :, 8:]], axis=-1)

        # update left wall
        if self.boundaryT2[3].type == 'ZF':
            gup = g[:, :, :1, :]
            gwall = (gup[:, :, :, 3:4] + gup[:, :, :, 6:7] + gup[:, :, :, 7:8]) \
                    / (self.W[:, :, :, 1:2] + self.W[:, :, :, 5:6] + self.W[:, :, :, 8:])
            gleft = tf.concat(values=[gup[:, :, :, 0:1], gwall * self.W[:, :, :, 1:2], \
                                      gup[:, :, :, 2:5], gwall * self.W[:, :, :, 5:6], \
                                       gup[:, :, :, 6:8],gwall * self.W[:, :, :, 8:]], axis=-1)

        res = tf.concat(
            values=[guup, tf.concat(values=[gleft[:, 1:-1, :, :], g_inn, gright[:, 1:-1, :, :]], axis=2), gdown],
            axis=1)
        updbc_step = self.g[0].assign(res)
        return updbc_step

    def ApplyBC(self):
        # upper boundary
        g = self.g[0]
        g_inn = g[:, 1:-1, 1:-1, :]
        # g = self.gtemp[0]
        # update upper wall
        if self.boundaryT2[0].type == 'CT':
            gup = g[:, :1, :, :]
            gwall = (tf.ones_like(gup[:, :, :, 0:1]) * self.boundaryT2[0].value \
                     - gup[:, :, :, 0:1] + gup[:, :, :, 1:2] + gup[:, :, :, 3:4] + gup[:, :, :, 4:5] + gup[:, :, :,7:8]\
                     + gup[:, :,:, 8:]) \
                    / (self.W[:, :, :, 2:3] + self.W[:, :, :, 5:6] + self.W[:, :, :, 6:7])
            guup = tf.concat(values=[gup[:, :, :, 0:2], gwall * self.W[:, :, :, 2:3], \
                                     gup[:, :, :, 3:5], gwall * self.W[:, :, :, 5:6], \
                                     gwall * self.W[:, :, :, 6:7], gup[:, :, :, 7:]], axis=-1)
        # update down wall
        if self.boundaryT2[2].type == 'CT':
            gup = g[:, -2:-1, :, :]
            gwall = (tf.ones_like(gup[:, :, :, 0:1]) * self.boundaryT2[2].value \
                     - gup[:, :, :, 0:1] + gup[:, :, :, 1:2] + gup[:, :, :, 2:3] + gup[:, :, :, 3:4] + gup[:, :, :,5:6]\
                     + gup[:, :, :, 6:7]) \
                    / (self.W[:, :, :, 4:5] + self.W[:, :, :, 7:8] + self.W[:, :, :, 8:])
            gdown = tf.concat(values=[gup[:, :, :, 0:4], gwall * self.W[:, :, :, 4:5], \
                                      gup[:, :, :, 5:7], gwall * self.W[:, :, :, 7:8], \
                                      gwall * self.W[:, :, :, 8:]], axis=-1)

        # update right wall
        if self.boundaryT2[1].type == 'ZF':
            gup = g[:, :, -2:-1, :]
            gwall = (gup[:, :, :, 1:2] + gup[:, :, :, 5:6] + gup[:, :, :, 8:]) \
                    / (self.W[:, :, :, 3:4] + self.W[:, :, :, 6:7] + self.W[:, :, :, 7:8])
            gright = tf.concat(values=[gup[:, :, :, 0:3], gwall * self.W[:, :, :, 3:4], \
                                       gup[:, :, :, 4:6], gwall * self.W[:, :, :, 6:7], \
                                       gwall * self.W[:, :, :, 7:8], gup[:, :, :, 8:]], axis=-1)

        # update left wall
        if self.boundaryT2[3].type == 'ZF':
            gup = g[:, :, :1, :]
            gwall = (gup[:, :, :, 3:4] + gup[:, :, :, 6:7] + gup[:, :, :, 7:8]) \
                    / (self.W[:, :, :, 1:2] + self.W[:, :, :, 5:6] + self.W[:, :, :, 8:])
            gleft = tf.concat(values=[gup[:, :, :, 0:1], gwall * self.W[:, :, :, 1:2], \
                                      gup[:, :, :, 2:5], gwall * self.W[:, :, :, 5:6], \
                                       gup[:, :, :, 6:8],gwall * self.W[:, :, :, 8:]], axis=-1)

        res = tf.concat(
            values=[guup, tf.concat(values=[gleft[:, 1:-1, :, :], g_inn, gright[:, 1:-1, :, :]], axis=2), gdown],
            axis=1)
        updbc_step = self.g[0].assign(res)
        return updbc_step



    def MomentsUpdate_T(self, graph_unroll=False):
        g_pad = self.g[0]
        T = tf.expand_dims(tf.reduce_sum(g_pad, self.Dim + 1), self.Dim + 1)
        if not graph_unroll:
            # create steps
            T_step = self.T[0].assign(T)
            return T_step
        else:
            self.T[0] = T

    def MomentsUpdate(self, graph_unroll=False):
        f_pad = self.F[0]
        Force = self.BForce[0]
        Rho = tf.expand_dims(tf.reduce_sum(f_pad, self.Dim + 1), self.Dim + 1)
        Vel = simple_conv(f_pad, self.C)
        Vel = Vel / (self.Cs * Rho) + Force / 2.0 / Rho
        if not graph_unroll:
            # create steps
            Rho_step = self.Rho[0].assign(Rho)
            Vel_step = self.Vel[0].assign(Vel)
            # force_step = self.BForce[0].assign(Force)
            step = tf.group(*[Rho_step, Vel_step])
            return step
        else:
            self.Rho_step[0] = Rho
            self.Vel_step[0] = Vel

    def StreamSC(self, graph_unroll=False):
        f_pad = pad_mobius(self.Ftemp[0])
        f_pad = simple_conv(f_pad, self.St)
        if not graph_unroll:
            step = self.F[0].assign(f_pad)
            return step
        else:
            self.F[0] = f_pad

    def Stream_T(self, graph_unroll=False):
        # stream f
        g_pad = pad_mobius(self.gtemp[0])
        g_pad = simple_conv(g_pad, self.St)
        # calc new velocity and density
        if not graph_unroll:
            # create steps
            stream_step = self.g[0].assign(g_pad)
            return stream_step
        else:
            self.g[0] = g_pad


    def Initialize(self, graph_unroll=False):
        np_f_zeros = np.zeros([1] + self.Ndim + [self.Nneigh], dtype=np.float32)
        f_zero = tf.constant(np_f_zeros)
        f_zero = f_zero + self.W
        if not graph_unroll:
            assign_step = self.F[0].assign(f_zero)
            assign_step_temp = self.Ftemp[0].assign(f_zero)
            return tf.group(*[assign_step, assign_step_temp])
        else:
            self.F[0].assign(f_zero)
            self.Ftemp[0].assign(f_zero)

    def Initialize_T(self, graph_unroll=False):
        np_f_zeros = np.zeros([1] + self.Ndim + [self.Nneigh], dtype=np.float32)
        g_zero = tf.constant(np_f_zeros)
        g_zero = g_zero + self.W * self.Tref
        if not graph_unroll:
            assign_step = self.g[0].assign(g_zero)
            assign_step_temp = self.gtemp[0].assign(g_zero)
            return tf.group(*[assign_step, assign_step_temp])
        else:
            self.g[0].assign(g_zero)
            self.gtemp[0].assign(g_zero)

    def Solve(self,
              sess,
              Tf,  # final time
              initialize_step,
              initialize_step_T,
              force_update,
              save_step,
              setup_step=None,

              save_interval=25):

        # make steps
        assign_step = self.Initialize()
        assign_step_T = self.Initialize_T()

        stream_step = self.StreamSC()
        stream_step_T = self.Stream_T()

        update_moments_step = self.MomentsUpdate()
        update_moments_T_step = self.MomentsUpdate_T()

        # force_update = self.ForceUpdate()

        collide_step = self.CollideSC()
        collide_step_T = self.Collide_T()
        bc_update_T = self.ApplyBC()

        # create_polygon_tf(self.idxs, self.objects[0].vertices)
        z = self.objects[0].create_polygon(self.idxs)
        # run solver
        sess.run(assign_step)
        sess.run(assign_step_T)

        sess.run(initialize_step)
        sess.run(initialize_step_T)

        sess.run(stream_step)
        sess.run(stream_step_T)

        sess.run(update_moments_step)
        sess.run(update_moments_T_step)

        num_steps = int(Tf / self.dt_real)
        # save_step(self, sess)
        # plt.show()
        # the status bar initializer
        for i in tqdm(range(num_steps)):
            if int(self.step_count % save_interval) == 0:
                save_step(self, sess)
            # sess.run(setup_step)
            sess.run(force_update)
            sess.run(collide_step)
            sess.run(collide_step_T)
            sess.run(stream_step)
            sess.run(stream_step_T)
            sess.run(bc_update_T)

            sess.run(update_moments_step)
            sess.run(update_moments_T_step)

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






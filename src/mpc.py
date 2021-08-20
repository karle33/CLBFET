import numpy as np
import osqp
import scipy.sparse as sparse
import time
# import scipy.linalg as sl
# import matplotlib.pyplot as plt
import rospy
# from dynamics import DynamicsAckermannZModified

class MPC():
    def __init__(self, dt, stepsize, N=10):
        self.N = N
        self.xdim = 6
        self.udim = 3
        self.dt = dt
        self.stepsize = stepsize
        self.update_params(dt, stepsize, N)
        
        self.res = None
    
    def update_params(self, dt, stepsize, N):
        self.N = N
        self.dt = dt
        self.stepsize = stepsize

        A0 = np.block([[np.zeros((self.xdim//2,self.xdim//2)),np.eye(self.xdim//2)],[np.zeros((self.xdim//2,self.xdim//2)),np.zeros((self.xdim//2,self.xdim//2))]])#.astype(np.float32)
        self.A1 = (np.eye(self.xdim) + self.dt * self.stepsize * A0)
        B0 = np.block([[np.zeros((self.xdim//2,self.xdim//2))],[np.eye(self.xdim//2)]]).astype(np.float32)
        self.B1 = self.dt * self.stepsize * B0
        self.A_all = np.empty((0, self.xdim))
        self.B_all = np.empty((0, self.udim * self.N))
        A1_temp = self.A1
        for i in range(self.N):
            self.A_all = np.concatenate((self.A_all, A1_temp), axis=0)
            A1_temp = np.matmul(A1_temp, self.A1)

            B1_temp = np.empty((self.xdim, 0))
            for j in range(self.N):
                if i > j:
                    B1_temp = np.concatenate((B1_temp, np.matmul(self.A_all[self.xdim*(i-j-1):self.xdim*(i-j),:],self.B1)),axis=1)
                elif i == j:
                    B1_temp = np.concatenate((B1_temp, self.B1),axis=1)
                elif i < j:
                    B1_temp = np.concatenate((B1_temp, np.zeros((self.xdim, self.udim))),axis=1)
            self.B_all = np.concatenate((self.B_all, B1_temp), axis=0)
        self.Q_mpc = np.eye(self.xdim * self.N)
        for i in range(self.xdim * self.N):
            if i % self.xdim >= 0 and i % self.xdim < self.xdim/2:
                self.Q_mpc[i,i] = 10
            else:
                self.Q_mpc[i,i] = .5
        self.R_mpc = np.eye(self.udim * self.N) * 0.5

        # self.P = self.B_all.T.dot(self.Q_mpc).dot(self.B_all) + self.R_mpc
        self.P = np.matmul(np.matmul(self.B_all.T, self.Q_mpc), self.B_all) + self.R_mpc
        self.P_csc = sparse.csc_matrix(self.P)
        self.q1 = np.matmul(self.Q_mpc, self.B_all)

        self.A = sparse.csc_matrix(np.eye(self.udim*self.N))

    # def set_reference_trajectory(self, z_d):
    #     self.z_d = z_d

    def update_barrier_locations(self, x, y, radius):
        self.barrier_x = x
        self.barrier_y = y
        self.barrier_radius = radius 

    def get_control(self, z_0, ref_traj):
        stepsize = self.stepsize
        # overflow: horizon exceed the end of trajectory
        if self.N * stepsize > ref_traj.shape[1]:
            self.N = (ref_traj.shape[1]) // stepsize
            self.update_params(self.dt, self.stepsize, self.N)

        dt = self.dt
        #z_ref: 4N x 1
        N = self.N
        z_ref = ref_traj[:,:N*stepsize:stepsize].ravel(order='F')
        q = np.matmul(np.matmul(z_0[:].T, self.A_all.T) - z_ref.T, self.q1).T
        
        # todo: correct the constraint precisely according to z0
        self.u = np.ones((self.udim*self.N, 1)) * 14.9375
        self.l = np.ones((self.udim*self.N, 1)) * -14.9375

        self.prob = osqp.OSQP()
        self.prob.setup(P=self.P_csc, q=q, A=self.A, u=self.u, l=self.l, verbose=False)
        self.res = self.prob.solve()

        if self.res.x[0] is None or np.isnan(self.res.x).any():
            self.res = None
            rospy.logwarn("MPC QP failed!")

        # print ('mpc osqp solve time', self.res.info.solve_time * 1000)
        
        z_next = np.matmul(self.A1, z_0[:]) + np.matmul(self.B1, self.res.x[0:self.udim]).reshape((self.xdim,1))
        # z_next = np.concatenate((z_next, [[1]]), axis=0)
        z_next_dot = (z_next - z_0) / dt / stepsize
        # print ('mpc.z_ref', z_next.T)

        # cons = np.matmul(np.matmul(np.matmul(z_0[:].T, self.A_all.T) - z_ref.T, self.Q_mpc), (np.matmul(self.A_all, z_0[:]) - z_ref.reshape((self.xdim*N,1))))
        # res = self.res.x.reshape((self.udim*N))
        # print cons[0,0]
        # print np.matmul(res.T, np.matmul(P, res)) + np.matmul(q.T, res)

        # if iters > 1185:
        #     zall = np.matmul(self.A_all, z_0[:]) + np.matmul(self.B_all, self.res.x.reshape((self.udim*N, 1)))
        #     zall = zall.reshape(N, 4).T
        #     z_ref = self.z_d[:,iters+stepsize:iters+stepsize+N*stepsize:stepsize]
        #     fig = plt.figure()
        #     plt.rcParams.update({'font.size': 12})
        #     plt.plot(z_ref[0,:],z_ref[1,:],'k-',label='ref')
        #     plt.plot(zall[0,:],zall[1,:],'g-',alpha=0.9,label='mpc',linewidth=3.0)
        #     plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower center", ncol=2)
        #     plt.xlabel('X Position')
        #     plt.ylabel('Y Position')
        #     plt.show()

        return z_next, z_next_dot


# T = 40
# dt = .1
# N = int(round(T/dt))
# t = np.linspace(0,T-2*dt,N-1)
# xdim=4
# udim=2

# x_d = np.stack((t, np.sin(2 * np.pi * t * .1),np.zeros(N-1), np.zeros(N-1)))
# x_d[2,:-1] = np.arctan2(np.diff(x_d[1,:]),np.diff(x_d[0,:]))
# x_d[3,:-1] = np.sqrt(np.diff(x_d[0,:])**2 + np.diff(x_d[1,:])**2)/dt
# x_d[2,-1]=x_d[2,-2]
# x_d[3,-1]=x_d[3,-2]

# true_dyn = DynamicsAckermannZModified(disturbance_scale_pos = 0.0, disturbance_scale_vel = -1.0, control_input_scale = 1.0)

# x0=np.array([[0.0],[0.0],[0.0],[0.0001]])
# z0 = true_dyn.convert_x_to_z(x0)


# z_d = np.zeros((xdim+1,N-1))
# z_1 = np.zeros((xdim+1,N-1))
# z_1[:,0:1] = z0
# i=0
# z_d[:,i+1:i+2] = true_dyn.convert_x_to_z(x_d[:,i+1:i+2])
# # for mpc
# for i in range(N-3):
# 	z_d[:,i+2:i+3] = true_dyn.convert_x_to_z(x_d[:,i+2:i+3])



# mpc = MPC(dt=.1, stepsize=1)
# mpc.set_reference_trajectory(z_d)
# print mpc.get_control(.1, 1, z0, 0)

import numpy as np
import copy

import rospy
import actionlib
import clbfet.msg
from clbfet.srv import *

from qp_solver import QPSolve
from dynamics import DynamicsQuadrotor
from cbf import BarrierQuadrotorPoint
import time
# from lyapunov import LyapunovAckermannZ

class AdaptiveClbf(object):
    def __init__(self, use_service = True, use_mpc=False, use_nmpc=False, use_trigger=False, use_IGPR=False):
        self.xdim = 6
        self.udim = 3
        self.use_service = use_service
        self.use_mpc = use_mpc
        self.use_nmpc = use_nmpc
        self.use_trigger = use_trigger

        # the value of u_lim will be change in self.update_param()
        self.u_lim = np.array([[-2.0,2.0],[-2.0,2.0],[-2.0,2.0]])
        self.K=np.block([[np.zeros((3,3)), np.zeros((3,3))],[np.eye(3), np.eye(3)]])

        self.dyn = DynamicsQuadrotor()

        self.model_trained = False
        
        if self.use_mpc:
            from mpc import MPC
            self.mpc = MPC(dt=.01, stepsize=1, N=20)
            
        if self.use_nmpc:
            from nmpc import MPC
            self.mpc = MPC(dt=.01, stepsize=1, N=10)

        if self.use_service:
            ## train model action
            self.train_model_action_client = actionlib.SimpleActionClient('train_model_service', clbfet.msg.TrainModelAction)
            self.train_model_action_client.wait_for_server()
            self.train_model_goal = clbfet.msg.TrainModelGoal()

            ## add data srv
            rospy.wait_for_service('add_data_2_model')
            self.model_add_data_srv = rospy.ServiceProxy('add_data_2_model', AddData2Model)

            ## predict srv
            rospy.wait_for_service('predict_model')
            self.model_predict_srv = rospy.ServiceProxy('predict_model', PredictModel)
        else:
            # setup non-service model object
            if not use_trigger:
                if not use_IGPR:
                    from model_service import ModelGPService
                    self.model = ModelGPService(self.xdim,use_service=False)
                else:
                    from model_service import ModelIGPRService
                    self.model = ModelIGPRService(self.xdim,self.odim,use_obs=True,use_service=False)
            else:
                if not use_IGPR:
                    from model_service import ModelGPService
                    self.model = ModelGPService(self.xdim,use_service=False)
                else:
                    from model_service import ModelIGPR2Service
                    self.model = ModelIGPR2Service(self.xdim,self.odim,use_obs=False,use_service=False)


        self.qpsolve = QPSolve(dyn=self.dyn,cbf_list=[],u_lim=self.u_lim,u_cost=0.0,u_prev_cost=1.0,p1_cost=1.0e8,p2_cost=1.0e8,verbose=False)

        x_init = np.zeros((self.xdim,1))
        x_init[3] = 0.01
        self.z_ref = self.dyn.convert_x_to_z(x_init)
        self.z = copy.copy(self.z_ref)
        self.z_ref_dot = np.zeros((self.xdim,1))
        self.z_dot = np.zeros((self.xdim//2,1))
        self.z_prev = copy.copy(self.z)
        self.y_out = np.zeros((self.xdim//2,1))
        self.mu_prev = np.zeros((self.xdim,1))
        self.u_prev = np.zeros((self.udim,1))
        self.u_prev_prev = np.zeros((self.udim,1))
        self.dt = 0.01
        self.max_error = 1.0

        self.barrier_locations={}
        self.barrier_locations["x"]=np.array([])
        self.barrier_locations["y"]=np.array([])
        self.barrier_locations["z"]=np.array([])
        self.barrier_radius = 1.0

        self.measurement_noise = 1.0
        self.true_dyn = None
        self.true_predict_error = 0.0
        self.predict_error = 0
        self.predict_var = np.zeros((self.xdim//2,1))

        self.debug={}

    def wrap_angle(self,a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def saturate(self,u,ulim):
        return np.array([np.clip(u[0],ulim[0][0],ulim[0][1]), np.clip(u[1],ulim[1][0],ulim[1][1]), np.clip(u[2],ulim[2][0],ulim[2][1])])

    def update_params(self,params):
        self.params = params
        print("updated params!")
        # todo
        self.u_lim_param = self.params["a_lim"]
        self.u_lim = np.array([[-self.u_lim_param,self.u_lim_param],
                                [-self.u_lim_param,self.u_lim_param],
                                [-self.u_lim_param,self.u_lim_param]],dtype=np.float32)
        self.qpsolve.u_lim = self.u_lim
        self.thrust_lim = self.params["thrust_lim"]

        self.k1 = self.params["kp_z"]
        self.k2 = self.params["kd_z"]
        self.A=np.block([[np.zeros((3,3),dtype=np.float32), np.eye(3,dtype=np.float32)],[-self.k1*np.eye(3,dtype=np.float32), -self.k2*np.eye(3,dtype=np.float32)]])
        self.qpsolve.update_ricatti(self.A)
        self.K=np.block([[self.k1*np.eye(3,dtype=np.float32), self.k2*np.eye(3,dtype=np.float32)]])
        self.max_error = self.params["max_error"]

        self.qpsolve.epsilon = self.params["clf_epsilon"]
        self.measurement_noise = self.params["measurement_noise"]

        self.qpsolve.u_cost = self.params["qp_u_cost"]
        self.qpsolve.u_prev_cost = self.params["qp_u_prev_cost"]
        self.qpsolve.p1_cost = self.params["qp_p1_cost"]
        self.qpsolve.p2_cost = self.params["qp_p2_cost"]
        self.qpsolve.verbose = self.params["qp_verbose"]
        self.qpsolve.ksig = self.params["qp_ksig"]
        self.qpsolve.max_var = self.params["qp_max_var"]
        self.verbose = self.params["verbose"]

        self.dt = self.params["dt"]
        if self.use_mpc:
            self.mpc.update_params(self.dt, self.params["mpc_stepsize"], self.params["mpc_N"])

        # update model params if not using service calls
        if not self.use_service:
            self.model.N_data = self.params["N_data"]
            self.model.verbose = self.params["learning_verbose"]
            self.model.N_updates = self.params["N_updates"]

    def update_barrier_locations(self,x,y,z,radius):
        self.barrier_locations["x"] = x
        self.barrier_locations["y"] = y
        self.barrier_locations["z"] = z
        self.barrier_radius = radius
        # if self.use_mpc:
        # 	self.mpc.update_barrier_locations(x, y, z, radius)

    def update_barriers(self):
        cbf_list = []
        bar_loc_x = copy.copy(self.barrier_locations["x"])
        bar_loc_y = copy.copy(self.barrier_locations["y"])
        bar_loc_z = copy.copy(self.barrier_locations["z"])
        bar_rad = self.barrier_radius

        # if self.params["use_barrier_vel"]:
        # 	cbf_list = cbf_list + \
        # 					[BarrierAckermannVelocityZ(bound_from_above=True, v_lim = self.params["max_velocity"], gamma=self.params["barrier_vel_gamma"]),
        # 					BarrierAckermannVelocityZ(bound_from_above=False, v_lim = self.params["min_velocity"], gamma=self.params["barrier_vel_gamma"])]

        if self.params["use_barrier_pointcloud"]:
            cbf_list = cbf_list + \
                        [BarrierQuadrotorPoint(x=bar_loc_x[i],y=bar_loc_y[i],z=bar_loc_z[i], radius=bar_rad[i], gamma_p=self.params["barrier_pc_gamma_p"], gamma=self.params["barrier_pc_gamma"]) for i in range(bar_loc_x.size)]
        self.qpsolve.cbf_list = cbf_list

    def get_control(self,z,z_ref,z_ref_dot,dt,use_model=False,add_data=True,check_model=True,use_qp=True,ref_traj=None):
        # assert z.shape[0] == self.xdim + 1
        # assert z_ref_dot.shape[0] == self.xdim + 1
        # assert z_ref.shape[0] == self.xdim + 1

        self.update_barriers()

        self.z = copy.copy(z.astype(np.float32))
        self.z_ref = copy.copy(z_ref.astype(np.float32))

        # self.z_ref_dot = (self.z_ref_next[:-1,:] - self.z_ref[:-1,:]) / dt # use / self.dt for test_adaptive_clbf
        self.z_ref_dot = copy.copy(z_ref_dot)

        if self.use_mpc:
            self.z_ref, self.z_ref_dot = self.mpc.get_control(z_0=self.z, ref_traj=ref_traj)

        mu_ad = np.zeros((self.xdim//2,1),dtype=np.float32)
        mDelta = np.zeros((self.xdim//2,1),dtype=np.float32)
        sigDelta = np.zeros((self.xdim//2,1),dtype=np.float32)
        trueDelta = np.zeros((self.xdim//2,1),dtype=np.float32)

        e = self.z_ref[:,:]-self.z[:,:]
        mu_pd = np.matmul(self.K,e)
        mu_pd = np.clip(mu_pd,-self.max_error,self.max_error)

        mu_rm = self.z_ref_dot[self.xdim//2:]

        mu_model = np.matmul(self.dyn.g(self.z_prev),self.u_prev) + self.dyn.f(self.z_prev)

        if add_data:
            if self.use_service:
                try:
                    self.model_add_data_srv(self.z.flatten(),self.z_prev.flatten(),mu_model.flatten(),dt)
                except:
                    print("add data service unavailable")
            else:
                req = AddData2Model()
                req.x_next = self.z.flatten()
                req.x = self.z_prev.flatten()
                req.mu_model = mu_model.flatten()
                req.dt = dt
                self.model.add_data(req)

        self.z_dot = (self.z[3:,:]-self.z_prev[3:,:])/dt - mu_model

        # if check_model and self.model.model_trained:
        if check_model and self.model_trained:
            # check how the model is doing.  compare the model's prediction with the actual sampled data.
            predict_service_success = False
            result = None
            if self.use_service:
                try:
                    result = self.model_predict_srv(self.z_prev.flatten())
                    if result.result:
                        predict_service_success = True
                except:
                    print("predict service unavailable")
            else:
                # print 'check model'
                req = PredictModel()
                req.x = self.z_prev.flatten()
                result = self.model.predict(req)
                predict_service_success = True

            if predict_service_success:
                self.y_out = np.expand_dims(result.y_out, axis=0).T
                var = np.expand_dims(result.var, axis=0).T

                if self.verbose:
                    print("predicted y_out: ", self.y_out)
                    print("predicted ynew: ", self.z_dot)
                    print("predicted var: ", var)

                self.predict_error = np.linalg.norm(self.y_out - self.z_dot)
                self.predict_var = var
                # print 'predict_error', self.predict_error


        if use_model and self.model_trained:
            predict_service_success = False
            result = None
            if self.use_service:
                try:
                    # print 'try request srv'
                    # start = time.time()
                    result = self.model_predict_srv(self.z.flatten())
                    # print 'cost time ', time.time() - start
                    if result.result:
                        predict_service_success = True
                except:
                    print("predict service unavailable")
            else:
                # print 'use model'
                req = PredictModel()
                req.x = self.z.flatten()
                result = self.model.predict(req)
                predict_service_success = True

            if predict_service_success:
                mDelta = np.expand_dims(result.y_out, axis=0).T
                sigDelta = np.expand_dims(result.var, axis=0).T

                # log error if true system model is available
                if self.true_dyn is not None:
                    trueDelta = self.true_dyn.f(self.z) - self.dyn.f(self.z)
                    # print ''
                    # print 'trueDelta', trueDelta.reshape((3,1)).T
                    # print 'mDelta', result.y_out
                    self.true_predict_error = np.linalg.norm(trueDelta - mDelta)

                mu_ad = mDelta
        else:
            sigDelta = np.ones((self.xdim//2,1))
            
        mu_d = mu_rm + mu_pd - mu_ad
        self.mu_qp = np.zeros((self.xdim//2,1))
        if use_qp:
                        # print("sigDelta: ", sigDelta)
            self.mu_qp = self.qpsolve.solve(self.z,self.z_ref,mu_d,sigDelta)
            # self.triggered = self.qpsolve.triggered

        self.mu_new = mu_d + self.mu_qp
        self.u_new = np.matmul(np.linalg.inv(self.dyn.g(self.z)), (self.mu_new-self.dyn.f(self.z)))

        u_new_unsaturated = copy.copy(self.u_new)
        #saturate in case constraints are not met
        self.u_new = self.saturate(self.u_new,self.u_lim)

        con = self.dyn.convert_mu_to_control(self.u_new)
        con[0] = min(con[0], self.thrust_lim)

        self.u_sat = self.dyn.convert_control_to_mu(con)


        # print('mu_rm', mu_rm.T)
        # print('mu_pd', mu_pd.T)
        # print('mu_ad', mu_ad.T)
        # print('mu_qp', self.mu_qp.T)
        # print '-----------------------------------------'
        # print('u_new', self.u_new)
        # print('u_sat', self.u_sat)
        # print('control', self.dyn.convert_mu_to_control(self.u_new))
        # print ('u_diff', self.u_new.reshape((3)) - self.u_sat)

        if self.verbose:
            print('z: ', self.z.T)
            print('z_ref: ', self.z_ref.T)
            print('mu_rm', mu_rm)
            print('mu_pd', mu_pd)
            print('mu_ad', mu_ad)
            print('mu_d', mu_d)
            print('mu_model', mu_model)
            print('rho', rho)
            print('mu_qp', self.mu_qp)
            print('mu',self.mu_new)
            print('u_new', self.u_new)
            print('u_unsat', u_new_unsaturated)
            print('trueDelta',trueDelta)
            print('true predict error', self.true_predict_error)
            print('mDelta', mDelta)
            print('sigDelta', sigDelta)

        self.debug["z"] = self.z.flatten().tolist()
        self.debug["z_ref"] = self.z_ref.flatten().tolist()
        self.debug["z_dot"] = self.z_dot.flatten().tolist()
        self.debug["y_out"] = self.y_out.flatten().tolist()
        self.debug["mu_rm"] = self.z_ref_dot.flatten().tolist()
        self.debug["mu_pd"] = mu_pd.flatten().tolist()
        self.debug["mu_ad"] = mu_ad.flatten().tolist()
        self.debug["mu_model"] = mu_model.flatten().tolist()
        self.debug["mu_qp"] = self.mu_qp.flatten().tolist()
        self.debug["mu"] = self.mu_new.flatten().tolist()
        self.debug["u_new"] = self.u_new.flatten().tolist()
        self.debug["u_unsat"] = u_new_unsaturated.flatten().tolist()
        self.debug["trueDelta"] = trueDelta.flatten().tolist()
        self.debug["true_predict_error"] = self.true_predict_error
        self.debug["mDelta"] = mDelta.flatten().tolist()
        self.debug["sigDelta"] = sigDelta.flatten().tolist()

        self.mu_prev = copy.copy(self.mu_new)
        self.u_prev_prev = copy.copy(self.u_prev)
        # self.u_prev = copy.copy(self.u_new)
        self.u_prev = copy.copy(self.u_sat.reshape((self.udim,1)))
        self.z_prev = copy.copy(self.z)

        return self.u_sat

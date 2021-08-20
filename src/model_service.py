#! /usr/bin/env python
import os

# don't use gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import rospy
import actionlib
import numpy as np

from dynamic_reconfigure.client import Client as DynamicReconfigureClient

import clbfet.msg
from clbfet.srv import *

import numpy as np
import matplotlib.pyplot as plt
from scaledgp import ScaledGP
from scipy import signal
# from progress.bar import Bar
import random

# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import ConstantKernel, RBF
import time

from IGPR import IGPR

BASE_PATH = os.path.expanduser('~/Documents')

class ModelService(object):
	_train_result = clbfet.msg.TrainModelResult()

	def __init__(self, xdim, use_service = True):

		self.xdim=xdim
		self.verbose = True

		if use_service:
			# train action server
			self._action_service = actionlib.SimpleActionServer('train_model_service', clbfet.msg.TrainModelAction, execute_cb=self.train, auto_start = False)
			self._action_service.start()
			# add data service
			self._add_data_srv = rospy.Service('add_data_2_model', AddData2Model, self.add_data)
			# predict service
			self._predict_srv = rospy.Service('predict_model', PredictModel, self.predict)

			# Create a dynamic reconfigure client
			# self.dyn_reconfig_client = DynamicReconfigureClient('controller_adaptiveclbf_reconfig', timeout=30, config_callback=self.reconfigure_cb)

			# self.N_data = rospy.get_param('/controller_adaptiveclbf/N_data',200)
			# self.verbose = rospy.get_param('/controller_adaptiveclbf/learning_verbose',False)


	def reconfigure_cb(self, config):
		self.N_data = config["N_data"]
		self.verbose = config["learning_verbose"]
		self.N_updates = config["N_updates"]

	def predict(self,req):
		# overload
		return None

	def train(self,goal):
		# overload
		return None

	def add_data(self,req):
		# overload
		return None


	def scale_data(self,x,xmean,xstd):
		if (xstd == 0).any():
			return (x-xmean)
		else:
			return (x - xmean) / xstd

	def unscale_data(self,x,xmean,xstd):
		if (xstd == 0).any():
			return x + xmean
		else:
			return x * xstd + xmean


class ModelGPService(ModelService):
	def __init__(self,xdim,use_service=True):
		ModelService.__init__(self,xdim,use_service)
		model_xdim=self.xdim
		model_ydim=self.xdim//2

		self.m = ScaledGP(xdim=model_xdim,ydim=model_ydim)
		self.y = np.zeros((0,model_ydim))
		self.Z = np.zeros((0,model_xdim))
		self.N_data = 60

	def predict(self,req):
		# rospy.loginfo('predicting')
		if not hasattr(self, 'm'):
			resp = PredictModelResponse()
			resp.result = False
			return resp
		x = np.expand_dims(req.x, axis=0).T

		# format the input and use the model to make a prediction.
		y, var = self.m.predict(x.T)
		y_out = y.T

		resp = PredictModelResponse()
		resp.y_out = y_out.flatten()
		resp.var = var.T.flatten()
		resp.result = True
		return resp

	def train(self, goal=None):
		success = True
		
		# print ''
		# print '--------------------------------'
		# print 'y', self.y
		# print '--------------------------------'
		if goal is not None:
			# goal was cancelled
			if self._action_service.is_preempt_requested():
				print("Prempt training request")
				self._action_service.set_preempted()
				success = False

		# train model.  this gets called by the training thread on timer_cb() in adaptive_clbf_node.
		if success and self.Z.shape[0] > 0 and self.Z.shape[0] == self.y.shape[0]:
			self.m.optimize(self.Z,self.y)
			if goal is not None:
				self._train_result.model_trained = True
				self._action_service.set_succeeded(self._train_result)
		else:
			if goal is not None:
				self._train_result.model_trained = False
				self._action_service.set_succeeded(self._train_result)

	def add_data(self,req):
		# rospy.loginfo('adding data')
		if not hasattr(self, 'y'):
			return AddData2ModelResponse(False)

		x_next = np.expand_dims(req.x_next, axis=0).T
		x = np.expand_dims(req.x, axis=0).T
		mu_model = np.expand_dims(req.mu_model, axis=0).T
		# obs = np.expand_dims(req.obs, axis=0).T
		dt = req.dt

		# add a sample to the history of data
		x_dot = (x_next[3:,:]-x[3:,:])/dt
		ynew = x_dot - mu_model
		self.Z = np.concatenate((self.Z,x.T))
		self.y = np.concatenate((self.y,ynew.T))

		# throw away old samples if too many samples collected.
		if self.y.shape[0] > self.N_data:
			self.y = self.y[-self.N_data:,:]
			self.Z = self.Z[-self.N_data:,:]
			# self.y = np.delete(self.y,random.randint(0,self.N_data-1),axis=0)
			# self.Z = np.delete(self.Z,random.randint(0,self.N_data-1),axis=0)

		if self.verbose:
			print("ynew",ynew.T)
			# print("Znew",Znew)
			# print("x_dot",x_dot)
			print('x_next', x_next)
			print('x', x)
			# print("mu_model",mu_model)
			print("dt",dt)
			# print("n data:", self.y.shape[0])
			# print("prediction error:", self.predict_error)
			# print("predict var:", self.predict_var)

		return AddData2ModelResponse(True)

class ModelsklGPService(ModelService):
	def __init__(self,xdim,odim,use_obs=False,use_service=True):
		ModelService.__init__(self,xdim,odim,use_obs,use_service)
		model_xdim=self.xdim
		model_ydim=self.xdim/2

		kernel = ConstantKernel(constant_value=1., constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=1, length_scale_bounds=(1e-2, 1e2))
		self.m = []
		for i in range(model_ydim):
			self.m.append(GaussianProcessRegressor(kernel=kernel, alpha=1e-6, optimizer=None))
		self.y = np.zeros((0,model_ydim))
		self.Z = np.zeros((0,model_xdim))
		self.N_data = 400

	def predict(self,req):
		if not hasattr(self, 'm'):
			resp = PredictModelResponse()
			resp.result = False
			return resp
		x = np.expand_dims(req.x, axis=0).T
		
		# print self.y

		# format the input and use the model to make a prediction.
		y = np.empty((3,1))
		var = np.empty((3,1))
		# print '---'
		# print self.m[2].X_train_
		# print self.m[2].y_train_
		for i in range(self.xdim/2):
			yt, vart = self.m[i].predict(x.T, return_std=True)
			# print yt
			y[i,0] = yt[0]
			var[i,0] = vart[0]
		# print y

		y_out = y.T
		# print 'y_out', y_out
		# print y_out.flatten()
		resp = PredictModelResponse()
		resp.y_out = y_out.flatten()
		resp.var = var.T.flatten()
		resp.result = True

		return resp

	def train(self, goal=None):
		success = True
		
		# print ''
		# print '--------------------------------'
		# print 'y', self.y
		# print '--------------------------------'
		if goal is not None:
			# goal was cancelled
			if self._action_service.is_preempt_requested():
				print("Preempt training request")
				self._action_service.set_preempted()
				success = False

		# train model.  this gets called by the training thread on timer_cb() in adaptive_clbf_node.
		if success and self.Z.shape[0] > 0 and self.Z.shape[0] == self.y.shape[0]:
			# self.m.optimize(self.Z,self.y)
			for i in range(self.xdim/2):
				self.m[i].fit(self.Z, self.y[:,i].T)
			# print ' '
			# print '*******************************'
			# print self.y[:,2].T
			# print '*******************************'
			if goal is not None:
				self._train_result.model_trained = True
				self._action_service.set_succeeded(self._train_result)
		else:
			if goal is not None:
				self._train_result.model_trained = False
				self._action_service.set_succeeded(self._train_result)

	def add_data(self,req):
		if not hasattr(self, 'y'):
			return AddData2ModelResponse(False)

		x_next = np.expand_dims(req.x_next, axis=0).T
		x = np.expand_dims(req.x, axis=0).T
		mu_model = np.expand_dims(req.mu_model, axis=0).T
		# obs = np.expand_dims(req.obs, axis=0).T
		dt = req.dt

		# add a sample to the history of data
		x_dot = (x_next[3:,:]-x[3:,:])/dt
		ynew = x_dot - mu_model
		self.Z = np.concatenate((self.Z,x.T))
		self.y = np.concatenate((self.y,ynew.T))
		# print ''
		# print '--------------------------------'
		# # print 'x_next', x_next
		# # print 'x', x
		# print 'ynew.T', ynew.T
		# print 'x', x
		# print 'Znew', Znew
		# # print 'obs', obs
		# print '--------------------------------'

		# throw away old samples if too many samples collected.
		if self.y.shape[0] > self.N_data:
			self.y = self.y[-self.N_data:,:]
			self.Z = self.Z[-self.N_data:,:]
			# self.y = np.delete(self.y,random.randint(0,self.N_data-1),axis=0)
			# self.Z = np.delete(self.Z,random.randint(0,self.N_data-1),axis=0)
		
		# print 'self.y[:,2].T', self.y[:,2].T
		# for i in range(self.xdim/2):
		# 	self.m[i].fit(self.Z, self.y[:,i].T)

		if self.verbose:
			print("obs", obs)
			print("ynew",ynew)
			print("ynew_rotated", ynew_rotated)
			print("Znew",Znew)
			print("x_dot",x_dot)
			print("mu_model",mu_model)
			print("dt",dt)
			print("n data:", self.y.shape[0])
			# print("prediction error:", self.predict_error)
			# print("predict var:", self.predict_var)

		return AddData2ModelResponse(True)

# update date instead of optimize when train
class ModelGPService2(ModelService):
	def __init__(self,xdim,odim,use_obs=False,use_service=True):
		ModelService.__init__(self,xdim,odim,use_obs,use_service)
		# note:  use use_obs and observations with caution.  model may overfit to this input.
		model_xdim=self.xdim/2
		if self.use_obs:
			 model_xdim += self.odim
		model_ydim=self.xdim/2

		self.m = ScaledGP(xdim=model_xdim,ydim=model_ydim)
		self.y = np.zeros((0,model_ydim))
		self.Z = np.zeros((0,model_xdim))
		self.y_hist = np.zeros((0,model_ydim))
		self.Z_hist = np.zeros((0,model_xdim))
		self.N_data = 400

	def rotate(self,x,theta):
		x_body = np.zeros((2,1))
		x_body[0] = x[0] * np.cos(theta) + x[1] * np.sin(theta)
		x_body[1] = -x[0] * np.sin(theta) + x[1] * np.cos(theta)
		return x_body

	def make_input(self,x,obs):
		# format input vector
		theta = obs[0]
		x_body = self.rotate(x[2:-1,:],theta)
		if self.use_obs:
			Z = np.concatenate((x_body,obs[1:,:])).T
		else:
			Z = np.concatenate((x_body)).T

		#normalize input by mean and variance
		# Z = (Z - self.Zmean) / self.Zvar

		return Z

	def predict(self,req):
		if not hasattr(self, 'm'):
			resp = PredictModelResponse()
			resp.result = False
			return resp
		x = np.expand_dims(req.x, axis=0).T
		obs = np.expand_dims(req.obs, axis=0).T

		# format the input and use the model to make a prediction.
		Z = self.make_input(x,obs)
		y, var = self.m.predict(Z)
		# theta = np.arctan2(x[3]*x[4],x[2]*x[4])
		theta=obs[0]
		y_out = self.rotate(y.T,-theta)

		resp = PredictModelResponse()
		resp.y_out = y_out.flatten()
		resp.var = var.T.flatten()
		resp.result = True

		return resp

	def train(self, goal=None):
		success = True

		if goal is not None:
			# goal was cancelled
			if self._action_service.is_preempt_requested():
				print("Preempt training request")
				self._action_service.set_preempted()
				success = False

		# train model.  this gets called by the training thread on timer_cb() in adaptive_clbf_node.
		if success and self.Z.shape[0] > 0 and self.Z.shape[0] == self.y.shape[0]:
			self.update()
			self.m.optimize(self.Z,self.y)
			if goal is not None:
				self._train_result.model_trained = True
				self._action_service.set_succeeded(self._train_result)
		else:
			if goal is not None:
				self._train_result.model_trained = False
				self._action_service.set_succeeded(self._train_result)

	def optimize(self, goal=None):
		success = True

		if goal is not None:
			# goal was cancelled
			if self._action_service.is_preempt_requested():
				print("Preempt training request")
				self._action_service.set_preempted()
				success = False

		# train model.  this gets called by the training thread on timer_cb() in adaptive_clbf_node.
		if success and self.Z.shape[0] > 0 and self.Z.shape[0] == self.y.shape[0]:
			self.m.optimize(self.Z,self.y)
			if goal is not None:
				self._train_result.model_trained = True
				self._action_service.set_succeeded(self._train_result)
		else:
			if goal is not None:
				self._train_result.model_trained = False
				self._action_service.set_succeeded(self._train_result)
	
	def add_data(self,req):
		if not hasattr(self, 'y'):
			return AddData2ModelResponse(False)

		x_next = np.expand_dims(req.x_next, axis=0).T
		x = np.expand_dims(req.x, axis=0).T
		mu_model = np.expand_dims(req.mu_model, axis=0).T
		obs = np.expand_dims(req.obs, axis=0).T
		dt = req.dt

		# add a sample to the history of data
		x_dot = (x_next[2:-1,:]-x[2:-1,:])/dt
		ynew = x_dot - mu_model
		Znew = self.make_input(x,obs)
		# theta = np.arctan2(x[3]*x[4],x[2]*x[4])
		theta=obs[0]
		ynew_rotated = self.rotate(ynew,theta)
		self.y_hist = np.concatenate((self.y,ynew_rotated.T))
		self.Z_hist = np.concatenate((self.Z,Znew))

		# throw away old samples if too many samples collected.
		# if self.y.shape[0] > self.N_data:
		# 	self.y = self.y[-self.N_data:,:]
		# 	self.Z = self.Z[-self.N_data:,:]
			# self.y = np.delete(self.y,random.randint(0,self.N_data-1),axis=0)
			# self.Z = np.delete(self.Z,random.randint(0,self.N_data-1),axis=0)

		if self.verbose:
			print("obs", obs)
			print("ynew",ynew)
			print("ynew_rotated", ynew_rotated)
			print("Znew",Znew)
			print("x_dot",x_dot)
			print("mu_model",mu_model)
			print("dt",dt)
			print("n data:", self.y.shape[0])
			# print("prediction error:", self.predict_error)
			# print("predict var:", self.predict_var)

		return AddData2ModelResponse(True)
	
	def update(self):
		if self.y_hist.shape[0] < self.N_data:
			self.y = self.y_hist
			self.Z = self.Z_hist
		else:
			self.y = self.y[-self.N_data:,:]
			self.Z = self.Z[-self.N_data:,:]

class ModelIGPRService(ModelService):
	def __init__(self,xdim,use_service=True):
		ModelService.__init__(self,xdim,use_service)
		self.ydim=self.xdim/2
		self.max_k_matrix_size = 20

		self.m = []
		for i in range(self.ydim):
			self.m.append(IGPR(max_k_matrix_size=self.max_k_matrix_size))
		self.y = np.zeros((0,self.ydim))
		self.Z = np.zeros((0,self.xdim))
		self.N_data = self.max_k_matrix_size

	def predict(self,req):
		if not hasattr(self, 'm'):
			resp = PredictModelResponse()
			resp.result = False
			return resp
		x = np.expand_dims(req.x, axis=0).T

		# format the input and use the model to make a prediction.
		y = np.zeros((self.ydim, 1))
		var = np.zeros((self.ydim, 1))
		for i in range(self.ydim):
			yt, vart = self.m[i].predict(x.reshape((1,self.xdim)))
			y[i,0] = yt[0]
			var[i,0] = vart# np.sqrt(vart)
		# var = np.hstack((var, var, var))

		resp = PredictModelResponse()
		resp.y_out = y.flatten()
		resp.var = var.T.flatten()
		resp.result = True

		return resp

	def train(self, goal=None):
		success = True

		if goal is not None:
			# goal was cancelled
			if self._action_service.is_preempt_requested():
				print("Preempt training request")
				self._action_service.set_preempted()
				success = False

		# train model.  this gets called by the training thread on timer_cb() in adaptive_clbf_node.
		if success and self.Z.shape[0] > 0 and self.Z.shape[0] == self.y.shape[0]:
			# self.m.optimize(self.Z,self.y)
			if goal is not None:
				self._train_result.model_trained = True
				self._action_service.set_succeeded(self._train_result)
		else:
			if goal is not None:
				self._train_result.model_trained = False
				self._action_service.set_succeeded(self._train_result)
	def add_data(self,req):
		if not hasattr(self, 'y'):
			return AddData2ModelResponse(False)

		x_next = np.expand_dims(req.x_next, axis=0).T
		x = np.expand_dims(req.x, axis=0).T
		mu_model = np.expand_dims(req.mu_model, axis=0).T
		dt = req.dt

		# add a sample to the history of data
		x_dot = (x_next[3:,:]-x[3:,:])/dt
		ynew = x_dot - mu_model
		self.y = np.concatenate((self.y,ynew.T))
		self.Z = np.concatenate((self.Z,x.T))
		print ('ynew.T', ynew.T)
		for i in range(self.ydim):
			self.m[i].learn(x.reshape((self.xdim)), ynew[i,0])

		# throw away old samples if too many samples collected.
		if self.y.shape[0] > self.N_data:
			self.y = self.y[-self.N_data:,:]
			self.Z = self.Z[-self.N_data:,:]
			# self.y = np.delete(self.y,random.randint(0,self.N_data-1),axis=0)
			# self.Z = np.delete(self.Z,random.randint(0,self.N_data-1),axis=0)

		if self.verbose:
			print("obs", obs)
			print("ynew",ynew)
			print("ynew_rotated", ynew_rotated)
			print("Znew",Znew)
			print("x_dot",x_dot)
			print("mu_model",mu_model)
			print("dt",dt)
			print("n data:", self.y.shape[0])
			# print("prediction error:", self.predict_error)
			# print("predict var:", self.predict_var)

		return AddData2ModelResponse(True)

class ModelIGPR2Service(ModelService):
	def __init__(self,xdim,odim,use_obs=False,use_service=True):
		ModelService.__init__(self,xdim,odim,use_obs,use_service)
		self.ydim=self.xdim/2
		self.max_k_matrix_size = 20
		self.data_counts_need_update = 0

		self.m = []
		for i in range(self.ydim):
			self.m.append(IGPR(max_k_matrix_size=self.max_k_matrix_size))
		self.y = np.zeros((0,self.ydim))
		self.Z = np.zeros((0,self.xdim))
		self.N_data = self.max_k_matrix_size

	def predict(self,req):
		if not hasattr(self, 'm'):
			resp = PredictModelResponse()
			resp.result = False
			return resp
		x = np.expand_dims(req.x, axis=0).T

		# format the input and use the model to make a prediction.
		y = np.zeros((self.ydim, 1))
		var = np.zeros((self.ydim, 1))
		for i in range(self.ydim):
			yt, vart = self.m[i].predict(x.reshape((1,self.xdim)))
			y[i,0] = yt[0]
			var[i,0] = vart# np.sqrt(vart)
		# var = np.hstack((var, var, var))

		resp = PredictModelResponse()
		resp.y_out = y.flatten()
		resp.var = var.T.flatten()
		resp.result = True

		return resp

	def train(self, goal=None):
		success = True

		if goal is not None:
			# goal was cancelled
			if self._action_service.is_preempt_requested():
				print("Preempt training request")
				self._action_service.set_preempted()
				success = False

		# train model.  this gets called by the training thread on timer_cb() in adaptive_clbf_node.
		if success and self.Z.shape[0] > 0 and self.Z.shape[0] == self.y.shape[0]:
			# self.m.optimize(self.Z,self.y)
			self.data_counts_need_update = min(self.data_counts_need_update, self.max_k_matrix_size)
			for i in range(self.y.shape[0] - self.data_counts_need_update, self.y.shape[0]):
				for j in range(self.ydim):
					self.m[j].learn(self.Z[i].reshape((self.xdim)), self.y[i,j])
			self.data_counts_need_update = 0

			for i in range(self.ydim):
				self.m[i].hyperparam_optimization()

			if goal is not None:
				self._train_result.model_trained = True
				self._action_service.set_succeeded(self._train_result)
		else:
			if goal is not None:
				self._train_result.model_trained = False
				self._action_service.set_succeeded(self._train_result)
	def add_data(self,req):
		if not hasattr(self, 'y'):
			return AddData2ModelResponse(False)

		x_next = np.expand_dims(req.x_next, axis=0).T
		x = np.expand_dims(req.x, axis=0).T
		mu_model = np.expand_dims(req.mu_model, axis=0).T
		dt = req.dt

		# add a sample to the history of data
		x_dot = (x_next[3:,:]-x[3:,:])/dt
		ynew = x_dot - mu_model
		self.y = np.concatenate((self.y,ynew.T))
		self.Z = np.concatenate((self.Z,x.T))
		# print 'ynew.T', ynew.T
		# for i in range(self.ydim):
		# 	self.m[i].learn(x.reshape((self.xdim)), ynew[i,0])
		
		self.data_counts_need_update += 1

		# throw away old samples if too many samples collected.
		if self.y.shape[0] > self.N_data:
			self.y = self.y[-self.N_data:,:]
			self.Z = self.Z[-self.N_data:,:]
			# self.y = np.delete(self.y,random.randint(0,self.N_data-1),axis=0)
			# self.Z = np.delete(self.Z,random.randint(0,self.N_data-1),axis=0)
		
		if self.y.shape[0] == 1:
			self.train()

		if self.verbose:
			print("obs", obs)
			print("ynew",ynew)
			print("ynew_rotated", ynew_rotated)
			print("Znew",Znew)
			print("x_dot",x_dot)
			print("mu_model",mu_model)
			print("dt",dt)
			print("n data:", self.y.shape[0])
			# print("prediction error:", self.predict_error)
			# print("predict var:", self.predict_var)

		return AddData2ModelResponse(True)

if __name__ == '__main__':
    rospy.init_node('model_service')
    # server = ModelVanillaService(4,6, use_obs = True) # TODO: put this in yaml or somewhere else
    # server = ModelALPaCAService(4,6, use_obs = True) # TODO: put this in yaml or somewhere else
    server = ModelGPService(6, use_service = True) # TODO: put this in yaml or somewhere else
    # server = ModelIGPRService(6, use_service = True) # TODO: put this in yaml or somewhere else
    rospy.spin()

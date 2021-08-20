#! /usr/bin/env python

import numpy as np
import copy
from adaptive_clbf import AdaptiveClbf
from dynamics import DynamicsQuadrotorModified
from progress.bar import Bar
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


np.random.seed(0)
adaptive_clbf_mpc_trigger = AdaptiveClbf(use_service = False, use_mpc=True, use_trigger=True, use_IGPR=False)

params={}
params["a_lim"] = 13.93
params["thrust_lim"] = 25#0.5
params["kp_z"] = 1.0
params["kd_z"] = 1.0
params["clf_epsilon"] = 100.0


params["qp_u_cost"] = 100.0
params["qp_u_prev_cost"] = 1.0
# params["qp_p1_cost"] = 1.0
params["qp_p1_cost"] = 1.0e8
params["qp_p2_cost"] = 1.0e12
params["qp_max_var"] = 1.5
params["qp_verbose"] = False
params["max_velocity"] = 2.0
params["min_velocity"] = 0.5
params["barrier_vel_gamma"] = 10.0
params["use_barrier_vel"] = True
params["use_barrier_pointcloud"] = True
params["barrier_radius"] = 1
params["barrier_radius_velocity_scale"] = 0.0
params["barrier_pc_gamma_p"] = 5.0
params["barrier_pc_gamma"] = 0.08
params["verbose"] = False
params["dt"] = 0.02
params["max_error"] = 10.0

# alpaca params
# params["qp_ksig"] = 2.0e3
# params["measurement_noise"] = 1.0e-3

# gp params
params["qp_ksig"] = 1.0e5
params["measurement_noise"] = 1.0

# vanilla nn params
params["qp_ksig"] = 1.0e2
params["measurement_noise"] = 1.0

#600 60 500 600 with obstacles
params["N_data"] = 60#600
params["learning_verbose"] = False
params["N_updates"] = 50

params["mpc_stepsize"] = 1
params["mpc_N"] = 20

true_dyn = DynamicsQuadrotorModified(disturbance_scale_pos = 0.0, disturbance_scale_vel = -1.0, control_input_scale = 1.0)

adaptive_clbf_mpc_trigger.update_params(params)

adaptive_clbf_mpc_trigger.true_dyn = true_dyn

# T = 40
T = 10
dt = params["dt"]
N = int(round(T/dt))
t = np.linspace(0,T-2*dt,N-1)
xdim=6
udim=3

width = 1.0
speed = 1.0
freq = 1.0/10
# x_d = np.stack((t * speed, width * np.sin(2 * np.pi * t * freq), t*speed*.1, np.zeros(N-1), np.zeros(N-1), np.zeros(N-1))) # no use
# x_d = np.stack((0.25 * t * np.cos(0.2 * np.pi * t), 0.25 * t * np.sin(0.2 * np.pi * t), 20 - 0.5 * t, np.zeros(N-1), np.zeros(N-1), np.zeros(N-1))) # luoxuan xiangxia
# x_d = np.stack((4 * np.cos(0.2 * t), 2 * np.sin(0.2 * 2 * t), 2 * np.sin(0.2 * 2 * t), np.zeros(N-1), np.zeros(N-1), np.zeros(N-1)))	# 8
# x_d = np.stack((t, t, t, np.zeros(N-1), np.zeros(N-1), np.zeros(N-1)))	# lined
# x_d = np.stack((1*t, 2*t, t/(t+1), np.zeros(N-1), np.zeros(N-1), np.zeros(N-1)))	# curved
# x_d = np.stack((-2 * np.sin(0.5 * t), 5 + 2 * np.cos(0.5 * t), 2*np.cos(t), np.zeros(N-1), np.zeros(N-1), np.zeros(N-1)))	# circular
# x_d = np.stack((2 * np.sin(0.5 * t), 2 - 2 * np.cos(0.5 * t), 0.2 * t, np.zeros(N-1), np.zeros(N-1), np.zeros(N-1)))	# spiral curve
# x_d = np.stack((np.ones(N-1)*4, np.zeros(N-1), np.zeros(N-1), np.zeros(N-1), np.zeros(N-1), np.zeros(N-1)))
x_d = np.stack((np.zeros(N-1), np.zeros(N-1), np.ones(N-1)*.5, np.zeros(N-1), np.zeros(N-1), np.zeros(N-1)))
x_d[2,-300:] = 0.13
print (x_d.shape)
print (x_d[:,:].shape)

# x = [px, py, theta, v]
# x_d[2,:-1] = np.arctan2(np.diff(x_d[1,:]),np.diff(x_d[0,:]))
# x_d[3,:-1] = np.sqrt(np.diff(x_d[0,:])**2 + np.diff(x_d[1,:])**2)/dt
# x = [px, py, pz, vx, vy, vz]
x_d[3,:-1] = np.diff(x_d[0,:])
x_d[4,:-1] = np.diff(x_d[1,:])
x_d[5,:-1] = np.diff(x_d[2,:])
x_d[3,-1]=x_d[3,-2]
x_d[4,-1]=x_d[4,-2]
x_d[5,-1]=x_d[5,-2]


barrier_x = np.array([-1.5, -4, 2])
barrier_y = np.array([-1.39, 0, -1.7])
barrier_z = np.array([-1.39, 0, -1.7])
# barrier_x = np.array([x_d[0, 100], x_d[0, 190], x_d[0, 280]])
# barrier_y = np.array([x_d[1, 100], x_d[1, 190], x_d[1, 280]])
# barrier_z = np.array([x_d[2, 100], x_d[2, 190], x_d[2, 280]])
# barrier_r = np.array([1, 1, 1])
barrier_x = np.array([-100])
barrier_y = np.array([-100])
barrier_z = np.array([-100])
barrier_r = np.array([1])
# barrier_x = np.array([])
# barrier_y = np.array([])
# barrier_z = np.array([])
adaptive_clbf_mpc_trigger.update_barrier_locations(barrier_x,barrier_y,barrier_z,barrier_r)

x0=np.array([[0.0],[0.0],[0.0],[0.0001],[0.0001],[0.0001]])
z0 = true_dyn.convert_x_to_z(x0)


train_interval = 10#40
start_training = 100
# last_training = -1


x0 = x_d[:,0:1]
x0 = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
z0 = true_dyn.convert_x_to_z(x0)

u_1 = np.zeros((udim,N))
x_1 = np.zeros((xdim,N-1))
x_1[:,0:1] = x0


z_d = np.zeros((xdim,N-1))
z_1 = np.zeros((xdim,N-1))
z_1[:,0:1] = z0


z_d_dot = np.zeros((xdim,1))


prediction_error_1 = np.zeros(N)
prediction_error_true_1 = np.zeros(N)
prediction_var_1 = np.zeros((xdim//2,N))
# trGssGP_1 = np.zeros(N)
# trGssGP_2 = np.zeros(N)
# trGssGP_3 = np.zeros(N)
# trGssGP_4 = np.zeros(N)

control_time_log_1 = []
update_time_log_1 = []
trigger_iter_log_1 = []
constraint_value_log_1 = []

i=0
z_d[:,i+1:i+2] = true_dyn.convert_x_to_z(x_d[:,i+1:i+2])
# for mpc
for i in range(N-3):
	z_d[:,i+2:i+3] = true_dyn.convert_x_to_z(x_d[:,i+2:i+3])

# adaptive_clbf_mpc_trigger.model.verbose = True
bar = Bar(max=N-1)
for i in range(N-2):

	# if i == 220:
	# 	clbf_mpc.true_dyn.set_disturbance_type(1)
	# 	adaptive_mpc.true_dyn.set_disturbance_type(1)
	# 	adaptive_clbf_mpc_trigger.true_dyn.set_disturbance_type(1)
	# 	adaptive_ad.true_dyn.set_disturbance_type(1)
	# elif i == 750:
	# 	clbf_mpc.true_dyn.set_disturbance_type(0)
	# 	adaptive_mpc.true_dyn.set_disturbance_type(0)
	# 	adaptive_clbf_mpc_trigger.true_dyn.set_disturbance_type(0)
	# 	adaptive_ad.true_dyn.set_disturbance_type(0)

	bar.next()
	print()
	start0 = time.time()

	if i < N-3:
		z_d[:,i+2:i+3] = true_dyn.convert_x_to_z(x_d[:,i+2:i+3])
		z_d_dot = (z_d[:,i+2:i+3] - z_d[:,i+1:i+2])/dt

	if i == 0:
		add_data = False
	else:
		add_data = True

	start = time.time()
	u_1[:,i+1] = adaptive_clbf_mpc_trigger.get_control(z_1[:,i:i+1],z_d[:,i+1:i+2],z_d_dot,dt=dt,use_model=False,add_data=add_data,use_qp=True,ref_traj=z_d[:,i+1:i+21])
	control_time_log_1.append((time.time() - start)*1000)
	start = time.time()
	if (i - 1 == train_interval) or adaptive_clbf_mpc_trigger.qpsolve.triggered:
	# if True:
	# if (i - 1 == train_interval):
		# adaptive_clbf_mpc_trigger.model.update()
		trigger_iter_log_1.append(i)
		adaptive_clbf_mpc_trigger.model.train()
		adaptive_clbf_mpc_trigger.model_trained = True
	update_time_log_1.append((time.time() - start)*1000)
	constraint_value_log_1.append(adaptive_clbf_mpc_trigger.qpsolve.constraint_value)
	prediction_error_1[i] = adaptive_clbf_mpc_trigger.predict_error
	prediction_error_true_1[i] = adaptive_clbf_mpc_trigger.true_predict_error
	prediction_var_1[:,i:i+1] = np.clip(adaptive_clbf_mpc_trigger.predict_var,0,params["qp_max_var"])
	# trGssGP_mpc_trigger[i] = adaptive_clbf_mpc_trigger.qpsolve.trGssGP

	# dt = np.random.uniform(0.05,0.15)
	c_1 = copy.copy(u_1[:,i+1:i+2])
	
	z_1[:,i+1:i+2] = true_dyn.step(z_1[:,i:i+1],c_1,dt)

	x_1[:,i+1:i+2] = true_dyn.convert_z_to_x(z_1[:,i+1:i+2])
	
	print('Iteration ', i, ', Time elapsed (ms): ', (time.time() - start0)*1000)
	print('control', u_1[:,i+1].T)
	print('x', x_1[:,i+1].T)

path_to_save = '/home/wuzhixuan/Desktop/cpr_data/'
# todo: plot mpc, trigger, mpctrigger
plt.figure()
plt.rcParams.update({'font.size': 12})
plt.rcParams['axes.unicode_minus'] = False
plt.semilogy(t[:-1],prediction_var_1[0,:-2],'g-',alpha=0.9)
plt.semilogy(t[:-1],prediction_var_1[1,:-2],'g:',alpha=0.9)
plt.semilogy(t[:-1],prediction_var_1[2,:-2],'g--',alpha=0.9)
plt.xlabel("Time(s)")
plt.ylabel(r"$\sigma_{\bar{\Delta}}(x,\mu)$")
# plt.legend(['all[1]','all[2]','without_qp[1]','without_qp[2]','without_mpc[1]','without_mpc[2]'],bbox_to_anchor=(0,1.2,1,0.2), loc="upper center", ncol=2)
plt.legend(['all[1]','all[2]','all[3]'],bbox_to_anchor=(0,1.2,1,0.2), loc="upper center", ncol=2)
plt.plot([t[0],t[-1]],[params["measurement_noise"],params["measurement_noise"]],'r--')
plt.savefig(path_to_save+'1.1.eps', dpi=600, format='eps',bbox_inches='tight')
plt.savefig(path_to_save+'1.1.png', dpi=600, format='png',bbox_inches='tight')
# plt.subplot(312)
# plt.plot(t[:-1],trGssGP[:-2],'g--',alpha=0.9)
# plt.ylabel("trace(GssGP)")
# plt.xlabel("Time(s)")
plt.figure()
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 12})
plt.plot(t[:-1],prediction_error_1[:-2],'g-',alpha=0.9, label='all')
plt.plot(t[:-1],prediction_error_true_1[:-2],'g:',alpha=0.9, label='all')
# plt.ylim([0,1.0])
plt.ylabel("Prediction error")
plt.xlabel("Time(s)")
# plt.legend(['all[1]','all[2]','without_qp[1]','without_qp[2]','without_mpc[1]','without_mpc[2]'],bbox_to_anchor=(0,1.2,1,0.2), loc="upper center", ncol=2)
plt.legend(bbox_to_anchor=(0,1.2,1,0.2), loc="upper center", ncol=1)
plt.savefig(path_to_save+'1.2.eps', dpi=600, format='eps',bbox_inches='tight')
plt.savefig(path_to_save+'1.2.png', dpi=600, format='png',bbox_inches='tight')
np.savetxt(path_to_save+'sim_predict_error_log_1.txt', prediction_error_1)
np.savetxt(path_to_save+'sim_predict_var_log_1.txt', prediction_var_1)


fig = plt.figure()
plt.rcParams.update({'font.size': 12})
plt.rcParams['axes.unicode_minus'] = False

ax = plt.axes(projection='3d')
ax.plot3D(x_1[0,:], x_1[1,:], x_1[2,:], 'g-',alpha=0.9,label='LB-FBLC-QP-MPC')
ax.plot3D(x_d[0,:], x_d[1,:], x_d[2,:], 'k-',label='ref')
# ax.scatter3D(barrier_x[:], barrier_y[:], barrier_z[:], c='r', s=5000)

# to = np.linspace(0, np.pi * 2, 100)
# so = np.linspace(0, np.pi, 100)
# to, so = np.meshgrid(to, so)
# xo = np.cos(to) * np.sin(so)
# yo = np.sin(to) * np.sin(so)
# zo = np.cos(so)
# ax.plot_surface(xo, yo, zo, rstride=1, cstride=1, cmap='r')

# plt.plot(x_1[0,:],x_1[1,:],'g-',alpha=0.9,label='LB-FBLC-QP-MPC')
# plt.plot(x_2[0,:],x_2[1,:],'c-',alpha=0.9,label='FBLC-QP-MPC')
# plt.plot(x_3[0,:],x_3[1,:],'b-',alpha=0.9,label='LB-FBLC-MPC')
# plt.plot(x_4[0,:],x_4[1,:],'m-',alpha=0.9,label='LB-FBLC-QP')
# # plt.plot(x_ad[0,:],x_ad[1,:],'m--',alpha=0.9,label='ad')
# plt.plot(x_d[0,:],x_d[1,:],'k-',label='ref')
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower center", ncol=3)
ax = fig.gca()
# for i in range(barrier_x.size):
# 	circle = plt.Circle((barrier_x[i],barrier_y[i]), params["barrier_radius"], color='r')
# 	ax.add_artist(circle)
# plt.xlabel('X Position')
# plt.ylabel('Y Position')
plt.savefig(path_to_save+'2.eps', dpi=600, format='eps',bbox_inches='tight')
plt.savefig(path_to_save+'2.png', dpi=600, format='png',bbox_inches='tight')
np.savetxt(path_to_save+'sim_x_log_1.txt', x_1)
np.savetxt(path_to_save+'sim_x_log_ref.txt', x_d)


plt.figure()
plt.rcParams.update({'font.size': 12})
plt.rcParams['axes.unicode_minus'] = False
# plt.subplot(211)
plt.plot(t,u_1[0,:-1],'g-',alpha=0.9)
# plt.plot(t,u_ad[0,:-1],'m--',alpha=0.9)
plt.legend(['LB-FBLC-QP-MPC'],bbox_to_anchor=(0,1.1,1,0.2), loc="upper center", ncol=4)
plt.plot([t[0],t[-1]],[params["a_lim"],params["a_lim"]],'r--')
plt.plot([t[0],t[-1]],[-params["a_lim"],-params["a_lim"]],'r--')
plt.ylabel('acc_x')
plt.xlabel('Time (s)')
plt.savefig(path_to_save+'3.1.eps', dpi=600, format='eps',bbox_inches='tight')
plt.savefig(path_to_save+'3.1.png', dpi=600, format='png',bbox_inches='tight')
plt.figure()
plt.rcParams.update({'font.size': 12})
plt.rcParams['axes.unicode_minus'] = False
plt.plot(t,u_1[1,:-1],'g-',alpha=0.9)
# plt.plot(t,u_ad[1,:-1],'m--',alpha=0.9)
plt.legend(['LB-FBLC-QP-MPC'],bbox_to_anchor=(0,1.1,1,0.2), loc="upper center", ncol=4)
plt.plot([t[0],t[-1]],[params["a_lim"],params["a_lim"]],'r--')
plt.plot([t[0],t[-1]],[-params["a_lim"],-params["a_lim"]],'r--')
plt.ylabel('acc_y (m/s^2)')
plt.xlabel('Time (s)')
plt.savefig(path_to_save+'3.2.eps', dpi=600, format='eps',bbox_inches='tight')
plt.savefig(path_to_save+'3.2.png', dpi=600, format='png',bbox_inches='tight')
plt.figure()
plt.rcParams.update({'font.size': 12})
plt.rcParams['axes.unicode_minus'] = False
plt.plot(t,u_1[2,:-1],'g-',alpha=0.9)
# plt.plot(t,u_ad[1,:-1],'m--',alpha=0.9)
plt.legend(['LB-FBLC-QP-MPC'],bbox_to_anchor=(0,1.1,1,0.2), loc="upper center", ncol=4)
plt.plot([t[0],t[-1]],[params["a_lim"],params["a_lim"]],'r--')
plt.plot([t[0],t[-1]],[-params["a_lim"],-params["a_lim"]],'r--')
plt.ylabel('acc_z (m/s^2)')
plt.xlabel('Time (s)')
plt.savefig(path_to_save+'3.3.eps', dpi=600, format='eps',bbox_inches='tight')
plt.savefig(path_to_save+'3.3.png', dpi=600, format='png',bbox_inches='tight')
np.savetxt(path_to_save+'sim_acc_log_1.txt', u_1)


barrier_dist_1 = np.min(np.stack([np.sqrt((barrier_x[i]-x_1[0,:])**2 + (barrier_y[i]-x_1[1,:])**2 + (barrier_z[i]-x_1[2,:])**2) for i in range(len(barrier_x))]),axis=0)
np.savetxt(path_to_save+'sim_barrier_dist_log_1.txt', barrier_dist_1)

plt.figure()
plt.rcParams.update({'font.size': 12})
plt.rcParams['axes.unicode_minus'] = False
plt.plot(t,np.sqrt((x_d[0,:]-x_1[0,:])**2 + (x_d[1,:]-x_1[1,:])**2 + (x_d[2,:]-x_1[2,:])**2),'g-',alpha=0.9)
# plt.plot(t,np.sqrt((x_d[0,:]-x_ad[0,:])**2 + (x_d[1,:]-x_ad[1,:])**2),'m--',alpha=0.9)
plt.plot([0],[0],'r--')
plt.ylabel("Tracking error (m)")
plt.xlabel("Time(s)")
# plt.legend(['LB-FBLC-QP-MPC','FBLC-QP-MPC','LB-FBLC-MPC','LB-FBLC-QP'],bbox_to_anchor=(0,1.2,1,0.2), loc="upper right", ncol=2)
plt.legend(['LB-FBLC-QP-MPC'], loc="upper right", ncol=2)
# plt.ylim([0,1.0])
plt.savefig(path_to_save+'4.1.eps', dpi=600, format='eps',bbox_inches='tight')
plt.savefig(path_to_save+'4.1.png', dpi=600, format='png',bbox_inches='tight')
plt.figure()
plt.rcParams.update({'font.size': 12})
plt.rcParams['axes.unicode_minus'] = False
plt.plot(t,x_d[3,:],'k-')
plt.plot(t,x_1[3,:],'g-',alpha=0.9)
# plt.plot(t,x_ad[3,:],'m--',alpha=0.9)
plt.ylabel('Vel (m/s)')
plt.plot([t[0],t[-1]],[params["max_velocity"],params["max_velocity"]],'r--')
plt.plot([t[0],t[-1]],[params["min_velocity"],params["min_velocity"]],'r--')
plt.legend(['ref','LB-FBLC-QP-MPC','barrier'],bbox_to_anchor=(0,1.2,1,0.2), loc="upper center", ncol=3)
plt.xlabel('Time (s)')
plt.savefig(path_to_save+'4.2.eps', dpi=600, format='eps',bbox_inches='tight')
plt.savefig(path_to_save+'4.2.png', dpi=600, format='png',bbox_inches='tight')
plt.figure()
plt.rcParams.update({'font.size': 12})
plt.rcParams['axes.unicode_minus'] = False
plt.plot(t,barrier_dist_1-params["barrier_radius"],'g-',alpha=0.9)
# plt.plot(t,barrier_dist_ad-params["barrier_radius"],'m--',alpha=0.9)
plt.plot([t[0],t[-1]],[0,0],'r--')
plt.ylabel("Distance to obstacles (m)")
plt.xlabel("Time(s)")
# plt.legend(['LB-FBLC-QP-MPC','FBLC-QP-MPC','LB-FBLC-MPC','LB-FBLC-QP','barrier'],bbox_to_anchor=(0,1.2,1,0.2), loc="center right", ncol=2)
plt.legend(['LB-FBLC-QP-MPC'], loc="upper right", ncol=2)
plt.savefig(path_to_save+'4.3.eps', dpi=600, format='eps',bbox_inches='tight')
plt.savefig(path_to_save+'4.3.png', dpi=600, format='png',bbox_inches='tight')


np.savetxt(path_to_save+'sim_constraint_value_log_1.txt', constraint_value_log_1)
constraint_value_log_1 = np.clip(constraint_value_log_1, -1e10, 1e-4)
plt.figure()
plt.rcParams.update({'font.size': 12})
plt.rcParams['axes.unicode_minus'] = False
plt.plot(t[40:-1],constraint_value_log_1[40:],'g-',alpha=.9)
plt.plot([t[0],t[-1]],[-5e-4, -5e-4],'r--',alpha=.9)
plt.ylabel(r'$\lambda$')
plt.xlabel("Time(s)")
plt.savefig(path_to_save+'5.1.eps', dpi=600, format='eps',bbox_inches='tight')
plt.savefig(path_to_save+'5.1.png', dpi=600, format='png',bbox_inches='tight')

print('1')
print('control time: ', sum(control_time_log_1))
print('mean control time: ', sum(control_time_log_1) / len(control_time_log_1))
print('update time: ', sum(update_time_log_1))
print('mean update time: ', sum(update_time_log_1) / len(update_time_log_1))
print("mean prediction error: ", np.mean(prediction_error_1[start_training+2:]))
print("mean true error: ", np.mean(prediction_error_true_1[start_training+2:]))
print("mean position error: ", np.mean(np.sqrt((x_d[0,:]-x_1[0,:])**2 + (x_d[1,:]-x_1[1,:])**2)))
print("trigger counts: ", len(trigger_iter_log_1))
# for i in trigger_iter_log_1:
# 	print i,' '
print('')

plt.show()

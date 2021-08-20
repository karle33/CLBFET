import numpy as np
import matplotlib.pyplot as plt

class Dynamics():
    def __init__(self,xdim,udim):
        self.xdim=xdim
        self.udim=udim

    def f(self,x):
        # overload
        return None

    def g(self,x):
        # overload
        return None

    def pseudoinv(self,x):
        return np.matmul(np.linalg.inv(np.matmul(x.T,x)),x.T)

    def convert_z_to_x(self,z):
        v=np.sqrt(z[2,:]**2 + z[3,:]**2) * z[4,:]
        return np.stack((z[0,:],z[1,:],np.arctan2(z[4,:]*z[3,:],z[4,:]*z[2,:]),v),axis=0)

    def convert_x_to_z(self,x):
        v_sign = 1.0
        if x[3,:] < 0:
            v_sign = -1
        return np.stack((x[0,:],x[1,:],x[3,:]*np.cos(x[2,:]),x[3,:]*np.sin(x[2,:]),np.array([v_sign])),axis=0)

class DynamicsQuadrotor(Dynamics):
    def __init__(self,xdim=6,udim=3,epsilon=1e-6,disturbance_scale_pos = 0.5,disturbance_scale_vel = 2.0,control_input_scale = 2.0):
        self.epsilon = epsilon
        Dynamics.__init__(self,xdim=xdim,udim=udim)

        self.ga = 9.8
        # self.m = 1550.10/1000
        self.m = 35.89/1000

        self.disturbance_scale_pos = disturbance_scale_pos
        self.disturbance_scale_vel = disturbance_scale_vel
        self.control_input_scale = control_input_scale
        self.disturbance_type = 0
    
    def set_disturbance_type(self, type):
        self.disturbance_type = type
        
    def f(self,z):
        return np.array([0.0, 0.0, -self.ga]).reshape((3,1))
        
    def g(self,z):
        # v=(np.sqrt(z[2,:]**2 + z[3,:]**2)  + self.epsilon) * z[4,:]
        # return self.control_input_scale * np.stack((np.concatenate((-z[3,:]*v,z[2,:]/v)),np.concatenate((z[2,:]*v,z[3,:]/v))))
        return np.eye(self.xdim//2)

    def step(self,z,u,dt):
        # v=np.sqrt(z[2,:]**2 + z[3,:]**2)
        znext = np.zeros((self.xdim,1))
        znext[0:3,:] = z[0:3,:] + dt*(z[3:,:])# + np.array([np.sin(v**3)*self.disturbance_scale_pos, np.cos(-v)*self.disturbance_scale_pos]))
        znext[3:,:] = z[3:,:] + dt*(self.f(z) + np.matmul(self.g(z),u))
        # znext[-1] = z[-1]
        return znext
    
    def convert_mu_to_control(self, mu):
        beta1 = mu[0]
        beta2 = mu[2]
        beta3 = -mu[1]
        theta = np.arctan2(beta1, beta2)	#pitch
        phi = np.arctan2(beta3, np.sqrt(beta1 ** 2 + beta2 ** 2))	#roll
        psi = 0	#yaw
        a_temp = np.linalg.norm(mu)
        thrust = a_temp * self.m
        return np.array([thrust, phi[0], theta[0], psi])
    
    def convert_control_to_mu(self, control):
        thrust = control[0]
        phi = control[1]
        theta = control[2]
        psi = control[3]
        ax = thrust * (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) / self.m
        ay = thrust * (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) / self.m
        az = thrust * (np.cos(phi) * np.cos(theta)) / self.m
        return np.array([ax, ay, az])
    
    def convert_z_to_x(self,z):
        return z
    
    def convert_x_to_z(self,x):
        return x

class DynamicsQuadrotorModified(Dynamics):
    def __init__(self,xdim=6,udim=3,epsilon=1e-6,disturbance_scale_pos = 0.5,disturbance_scale_vel = 2.0,control_input_scale = 2.0):
        self.epsilon = epsilon
        Dynamics.__init__(self,xdim=xdim,udim=udim)

        self.ga = 9.8
        self.wind_Data = np.load('/home/wuzhixuan/pro/LB-FBLC-CLBF_ws/src/LB-FBLC-CLBF/src/wind.npy')

        self.disturbance_scale_pos = disturbance_scale_pos
        self.disturbance_scale_vel = disturbance_scale_vel
        self.control_input_scale = control_input_scale
        self.disturbance_type = 0
    
    def set_disturbance_type(self, type):
        self.disturbance_type = type
        
    def f(self,z):
        # v=np.sqrt(z[2,:]**2 + z[3,:]**2) * z[4,:]
        # theta = np.arctan2(z[3]*z[4],z[2]*z[4])
        # if self.disturbance_type == 0:
        # 	v_disturbance_body = [np.tanh(v**2)*self.disturbance_scale_vel, (0.1+v)*self.disturbance_scale_vel]
        # 	v_disturbance_world = [v_disturbance_body[0] * np.cos(theta) - v_disturbance_body[1] * np.sin(theta),
        # 					   v_disturbance_body[0] * np.sin(theta) + v_disturbance_body[1] * np.cos(theta)]
        # elif self.disturbance_type == 1:
        # 	v_disturbance_body = [np.tanh(v**2)*self.disturbance_scale_vel*.5, (0.1+v)*self.disturbance_scale_vel*1.4]
        # 	v_disturbance_world = [v_disturbance_body[0] * np.cos(theta) - v_disturbance_body[1] * np.sin(theta),
        # 					   v_disturbance_body[0] * np.sin(theta) + v_disturbance_body[1] * np.cos(theta)]
        # return np.array([v_disturbance_world[0], v_disturbance_world[1]])
        # todo: more complex disturbances
        dw = np.array([ 5.6471  - 2.1487 * (1 - np.cos(z[0])[0]),
                    -0.1755 + 2.4944 * (1 - np.sin(z[1])[0]),
                    7.6034])
        windindex = min(int(z[0] * 50), 4000)
        dw += np.array([self.wind_Data[0, windindex], self.wind_Data[1, windindex], self.wind_Data[2, windindex]])
        dw += np.array([-z[3,0], -z[4,0], -z[5,0]])
        wind = 0.02 * dw + 0.005 * (2 * np.random.rand() - 1)
        # wind = np.array([0, 0, 0])
        return (np.array([0.0, 0.0, -self.ga]) + 0*wind).reshape((3,1))
        
    def g(self,z):
        # v=(np.sqrt(z[2,:]**2 + z[3,:]**2)  + self.epsilon) * z[4,:]
        # return self.control_input_scale * np.stack((np.concatenate((-z[3,:]*v,z[2,:]/v)),np.concatenate((z[2,:]*v,z[3,:]/v))))
        return np.eye(self.xdim//2)

    def step(self,z,u,dt):
        # v=np.sqrt(z[2,:]**2 + z[3,:]**2)
        znext = np.zeros((self.xdim,1))
        znext[0:3,:] = z[0:3,:] + dt*(z[3:,:])# + np.array([np.sin(v**3)*self.disturbance_scale_pos, np.cos(-v)*self.disturbance_scale_pos]))
        znext[3:,:] = z[3:,:] + dt*(self.f(z) + np.matmul(self.g(z),u))
        # znext[-1] = z[-1]
        return znext
    
    def convert_z_to_x(self,z):
        return z
    
    def convert_x_to_z(self,x):
        return x

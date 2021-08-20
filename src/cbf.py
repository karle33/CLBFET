import numpy as np

class Barrier():
	def __init__(self,dim,gamma):
		self.dim=dim
		self.gamma = gamma

	def h(self,x):
		# overload
		return None

	def dh(self,x):
		# overload
		return None

	def B(self,x):
		# TODO:  parameterize this, and craft something better.
		hx = self.h(x)
		# return np.exp(-hx+1)
		if hx == 0:
			hx = 1e-3
		return 1.0/hx

	def dB(self,x):
		hx = self.h(x)
		if hx == 0:
			hx = 1e-3
		return -1.0/(hx**2)*self.dh(x)
		# return -np.exp(-hx+1)*self.dh(x)

	def d2B(self,x):
		hx = self.h(x)
		if hx == 0:
			hx = 1e-3
		# return -1.0/(hx**3)*(self.d2h(x) -2*np.matmul(self.dh(x),self.dh(x).T))
		# can ignore d2h because taking trace(G*sig*sig^T*G^T d2B).
		dh = self.dh(x)[2:].T
		return 1.0/(hx**3)*(2*np.outer(dh,dh))

	def get_B_derivatives(self,x):
		hx = self.h(x)
		if hx == 0:
			hx = 1e-3

		dh = self.dh(x)
		return hx, -1.0/(hx*hx)*dh.T, 2.0/(hx*hx*hx)*(np.outer(dh[2:],dh[2:]))

# todo: d2h
class BarrierQuadrotorPoint(Barrier):
	def __init__(self,dim=6,gamma=1.0, x=0.0, y=0.0, z=0.0, radius = 1.0, gamma_p = 1.0):
		self.x = x
		self.y = y
		self.z = z
		self.radius = radius
		self.gamma_p = gamma_p

		Barrier.__init__(self,dim,gamma)

	def h(self,x):
		sgn = 1.0
		if self.radius < 0.0:
			sgn = -1.0

		d = np.sqrt((x[0] - self.x)*(x[0] - self.x) + (x[1] - self.y)*(x[1] - self.y) + (x[2] - self.z)*(x[2] - self.z)) + 1.0e-6
		return sgn * (self.gamma_p * (d - self.radius) + (x[0] - self.x) / d * x[3] + (x[1] - self.y) / d * x[4] + (x[2] - self.z) / d * x[5])
		 
	def dh(self,x):
		sgn = 1.0
		if self.radius < 0.0:
			sgn = -1.0

		d = np.sqrt((x[0] - self.x)*(x[0] - self.x) + (x[1] - self.y)*(x[1] - self.y) + (x[2] - self.z)*(x[2] - self.z)) + 1.0e-6
		d_2 = d*d
		d_3 = d*d*d
		px_m_pxo = x[0] - self.x
		py_m_pyo = x[1] - self.y
		pz_m_pzo = x[2] - self.z
		return sgn * np.array((
			(self.gamma_p * px_m_pxo + x[3]) / d - (x[3] * px_m_pxo + x[4] * py_m_pyo + x[5] * pz_m_pzo) * px_m_pxo / d_3,
			(self.gamma_p * py_m_pyo + x[4]) / d - (x[3] * px_m_pxo + x[4] * py_m_pyo + x[5] * pz_m_pzo) * py_m_pyo / d_3,
			(self.gamma_p * pz_m_pzo + x[5]) / d - (x[3] * px_m_pxo + x[4] * py_m_pyo + x[5] * pz_m_pzo) * pz_m_pzo / d_3,
			px_m_pxo / d,
			py_m_pyo / d,
			pz_m_pzo / d), dtype=np.float32)

	def d2h(self,x):
		# sgn = 1.0
		# if self.radius < 0.0:
		# 	sgn = -1.0

		# d = np.sqrt((z[0,:] - self.x)**2 + (z[1,:] - self.y)**2) + 1.0e-6
		# z1 = z[0,:]
		# z2 = z[1,:]
		# z3 = z[2,:]
		# z4 = z[3,:]
		# z5 = z[4,:]
		# gamma_p = self.gamma_p
		# x_pos = self.x
		# y_pos = self.y
		# y_pos_m_z2 = (y_pos - z2)
		# x_pos_m_z1 = (x_pos - z1)
		# d2 = d**2
		# d3 = d**3
		# d5 = d**5
		# z1_2 = z1**2
		# z2_2 = z2**2
		# x_pos_2 = x_pos**2
		# y_pos_2 = y_pos**2
		# a11 = (y_pos_m_z2*(gamma_p *(x_pos_2*y_pos - x_pos_2*z2 + 3*y_pos*z2_2 - 2*x_pos*y_pos*z1 + 2*x_pos*z1*z2 + y_pos**3 - 3*y_pos_2*z2 + y_pos*z1_2 - z1_2*z2 - z2**3) - 2*z4*x_pos_2 + 3*z3*x_pos*y_pos + 4*z4*x_pos*z1 - 3*z3*x_pos*z2 + z4*y_pos_2 - 3*z3*y_pos*z1 - 2*z4*y_pos*z2 - 2*z4*z1_2 + 3*z3*z1*z2 + z4*z2_2))/(d5)
		# a12 = x_pos_m_z1 * y_pos_m_z2 * (z4 + z3 - gamma_p - (3*z3*x_pos_m_z1)/(d2) - (3*z4*y_pos_m_z2)/(d2))/(d3)
		# a13 = y_pos_m_z2**2/(d3)
		# a14 = -(x_pos_m_z1*y_pos_m_z2)/(d3)
		# a22 = (x_pos_m_z1*(gamma_p*(x_pos**3 - 3*x_pos_2*z1 + x_pos*y_pos_2 - 2*x_pos*y_pos*z2 + 3*x_pos*z1_2 + x_pos*z2_2 - y_pos_2*z1  + 2*y_pos*z1*z2 - z1**3 - z1*z2_2)+ z3*x_pos_2 + 3*z4*x_pos*y_pos - 2*z3*x_pos*z1 - 3*z4*x_pos*z2 - 2*z3*y_pos_2 - 3*z4*y_pos*z1 + 4*z3*y_pos*z2 + z3*z1_2 + 3*z4*z1*z2 - 2*z3*z2_2))/(d5)
		# a23 = -(y_pos_m_z2*x_pos_m_z1)/(d3)
		# a24 = x_pos_m_z1**2/(d3)

		# return sgn * np.block([
		# 	[ a11, a12, a13, a14],
		# 	[ a12, a22, a23, a24],
		# 	[ a13, a23, 0, 0],
		# 	[ a14, a24, 0, 0]], dtype=np.float32)
		return 0

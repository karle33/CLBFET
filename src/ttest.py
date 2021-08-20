import osqp
import numpy as np
from scipy import sparse

# Define problem data
wind_Data = np.load('/home/wuzhixuan/pro/LB-FBLC-CLBF_ws/src/LB-FBLC-CLBF/src/wind.npy')
print wind_Data
print wind_Data.shape

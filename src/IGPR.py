from scipy.optimize import minimize
import numpy as np
import csv
from collections import deque
from scipy.spatial.distance import cdist
import time

class HyperParam(object):
    def __init__(self, theta_f=1., len=1.):
        self.theta_f = theta_f       # for squared exponential kernel
        self.len = len           # for squared exponential kernel

class IGPR(object):
    """
    Incremental Gaussian Process Regressian

    kernel: RBF

    * Before size of datas reach max size(self.max_k_matrix_size, default=20), 
        new data will be added when fiting.
    * When size of datas reach max size, new data will probably be added when fitting,
        and size of k_matrix remain max size.
    
    Parameters
    ----------
    hyperparam_len: float, default=1.
        hyperparam "len" of kernel RBF

    hyperparam_theta_f: float, default=1.
        hyperparam "theta_f" of kernel RBF

    alpha: float, default=1e-6
        Value added to the diagonal of the kernel matrix
    
    update_mode: string, default="FIFO"
        Strategy to sellect which data to replace
        "FIFO": replace the oldest one
        "max_delta": replace the least relevant one with new data
    
    optimize: bool, default=False
        Whether optimize hyperparams of RBF
        True: optimize hyperparams when fit, with cost of 
        False: no optimize
    
    k_matrix_adjust: bool, default=False
        Whether recalculate k matrix per "self.k_matrix_adjust_count" counts

    ---------

    Examples:
    igpr = IGPR(x[0], y[0], update_mode="FIFO", alpha=1e-6, optimize=False)
    for i in range(1,N-1):
        igpr.learn(x[i], y[i])
        ypredict, ycov = igpr.predict(x[i+1])
    """
    def __init__(self, target_counts = 1, hyperparam_len=1., hyperparam_theta_f=1., alpha=1e-6,\
                update_mode="FIFO", max_k_matrix_size=20, optimize=False, k_matrix_adjust = False):
        self.hyperparam = HyperParam(hyperparam_theta_f, hyperparam_len)
        self.alpha = alpha
        self.target_counts = target_counts
        self.update_mode = update_mode # "max_delta" or "FIFO"
        self.optimize = optimize
        self.optimize_count = 1
        self.k_matrix_adjust = k_matrix_adjust
        self.k_matrix_adjust_count = 100
        self.reset_count = self.optimize_count * self.k_matrix_adjust_count
        self.max_k_matrix_size = max_k_matrix_size
        self.lamda = 1
        self.count = 0
        self.is_av = False
        self.initial = False

    def is_available(self):
        if not self.initial:
            return False
        n = len(self.kernel_x)
        if n >= 2:
            self.is_av = True
        return self.is_av

    def load_csv(self, file_name):
        with open(file_name, "r") as f:
            reader = csv.reader(f)
            columns = [row for row in reader]
        columns = np.array(columns)
        m_x, n_x = columns.shape
        data_set = np.zeros((m_x,n_x))
        for i in range(m_x):
            for j in range(n_x):
                data_set[i][j] = float(columns[i][j])
        return data_set

    """
    new_x: array-like of shape (n_features,)

    new_y: float
    """
    def learn(self, new_x, new_y):
        if not self.initial:
            self.feature_counts = len(new_x) if (np.array(new_x).ndim == 1) else len(new_x[0])
            self.kernel_x = np.empty((0, self.feature_counts))
            self.kernel_x = np.vstack((self.kernel_x, new_x))
            self.kernel_y = np.empty((0, self.target_counts))
            self.kernel_y = np.vstack((self.kernel_y, new_y))
            self.k_matrix = np.ones((1, 1)) + self.alpha
            self.inv_k_matrix = np.ones((1, 1)) / self.alpha
            temp = np.sum(self.k_matrix, axis=0)
            # sum of each row/column of k_matrix
            self.delta = deque()
            for i in range(temp.shape[0]):
                self.delta.append(temp[i])
            self.initial = True
            return

        for i in range(len(self.delta)):
            self.delta[i] = self.delta[i]*self.lamda

        if self.is_available():
            #print('available')
            if len(self.kernel_x) < self.max_k_matrix_size:
                # print('aug_update_SE_kernel')
                self.aug_update_SE_kernel(new_x, new_y)
            else:
                if self.update_mode == "max_delta":
                    # choose max cov one with the new data to replace
                    new_delta = self.count_delta(new_x)
                    max_value, max_index = self.get_max(self.delta)
                    if new_delta < max_value:
                        # print('SM_update_SE_kernel')
                        self.SM_update_SE_kernel(new_x, new_y, max_index)
                        self.count = self.count + 1
                elif self.update_mode == "FIFO":
                    # print('schur_update_SE_kernel')
                    self.schur_update_SE_kernel(new_x, new_y)
                    self.count = self.count + 1
                if self.optimize and self.count % self.optimize_count == 0:
                    # bad performance
                    self.hyperparam_optimization()
                elif self.k_matrix_adjust and self.count % self.k_matrix_adjust_count == 0:
                    self.calculate_SE_kernel()
                    self.inv_k_matrix = np.linalg.inv(self.k_matrix)
                if self.count == self.reset_count:
                    self.count = 0


        else:
            # print('not available')
            self.kernel_x = np.vstack((self.kernel_x, new_x))
            self.kernel_y = np.vstack((self.kernel_y, new_y))
            self.calculate_SE_kernel()
            self.inv_k_matrix = np.linalg.inv(self.k_matrix)

    def calculate_SE_kernel(self):
        self.k_matrix = np.exp(-.5 * cdist(self.kernel_x / self.hyperparam.len, self.kernel_x / self.hyperparam.len,
                          metric='sqeuclidean')).T * self.hyperparam.theta_f ** 2 + self.alpha * np.eye(len(self.kernel_x))
        temp = np.sum(self.k_matrix, axis=0)
        self.delta = deque()
        for i in range(temp.shape[0]):
            self.delta.append(temp[i])
    
    def update_SE_kernel_hyperparam(self, new_theta_f, new_len, op=1):
        if op == 0:
            self.hyperparam = HyperParam(new_theta_f, new_len)
            self.calculate_SE_kernel()
            self.inv_k_matrix = np.linalg.inv(self.k_matrix)
        else:
            # todo: result may need checked
            n = len(self.kernel_x)
            self.k_matrix = ((self.k_matrix - self.alpha * np.eye(n)) / self.hyperparam.theta_f ** 2) ** (self.hyperparam.len ** 2 / new_len ** 2) * new_theta_f ** 2 + self.alpha * np.eye(n)
            # todo: optimize it
            self.inv_k_matrix = np.linalg.inv(self.k_matrix)
            self.hyperparam = HyperParam(new_theta_f, new_len)
            temp = np.sum(self.k_matrix, axis=0)
            self.delta = deque()
            for i in range(temp.shape[0]):
                self.delta.append(temp[i])
    
    def hyperparam_optimization(self):
        # optimization needs K_inv for different hyperparam, which makes igpr no sense if optimize for each step.
        # print("optimize here")
        def negative_log_likelihood_loss(params):
            self.update_SE_kernel_hyperparam(params[0], params[1], 0)
            ky = self.kernel_y.flatten()
            return (0.5 * ky.dot(self.inv_k_matrix).dot(ky) + 0.5 * np.linalg.slogdet(self.k_matrix)[1] + 0.5 * len(self.kernel_x) * np.log(2 * np.pi))
        # print ("neg_log_llh_l_before=", negative_log_likelihood_loss(np.array([self.hyperparam.theta_f, self.hyperparam.len])))
        res = minimize(negative_log_likelihood_loss, np.array([self.hyperparam.theta_f, self.hyperparam.len]), bounds=((1e-3, 1e3), (1e-2, 1e2)), method='L-BFGS-B')
        self.hyperparam = HyperParam(res.x[0], res.x[1])
        # print ("neg_log_llh_l_after=", negative_log_likelihood_loss(np.array([self.hyperparam.theta_f, self.hyperparam.len])))
        # print ("theta_f=", self.hyperparam.theta_f, "len=", self.hyperparam.len)

    """
    coming_x: array-like of shape (n_samples, n_features)

    return_cov: bool
        True: return mean and variance
        False: return mean
        
    """
    def predict(self, coming_x, return_cov=True):
        if self.is_available():
            if coming_x.ndim == 1:
                coming_x = np.array([coming_x])
            data_size = len(coming_x)
            # calculate cross kernel
            cross_kernel_k = np.exp(-.5 * cdist(self.kernel_x / self.hyperparam.len, coming_x / self.hyperparam.len,
                          metric='sqeuclidean')).T * self.hyperparam.theta_f ** 2
            prediction = cross_kernel_k.dot(self.inv_k_matrix.dot(self.kernel_y))

            if return_cov:
                kyy = self.hyperparam.theta_f * self.hyperparam.theta_f
                variance = cross_kernel_k.dot(self.inv_k_matrix).dot(cross_kernel_k.T)
                variance = kyy - np.diag(variance).reshape(data_size,1)
                return prediction, variance
            else:
                return prediction
        else:
            if return_cov:
                return self.kernel_y[0], 0.0
            else:
                return self.kernel_y[0]


    def aug_update_SE_kernel(self, new_x, new_y):
        n = len(self.kernel_x)
        self.kernel_x = np.vstack((self.kernel_x, new_x))
        self.kernel_y = np.vstack((self.kernel_y, new_y))
        self.k_matrix = np.hstack((self.k_matrix, np.zeros((n, 1))))
        self.k_matrix = np.vstack((self.k_matrix, np.zeros((1, n+1))))

        cross_kernel_k = np.exp(-.5 * cdist(self.kernel_x / self.hyperparam.len, np.array([new_x]) / self.hyperparam.len,
                          metric='sqeuclidean')).T * self.hyperparam.theta_f ** 2
        self.k_matrix[:, n] = cross_kernel_k
        self.k_matrix[n, :] = cross_kernel_k
        self.k_matrix[n, n] += self.alpha
        
        b = self.k_matrix[0:n, n].reshape((n, 1))
        d = self.k_matrix[n, n]
        e = self.inv_k_matrix.dot(b)
        g = 1 / (d - (b.T).dot(e))
        h11 = self.inv_k_matrix + g[0][0]*e.dot(e.T)
        h12 = -g[0][0]*e
        h21 = -g[0][0]*(e.T)
        h22 = g


        temp_1 = np.hstack((h11, h12))
        temp_2 = np.hstack((h21, h22))
        self.inv_k_matrix = np.vstack((temp_1, temp_2))

        # udpate delta
        for i in range(n):
            self.delta[i] = self.delta[i] + self.k_matrix[i, n]
        self.delta.append(0)

        for i in range(n+1):
            self.delta[n] = self.delta[n] + self.k_matrix[i, n]

    def schur_update_SE_kernel(self, new_x, new_y):
        n = len(self.kernel_x)

        self.kernel_x = np.vstack((self.kernel_x, new_x))
        self.kernel_y = np.vstack((self.kernel_y, new_y))
        self.kernel_x = self.kernel_x[1:]
        self.kernel_y = self.kernel_y[1:]

        K2 = np.zeros((n, n))
        K2[0:n-1, 0:n-1] = self.k_matrix[1:n, 1:n]

        cross_kernel_k = np.exp(-.5 * cdist(self.kernel_x / self.hyperparam.len, np.array([new_x]) / self.hyperparam.len,
                          metric='sqeuclidean')).T * self.hyperparam.theta_f ** 2
        K2[:, n-1] = cross_kernel_k
        K2[n-1, :] = cross_kernel_k
        K2[n-1, n-1] += self.alpha

        e = self.inv_k_matrix[0][0]
        f = self.inv_k_matrix[1:n, 0].reshape((n-1, 1))
        g = K2[n-1, n-1]
        h = K2[0:n-1, n-1].reshape((n-1, 1))
        H = self.inv_k_matrix[1:n, 1:n]
        B = H - (f.dot(f.T)) / e
        s = 1 / (g - (h.T).dot(B.dot(h)))
        h11 = B + (B.dot(h)).dot((B.dot(h)).T) * s
        h12 = -B.dot(h) * s
        h21 = -(B.dot(h)).T * s
        h22 = s
        temp_1 = np.hstack((h11, h12))
        temp_2 = np.hstack((h21, h22))
        self.inv_k_matrix = np.vstack((temp_1, temp_2))

        # update delta
        self.delta.popleft()
        self.delta.append(0)
        for i in range(n-1):
            self.delta[i] = self.delta[i] - self.k_matrix[0, i+1]

        for i in range(n-1):
            self.delta[i] = self.delta[i] + K2[n-1, i]

        for i in range(n):
            self.delta[n-1] = self.delta[n-1] + K2[i, n-1]

        self.k_matrix = K2

    def SM_update_SE_kernel(self, new_x, new_y, index):
        n = len(self.kernel_x)
        self.kernel_x[index] = new_x
        self.kernel_y[index] = new_y
        new_k_matrix = self.k_matrix.copy()
        
        # calculate K using SE kernel
        cross_kernel_k = np.exp(-.5 * cdist(self.kernel_x / self.hyperparam.len, np.array([self.kernel_x[index]]) / self.hyperparam.len,
                          metric='sqeuclidean')).T * self.hyperparam.theta_f ** 2
        new_k_matrix[:, index] = cross_kernel_k
        new_k_matrix[index, :] = cross_kernel_k
        new_k_matrix[index, index] += self.alpha
        
        r = new_k_matrix[:, index].reshape((n, 1)) - self.k_matrix[:, index].reshape((n, 1))
        A = self.inv_k_matrix - (self.inv_k_matrix.dot(r.dot(self.inv_k_matrix[index, :].reshape((1, n)))))/(1 + r.transpose().dot(self.inv_k_matrix[:, index].reshape((n, 1)))[0, 0])
        self.inv_k_matrix = A - ((A[:, index].reshape((n, 1))).dot(r.transpose().dot(A)))/(1 + (r.transpose().dot(A[:, index].reshape((n, 1))))[0, 0])

        # update delta

        for i in range(n):
            if i!=index:
                self.delta[i] = self.delta[i] - self.k_matrix[index, i]

        for i in range(n):
            if i != index:
                self.delta[i] = self.delta[i] + new_k_matrix[index, i]

        self.delta[index] = 0
        for i in range(n):
            self.delta[index] = self.delta[index] + new_k_matrix[i, index]

        self.k_matrix = new_k_matrix

    def count_delta(self, new_x):
        n = len(self.kernel_x)
        temp_delta = np.zeros((1, n))
        for i in range(n):
            temp_delta[0, i] = np.sum(np.square(self.kernel_x[i] - new_x))
            temp_delta[0, i] = temp_delta[0, i] / -2
            temp_delta[0, i] = temp_delta[0, i] / self.hyperparam.len
            temp_delta[0, i] = temp_delta[0, i] / self.hyperparam.len
            temp_delta[0, i] = np.exp(temp_delta[0, i])
            temp_delta[0, i] = temp_delta[0, i] * self.hyperparam.theta_f
            temp_delta[0, i] = temp_delta[0, i] * self.hyperparam.theta_f
        temp_delta = np.sum(temp_delta)
        return temp_delta

    def get_max(self, delta):
        max_index = 0
        max_value = delta[0]
        for i in range(1, len(delta)):
            if delta[i] > max_index:
                max_index = i
                max_value = delta[i]
        return max_value, max_index
    
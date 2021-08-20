import numpy as np
import osqp
import scipy.sparse as sparse
import scipy.linalg as sl
import rospy

class QPSolve():
    def __init__(self,dyn,cbf_list,u_lim,u_cost=0.0,u_prev_cost=1.0,p1_cost=1.0e10,p2_cost=1.0e10,verbose=True):
        self.xdim = dyn.xdim
        self.udim = dyn.udim
        self.dyn = dyn
        self.epsilon = 100.0
        self.cbf_list = cbf_list
        self.u_lim = u_lim
        self.p1_cost = p1_cost
        self.p2_cost = p2_cost
        self.verbose = verbose
        self.u_cost = u_cost
        self.u_prev_cost = u_prev_cost
        self.K = 0.0
        self.ksig = 1.0
        self.max_var = 1.0
        self.mu_qp_prev = np.zeros((self.xdim//2,1), dtype=np.float32)
        self.P = np.eye(self.xdim, dtype=np.float32)
        self.A = np.zeros((self.xdim,self.xdim), dtype=np.float32)
        self.A0 = np.block([[np.zeros((self.xdim//2,self.xdim//2)),np.eye(self.xdim//2)],[np.zeros((self.xdim//2,self.xdim//2)),np.zeros((self.xdim//2,self.xdim//2))]]).astype(np.float32)
        self.G = np.block([[np.zeros((self.xdim//2,self.xdim//2))],[np.eye(self.xdim//2)]]).astype(np.float32)
        self.res = None
        self.max_error = 1.0

    def update_ricatti(self,A):
        self.A = A
        Q = np.eye(self.xdim, dtype=np.float32)
        self.P = sl.solve_continuous_are(self.A,np.zeros((self.xdim,self.xdim), dtype=np.float32),Q,np.eye(self.xdim, dtype=np.float32))

    def solve(self,x,x_d,mu_d,sigDelta):
        sigDelta = sigDelta * self.ksig
        sigDelta = np.clip(sigDelta,0.0,self.max_var)
        # sigDelta = np.ones((self.xdim//2,1)) * self.max_var # for testing

        # build Q and p matrices to specify minimization expression
        Q = np.diag(np.append(np.append(np.ones(self.xdim//2, dtype=np.float32)*(self.u_cost + self.u_prev_cost),self.p1_cost),self.p2_cost))
        self.Q = sparse.csc_matrix(Q)
        self.p = 2*np.append(np.append(-self.mu_qp_prev*self.u_prev_cost,0),0)

        #error dynamics for clf
        e = x[:,:]-x_d[:,:]
        e = np.clip(e,-self.max_error,self.max_error)
        eTPG = np.matmul(e.T,np.matmul(self.P,self.G))
        G_dyn = np.expand_dims(np.append(np.append(2*eTPG,1),0),0) #append 1 for clf < d
        # G_dyn = np.expand_dims(np.append(np.append(-2*eTPG,1),0),0) #append 1 for clf < d
        Gsig = np.matmul(self.G,sigDelta)
        GssG = np.matmul(Gsig,Gsig.T)
        self.trGssGP = np.trace(np.matmul(GssG,self.P))
        # yr ver: always trigger using this clf no matter how the distrubunce and barriers are
        # h_dyn = -1 * ( #-0.5*np.matmul(e.T,np.matmul(Q,e))
        #             + 0.5*np.matmul(e.T,np.matmul(self.P,e)) / self.clf.epsilon
        #             + np.dot(np.linalg.norm(eTPG), np.linalg.norm(sigDelta)) 
        #             #+ 0.5*self.trGssGP
        #             )
        h_dyn = -np.matmul(e.T,np.matmul(self.P,e)) / self.epsilon - 2 * np.dot(np.linalg.norm(eTPG), np.linalg.norm(sigDelta)) 
        # my ver: never trigger using this clf no matter how the distrubunce and barriers are
        # h_dyn = -2 * (# -0.5*np.matmul(e.T,np.matmul(Q,e))
        #             # + 0.5*np.matmul(e.T,np.matmul(self.P,e)) / self.clf.epsilon
        #             [[0.]]
        #             + np.dot(np.linalg.norm(eTPG), np.linalg.norm(sigDelta)) 
        #             #+ 0.5*self.trGssGP
        #             )
        # balsa ver
        # h_dyn = -1 * ( -0.5*np.matmul(e.T,np.matmul(Q,e))
        #             + 0.5*np.matmul(e.T,np.matmul(self.P,e)) / self.clf.epsilon
        #             + 0.5*self.trGssGP)

        # build constraints for barriers
        N_cbf = len(self.cbf_list)
        G_cbf = np.zeros((N_cbf,self.xdim//2+2), dtype=np.float32)
        h_cbf = np.zeros((N_cbf,1), dtype=np.float32)
        A0x_Gmud = np.matmul(self.A0,x[:,:]) + np.matmul(self.G,mu_d) #- np.matmul(self.G, sigDelta)
        GssG_22 = GssG[2:,2:]
        # todo: rewrite cbf to 3d and use it with cbf trigger
        for i, cbf in enumerate(self.cbf_list):
            h_x, dB, d2B = cbf.get_B_derivatives(x)
            G_cbf[i,:] = np.append(np.append(np.einsum('ij,jk',dB,self.G),0),1)
            trGssGd2B = np.einsum('ii',np.einsum('ij,jk',GssG_22,d2B))
            h_cbf[i,:] = -1 * (np.einsum('ij,jk',dB,A0x_Gmud)
                                - cbf.gamma * h_x
                                + np.dot(np.linalg.norm(np.dot(dB,self.G)),np.linalg.norm(sigDelta))
                                #+ 0.5*trGssGd2B
                                )
            h_cbf[i,:] = -1 * (np.einsum('ij,jk',dB,A0x_Gmud)
                                - cbf.gamma * h_x
                                + 0.5*trGssGd2B)


        # build constraints for control limits
        ginv = np.linalg.inv(self.dyn.g(x))
        l_ctrl = np.matmul(ginv, mu_d - self.dyn.f(x))
        A_ctrl = ginv

        G_ctrl = np.zeros((self.udim*2,self.xdim//2+2), dtype=np.float32)
        h_ctrl = np.zeros((self.udim*2,1), dtype=np.float32)
        for i in range(self.udim):
            G_ctrl[i*2,:self.xdim//2] = - A_ctrl[i,:]
            h_ctrl[i*2] = - self.u_lim[i,0] + l_ctrl[i]
            G_ctrl[i*2+1,:self.xdim//2] = A_ctrl[i,:]
            h_ctrl[i*2+1] = self. u_lim[i,1] - l_ctrl[i]

        # stack into one matrix and vector
        G = np.concatenate((G_dyn,G_cbf,G_ctrl),axis=0)
        h = np.concatenate((h_dyn,h_cbf,h_ctrl),axis=0)
        # G = np.concatenate((G_dyn,G_ctrl),axis=0)
        # h = np.concatenate((h_dyn,h_ctrl),axis=0)
        # G = np.concatenate((G_dyn),axis=0)
        # h = np.concatenate((h_dyn),axis=0)

        self.G_csc = sparse.csc_matrix(G)
        self.h = h

        # dummy lower bound
        l = np.ones(h.shape, dtype=np.float32)*np.inf * -1

        #Solve QP
        self.prob = osqp.OSQP()
        exception_called = False
        mu_bar = np.zeros((self.xdim+1), dtype=np.float32)
        # try:
        self.prob.setup(P=self.Q, q=self.p, A=self.G_csc, l=l, u=self.h, verbose=self.verbose)
        # print 'Q', self.Q.toarray()
        # print 'P', self.p
        # print 'A', self.G_csc.toarray()
        # print 'l', l
        # print 'u', self.h
        # self.prob.setup(P=self.Q, q=self.p, A=self.G_csc, l=l, u=self.h, eps_rel=1e-8, verbose=self.verbose)
        self.res = self.prob.solve()
        # except:
            # exception_called = True
        # else:
        mu_bar = self.res.x
        # if exception_called or u_bar[0] is None or np.isnan(u_bar).any():
        if mu_bar[0] is None or np.isnan(mu_bar).any():
            mu_bar = np.zeros((self.xdim+1))
            self.res = None
            rospy.logwarn("QP fail ed!")
        

        self.mu_qp_prev = np.expand_dims(mu_bar[0:self.xdim//2],axis=0).T

        self.triggered = False

        # print(h_dyn - np.matmul(eTPG, self.mu_qp_prev))
        
        # print('h_dyn', h_dyn)
        # print('eTPG', eTPG)
        # print('mu_bar', mu_bar)
        # print('mu_qp_prev', self.mu_qp_prev)
        # print('clf_constraint', h_dyn - np.matmul(2*np.abs(eTPG), self.mu_qp_prev))
        # print('h', self.h)
        # print('G', G)
        # print('res.x', self.res.x)
        # print 'res.x', self.res.x
        # print 'A@X', np.matmul(G, self.res.x)
        # print 'constraint', h_dyn - np.matmul(2*np.abs(eTPG), self.mu_qp_prev)
        self.constraint_value = (h_dyn - np.matmul(2*np.abs(eTPG), self.mu_qp_prev))[0][0]
        trigger_bound = -5e-4
        # print ''
        # print self.constraint_value
        # print self.constraint_value + 2 * np.dot(np.linalg.norm(eTPG), np.linalg.norm(sigDelta))
        if self.constraint_value < trigger_bound and self.constraint_value + 2 * np.dot(np.linalg.norm(eTPG), np.linalg.norm(sigDelta)) > trigger_bound:
            self.triggered = True
        
        # for i, cbf in enumerate(self.cbf_list):
        #     cbf_triggered = (h_cbf[i] - np.matmul(G_cbf[i,:2], self.mu_qp_prev))[0]
        #     # print(cbf_triggered)
        #     if cbf_triggered < 0 and abs(cbf_triggered) > 5e-4:#1e-2
        #         self.triggered = False

        if self.verbose:
            print('z_ref: ', x.T)
            print('z_des: ', x_d.T)
            print('u_lim', self.u_lim)
            print('V: ', self.V)
            print('Q:', Q)
            print('p:', np.array(self.p))
            print('G_dyn:', G_dyn)
            print('h_dyn:', h_dyn)
            print('trGssGP',self.trGssGP)
            if h_cbf.shape[0] < 10:
                print('G_cbf:', G_cbf)
                print('h_cbf:', h_cbf)
            print('G_ctrl:', G_ctrl)
            print('h_ctrl:', h_ctrl)
            print('result:', mu_bar)

        return self.mu_qp_prev

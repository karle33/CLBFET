#! /usr/bin/env python

import rospy
import numpy as np
from dynamics import DynamicsQuadrotorModified
from nav_msgs.msg import Odometry
from clbfet.msg import hjlcon
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class sim_odom():
    def __init__(self):
        self.m = 35.89/1000
        self.x_log = np.zeros((6,500-1))
        self.x = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
        self.x_log[:,0:1] = self.x
        self.iters = 0
        self.current_time = rospy.get_rostime()
        self.last_time = self.current_time
        self.last_acc = np.zeros((3,1))
        self.true_dyn = DynamicsQuadrotorModified(disturbance_scale_pos = 0.0, disturbance_scale_vel = -1.0, control_input_scale = 1.0)
        rospy.Subscriber('hjl_con', hjlcon, self.con_cb)
        self.odom_pub = rospy.Publisher('odom', Odometry, queue_size = 10)
        self.timer = rospy.Timer(rospy.Duration(1/250.0), self.timer_cb)
        self.end = False

    def con_cb(self, con):
        print ('control time: ', (rospy.get_rostime() - self.last_send_odom_time).to_sec())
        if self.end:
            return
        # rospy.loginfo('receive control')
        control = np.array([con.thrust, con.roll, con.pitch, con.yaw])
        self.last_acc = self.convert_control_to_mu(control).reshape((3,1))
        # print 'receive acc:', self.last_acc.T


        self.current_time = rospy.get_rostime()
        dt = (self.current_time - self.last_time).to_sec()
        dt = 0.02
        print ('dt', dt)
        next_x = self.true_dyn.step(self.x, self.last_acc, dt)
        print (' with acc=', self.last_acc.T, ' after ', dt)
        print (' to ', next_x.T)
        print ('--------------------------------------------------------')
        self.x = next_x
        self.last_time = self.current_time
        self.x_log[:,self.iters+1:self.iters+2] = self.x
        self.iters += 1

        if con.end:
            self.timer.shutdown()
            print ('x_log.shape', self.x_log.shape)
            self.savefig()
            self.end = True
    
    def timer_cb(self, _event):
        # self.current_time = rospy.get_rostime()
        # dt = (self.current_time - self.last_time).to_sec()
        # # dt = 0.02
        # print 'dt', dt
        # next_x = self.true_dyn.step(self.x, self.last_acc, dt)
        # print ' with acc=', self.last_acc.T, ' after ', dt
        # print ' to ', next_x.T
        # print '--------------------------------------------------------'
        # self.x = next_x
        # self.last_time = self.current_time
        # self.x_log[:,self.iters+1:self.iters+2] = self.x
        # self.iters += 1

        odom = Odometry()
        odom.header.stamp = rospy.get_rostime()
        odom.header.frame_id = 'odom'
        odom.pose.pose.position.x = self.x[0,0]
        odom.pose.pose.position.y = self.x[1,0]
        odom.pose.pose.position.z = self.x[2,0]
        odom.twist.twist.linear.x = self.x[3,0]
        odom.twist.twist.linear.y = self.x[4,0]
        odom.twist.twist.linear.z = self.x[5,0]
        # print 'sending pos', self.x.T
        self.last_send_odom_time = rospy.get_rostime()
        self.odom_pub.publish(odom)
    
    def convert_control_to_mu(self, control):
        thrust = control[0]
        phi = control[1]
        theta = control[2]
        psi = control[3]
        ax = thrust * (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) / self.m
        ay = thrust * (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) / self.m
        az = thrust * (np.cos(phi) * np.cos(theta)) / self.m
        return np.array([ax, ay, az])

    def savefig(self):
        self.x = np.array(self.x)
        fig = plt.figure()
        plt.rcParams.update({'font.size': 12})
        plt.rcParams['axes.unicode_minus'] = False
        ax = plt.axes(projection='3d')
        ax.plot3D(self.x_log[0,:], self.x_log[1,:], self.x_log[2,:], 'g-',alpha=0.9,label='traj')
        ax = fig.gca()
        plt.savefig('odom.png', dpi=600, format='png',bbox_inches='tight')
        rospy.loginfo('fig saved.')

if __name__ == '__main__':
    try:
        rospy.init_node("sim_odom_node")
        simple_odom = sim_odom()
        rospy.loginfo("sim_odom_node is starting...")
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down sim_odom node.")

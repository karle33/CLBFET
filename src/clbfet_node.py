#! /usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import time
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import numpy as np

from adaptive_clbf import AdaptiveClbf
from actionlib_msgs.msg import GoalStatus

import cflib.crtp
from cflib.crazyflie import Crazyflie

from clbfet.msg import hjlcon

steps = 0
pre_update_step = 0

class CLBFET():
    def __init__(self):
        #初始化订阅/odom的实际xyz
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_z = 0.0
        self.linear_x = 0.0
        self.linear_y = 0.0
        self.linear_z = 0.0
        
        self.prev_odom_timestamp = rospy.Time(0)
        self.prev_goal_timestamp = rospy.Time(0)

        self.x_ref = np.zeros((6,1))
        self.x_ref_dot = np.zeros((6,1))

        self.controller = AdaptiveClbf(use_mpc=True, use_trigger=True)
        rospy.loginfo('clbfet controller initial finish')

        params={}
        params["a_lim"] = 13.93
        params["thrust_lim"] = 25#0.5
        params["kp_z"] = 1.0
        params["kd_z"] = 1.0
        params["clf_epsilon"] = 100.0

        params["qp_u_cost"] = 100.0
        params["qp_u_prev_cost"] = 1.0
        params["qp_p1_cost"] = 1.0e8
        params["qp_p2_cost"] = 1.0e12
        params["qp_max_var"] = 1.5
        params["qp_verbose"] = False
        params["max_velocity"] = 2.0
        params["min_velocity"] = 0.5
        params["barrier_vel_gamma"] = 10.0
        params["use_barrier_vel"] = True
        params["use_barrier_pointcloud"] = True
        params["barrier_radius_velocity_scale"] = 0.0
        params["barrier_pc_gamma_p"] = 5.0
        params["barrier_pc_gamma"] = 0.08
        params["verbose"] = False
        params["dt"] = 1.0 / 50.0
        params["max_error"] = 10.0

        params["mpc_stepsize"] = 1
        params["mpc_N"] = 20

        # gp params
        params["qp_ksig"] = 1.0e5
        params["measurement_noise"] = 1.0

        # vanilla nn params
        params["qp_ksig"] = 1.0e2
        params["measurement_noise"] = 1.0

        params["N_data"] = 20#600
        params["learning_verbose"] = False
        params["N_updates"] = 50

        
        self.controller.update_params(params)
        self.controller.update_barrier_locations(np.array([-100]),np.array([-100]),np.array([-100]),np.array([1]))

        self.dt = params["dt"]
        self.iters = 0
        self.sent_train_goal = False
        self.ref_traj = np.zeros((6,20))
        self.add_data = False
        self.use_model = False

        self.cf_connected = False

        # TODO
        self.control_pub = rospy.Publisher('hjl_con', hjlcon, queue_size = 10)

        cflib.crtp.init_drivers()
        self.crazyflie = Crazyflie()
        self.crazyflie.connected.add_callback(self.crazyflie_connected)
        # 0xE7E7E7E060
        self.crazyflie.open_link("radio://0/60/2M/E7E7E7E060")
        while not self.cf_connected:
            time.sleep(1)
        wait_for_param_download(self.crazyflie)
        set_initial_position(self.crazyflie)
        reset_estimator(self.crazyflie)

        # for simple test
        T = 10
        N = int(round(T/self.dt))
        t = np.linspace(0, T-2*self.dt, N-1)
        self.x_d = np.stack((4 * np.cos(0.2 * t), 2 * np.sin(0.2 * 2 * t), 2 * np.sin(0.2 * 2 * t), np.zeros(N-1), np.zeros(N-1), np.zeros(N-1)))	# 8
        # self.x_d = np.stack((np.zeros(N-1), np.zeros(N-1), np.ones(N-1)*.5, np.zeros(N-1), np.zeros(N-1), np.zeros(N-1)))
        # self.x_d[2,-300:] = 0.13
        self.x_d[3,:-1] = np.diff(self.x_d[0,:])
        self.x_d[4,:-1] = np.diff(self.x_d[1,:])
        self.x_d[5,:-1] = np.diff(self.x_d[2,:])
        self.x_d[3,-1]=self.x_d[3,-2]
        self.x_d[4,-1]=self.x_d[4,-2]
        self.x_d[5,-1]=self.x_d[5,-2]
        self.ref_traj = self.x_d[:,self.iters:self.iters+20]
        # self.ref_traj = self.x_d[:,:20]
        
        self.x_log = []
        self.acc_log = []
        self.predict_error_log = []
        self.predict_var_log = []
        self.trigger_log = []

        # for cmd
        self.control_mode = -1
        self.setthrust = 0
        self.op = 'p'
        self.oplock = False
                
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.timer_cb)
        rospy.Subscriber("odom", Odometry, self.odom_cb)
        rospy.Subscriber("hjl_cmd", String, self.cmd_cb)
        # TODO
        self.timer_control = rospy.Timer(rospy.Duration(self.dt), self.timer_con_cb)
        # TODO
        # rospy.Subscriber("hjl_obstacles", Odometry, self.obstacles_cb)
        # rospy.Subscriber("hjl_goal", Odometry, self.goal_cb)



    def crazyflie_connected(self, _event):
        print ('connected to crazyflie.')
        self.cf_connected = True
        
    def wait_for_param_download(cf):
        while not cf.param.is_updated:
            time.sleep(1.0)
        print('Parameters downloaded')
    
    def set_initial_position(cf):
        print('set_initial_position')

        (x_real, y_real, z_real) = (0.0, 0.0, 0.0)

        cf.param.set_value('kalman.initialX', x_real)
        cf.param.set_value('kalman.initialY', y_real)
        cf.param.set_value('kalman.initialZ', z_real)

        print('set value for [kalman.initialYaw]')
        yaw_radians = math.radians(0)
        scf.cf.param.set_value('kalman.initialYaw', yaw_radians)

    def reset_estimator(cf):
        print('reset_estimator')
        # print('sequence in reset_estimator = ', sequence)
        cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        cf.param.set_value('kalman.resetEstimation', '0')
        wait_for_position_estimator(scf, sequence)
    
    def wait_for_position_estimator(cf):
        # print('sequence = ',sequence)
        print('Waiting for estimator to find position...')
        print('scf = ', cf)

        log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
        log_config.add_variable('kalman.varPX', 'float')
        log_config.add_variable('kalman.varPY', 'float')
        log_config.add_variable('kalman.varPZ', 'float')
        log_config.add_variable('kalman.stateX', 'float')
        log_config.add_variable('kalman.stateY', 'float')
        log_config.add_variable('kalman.stateZ', 'float')

        var_y_history = [1000] * 10
        var_x_history = [1000] * 10
        var_z_history = [1000] * 10

        # threshold = 0.001
        threshold = 0.005

        with SyncLogger(cf, log_config) as logger:
            for log_entry in logger:
                [x_real, y_real, z_real] = [0.0, 0.0, 0.0]
                cf.extpos.send_extpos(x_real, y_real, z_real)

                data = log_entry[1]

                print("pos_mcu: ", data['kalman.stateX'], data['kalman.stateY'], data['kalman.stateZ'])
                var_x_history.append(data['kalman.varPX'])
                var_x_history.pop(0)
                var_y_history.append(data['kalman.varPY'])
                var_y_history.pop(0)
                var_z_history.append(data['kalman.varPZ'])
                var_z_history.pop(0)

                min_x = min(var_x_history)
                max_x = max(var_x_history)
                min_y = min(var_y_history)
                max_y = max(var_y_history)
                min_z = min(var_z_history)
                max_z = max(var_z_history)

                print("{} {} {}".
                    format(max_x - min_x, max_y - min_y, max_z - min_z))

                if (max_x - min_x) < threshold and (
                        max_y - min_y) < threshold and (
                        max_z - min_z) < threshold:
                    break

    def cf_takeoff(cf):
        take_off_time = 1.5 # s
        take_off_height = 0.7 # m
        take_off_x = 0.0
        take_off_y = 0.0
        end_time = time.time() + take_off_time
        while time.time() < end_time:
            [x_real, y_real, z_real] = [0.0, 0.0, 0.0]
            cf.extpos.send_extpos(x_real, y_real, z_real)
            cf.commander.send_position_setpoint(take_off_x, take_off_y, take_off_height, 0)
            time.sleep(0.1)
    
    def cf_land(cf):
        landing_time = 1.5
        sleep_time = 0.1
        steps = int(landing_time / sleep_time)
        vz = -0.7/ landing_time
        for _ in range(steps):
            [x_real, y_real, z_real] = [0.0, 0.0, 0.0]
            cf.extpos.send_extpos(x_real, y_real, z_real)
            cf.commander.send_velocity_world_setpoint(0, 0, vz, 0)
            time.sleep(sleep_time)
        # cf.high_level_commander.land(0.0, 2.0)
        cf.commander.send_stop_setpoint()

    def save_log(self):
        fp = open('x_log.txt', 'w+')
        for x in self.x_log:
            fp.write('px: ' + str(x[0]) + '\n')
            fp.write('py: ' + str(x[1]) + '\n')
            fp.write('pz: ' + str(x[2]) + '\n')
            fp.write('vx: ' + str(x[3]) + '\n')
            fp.write('vy: ' + str(x[4]) + '\n')
            fp.write('vz: ' + str(x[5]) + '\n')
        fp.close()
        fp = open('acc_log.txt', 'w+')
        for a in self.acc_log:
            fp.write('ax: ' + str(a[0]) + '\n')
            fp.write('ay: ' + str(a[1]) + '\n')
            fp.write('az: ' + str(a[2]) + '\n')
        fp.close()
        fp = open('predict_error_log.txt', 'w+')
        for e in self.predict_error_log:
            fp.write(str(e) + '\n')
        fp.close()
        fp = open('predict_var_log.txt', 'w+')
        for v in self.predict_var_log:
            fp.write(str(v) + '\n')
        fp.close()
        fp = open('trigger_log.txt', 'w+')
        for t in self.trigger_log:
            fp.write(str(t) + '\n')
        fp.close()

    def cmd_cb(self, key):
        print (key)
        op = key.data
        if op == '0':
            self.control_mode = -1
            self.crazyflie.commander.send_stop_setpoint()
            self.crazyflie.commander.send_stop_setpoint()
            self.crazyflie.commander.send_stop_setpoint()
        elif op == '1':
            self.control_mode = 1
            rospy.loginfo('shoudong controller start.')
        elif op == '2':
            self.control_mode = 2
            rospy.loginfo('clbfet controller start.')
        elif op == 'v':
            self.add_data = True
            rospy.loginfo('clbfet controller add data true.')
        elif op == 'b':
            self.add_data = False
            rospy.loginfo('clbfet controller add data false.')
        elif op == 'm':
            self.use_model = True
            rospy.loginfo('clbfet controller use model true.')
        elif op == 'n':
            self.use_model = False
            rospy.loginfo('clbfet controller use model false.')
        elif op == 'y':
            self.control_mode = 3
            # self.crazyflie.high_level_commander.takeoff(0.7, 2)
            cf_takeoff(self.crazyflie)
            rospy.loginfo('take off')
        elif op == 'u':
            self.control_mode = 3
            # self.crazyflie.high_level_commander.land(0, 2)
            cf_land(self.crazyflie)
            rospy.loginfo('land')
        elif op == 't':
            self.crazyflie.close_link()
            rospy.logwarn('crazyflie link close.')
        elif self.control_mode == 1:
            if not self.oplock:
                self.op = op
                self.oplock = True
                rospy.loginfo('receive ' + op)
            else:
                rospy.loginfo('locked!')
        else:
            rospy.loginfo('wrong mode.')
    
    def timer_cb(self, _event):
        if self.sent_train_goal:
            # states 0 = PENDING, 1 = ACTIVE, 2 = PREEMPTED, 3 = SUCCEEDED
            train_state = self.controller.train_model_action_client.get_state()
            # print("State:",train_state)
            if train_state == GoalStatus.SUCCEEDED:
                train_result = self.controller.train_model_action_client.get_result() 
                if hasattr(train_result, 'model_trained'):
                    self.controller.model_trained = self.controller.train_model_action_client.get_result().model_trained
                    self.sent_train_goal = False
                    end_time = rospy.get_rostime()
                    rospy.loginfo('trained')
                    # rospy.logwarn(["training latency (ms): ", (end_time-self.train_start_time).to_sec() * 1000.0])
                # else:
                #     self.controller.model_trained = False

    def odom_cb(self, odom):
        # if self.iters == 299:
        #     con = hjlcon()
        #     con.end = True
        #     self.control_pub.publish(con)
        #     return
        # elif self.iters > 299:
        #     return

        self.pose_x = odom.pose.pose.position.x
        self.pose_y = odom.pose.pose.position.y
        self.pose_z = odom.pose.pose.position.z
        self.linear_x = odom.twist.twist.linear.x
        self.linear_y = odom.twist.twist.linear.y
        self.linear_z = odom.twist.twist.linear.z

        self.x_log.append([self.pose_x, self.pose_y, self.pose_z, self.linear_x, self.linear_y, self.linear_z])

        # # print 'iters: ', self.iters

        # # TODO:  theory - maybe dt of rostime and header does not match dt of actual odometry data.  this might cause big problems.
        # dt = (odom.header.stamp - self.prev_odom_timestamp).to_sec()
        # # dt = 0.02

        # if dt < 0:
        #     rospy.logwarn("detected jump back in time!  resetting prev_odom_timestamp.")
        #     self.prev_odom_timestamp = self.odom.header.stamp
        #     self.sent_train_goal = False
        #     return

        # if dt < self.dt:
        #     rospy.logwarn("dt is too small! (%f)  skipping this odometry callback!", dt)
        #     return
        
        # self.prev_odom_timestamp = odom.header.stamp
        # x_cur = np.array([[self.pose_x, self.pose_y, self.pose_z,
        #                 self.linear_x, self.linear_y, self.linear_z]]).T
        # # print 'receive x', x_cur
        
        # self.x_ref_dot = (self.ref_traj[:,:1] - self.x_ref) / dt
        # self.x_ref = self.ref_traj[:,:1]

        # acc = self.controller.get_control(x_cur,self.x_ref,self.x_ref_dot,dt=dt,use_model=True,add_data=self.add_data,use_qp=True,ref_traj=self.ref_traj)
        # self.add_data = True
        
        # if (self.iters - 1 == 10) or self.controller.qpsolve.triggered:
        #     if not self.sent_train_goal:
        #         self.train_start_time = rospy.get_rostime()
        #         print("sending training goal")
        #         self.controller.train_model_action_client.send_goal(self.controller.train_model_goal)
        #         self.sent_train_goal = True
        
        # self.iters += 1
        # # self.ref_traj = self.x_d[:,self.iters:self.iters+20]

        # c = self.controller.dyn.convert_mu_to_control(acc.reshape((3,1)))

        # # TODO
        # crazyflie.commander.send_setpoint(np.rad2deg(c[1]), np.rad2deg(c[2]), yawrate, thrust)
        # # con = hjlcon()
        # # con.thrust = c[0]
        # # con.roll = c[1]
        # # con.pitch = c[2]
        # # con.yaw = c[3]
        # # con.end = False
        # # self.control_pub.publish(con)

    def timer_con_cb(self, _event):
        if self.control_mode == -1:
            return
        elif self.control_mode == 1:
            if self.oplock:
                if self.op == 'w':
                    self.crazyflie.commander.send_velocity_world_setpoint(0.5, 0.0, 0.0, 0)
                elif self.op == 's':
                    self.crazyflie.commander.send_velocity_world_setpoint(-0.5, 0.0, 0.0, 0)
                elif self.op == 'a':
                    self.crazyflie.commander.send_velocity_world_setpoint(0.0, -0.5, 0.0, 0)
                elif self.op == 'd':
                    self.crazyflie.commander.send_velocity_world_setpoint(0.0, 0.5, 0.0, 0)
                elif self.op == 'i':
                    self.crazyflie.commander.send_velocity_world_setpoint(0.0, 0.0, 0.5, 0)
                elif self.op == 'j':
                    self.crazyflie.commander.send_velocity_world_setpoint(0.0, 0.0, 0.0, -10)
                elif self.op == 'k':
                    self.crazyflie.commander.send_velocity_world_setpoint(0.0, 0.0, -0.5, 0)
                elif self.op == 'l':
                    self.crazyflie.commander.send_velocity_world_setpoint(0.0, 0.0, 0.0, 10)
                self.oplock = False
                print ('unlocked!')
            else:
                self.crazyflie.commander.send_velocity_world_setpoint(0.0, 0.0, 0.0, 0)
                # pass
        elif self.control_mode == 2:
            if self.iters == 999:
                con = hjlcon()
                con.end = True
                self.control_pub.publish(con)
                # self.crazyflie.commander.send_setpoint(0, 0, 0, 0)
                # time.sleep(0.1)
                # self.crazyflie.close_link()

                self.save_log()
                return
            elif self.iters > 999:
                return
            
            x_cur = np.array([[self.pose_x, self.pose_y, self.pose_z,
                            self.linear_x, self.linear_y, self.linear_z]]).T
            # print 'receive x', x_cur
            
            self.x_ref_dot = (self.ref_traj[:,:1] - self.x_ref) / self.dt
            self.x_ref = self.ref_traj[:,:1]

            acc = self.controller.get_control(x_cur,self.x_ref,self.x_ref_dot,dt=self.dt,use_model=self.use_model,add_data=self.add_data,use_qp=True,ref_traj=self.ref_traj)
            self.acc_log.append(acc)
            self.predict_error_log.append(self.controller.predict_error)
            self.predict_var_log.append(self.controller.predict_var)
            self.trigger_log.append(self.controller.qpsolve.triggered)

            # if self.iters > 7:
            #     self.add_data = True
            
            if (self.iters - 1 == 10) or self.controller.qpsolve.triggered:
                if not self.sent_train_goal:
                    self.train_start_time = rospy.get_rostime()
                    print("sending training goal")
                    self.controller.train_model_action_client.send_goal(self.controller.train_model_goal)
                    self.sent_train_goal = True
            
            self.iters += 1
            # self.ref_traj = self.x_d[:,self.iters:self.iters+20]


            # TODO
            # c = self.controller.dyn.convert_mu_to_control(acc.reshape((3,1)))
            # self.crazyflie.commander.send_setpoint(np.rad2deg(c[1]), np.rad2deg(c[2]), yawrate, thrust)
            # desire_next_x =self.controller.dyn.step(self.controller.z, acc, self.dt)
            # self.crazyflie.commander.send_velocity_world_setpoint(desire_next_x[3], desire_next_x[4], desire_next_x[5], 0)

            c = self.controller.dyn.convert_mu_to_control(acc.reshape((3,1)))
            con = hjlcon()
            con.thrust = c[0]
            con.roll = c[1]
            con.pitch = c[2]
            con.yaw = c[3]
            con.end = False
            self.control_pub.publish(con)
    
    def goal_cb(self, goal):
        dt = (goal.header.stamp - self.prev_goal_timestamp).to_sec()

        if dt < 0:
            rospy.logwarn("detected jump back in time!  resetting prev_goal_timestamp.")
            self.prev_goal_timestamp = goal.header.stamp
            return

        if dt < self.dt / 4.0:
            rospy.logwarn("dt is too small! (%f)  skipping this goal callback!", dt)
            return

        # TODO: get current desired trajectory
        q = (goal.pose.pose.orientation.x,
             goal.pose.pose.orientation.y,
             goal.pose.pose.orientation.z,
             goal.pose.pose.orientation.w)
        euler = tr.euler_from_quaternion(q)
        desired_heading = euler[2]
        desired_vel = goal.twist.twist.linear.x

        x_ref = np.array([[goal.pose.pose.position.x,goal.pose.pose.position.y,desired_heading,desired_vel]]).T
        z_ref_new = self.adaptive_clbf.dyn.convert_x_to_z(x_ref)
        self.z_ref_dot = (z_ref_new - self.z_ref) / dt
        self.z_ref = z_ref_new
        self.prev_goal_timestamp = goal.header.stamp
    
    def obstacles_cb(self, obstacles):
        # TODO: update obstacles
        barrier_x = np.array([-100])
        barrier_y = np.array([-100])
        barrier_z = np.array([-100])
        barrier_r = np.array([1])
        # TODO: updates locations here or in odom_cb(), depends on the rate
        self.controller.update_barrier_locations(barrier_x,barrier_y,barrier_z,barrier_r)


if __name__ == '__main__':
    try:
        rospy.init_node("clbfet_node")
        clbfet = CLBFET()
        rospy.loginfo("clbfet_node is starting...")
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down clbfet node.")

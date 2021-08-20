#! /usr/bin/env python

import sys
import tty, termios
import rospy
from std_msgs.msg import String
from nav_msgs.msg import Odometry

class Commander():
    def __init__(self):
        # self.roll = 0.0
        # self.pitch = 0.0
        # self.yaw = 0.0
        # self.thrust = 0
        self.pub = rospy.Publisher('hjl_cmd', String, queue_size=10)
        # self.rate = rospy.Rate(1)
        self.timer = rospy.Timer(rospy.Duration(1/1.0), self.timer_cb)
  
    def timer_cb(self, _event):
        while not rospy.is_shutdown():  
            fd = sys.stdin.fileno()  
            old_settings = termios.tcgetattr(fd)  
            # no echo
            old_settings[3] = old_settings[3] & ~termios.ICANON & ~termios.ECHO  
            try :  
                tty.setraw( fd )  
                ch = sys.stdin.read( 1 )  
            finally :  
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            # ch = 'w'
            # print ch, ch, ch
            if ch == 'p':  
                self.timer.shutdown()
                exit()
            else:
                self.pub.publish(ch)
  
if __name__ == '__main__':
    try:
        rospy.init_node("cmd_node")
        commander = Commander()
        rospy.loginfo("cmd_node is starting...")
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down sim_odom node.")
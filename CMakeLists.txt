cmake_minimum_required(VERSION 2.8.3)
project(clbfet)

find_package(catkin REQUIRED COMPONENTS
    dynamic_reconfigure
    message_generation
    roscpp
    rospy
    std_msgs
    actionlib_msgs
    nav_msgs
)

add_service_files(DIRECTORY srv FILES AddData2Model.srv PredictModel.srv)
add_action_files(DIRECTORY action FILES TrainModel.action)
add_message_files(DIRECTORY msg FILES hjlcon.msg)

generate_messages(
  DEPENDENCIES
  std_msgs
  actionlib_msgs
  nav_msgs
)

catkin_package(CATKIN_DEPENDS message_runtime)

catkin_install_python(PROGRAMS src/clbfet_node.py src/model_service.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

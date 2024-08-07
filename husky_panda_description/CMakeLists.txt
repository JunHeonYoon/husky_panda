cmake_minimum_required(VERSION 3.0.2)
project(husky_panda_description)

find_package(catkin REQUIRED)

catkin_package(CATKIN_DEPENDS xacro)


include_directories(DIRECTORY robots
DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

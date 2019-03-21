cmake_minimum_required(VERSION 2.8.3)
project(lidar_camera_fusion)

add_compile_options(-std=c++11)

find_package(cmake_modules REQUIRED)
find_package(Eigen REQUIRED)
find_package(PCL REQUIRED)
find_package(catkin
    REQUIRED COMPONENTS
            roscpp
            sensor_msgs
	    std_msgs
            nav_msgs
            geometry_msgs
	    cv_bridge
	    pcl_ros
            visualization_msgs
)
find_package(OpenCV REQUIRED)

#add_definitions(${EIGEN_DEFINITIONS})

catkin_package(
    INCLUDE_DIRS
            include
    CATKIN_DEPENDS
            roscpp
            sensor_msgs
            nav_msgs
            geometry_msgs
            visualization_msgs
            yaml_cpp_0_3
)

include_directories(include
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

link_directories(${PCL_LIBRARY_DIRS} ${EIGEN3_LIBRARY_DIR})

add_executable(fusion src/main.cpp)
add_dependencies(fusion ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} ${PCL_LIBRARIES} ${EIGEN3_LIBRARY_DIR})
target_link_libraries(fusion ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${EIGEN3_LIBRARY_DIR} ${OpenCV_LIBRARIES})

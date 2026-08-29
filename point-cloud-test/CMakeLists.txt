cmake_minimum_required(VERSION 3.8)
project(point-cloud-test)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# set(PCL_DIR "/home/wafi/Software/pcl/build/" CACHE PATH "Directory containing PCLConfig.cmake")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(message_filters REQUIRED)
find_package(pcl_cstm_msg REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(zed_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(PCL 1.12 REQUIRED)

# Include PCL headers
include_directories(${PCL_INCLUDE_DIRS})

# Link PCL directories
link_directories(${PCL_LIBRARY_DIRS})

# Add PCL definitions
add_definitions(${PCL_DEFINITIONS})

add_executable(pcl_proc_node src/pcl_proc_node.cpp)

target_include_directories(pcl_proc_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(pcl_proc_node ${PCL_LIBRARIES})
ament_target_dependencies(pcl_proc_node
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  pcl_conversions
  message_filters
  pcl_cstm_msg
)

# Bounding-box perception path from Wafi-Afdi/PCL_Segmentation.  Keep the
# geometric PCL node above for Gazebo and legacy bag replay.
add_executable(bb_pcl_proc_node src/bb_pcl_proc_node.cpp)

target_include_directories(bb_pcl_proc_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(bb_pcl_proc_node Eigen3::Eigen)
ament_target_dependencies(bb_pcl_proc_node
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  message_filters
  pcl_cstm_msg
  tf2
  tf2_ros
  zed_msgs
)

install(TARGETS
  pcl_proc_node
  bb_pcl_proc_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

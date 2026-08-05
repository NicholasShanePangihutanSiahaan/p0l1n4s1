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

find_package(PCL 1.12 REQUIRED)
find_package(VTK REQUIRED)

# Include PCL headers
include_directories(${PCL_INCLUDE_DIRS})

# Link PCL directories
link_directories(${PCL_LIBRARY_DIRS})

# Add PCL definitions
add_definitions(${PCL_DEFINITIONS})

# pcl_proc executable (standalone PCD processor)
add_executable(pcl_proc src/main_pcl_proc.cpp)

target_include_directories(pcl_proc PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(pcl_proc ${PCL_LIBRARIES} ${VTK_LIBRARIES})
ament_target_dependencies(pcl_proc rclcpp pcl_cstm_msg)

if (VTK_VERSION VERSION_GREATER_EQUAL "8.90.0")
  vtk_module_autoinit(
    TARGETS pcl_proc
    MODULES ${VTK_LIBRARIES}
  )
endif()

# pcl_proc_node executable (ROS 2 point cloud processing node)
add_executable(pcl_proc_node src/pcl_proc_node.cpp)

target_include_directories(pcl_proc_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(pcl_proc_node ${PCL_LIBRARIES} ${VTK_LIBRARIES})
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

# pcl_norm_vis
add_executable(pcl_norm_vis src/pcl_norm_vis.cpp)

target_include_directories(pcl_norm_vis PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(pcl_norm_vis ${PCL_LIBRARIES} ${VTK_LIBRARIES})
ament_target_dependencies(pcl_norm_vis
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  pcl_conversions
  message_filters
  pcl_cstm_msg
)

if (VTK_VERSION VERSION_GREATER_EQUAL "8.90.0")
  vtk_module_autoinit(
    TARGETS pcl_proc_node
    MODULES ${VTK_LIBRARIES}
  )
endif()

install(TARGETS
  pcl_proc
  pcl_proc_node
  pcl_norm_vis
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

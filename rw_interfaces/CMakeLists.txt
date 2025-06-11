cmake_minimum_required(VERSION 3.8)
project(rw_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/NavigationStatus.msg"
  "msg/ProximityStatus.msg"  
  "srv/ControlNavigation.srv"
  "srv/SetWaypoints.srv"
  "srv/GetWaypoints.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE) # Assuming you'll add copyright/license
  set(ament_cmake_cpplint_FOUND TRUE) # Assuming
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
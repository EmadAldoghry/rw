cmake_minimum_required(VERSION 3.8)
project(rw_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED) # If your .msg files use Header etc. from std_msgs

# This is crucial for rosidl_generate_interfaces
find_package(rosidl_default_generators REQUIRED)

# Corrected path for msg file
# Remember you have:
# rw_interfaces/msg/NavigationStatus.msg
# rw_interfaces/srv/ControlNavigation.srv
# rw_interfaces/srv/SetWaypoints.srv
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/NavigationStatus.msg"
  "srv/ControlNavigation.srv"
  "srv/SetWaypoints.srv"
  DEPENDENCIES geometry_msgs std_msgs # Add std_msgs if NavigationStatus.msg uses Header
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE) # Assuming you'll add copyright/license
  set(ament_cmake_cpplint_FOUND TRUE) # Assuming
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
cmake_minimum_required(VERSION 3.8)
project(pm01_deploy)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(interface_protocol REQUIRED)
# 修复1：yaml-cpp查找（指定版本+NO_MODULE）
find_package(yaml-cpp 0.6 REQUIRED NO_MODULE)
find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  src
)

add_executable(pm01_controller
  src/pm01_controller.cpp
  src/components/message_handler.cc
)
ament_target_dependencies(pm01_controller
  rclcpp
  interface_protocol
)
target_link_libraries(pm01_controller
  # 修复2：yaml-cpp链接目标（去掉命名空间别名）
  yaml-cpp
  Eigen3::Eigen
  "${TORCH_LIBRARIES}"
)

install(TARGETS
  pm01_controller
  DESTINATION lib/${PROJECT_NAME}
)

set_property(TARGET pm01_controller PROPERTY CXX_STANDARD 14)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
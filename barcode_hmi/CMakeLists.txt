cmake_minimum_required(VERSION 3.8)
project(barcode_hmi)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BarcodeDetection.msg"
  "msg/CameraStatusArray.msg"
  "msg/SingleCameraStatus.msg"
  DEPENDENCIES std_msgs builtin_interfaces
)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Export runtime dependency so other packages can use our messages
ament_export_dependencies(rosidl_default_runtime)

# Finalize package
ament_package()

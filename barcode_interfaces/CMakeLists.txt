cmake_minimum_required(VERSION 3.8)
project(barcode_interfaces)

# === Dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

# === Generate Messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BarcodeDetection.msg"
  "msg/CameraStatusArray.msg"
  "msg/SingleCameraStatus.msg"
  DEPENDENCIES std_msgs builtin_interfaces
)

# === Export interface
ament_export_dependencies(rosidl_default_runtime)
ament_package()

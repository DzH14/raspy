# An array of statuses for all cameras in the system

# The header is useful for timestamping the entire status update
std_msgs/Header header

# An array containing the status of each individual camera
SingleCameraStatus[] cameras
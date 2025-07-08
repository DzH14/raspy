# Represents a single successful barcode detection event

# The data or value of the barcode that was scanned
string barcode_data

# The timestamp of when the detection occurred
builtin_interfaces/Time timestamp

# How long the detection process took in seconds
float32 detection_duration

# The name of the camera that performed the detection
string camera_name

# The processing FPS of the source camera at the moment of detection
float32 fps_at_detection
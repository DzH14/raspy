import rclpy
from rclpy.node import Node
import time
import cv2
from pyzbar import pyzbar
from std_msgs.msg import String as RosString

# Import the custom messages we created
from barcode_interfaces.msg import BarcodeDetection, CameraStatusArray, SingleCameraStatus

class LiveBackendNode(Node):
    def __init__(self):
        super().__init__('live_backend_node')
        self.get_logger().info('Live Backend Node started. Waiting for START command from HMI.')

        # --- Parameters ---
        self.declare_parameter('camera1_device', 0)
        self.declare_parameter('camera2_device', 2)
        self.declare_parameter('camera3_device', -1) # Disabled by default
        self.declare_parameter('use_v4l2', False)
        
        # === NEW PARAMETERS FOR IMAGE PROCESSING ===
        self.declare_parameter('brightness_adjustment', 10)  # Default: slightly brighter
        self.declare_parameter('contrast_adjustment', 25)   # Default: moderately higher contrast

        cam1_idx = self.get_parameter('camera1_device').value
        cam2_idx = self.get_parameter('camera2_device').value
        cam3_idx = self.get_parameter('camera3_device').value
        use_v4l2 = self.get_parameter('use_v4l2').value

        # Log the initial processing values
        self.get_logger().info(
            f"Image Processing Params -> "
            f"Brightness: {self.get_parameter('brightness_adjustment').value}, "
            f"Contrast: {self.get_parameter('contrast_adjustment').value}"
        )

        # --- State Variables ---
        self.detection_active = False
        self.caps = {}
        self.camera_names = {}
        self.camera_fps_trackers = {}
        self.seen_barcodes = set()
        
        # --- Camera Initialization ---
        self.init_camera("Camera 1", cam1_idx, use_v4l2)
        self.init_camera("Camera 2", cam2_idx, use_v4l2)
        self.init_camera("Camera 3", cam3_idx, use_v4l2)

        # --- ROS 2 Interfaces ---
        self.control_subscriber = self.create_subscription(
            RosString, 'hmi/control_cmd', self.control_cmd_callback, 10)
        self.detection_publisher = self.create_publisher(
            BarcodeDetection, 'barcode/detection', 10)
        self.status_publisher = self.create_publisher(
            CameraStatusArray, 'camera/status', 10)

        # --- Timers ---
        self.processing_timer = self.create_timer(1.0 / 30.0, self.process_frames_callback)
        self.status_timer = self.create_timer(1.0, self.publish_status_callback)

    def init_camera(self, name, index, use_v4l2):
        # ... (This method is unchanged)
        if index < 0:
            self.get_logger().info(f"{name} is disabled (index < 0).")
            return
        api_preference = cv2.CAP_V4L2 if use_v4l2 else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, api_preference)
        time.sleep(1.0)
        if cap.isOpened():
            self.caps[name] = cap
            self.camera_names[name] = f"{name} (idx {index})"
            self.get_logger().info(f"✅ {self.camera_names[name]} opened successfully.")
            self.camera_fps_trackers[name] = [0, time.time()]
        else:
            self.get_logger().error(f"❌ {name} (idx {index}) failed to open.")
            
    def control_cmd_callback(self, msg: RosString):
        # ... (This method is unchanged)
        command = msg.data.upper()
        self.get_logger().info(f"Received HMI command: '{command}'")
        if command == "START":
            if not self.detection_active:
                self.detection_active = True
                self.get_logger().info("--- Detection STARTED ---")
        elif command == "STOP":
            if self.detection_active:
                self.detection_active = False
                self.get_logger().info("--- Detection STOPPED ---")
        elif command == "RESET":
            self.seen_barcodes.clear()
            self.get_logger().info("--- Barcode history has been RESET ---")

    def publish_status_callback(self):
        # ... (This method is unchanged)
        status_msg = CameraStatusArray()
        status_msg.header.stamp = self.get_clock().now().to_msg()
        for name in ["Camera 1", "Camera 2", "Camera 3"]:
            single_status = SingleCameraStatus()
            single_status.camera_name = name
            if name in self.caps:
                if self.detection_active:
                    tracker = self.camera_fps_trackers[name]
                    elapsed_time = time.time() - tracker[1]
                    fps = tracker[0] / elapsed_time if elapsed_time > 0 else 0.0
                    single_status.fps = float(fps)
                    tracker[0] = 0; tracker[1] = time.time()
                    single_status.status = "Scanning"
                else:
                    single_status.status = "Connected"
                    single_status.fps = 0.0
            else:
                single_status.status = "Disconnected"
                single_status.fps = 0.0
            status_msg.cameras.append(single_status)
        self.status_publisher.publish(status_msg)

    # === NEW HELPER FUNCTION FOR IMAGE PROCESSING ===
    def preprocess_frame(self, frame):
        """
        Applies brightness and contrast adjustments based on ROS parameters.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get the current parameter values
        brightness = self.get_parameter('brightness_adjustment').value
        contrast = self.get_parameter('contrast_adjustment').value

        # Apply the brightness and contrast adjustment
        # The formula is: new_image = alpha * old_image + beta
        alpha = 1.0 + (contrast / 127.0)  # Controls contrast
        beta = float(brightness)          # Controls brightness
        
        adjusted_gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        
        return adjusted_gray

    def process_frames_callback(self):
        if not self.detection_active:
            return

        for name, cap in self.caps.items():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret: continue

            self.camera_fps_trackers[name][0] += 1

            # === UPDATED PROCESSING PIPELINE ===
            # 1. Pre-process the frame using our new function
            processed_frame = self.preprocess_frame(frame)
            
            # 2. Decode the *processed* frame
            barcodes = pyzbar.decode(processed_frame)

            if barcodes:
                for barcode in barcodes:
                    data = barcode.data.decode("utf-8")
                    btype = barcode.type
                    key = (data, btype)
                    if key not in self.seen_barcodes:
                        self.seen_barcodes.add(key)
                        duration = time.time() - start_time
                        self.get_logger().info(f"[NEW DETECTION] Found '{data}' on {name}")
                        
                        tracker = self.camera_fps_trackers[name]
                        elapsed_time = time.time() - tracker[1]
                        current_fps = tracker[0] / elapsed_time if elapsed_time > 0 else 0.0
                        
                        self.publish_detection(data, duration, name, current_fps)

    def publish_detection(self, barcode_data, duration, camera_name, fps):
        # ... (This method is unchanged)
        detection_msg = BarcodeDetection()
        detection_msg.barcode_data = barcode_data
        detection_msg.timestamp = self.get_clock().now().to_msg()
        detection_msg.detection_duration = float(duration)
        detection_msg.camera_name = camera_name
        detection_msg.fps_at_detection = float(fps)
        self.detection_publisher.publish(detection_msg)

    def cleanup(self):
        # ... (This method is unchanged)
        self.get_logger().info("Shutting down live backend node.")
        for name, cap in self.caps.items():
            cap.release()
            self.get_logger().info(f"{name} released.")

def main(args=None):
    rclpy.init(args=args)
    node = LiveBackendNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping node (Ctrl+C).")
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
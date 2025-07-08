import rclpy
from rclpy.node import Node
import time
import cv2
from pyzbar import pyzbar
import serial # Added for Arduino communication

from std_msgs.msg import String as RosString
from barcode_interfaces.msg import BarcodeDetection, CameraStatusArray, SingleCameraStatus

class LiveBackendNode(Node):
    def __init__(self):
        super().__init__('live_backend_node')
        self.get_logger().info('Live Backend Node started. Waiting for START command from HMI.')

        # --- System Parameters ---
        self.declare_parameter('camera1_device', 4)
        self.declare_parameter('camera2_device', 0)
        self.declare_parameter('camera3_device', -1)
        self.declare_parameter('use_v4l2', False)
        self.declare_parameter('brightness_adjustment', 10)
        self.declare_parameter('contrast_adjustment', 25)
        
        # === NEW PARAMETERS FOR ARDUINO CONTROL ===
        self.declare_parameter('serial_port', '/dev/ttyUSB0') # IMPORTANT: Change to your Arduino's port
        self.declare_parameter('baud_rate', 9600)
        self.declare_parameter('scan_quiet_time', 10.0) # Time in seconds with no detections to trigger a move

        cam1_idx = self.get_parameter('camera1_device').value
        cam2_idx = self.get_parameter('camera2_device').value
        cam3_idx = self.get_parameter('camera3_device').value
        use_v4l2 = self.get_parameter('use_v4l2').value
        
        # --- State Variables ---
        self.detection_active = False
        self.caps = {}
        self.camera_names = {}
        self.camera_fps_trackers = {}
        self.seen_barcodes = set()

        # === NEW STATE VARIABLES FOR ARM CONTROL ===
        self.arm_state = "IDLE" # States: IDLE, SCANNING_BOTTOM, MOVING_UP, SCANNING_TOP, MOVING_DOWN
        self.arm_position = "UNKNOWN" # Positions: UNKNOWN, POS_BAWAH, POS_ATAS
        self.last_detection_time = self.get_clock().now()
        self.arduino_serial = None

        # --- Camera Initialization ---
        self.init_camera("Camera 1", cam1_idx, use_v4l2)
        self.init_camera("Camera 2", cam2_idx, use_v4l2)
        self.init_camera("Camera 3", cam3_idx, use_v4l2)

        # --- Arduino Initialization ---
        self.init_arduino()

        # --- ROS 2 Interfaces ---
        self.control_subscriber = self.create_subscription(RosString, 'hmi/control_cmd', self.control_cmd_callback, 10)
        self.detection_publisher = self.create_publisher(BarcodeDetection, 'barcode/detection', 10)
        self.status_publisher = self.create_publisher(CameraStatusArray, 'camera/status', 10)
        # === NEW PUBLISHER FOR ARM STATUS ===
        self.arm_status_publisher = self.create_publisher(RosString, 'arm/status', 10)

        # --- Timers ---
        self.processing_timer = self.create_timer(1.0 / 30.0, self.process_frames_callback)
        self.status_timer = self.create_timer(1.0, self.publish_status_callback)
        # === NEW TIMER FOR READING ARDUINO SERIAL ===
        self.arduino_read_timer = self.create_timer(0.1, self.read_from_arduino_callback)

    def init_camera(self, name, index, use_v4l2):
        if index < 0: return
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2 if use_v4l2 else cv2.CAP_ANY)
        time.sleep(1.0)
        if cap.isOpened():
            self.caps[name] = cap
            self.get_logger().info(f"✅ {name} (idx {index}) opened successfully.")
            self.camera_fps_trackers[name] = [0, time.time()]
        else:
            self.get_logger().error(f"❌ {name} (idx {index}) failed to open.")

    def init_arduino(self):
        """Initializes the serial connection to the Arduino."""
        port = self.get_parameter('serial_port').value
        baud = self.get_parameter('baud_rate').value
        try:
            self.arduino_serial = serial.Serial(port, baud, timeout=1)
            time.sleep(2) # Wait for the connection to establish
            self.get_logger().info(f"✅ Arduino connected successfully on {port} at {baud} baud.")
        except serial.SerialException as e:
            self.get_logger().error(f"❌ Failed to connect to Arduino on {port}: {e}")
            self.arduino_serial = None

    def send_to_arduino(self, command):
        """Sends a command string to the Arduino."""
        if self.arduino_serial and self.arduino_serial.is_open:
            self.get_logger().info(f"Sending command to Arduino: '{command}'")
            self.arduino_serial.write(f"{command}\n".encode('utf-8'))
        else:
            self.get_logger().warn("Cannot send command, Arduino not connected.")

    def read_from_arduino_callback(self):
        """Periodically reads status messages from the Arduino."""
        if self.arduino_serial and self.arduino_serial.in_waiting > 0:
            try:
                line = self.arduino_serial.readline().decode('utf-8').strip()
                if line.startswith("STATE:"):
                    new_position = line.split(":")[1]
                    if new_position != self.arm_position:
                        self.arm_position = new_position
                        self.get_logger().info(f"Received arm position update: {self.arm_position}")
                        # Publish the new status
                        status_msg = RosString()
                        status_msg.data = self.arm_position
                        self.arm_status_publisher.publish(status_msg)
            except Exception as e:
                self.get_logger().warn(f"Error reading from Arduino: {e}")

    def control_cmd_callback(self, msg: RosString):
        command = msg.data.upper()
        self.get_logger().info(f"Received HMI command: '{command}'")
        if command == "START":
            if self.arm_state == "IDLE":
                self.get_logger().info("--- Starting Scan Cycle ---")
                self.detection_active = True
                self.arm_state = "SCANNING_BOTTOM"
                self.last_detection_time = self.get_clock().now()
        elif command == "STOP":
            self.get_logger().info("--- Detection STOPPED by HMI ---")
            self.detection_active = False
            self.arm_state = "IDLE"
        elif command == "RESET":
            self.seen_barcodes.clear()
            self.get_logger().info("--- Barcode history has been RESET ---")
            self.arm_state = "IDLE"
            self.detection_active = False


    def process_frames_callback(self):
        # The main logic loop is now driven by the arm's state
        if not self.detection_active:
            return

        quiet_time_sec = (self.get_clock().now() - self.last_detection_time).nanoseconds / 1e9
        scan_quiet_threshold = self.get_parameter('scan_quiet_time').value

        # --- STATE: SCANNING_BOTTOM ---
        if self.arm_state == "SCANNING_BOTTOM":
            self.scan_for_barcodes()
            if quiet_time_sec > scan_quiet_threshold:
                self.get_logger().info(f"No barcodes detected for {scan_quiet_threshold}s. Moving up.")
                self.send_to_arduino("MOVEUP")
                self.arm_state = "MOVING_UP"
        
        # --- STATE: MOVING_UP ---
        elif self.arm_state == "MOVING_UP":
            if self.arm_position == "POS_ATAS":
                self.get_logger().info("Arm has reached the top. Resuming scan.")
                self.arm_state = "SCANNING_TOP"
                self.last_detection_time = self.get_clock().now() # Reset timer for top scan

        # --- STATE: SCANNING_TOP ---
        elif self.arm_state == "SCANNING_TOP":
            self.scan_for_barcodes()
            if quiet_time_sec > scan_quiet_threshold:
                self.get_logger().info(f"No barcodes detected for {scan_quiet_threshold}s. Moving down.")
                self.send_to_arduino("MOVEDOWN")
                self.arm_state = "MOVING_DOWN"
        
        # --- STATE: MOVING_DOWN ---
        elif self.arm_state == "MOVING_DOWN":
            if self.arm_position == "POS_BAWAH":
                self.get_logger().info("Arm has returned to bottom. Scan cycle complete.")
                self.arm_state = "IDLE"
                self.detection_active = False # End the cycle

    def scan_for_barcodes(self):
        """The actual barcode detection logic, called by the state machine."""
        for name, cap in self.caps.items():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret: continue

            self.camera_fps_trackers[name][0] += 1
            processed_frame = self.preprocess_frame(frame)
            barcodes = pyzbar.decode(processed_frame)

            if barcodes:
                for barcode in barcodes:
                    data = barcode.data.decode("utf-8")
                    if (data, barcode.type) not in self.seen_barcodes:
                        self.seen_barcodes.add((data, barcode.type))
                        duration = time.time() - start_time
                        
                        tracker = self.camera_fps_trackers[name]
                        elapsed_time = time.time() - tracker[1]
                        current_fps = tracker[0] / elapsed_time if elapsed_time > 0 else 0.0
                        
                        self.publish_detection(data, duration, name, current_fps)

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = self.get_parameter('brightness_adjustment').value
        contrast = self.get_parameter('contrast_adjustment').value
        alpha = 1.0 + (contrast / 127.0)
        beta = float(brightness)
        return cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    def publish_detection(self, barcode_data, duration, camera_name, fps):
        self.get_logger().info(f"[NEW DETECTION] Found '{barcode_data}' on {camera_name}")
        # THIS IS THE KEY: Update the last detection time whenever a new barcode is found
        self.last_detection_time = self.get_clock().now()
        
        detection_msg = BarcodeDetection()
        detection_msg.barcode_data = barcode_data
        detection_msg.timestamp = self.get_clock().now().to_msg()
        detection_msg.detection_duration = float(duration)
        detection_msg.camera_name = camera_name
        detection_msg.fps_at_detection = float(fps)
        self.detection_publisher.publish(detection_msg)

    def publish_status_callback(self):
        # This function remains largely unchanged
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

    def cleanup(self):
        self.get_logger().info("Shutting down live backend node.")
        for name, cap in self.caps.items():
            cap.release()
            self.get_logger().info(f"{name} released.")
        if self.arduino_serial and self.arduino_serial.is_open:
            self.arduino_serial.close()
            self.get_logger().info("Arduino serial port closed.")

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
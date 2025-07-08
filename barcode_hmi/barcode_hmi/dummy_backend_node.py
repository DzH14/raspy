import rclpy
from rclpy.node import Node
import time
import random

# Import standard and custom messages
from std_msgs.msg import String as RosString
from barcode_interfaces.msg import BarcodeDetection, CameraStatusArray, SingleCameraStatus

class DummyBackendNode(Node):
    def __init__(self):
        super().__init__('dummy_backend_node')
        self.get_logger().info('Dummy Backend Node started.')

        # State variables
        self.detection_active = False
        self.camera_names = ["Camera 1", "Camera 2", "Camera 3"]
        self.camera_fps = {name: 0.0 for name in self.camera_names}

        # Subscribers
        self.control_subscriber = self.create_subscription(
            RosString,
            'hmi/control_cmd',
            self.control_cmd_callback,
            10)

        # Publishers
        self.detection_publisher = self.create_publisher(BarcodeDetection, 'barcode/detection', 10)
        self.status_publisher = self.create_publisher(CameraStatusArray, 'camera/status', 10)

        # Timers to periodically publish data
        self.status_timer = self.create_timer(1.0, self.publish_status) # Publish status every 1 second
        self.detection_timer = self.create_timer(0.5, self.publish_detection) # Attempt detection every 0.5 seconds

    def control_cmd_callback(self, msg: RosString):
        """Listens for commands from the HMI."""
        command = msg.data.upper()
        self.get_logger().info(f"Received command: '{command}'")
        if command == "START":
            self.detection_active = True
        elif command == "STOP":
            self.detection_active = False
        elif command == "RESET":
            # In a real node, you would reset your internal counters here
            self.get_logger().info("Reset command received. (No action in dummy node)")
        # No SAVE command needed here, as HMI saves its own log now.

    def publish_status(self):
        """Publishes the current status of all cameras."""
        status_msg = CameraStatusArray()
        status_msg.header.stamp = self.get_clock().now().to_msg()
        
        for cam_name in self.camera_names:
            single_status = SingleCameraStatus()
            single_status.camera_name = cam_name

            if not self.detection_active:
                single_status.status = "Disconnected"
                single_status.fps = 0.0
            else:
                single_status.status = "Scanning" # Default to scanning if active
                # Simulate fluctuating FPS
                self.camera_fps[cam_name] = random.uniform(25.0, 35.0)
                single_status.fps = self.camera_fps[cam_name]
            
            status_msg.cameras.append(single_status)

        self.status_publisher.publish(status_msg)

    def publish_detection(self):
        """Simulates a barcode detection event and publishes it."""
        if not self.detection_active:
            return

        # Simulate a 70% chance of detection
        if random.random() < 0.7:
            detection_msg = BarcodeDetection()
            
            # Choose a random camera for the detection
            detected_camera = random.choice(self.camera_names)

            # Populate the message
            detection_msg.barcode_data = f"PDB-{random.randint(1000000, 9999999)}"
            detection_msg.timestamp = self.get_clock().now().to_msg()
            detection_msg.detection_duration = random.uniform(0.1, 2.5)
            detection_msg.camera_name = detected_camera
            detection_msg.fps_at_detection = self.camera_fps[detected_camera]

            self.detection_publisher.publish(detection_msg)
            self.get_logger().info(f"Published detection: {detection_msg.barcode_data}")

            # Also update and send a new status immediately to show "Active" state
            self.update_and_send_active_status(detected_camera)

    def update_and_send_active_status(self, active_camera_name):
        """Sends a one-off status update to mark a camera as 'Active'."""
        status_msg = CameraStatusArray()
        status_msg.header.stamp = self.get_clock().now().to_msg()
        
        for cam_name in self.camera_names:
            single_status = SingleCameraStatus()
            single_status.camera_name = cam_name
            single_status.fps = self.camera_fps[cam_name]
            if not self.detection_active:
                 single_status.status = "Disconnected"
            elif cam_name == active_camera_name:
                single_status.status = "Active"
            else:
                single_status.status = "Connected"
            
            status_msg.cameras.append(single_status)

        self.status_publisher.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    dummy_backend_node = DummyBackendNode()
    rclpy.spin(dummy_backend_node)
    dummy_backend_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import datetime
import csv
from PIL import Image, ImageTk

# ROS 2 Imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import String as RosString # Renamed to avoid conflict with Python's string
from builtin_interfaces.msg import Time as RosTime

# Import your custom messages
# Make sure you have built your workspace with 'colcon build'
from barcode_interfaces.msg import BarcodeDetection, CameraStatusArray, SingleCameraStatus

class BarcodeHMI(Node):
    def __init__(self, master):
        # Initialize ROS 2 Node first
        super().__init__('barcode_hmi_node')

        # Then initialize the Tkinter UI
        self.master = master
        master.title("Barcode Detection HMI Dashboard (ROS 2 Connected)")
        master.geometry("1200x800")
        master.resizable(True, True)

        # --- Define Colors (Same as before) ---
        self.bg_color = "#E0F2F7"
        self.frame_bg_color = "#FFFFFF"
        self.primary_blue = "#2196F3"
        self.dark_blue = "#1976D2"
        self.text_color = "#333333"
        self.secondary_text_color = "#616161"
        self.success_green = "#4CAF50"
        self.danger_red = "#F44336"
        self.warning_orange = "#FF9800"
        self.accent_blue = "#4682B4"
        
        # --- UI Setup (Same as before) ---
        # This part is largely unchanged as it just defines the visual layout
        master.configure(bg=self.bg_color)
        self._configure_styles()

        # --- Variables to hold data and states ---
        # These are now updated by ROS callbacks, not simulation
        self.camera_statuses = {
            "Camera 1": {"status": "Disconnected", "indicator_widget": None, "scan_count": tk.IntVar(value=0), "fps_value": tk.StringVar(value="0.0")},
            "Camera 2": {"status": "Disconnected", "indicator_widget": None, "scan_count": tk.IntVar(value=0), "fps_value": tk.StringVar(value="0.0")},
            "Camera 3": {"status": "Disconnected", "indicator_widget": None, "scan_count": tk.IntVar(value=0), "fps_value": tk.StringVar(value="0.0")}
        }
        self.total_barcodes_detected = tk.IntVar(value=0)
        self.last_detected_barcode = tk.StringVar(value="N/A")
        self.last_detection_timestamp = tk.StringVar(value="N/A")
        self.last_detection_duration = tk.StringVar(value="N/A")
        self.active_detection_camera = tk.StringVar(value="None")
        self.system_status_msg = tk.StringVar(value="System Disconnected. Waiting for backend node...")

        self.barcode_log_data = [] # Local log for saving

        # --- Build the UI ---
        self._create_widgets()
        self._update_camera_status_indicators()

        # --- ROS 2 Publishers and Subscribers ---
        self.get_logger().info('HMI Node is setting up ROS 2 interfaces...')
        
        # Publisher for sending commands to the backend
        self.control_publisher = self.create_publisher(RosString, 'hmi/control_cmd', 10)

        # Subscriber to get barcode detection data
        self.detection_subscriber = self.create_subscription(
            BarcodeDetection,
            'barcode/detection',
            self._barcode_detected_callback,
            10)

        # Subscriber to get camera status updates
        self.status_subscriber = self.create_subscription(
            CameraStatusArray,
            'camera/status',
            self._camera_status_callback,
            10)
        
        self.get_logger().info('HMI Node is ready and waiting for data.')

    def _configure_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TLabel", background=self.frame_bg_color, foreground=self.text_color, font=("Arial", 10))
        self.style.configure("TFrame", background=self.frame_bg_color, relief="flat")
        self.style.configure("TLabelframe", background=self.frame_bg_color, foreground=self.dark_blue, font=("Arial", 12, "bold"))
        self.style.configure("TLabelframe.Label", background=self.frame_bg_color, foreground=self.dark_blue)
        self.style.configure("TButton", background=self.primary_blue, foreground="white", font=("Arial", 10, "bold"), padding=[15, 8], relief="flat")
        self.style.map("TButton", background=[('active', self.dark_blue), ('pressed', self.dark_blue)], foreground=[('active', 'white'), ('pressed', 'white')])
        self.style.configure("Treeview.Heading", font=("Arial", 11, "bold"), background=self.primary_blue, foreground="white", relief="flat")
        self.style.configure("Treeview", background="white", foreground=self.text_color, rowheight=28, fieldbackground="white")
        self.style.map("Treeview", background=[('selected', self.accent_blue)])
    
    # --- ROS 2 Callback Functions ---
    def _barcode_detected_callback(self, msg: BarcodeDetection):
        """Callback function for when a new barcode is detected by the backend."""
        self.get_logger().info(f'Received barcode: {msg.barcode_data}')
        
        # Convert ROS timestamp to human-readable format
        ts = msg.timestamp.sec + msg.timestamp.nanosec / 1e9
        detection_timestamp = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        
        # Update GUI variables
        self.last_detected_barcode.set(msg.barcode_data)
        self.last_detection_timestamp.set(detection_timestamp)
        self.last_detection_duration.set(f"{msg.detection_duration:.2f}")
        self.active_detection_camera.set(msg.camera_name)
        self.system_status_msg.set(f"SUCCESS: Barcode '{msg.barcode_data}' detected by {msg.camera_name}!")
        self.total_barcodes_detected.set(self.total_barcodes_detected.get() + 1)
        
        if msg.camera_name in self.camera_statuses:
            cam_data = self.camera_statuses[msg.camera_name]
            cam_data["scan_count"].set(cam_data["scan_count"].get() + 1)

        # Visual feedback for detection
        self.last_barcode_display_label.config(background=self.success_green, foreground="white")
        self.master.after(200, lambda: self.last_barcode_display_label.config(background=self.frame_bg_color, foreground=self.primary_blue))

        # Log to table and internal list for CSV saving
        log_entry = {
            "Barcode Series": msg.barcode_data,
            "Timestamp": detection_timestamp,
            "Detection Duration (s)": f"{msg.detection_duration:.2f}",
            "Camera Source": msg.camera_name,
            "FPS at Detection": f"{msg.fps_at_detection:.1f}"
        }
        self.barcode_log_data.insert(0, log_entry)
        self.barcode_table.insert("", "0", values=list(log_entry.values()))

    def _camera_status_callback(self, msg: CameraStatusArray):
        """Callback for updating the status of all cameras."""
        any_active = False
        for cam_status in msg.cameras:
            if cam_status.camera_name in self.camera_statuses:
                cam_data = self.camera_statuses[cam_status.camera_name]
                cam_data["status"] = cam_status.status
                cam_data["fps_value"].set(f"{cam_status.fps:.1f}")
                if cam_status.status == "Active":
                    any_active = True
        
        if not any_active:
             self.active_detection_camera.set("None")

        self._update_camera_status_indicators()

    # --- UI Control Functions (Publish ROS messages) ---
    def _send_control_command(self, command: str):
        """Helper function to publish a command string."""
        msg = RosString()
        msg.data = command
        self.control_publisher.publish(msg)
        self.get_logger().info(f"Published control command: '{command}'")
        self.system_status_msg.set(f"Sent '{command}' command to backend node.")

    def _start_detection(self):
        self._send_control_command("START")

    def _stop_detection(self):
        self._send_control_command("STOP")

    def _reset_data(self):
        if messagebox.askyesno("Confirm Reset", "This will clear the HMI display and send a RESET command to the backend node. Continue?"):
            # Send command to backend
            self._send_control_command("RESET")
            
            # Immediately clear local UI for responsiveness
            self.barcode_log_data.clear()
            for item in self.barcode_table.get_children():
                self.barcode_table.delete(item)

            self.total_barcodes_detected.set(0)
            for cam_name in self.camera_statuses.keys():
                self.camera_statuses[cam_name]["scan_count"].set(0)
            
            self.last_detected_barcode.set("N/A")
            self.last_detection_timestamp.set("N/A")
            self.last_detection_duration.set("N/A")
            self.active_detection_camera.set("None")
            self.system_status_msg.set("Sent 'RESET' command and cleared local data.")
            self.last_barcode_display_label.config(background=self.frame_bg_color)
            self.get_logger().info("HMI data has been reset.")

    def _save_log_to_csv(self):
        """The HMI now saves its own collected log."""
        if not self.barcode_log_data:
            messagebox.showinfo("Info", "No data to save. The log is empty.")
            return

        current_time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"hmi_barcode_log_{current_time_str}.csv"
        try:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = self.barcode_log_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.barcode_log_data)
            messagebox.showinfo("Save Successful", f"Data saved to {filename}")
            self.system_status_msg.set(f"Log saved successfully to {filename}.")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save data: {e}")
            self.system_status_msg.set(f"Error saving log: {e}")

    def _show_help(self):
        help_text = (
            "Welcome to the ROS 2 Barcode Detection Dashboard!\n\n"
            "This HMI communicates with a separate ROS 2 backend node.\n\n"
            "**Buttons:**\n"
            "• Start/Stop Detection: Sends a command to the backend node.\n"
            "• Reset Data: Clears the display and tells the backend to reset.\n"
            "• Save Log (CSV): Saves the data currently visible in the history table.\n\n"
            "**Camera Status Meanings:**\n"
            "  • Red: Disconnected (Backend reports not connected)\n"
            "  • Orange: Connected/Scanning (Idle)\n"
            "  • Green: Active (This camera just detected a barcode)\n\n"
            "Ensure the backend ROS 2 node is running to see data."
        )
        messagebox.showinfo("Help - ROS 2 HMI", help_text)

    # --- Unchanged UI building and updating methods ---
    def _create_widgets(self):
        # Master Grid Configuration
        self.master.grid_rowconfigure(0, weight=0); self.master.grid_rowconfigure(1, weight=0)
        self.master.grid_rowconfigure(2, weight=1); self.master.grid_columnconfigure(0, weight=1)

        header_frame = ttk.Frame(self.master, padding="15", style="TFrame", relief="solid", borderwidth=1)
        header_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=10)
        header_frame.grid_columnconfigure(1, weight=1)

        try:
            img = Image.open("polman_logo.png").resize((int(50 * 2.8), 50), Image.LANCZOS)
            self.polman_logo = ImageTk.PhotoImage(img)
            logo_label = ttk.Label(header_frame, image=self.polman_logo, background=self.frame_bg_color)
            logo_label.grid(row=0, column=0, padx=(0, 15), pady=5, sticky="w")
        except Exception as e:
            self.get_logger().error(f"Could not load logo: {e}")
            ttk.Label(header_frame, text="[Logo]", background=self.frame_bg_color).grid(row=0, column=0, padx=(0, 15))

        ttk.Label(header_frame, text="Polearm Barcode Detection", font=("Arial", 15, "bold"), foreground=self.primary_blue, background=self.frame_bg_color).grid(row=0, column=1, padx=10, sticky="w")
        cam_status_block_frame = ttk.Frame(header_frame, style="TFrame"); cam_status_block_frame.grid(row=0, column=2, padx=10, sticky="e")
        ttk.Label(cam_status_block_frame, text="Camera Status:", font=("Arial", 11, "bold")).pack(side="left", padx=(0, 10))
        for cam_name in self.camera_statuses.keys():
            cam_frame = ttk.Frame(cam_status_block_frame, style="TFrame"); cam_frame.pack(side="left", padx=10)
            indicator_canvas = tk.Canvas(cam_frame, width=18, height=18, bg=self.frame_bg_color, highlightthickness=0); indicator_canvas.pack(side="left", padx=(0, 5))
            indicator_oval = indicator_canvas.create_oval(3, 3, 15, 15, fill=self.danger_red, outline=self.text_color, width=1)
            self.camera_statuses[cam_name].update({"indicator_widget": indicator_oval, "canvas_widget": indicator_canvas})
            ttk.Label(cam_frame, text=cam_name, font=("Arial", 10, "bold")).pack(side="left")

        active_detector_frame = ttk.Frame(header_frame, style="TFrame"); active_detector_frame.grid(row=0, column=3, padx=10, sticky="e")
        ttk.Label(active_detector_frame, text="Active Detector: ", font=("Arial", 11, "bold")).pack(side="left")
        ttk.Label(active_detector_frame, textvariable=self.active_detection_camera, font=("Arial", 11, "bold"), foreground=self.primary_blue).pack(side="left")

        main_content_frame = ttk.Frame(self.master, padding="15", style="TFrame", relief="solid", borderwidth=1)
        main_content_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=10); main_content_frame.grid_columnconfigure(0, weight=1); main_content_frame.grid_columnconfigure(1, weight=2)

        metrics_frame = ttk.LabelFrame(main_content_frame, text="Overall Metrics", padding="15"); metrics_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew"); metrics_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(metrics_frame, text="Total Barcodes:", font=("Arial", 11, "bold")).grid(row=0, column=0, sticky="w", pady=5)
        ttk.Label(metrics_frame, textvariable=self.total_barcodes_detected, font=("Arial", 18, "bold"), foreground=self.dark_blue).grid(row=0, column=1, sticky="w", pady=5)
        ttk.Label(metrics_frame, text="Barcodes / Camera:", font=("Arial", 11, "bold")).grid(row=1, column=0, sticky="w", pady=5, columnspan=2)
        for i, cam_name in enumerate(self.camera_statuses.keys()):
            ttk.Label(metrics_frame, text=f"{cam_name}:", foreground=self.secondary_text_color).grid(row=2 + i, column=0, sticky="w", padx=10)
            ttk.Label(metrics_frame, textvariable=self.camera_statuses[cam_name]["scan_count"], font=("Arial", 10, "bold")).grid(row=2 + i, column=1, sticky="w")
        
        fps_start_row = 2 + len(self.camera_statuses)
        ttk.Label(metrics_frame, text="Current FPS:", font=("Arial", 11, "bold")).grid(row=fps_start_row, column=0, sticky="w", pady=(10,5), columnspan=2)
        for i, cam_name in enumerate(self.camera_statuses.keys()):
            ttk.Label(metrics_frame, text=f"{cam_name}:", foreground=self.secondary_text_color).grid(row=fps_start_row + 1 + i, column=0, sticky="w", padx=10)
            ttk.Label(metrics_frame, textvariable=self.camera_statuses[cam_name]["fps_value"], font=("Arial", 10, "bold"), foreground=self.dark_blue).grid(row=fps_start_row + 1 + i, column=1, sticky="w")

        barcode_controls_frame = ttk.Frame(main_content_frame, style="TFrame"); barcode_controls_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew"); barcode_controls_frame.grid_columnconfigure(0, weight=1)
        self.last_barcode_display_frame = ttk.LabelFrame(barcode_controls_frame, text="Last Detected Barcode", padding="15"); self.last_barcode_display_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10)); self.last_barcode_display_frame.grid_columnconfigure(0, weight=1)
        self.last_barcode_display_label = ttk.Label(self.last_barcode_display_frame, textvariable=self.last_detected_barcode, font=("Arial", 40, "bold"), foreground=self.primary_blue, background=self.frame_bg_color); self.last_barcode_display_label.grid(row=0, column=0, pady=20, sticky="nsew")

        details_frame = ttk.Frame(barcode_controls_frame, style="TFrame", padding="10"); details_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10)); details_frame.grid_columnconfigure(1, weight=1); details_frame.grid_columnconfigure(3, weight=1)
        ttk.Label(details_frame, text="Timestamp:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", padx=5); ttk.Label(details_frame, textvariable=self.last_detection_timestamp).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(details_frame, text="Duration (s):", font=("Arial", 10, "bold")).grid(row=0, column=2, sticky="w", padx=20); ttk.Label(details_frame, textvariable=self.last_detection_duration).grid(row=0, column=3, sticky="w", padx=5)

        button_frame = ttk.Frame(barcode_controls_frame, style="TFrame", padding="10"); button_frame.grid(row=2, column=0, sticky="ew")
        for i in range(5): button_frame.grid_columnconfigure(i, weight=1)
        ttk.Button(button_frame, text="Start Detection", command=self._start_detection).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(button_frame, text="Stop Detection", command=self._stop_detection).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(button_frame, text="Reset Data", command=self._reset_data).grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        ttk.Button(button_frame, text="Save Log (CSV)", command=self._save_log_to_csv).grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        ttk.Button(button_frame, text="Help", command=self._show_help).grid(row=0, column=4, padx=5, pady=5, sticky="ew")

        table_frame = ttk.LabelFrame(self.master, text="Barcode Detection History", padding="15"); table_frame.grid(row=2, column=0, sticky="nsew", padx=15, pady=10); table_frame.grid_rowconfigure(0, weight=1); table_frame.grid_columnconfigure(0, weight=1)
        columns = ("barcode_series", "timestamp", "detection_duration", "camera_source", "fps_at_detection")
        self.barcode_table = ttk.Treeview(table_frame, columns=columns, show="headings"); self.barcode_table.heading("barcode_series", text="Barcode Series"); self.barcode_table.heading("timestamp", text="Timestamp"); self.barcode_table.heading("detection_duration", text="Duration (s)"); self.barcode_table.heading("camera_source", text="Camera Source"); self.barcode_table.heading("fps_at_detection", text="FPS at Detection")
        self.barcode_table.column("barcode_series", width=200, stretch=tk.YES); self.barcode_table.column("timestamp", width=180, stretch=tk.YES); self.barcode_table.column("detection_duration", width=100, stretch=tk.YES); self.barcode_table.column("camera_source", width=100, stretch=tk.YES); self.barcode_table.column("fps_at_detection", width=100, stretch=tk.YES)
        self.barcode_table.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.barcode_table.yview); self.barcode_table.configure(yscrollcommand=scrollbar.set); scrollbar.grid(row=0, column=1, sticky="ns")
        status_bar = ttk.Label(self.master, textvariable=self.system_status_msg, relief="groove", anchor="w", font=("Arial", 10), background="white", foreground=self.text_color, padding=[10, 5]); status_bar.grid(row=3, column=0, sticky="ew", padx=15, pady=(0, 15))


    def _update_camera_status_indicators(self):
        for cam_name, cam_data in self.camera_statuses.items():
            color = self.danger_red
            if cam_data["status"] == "Active": color = self.success_green
            elif cam_data["status"] in ["Connected", "Scanning"]: color = self.warning_orange
            cam_data["canvas_widget"].itemconfig(cam_data["indicator_widget"], fill=color)

# --- Main Application Loop ---
def main(args=None):
    rclpy.init(args=args)
    
    # Create the main Tkinter window
    root = tk.Tk()
    
    # Create the HMI class instance, which is also a ROS 2 Node
    app = BarcodeHMI(root)
    
    # This is the crucial part that integrates Tkinter's event loop with ROS 2's
    def ros_update_loop():
        rclpy.spin_once(app, timeout_sec=0.01) # Process ROS 2 messages
        root.after(20, ros_update_loop) # Reschedule this function to run every 20ms

    # Start the integrated loop and the Tkinter main loop
    ros_update_loop()
    root.mainloop()
    
    # Cleanup
    app.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
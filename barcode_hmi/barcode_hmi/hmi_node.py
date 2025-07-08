import tkinter as tk
from tkinter import ttk, messagebox
import datetime
import csv
from PIL import Image, ImageTk
from collections import deque
import numpy as np

# Import libraries for graphing
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ROS 2 Imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import String as RosString
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
        master.geometry("1200x850")
        master.resizable(True, True)

        # --- Colors and Styles ---
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

        master.configure(bg=self.bg_color)
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TLabel", background=self.frame_bg_color, foreground=self.text_color, font=("Arial", 10))
        self.style.configure("TFrame", background=self.frame_bg_color, relief="flat")
        self.style.configure("TLabelframe", background=self.frame_bg_color, foreground=self.dark_blue, font=("Arial", 12, "bold"))
        self.style.configure("TLabelframe.Label", background=self.frame_bg_color, foreground=self.dark_blue)
        self.style.configure("TButton", background=self.primary_blue, foreground="white", font=("Arial", 10, "bold"), padding=[15, 8], relief="flat")
        self.style.map("TButton", background=[('active', self.dark_blue), ('pressed', self.dark_blue)])
        self.style.configure("Treeview.Heading", font=("Arial", 11, "bold"), background=self.primary_blue, foreground="white", relief="flat")
        self.style.configure("Treeview", background="white", foreground=self.text_color, rowheight=28, fieldbackground="white")
        self.style.map("Treeview", background=[('selected', self.accent_blue)])
        self.style.configure("TNotebook.Tab", font=("Arial", 10, "bold"), padding=[15, 8])
        self.style.map("TNotebook.Tab", background=[("selected", self.frame_bg_color)], foreground=[("selected", self.primary_blue)])


        # --- Data Variables ---
        self.camera_statuses = {
            "Camera 1": {"status": "Disconnected", "indicator_widget": None, "scan_count": tk.IntVar(value=0), "fps_value": tk.StringVar(value="0.0"), "fps_history": deque(maxlen=50)},
            "Camera 2": {"status": "Disconnected", "indicator_widget": None, "scan_count": tk.IntVar(value=0), "fps_value": tk.StringVar(value="0.0"), "fps_history": deque(maxlen=50)},
            "Camera 3": {"status": "Disconnected", "indicator_widget": None, "scan_count": tk.IntVar(value=0), "fps_value": tk.StringVar(value="0.0"), "fps_history": deque(maxlen=50)}
        }
        self.total_barcodes_detected = tk.IntVar(value=0)
        self.last_detected_barcode = tk.StringVar(value="N/A")
        self.last_detection_timestamp = tk.StringVar(value="N/A")
        self.last_detection_duration = tk.StringVar(value="N/A")
        self.active_detection_camera = tk.StringVar(value="None")
        self.system_status_msg = tk.StringVar(value="System Disconnected. Waiting for backend node...")
        self.barcode_log_data = []
        self.graph_widgets = {}
        self.avg_fps_history = deque(maxlen=50)

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
        # Start a periodic update for the graphs
        self.master.after(500, self._update_graphs)
        
    def _create_widgets(self):
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        notebook = ttk.Notebook(self.master)
        notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        dashboard_tab = ttk.Frame(notebook, style="TFrame")
        graphs_tab = ttk.Frame(notebook, style="TFrame")
        
        notebook.add(dashboard_tab, text="Dashboard")
        notebook.add(graphs_tab, text="Performance Graphs")

        self._create_dashboard_tab(dashboard_tab)
        self._create_graphs_tab(graphs_tab)
        
    def _create_dashboard_tab(self, parent_tab):
        parent_tab.grid_rowconfigure(2, weight=1)
        parent_tab.grid_columnconfigure(0, weight=1)

        header_frame = ttk.Frame(parent_tab, padding="15", style="TFrame", relief="solid", borderwidth=1)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        header_frame.grid_columnconfigure(1, weight=1)
        
        try:
            img = Image.open("polman_logo.png").resize((140, 50), Image.LANCZOS)
            self.polman_logo = ImageTk.PhotoImage(img)
            ttk.Label(header_frame, image=self.polman_logo, background=self.frame_bg_color).grid(row=0, column=0, padx=(0, 15), pady=5, sticky="w")
        except Exception as e:
            self.get_logger().warn(f"Could not load logo: {e}")

        ttk.Label(header_frame, text="Polearm Barcode Detection", font=("Arial", 15, "bold"), foreground=self.primary_blue, background=self.frame_bg_color).grid(row=0, column=1, sticky="w")
        cam_status_block_frame = ttk.Frame(header_frame, style="TFrame"); cam_status_block_frame.grid(row=0, column=2, padx=20, sticky="e")
        ttk.Label(cam_status_block_frame, text="Camera Status:", font=("Arial", 11, "bold")).pack(side="left")
        for cam_name in self.camera_statuses.keys():
            cam_frame = ttk.Frame(cam_status_block_frame, style="TFrame"); cam_frame.pack(side="left", padx=10)
            indicator_canvas = tk.Canvas(cam_frame, width=18, height=18, bg=self.frame_bg_color, highlightthickness=0); indicator_canvas.pack(side="left")
            indicator_oval = indicator_canvas.create_oval(3, 3, 15, 15, fill=self.danger_red, outline=self.text_color);
            self.camera_statuses[cam_name].update({"indicator_widget": indicator_oval, "canvas_widget": indicator_canvas})
            ttk.Label(cam_frame, text=cam_name, font=("Arial", 10, "bold")).pack(side="left")
        active_detector_frame = ttk.Frame(header_frame, style="TFrame"); active_detector_frame.grid(row=0, column=3, sticky="e")
        ttk.Label(active_detector_frame, text="Active Detector: ", font=("Arial", 11, "bold")).pack(side="left")
        ttk.Label(active_detector_frame, textvariable=self.active_detection_camera, font=("Arial", 11, "bold"), foreground=self.primary_blue).pack(side="left")

        main_content_frame = ttk.Frame(parent_tab, style="TFrame")
        main_content_frame.grid(row=1, column=0, sticky="ew", padx=10)
        main_content_frame.grid_columnconfigure(0, weight=1); main_content_frame.grid_columnconfigure(1, weight=2)
        
        metrics_frame = ttk.LabelFrame(main_content_frame, text="Overall Metrics", padding="15"); metrics_frame.grid(row=0, column=0, pady=10, sticky="nsew")
        ttk.Label(metrics_frame, text="Total Barcodes:", font=("Arial", 11, "bold")).grid(row=0, column=0, sticky="w", pady=5)
        ttk.Label(metrics_frame, textvariable=self.total_barcodes_detected, font=("Arial", 18, "bold"), foreground=self.dark_blue).grid(row=0, column=1, sticky="w")
        ttk.Label(metrics_frame, text="Barcodes / Camera:", font=("Arial", 11, "bold")).grid(row=1, column=0, sticky="w", columnspan=2)
        for i, cam_name in enumerate(self.camera_statuses.keys()):
            ttk.Label(metrics_frame, text=f"{cam_name}:", foreground=self.secondary_text_color).grid(row=2 + i, column=0, sticky="w", padx=10)
            ttk.Label(metrics_frame, textvariable=self.camera_statuses[cam_name]["scan_count"], font=("Arial", 10, "bold")).grid(row=2 + i, column=1, sticky="w")
        fps_start_row = 2 + len(self.camera_statuses)
        ttk.Label(metrics_frame, text="Current FPS:", font=("Arial", 11, "bold")).grid(row=fps_start_row, column=0, sticky="w", columnspan=2, pady=(10,5))
        for i, cam_name in enumerate(self.camera_statuses.keys()):
            ttk.Label(metrics_frame, text=f"{cam_name}:", foreground=self.secondary_text_color).grid(row=fps_start_row + 1 + i, column=0, sticky="w", padx=10)
            ttk.Label(metrics_frame, textvariable=self.camera_statuses[cam_name]["fps_value"], font=("Arial", 10, "bold"), foreground=self.dark_blue).grid(row=fps_start_row + 1 + i, column=1, sticky="w")

        barcode_controls_frame = ttk.Frame(main_content_frame, style="TFrame"); barcode_controls_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        barcode_controls_frame.grid_columnconfigure(0, weight=1)
        self.last_barcode_display_frame = ttk.LabelFrame(barcode_controls_frame, text="Last Detected Barcode", padding="15"); self.last_barcode_display_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        self.last_barcode_display_frame.grid_columnconfigure(0, weight=1)
        self.last_barcode_display_label = ttk.Label(self.last_barcode_display_frame, textvariable=self.last_detected_barcode, font=("Arial", 40, "bold"), foreground=self.primary_blue, background=self.frame_bg_color); self.last_barcode_display_label.grid(row=0, column=0, pady=20, sticky="nsew")
        details_frame = ttk.Frame(barcode_controls_frame, style="TFrame", padding="10"); details_frame.grid(row=1, column=0, sticky="ew")
        ttk.Label(details_frame, text="Timestamp:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", padx=5); ttk.Label(details_frame, textvariable=self.last_detection_timestamp).grid(row=0, column=1, sticky="w")
        ttk.Label(details_frame, text="Duration (s):", font=("Arial", 10, "bold")).grid(row=0, column=2, sticky="w", padx=20); ttk.Label(details_frame, textvariable=self.last_detection_duration).grid(row=0, column=3, sticky="w")
        button_frame = ttk.Frame(barcode_controls_frame, style="TFrame", padding="10"); button_frame.grid(row=2, column=0, sticky="ew")
        for i in range(5): button_frame.grid_columnconfigure(i, weight=1) 
        ttk.Button(button_frame, text="Start Detection", command=self._start_detection).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(button_frame, text="Stop Detection", command=self._stop_detection).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(button_frame, text="Reset Data", command=self._reset_data).grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        ttk.Button(button_frame, text="Save Log (CSV)", command=self._save_log_to_csv).grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        ttk.Button(button_frame, text="Help", command=self._show_help).grid(row=0, column=4, padx=5, pady=5, sticky="ew")

        table_frame = ttk.LabelFrame(parent_tab, text="Barcode Detection History", padding="15")
        table_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        table_frame.grid_rowconfigure(0, weight=1); table_frame.grid_columnconfigure(0, weight=1)
        columns = ("barcode_series", "timestamp", "detection_duration", "camera_source", "fps_at_detection")
        self.barcode_table = ttk.Treeview(table_frame, columns=columns, show="headings"); self.barcode_table.grid(row=0, column=0, sticky="nsew")
        self.barcode_table.heading("barcode_series", text="Barcode Series"); self.barcode_table.heading("timestamp", text="Timestamp"); self.barcode_table.heading("detection_duration", text="Duration (s)"); self.barcode_table.heading("camera_source", text="Camera Source"); self.barcode_table.heading("fps_at_detection", text="FPS at Detection")
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.barcode_table.yview); self.barcode_table.configure(yscrollcommand=scrollbar.set); scrollbar.grid(row=0, column=1, sticky="ns")

        status_bar = ttk.Label(parent_tab, textvariable=self.system_status_msg, relief="groove", anchor="w", font=("Arial", 10), background="white", foreground=self.text_color, padding=[10, 5])
        status_bar.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))

    def _create_graphs_tab(self, parent_tab):
        parent_tab.configure(style="TFrame")
        parent_tab.grid_columnconfigure(0, weight=1); parent_tab.grid_columnconfigure(1, weight=1)
        parent_tab.grid_rowconfigure(0, weight=1); parent_tab.grid_rowconfigure(1, weight=1)
        
        graph_titles = list(self.camera_statuses.keys()) + ["Average FPS"]
        self.graph_widgets.clear()

        for i, title in enumerate(graph_titles):
            row, col = divmod(i, 2)
            graph_frame = ttk.LabelFrame(parent_tab, text=f"{title} FPS", style="TLabelframe")
            graph_frame.grid(row=row, column=col, sticky="nsew", padx=10, pady=10)
            graph_frame.grid_rowconfigure(0, weight=1); graph_frame.grid_columnconfigure(0, weight=1)
            
            fig = Figure(figsize=(5, 4), dpi=100, facecolor=self.frame_bg_color)
            ax = fig.add_subplot(111)
            ax.set_facecolor(self.bg_color)
            fig.tight_layout(pad=2.0)
            canvas = FigureCanvasTkAgg(fig, master=graph_frame)
            canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
            self.graph_widgets[title] = {"fig": fig, "ax": ax, "canvas": canvas}

    def _update_graphs(self):
        """Periodically updates all graphs with new data from the deques."""
        for title, widgets in self.graph_widgets.items():
            ax = widgets["ax"]; canvas = widgets["canvas"]
            ax.clear()

            if title == "Average FPS":
                data = self.avg_fps_history
            else:
                data = self.camera_statuses.get(title, {}).get("fps_history")
            
            ax.plot(data, color=self.dark_blue, linewidth=2)
            ax.set_ylabel("FPS", color=self.secondary_text_color)
            ax.set_xlabel("Time (last 50 readings)", color=self.secondary_text_color)
            ax.set_ylim(0, 40)
            ax.grid(True, linestyle='--', linewidth=0.5)
            ax.set_title(f"{title} FPS Trend", color=self.primary_blue)
            canvas.draw()
        
        # Reschedule the update
        self.master.after(500, self._update_graphs)

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
            # Mark the detecting camera as 'Active'
            cam_data["status"] = "Active"

        # Set other cameras to 'Connected'
        for cam_name, cam_data in self.camera_statuses.items():
            if cam_name != msg.camera_name and cam_data["status"] != "Disconnected":
                cam_data["status"] = "Connected"

        self._update_camera_status_indicators()

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
        """Callback for updating the status of all cameras and collecting FPS data."""
        any_active = False
        active_fps_values = []
        for cam_status in msg.cameras:
            if cam_status.camera_name in self.camera_statuses:
                cam_data = self.camera_statuses[cam_status.camera_name]
                # Only update status if it's not 'Active' from a recent detection
                if cam_data["status"] != "Active":
                    cam_data["status"] = cam_status.status
                
                # Update FPS value and history
                cam_data["fps_value"].set(f"{cam_status.fps:.1f}")
                cam_data["fps_history"].append(cam_status.fps)
                
                if cam_status.status in ["Active", "Connected", "Scanning"]:
                    any_active = True
                    active_fps_values.append(cam_status.fps)
        
        # Update average FPS history if there are active cameras
        if active_fps_values:
            self.avg_fps_history.append(np.mean(active_fps_values))
        else:
            self.avg_fps_history.append(0) # Append 0 if no cameras are active

        # Update system status message
        if any_active and "SUCCESS" not in self.system_status_msg.get():
             self.system_status_msg.set("System connected. Scanning for barcodes...")
        elif not any_active:
             self.system_status_msg.set("System Disconnected. Waiting for backend node...")

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
        for cam_data in self.camera_statuses.values():
            cam_data["fps_history"].clear()
        self.avg_fps_history.clear()


    def _stop_detection(self):
        self._send_control_command("STOP")
        self.active_detection_camera.set("None")

    def _reset_data(self):
        if messagebox.askyesno("Confirm Reset", "This will clear the HMI display and send a RESET command to the backend node. Continue?"):
            self._send_control_command("RESET")
            
            # Immediately clear local UI for responsiveness
            self.barcode_log_data.clear()
            for item in self.barcode_table.get_children():
                self.barcode_table.delete(item)

            self.total_barcodes_detected.set(0)
            for cam_name in self.camera_statuses.keys():
                self.camera_statuses[cam_name]["scan_count"].set(0)
                self.camera_statuses[cam_name]["fps_history"].clear()

            self.avg_fps_history.clear()
            self.last_detected_barcode.set("N/A")
            self.last_detection_timestamp.set("N/A")
            self.last_detection_duration.set("N/A")
            self.active_detection_camera.set("None")
            self.system_status_msg.set("Sent 'RESET' command and cleared local data.")
            self.get_logger().info("HMI data has been reset.")

    def _save_log_to_csv(self):
        if not self.barcode_log_data:
            messagebox.showinfo("Info", "No data to save. The log is empty.")
            return
        filename = f"hmi_barcode_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = self.barcode_log_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.barcode_log_data)
            messagebox.showinfo("Save Successful", f"Data saved to {filename}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save data: {e}")

    def _show_help(self):
        help_text = (
            "Welcome to the ROS 2 Barcode Detection Dashboard!\n\n"
            "This HMI communicates with a separate ROS 2 backend node.\n\n"
            "• Start/Stop/Reset: Sends a command to the backend node.\n"
            "• Save Log (CSV): Saves the data currently in the history table.\n\n"
            "The Performance Graphs tab shows live FPS data from the cameras."
        )
        messagebox.showinfo("Help - ROS 2 HMI", help_text)

    def _update_camera_status_indicators(self):
        """Updates the color of the camera status indicators."""
        for cam_data in self.camera_statuses.values():
            if not cam_data.get("canvas_widget"): continue
            color = self.danger_red # Default to Disconnected
            if cam_data["status"] == "Active":
                color = self.success_green
            elif cam_data["status"] in ["Connected", "Scanning"]:
                color = self.warning_orange
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
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32, Float64
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class FlightManager(Node):
    def __init__(self):
        super().__init__("flight_manager")

        self.current_state = State()
        self.local_alt = 0.0
        self.relative_alt = 0.0
        self.relative_alt_received = False
        self.target_takeoff_alt = 0.0

        self.declare_parameter("altitude_source", "local_position")
        self.declare_parameter("hover_tolerance", 0.2)
        self.altitude_source = str(self.get_parameter("altitude_source").value)
        self.hover_tolerance = float(self.get_parameter("hover_tolerance").value)
        if self.altitude_source not in ("local_position", "relative_alt"):
            raise ValueError(
                "altitude_source harus 'local_position' atau 'relative_alt'"
            )

        # ==========================================
        # 1. MAVROS SUBSCRIBERS (Membaca dari Pixhawk)
        # ==========================================
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.state_sub = self.create_subscription(
            State, "/mavros/state", self.state_callback, 10
        )
        self.pose_sub = self.create_subscription(
            PoseStamped, "/mavros/local_position/pose", self.pose_callback, qos_sensor
        )
        self.relative_alt_sub = self.create_subscription(
            Float64, "/mavros/global_position/rel_alt", self.relative_alt_callback,
            qos_sensor
        )

        # ==========================================
        # 2. COMMAND SUBSCRIBERS (Menerima dari State Machine)
        # ==========================================
        self.cmd_mode_sub = self.create_subscription(
            String, "/flight/cmd/set_mode", self.cmd_mode_cb, 10
        )
        self.cmd_arm_sub = self.create_subscription(
            Bool, "/flight/cmd/set_arm", self.cmd_arm_cb, 10
        )
        self.cmd_takeoff_sub = self.create_subscription(
            Float32, "/flight/cmd/takeoff", self.cmd_takeoff_cb, 10
        )
        self.cmd_land_sub = self.create_subscription(
            Bool, "/flight/cmd/land", self.cmd_land_cb, 10
        )

        # ==========================================
        # 3. TELEMETRY PUBLISHERS (Melapor ke State Machine)
        # ==========================================
        self.pub_armed = self.create_publisher(Bool, "/flight/telemetry/is_armed", 10)
        self.pub_mode = self.create_publisher(String, "/flight/telemetry/current_mode", 10)
        self.pub_alt = self.create_publisher(Float32, "/flight/telemetry/altitude", 10)
        self.pub_hover = self.create_publisher(Bool, "/flight/telemetry/is_hovering", 10)

        # ==========================================
        # 4. MAVROS SERVICE CLIENTS
        # ==========================================
        self.mode_client = self.create_client(SetMode, "/mavros/set_mode")
        self.arm_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.takeoff_client = self.create_client(CommandTOL, "/mavros/cmd/takeoff")
        self.land_client = self.create_client(CommandTOL, "/mavros/cmd/land")

        # Timer berjalan pada 10Hz (0.1s) murni untuk mempublikasikan status sensor
        self.timer = self.create_timer(0.1, self.publish_telemetry)

        self.get_logger().info(
            "Flight Manager aktif: sumber altitude=%s, toleransi hover=%.2fm"
            % (self.altitude_source, self.hover_tolerance)
        )

    # --- Callbacks Pembacaan Sensor ---
    def state_callback(self, msg):
        self.current_state = msg

    def pose_callback(self, msg):
        self.local_alt = msg.pose.position.z

    def relative_alt_callback(self, msg):
        # Nilai ini diterbitkan oleh flight controller dan bereferensi ke home.
        self.relative_alt = msg.data
        self.relative_alt_received = True

    def altitude_for_takeoff(self):
        if self.altitude_source == "relative_alt":
            return self.relative_alt if self.relative_alt_received else None
        return self.local_alt

    # --- Callbacks Perintah FSM ---
    def cmd_mode_cb(self, msg):
        mode = msg.data
        if self.current_state.mode == mode:
            return
        
        req = SetMode.Request()
        req.custom_mode = mode
        if self.mode_client.wait_for_service(timeout_sec=1.0):
            self.mode_client.call_async(req)
            self.get_logger().info(f"Eksekusi Ganti Mode: {mode}")

    def cmd_arm_cb(self, msg):
        state = msg.data
        if self.current_state.armed == state:
            return
            
        req = CommandBool.Request()
        req.value = state
        if self.arm_client.wait_for_service(timeout_sec=1.0):
            self.arm_client.call_async(req)
            self.get_logger().info(f"Eksekusi Arming: {state}")

    def cmd_takeoff_cb(self, msg):
        self.target_takeoff_alt = msg.data
        req = CommandTOL.Request()
        req.altitude = self.target_takeoff_alt
        
        if self.takeoff_client.wait_for_service(timeout_sec=1.0):
            self.takeoff_client.call_async(req)
            self.get_logger().info(f"Eksekusi Takeoff ke ketinggian {self.target_takeoff_alt}m")

    def cmd_land_cb(self, msg):
        if msg.data:
            req = CommandTOL.Request()
            if self.land_client.wait_for_service(timeout_sec=1.0):
                self.land_client.call_async(req)
                self.get_logger().info("Eksekusi Landing")

    # --- Pengiriman Umpan Balik (Telemetry) ---
    def publish_telemetry(self):
        msg_arm = Bool()
        msg_arm.data = self.current_state.armed
        self.pub_armed.publish(msg_arm)

        msg_mode = String()
        msg_mode.data = self.current_state.mode
        self.pub_mode.publish(msg_mode)

        current_alt = self.altitude_for_takeoff()
        msg_alt = Float32()
        msg_alt.data = float(current_alt) if current_alt is not None else float("nan")
        self.pub_alt.publish(msg_alt)
        
        # Sama seperti versi uji 0b8bedd: keputusan hover hanya berdasarkan
        # selisih altitude. Velocity vision ZED sengaja tidak digunakan.
        is_hovering = False
        if (self.current_state.armed
                and current_alt is not None):
            is_hovering = (
                abs(current_alt - self.target_takeoff_alt) < self.hover_tolerance
            )
        
        msg_hover = Bool()
        msg_hover.data = is_hovering
        self.pub_hover.publish(msg_hover)

def main(args=None):
    rclpy.init(args=args)
    node = FlightManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

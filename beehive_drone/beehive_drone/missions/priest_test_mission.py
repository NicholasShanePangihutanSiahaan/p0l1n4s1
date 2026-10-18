from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
import math
from copy import deepcopy
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, String, Float32
from uav_interfaces.msg import Tree
from beehive_drone.mission_params import MissionConfig
from beehive_drone.missions.base_mission import BaseMissionStrategy

if TYPE_CHECKING:
    from mission_state_machine import MissionStateMachine

class PriestTestMission(BaseMissionStrategy):
    @property
    def navigation_states(self) -> Sequence[str]:
        return (
            'POST_TAKEOFF_HOVER',
            'EXPLORE_ROW', 'ALIGN_TO_TREE', 'APPROACH_TREE', 'VERIFY_TREE', 'START_ORBIT', 
            'WAIT_ORBIT', 'POST_ORBIT_HOVER', 'ALIGN_HOME', 
            'END_OF_ROW',
            'CRAB_SCAN', 'RETURN_TO_HOME', 'HOME_HOVER', 'FINAL_SPIN', 'COMMAND_GOAL',
        )

    @property
    def timeout_exempt_states(self) -> Sequence[str]:
        return (
            'WAIT_START', 'EXPLORE_ROW', 'WAIT_ORBIT', 'LANDING', 'DONE',
            'ABORT', 'MANUAL_OVERRIDE'
        )
    def execute(self, fsm: MissionStateMachine, active: bool, elapsed: float, cx: float, cy: float):
        # --- FASE PRE-FLIGHT ---
        if fsm.state == 'WAIT_START':
            if fsm.start_requested and not fsm.frame_alignment_ready:
                fsm.get_logger().warning(
                    'WAIT_START: menunggu alignment ZED-FC terkunci.',
                    throttle_duration_sec=2.0)
            elif fsm.start_requested and not fsm.vision_ready():
                fsm.log_vision_wait()
            elif fsm.start_requested and (fsm.safety_ok or not fsm.require_safety):
                fsm.home_pose = (cx, cy, fsm.current_yaw())
                fsm.transition('INIT')
                fsm.get_logger().info('Start diterima dan sensor sehat; memulai preflight.')

        elif fsm.state == "INIT":
            mode_msg = String(); mode_msg.data = "GUIDED"
            fsm.cmd_mode_pub.publish(mode_msg)
            fsm.transition("WAIT_GUIDED")
            fsm.retry_counter = 0
            fsm.get_logger().info("Meminta transisi ke mode GUIDED...")

        elif fsm.state == "WAIT_GUIDED":
            if fsm.current_mode == "GUIDED":
                arm_msg = Bool(); arm_msg.data = True
                fsm.takeoff_yaw = fsm.current_yaw()
                fsm.cmd_arm_pub.publish(arm_msg)
                fsm.transition("WAIT_ARM")
                fsm.retry_counter = 0
                fsm.get_logger().info("Mode GUIDED aktif. Meminta Arming Motor...")
            else:
                fsm.retry_counter += 1
                if fsm.retry_counter > 20:
                    mode_msg = String(); mode_msg.data = "GUIDED"
                    fsm.cmd_mode_pub.publish(mode_msg)
                    fsm.retry_counter = 0

        elif fsm.state == "WAIT_ARM":
            if fsm.is_armed:
                if not hasattr(fsm, 'arm_delay_start'):
                    fsm.arm_delay_start = fsm.get_clock().now()
                elapsed_ns = (fsm.get_clock().now() - fsm.arm_delay_start).nanoseconds
                if elapsed_ns >= 350_000_000:
                    takeoff_msg = Float32(); takeoff_msg.data = fsm.flight_altitude
                    fsm.cmd_takeoff_pub.publish(takeoff_msg)
                    fsm.transition("WAIT_TAKEOFF")
                    fsm.retry_counter = 0
                    fsm.get_logger().info(f"Motor Bersenjata (Armed). Takeoff ke ketinggian {fsm.flight_altitude}m...")
            else:
                fsm.retry_counter += 1
                if fsm.retry_counter > 20:
                    arm_msg = Bool(); arm_msg.data = True
                    fsm.cmd_arm_pub.publish(arm_msg)
                    fsm.get_logger().info("Mencoba Arming ulang... (Menunggu Pre-arm good dari ArduPilot)")
                    fsm.retry_counter = 0

        elif fsm.state == "WAIT_TAKEOFF":
            if fsm.is_hovering:
                fsm.hold_x = cx
                fsm.hold_y = cy
                fsm.hold_yaw = fsm.takeoff_yaw
                fsm.navigation_altitude = fsm.current_pose.pose.position.z
                fsm.last_tree_x = cx
                fsm.last_tree_y = cy
                fsm.transition("POST_TAKEOFF_HOVER")
                fsm.get_logger().info(
                    "Altitude takeoff tercapai. Menahan posisi selama "
                    f"{fsm.post_takeoff_hover_time:.1f} detik.")

        elif fsm.state == "POST_TAKEOFF_HOVER":
            fsm.publish_goal(fsm.hold_x, fsm.hold_y, fsm.hold_yaw)
            if elapsed >= fsm.post_takeoff_hover_time:
                fsm.transition("COMMAND_GOAL")
                fsm.get_logger().info("Hover pasca-takeoff selesai. Menunggu Perintah COMMAND_GOAL")

        elif fsm.state == "COMMAND_GOAL":
            target_goal_x = 15.0
            target_goal_y = 0.0
            yaw_target = fsm.takeoff_yaw
            fsm.get_logger().info(f"Bergerak ke ({target_goal_x}, {target_goal_y}) dengan yaw {yaw_target}", throttle_duration_sec=1.0)

            if fsm.distance(cx, cy, target_goal_x, target_goal_y) < 0.2:
                fsm.transition("RETURN_TO_HOME")
                fsm.get_logger().info("Tiba di titik goal. Sekarang return to home")
            else:
                fsm.publish_goal(target_goal_x, target_goal_y, yaw_target)



        elif fsm.state == "RETURN_TO_HOME":
            if fsm.home_pose is None:
                return
            home_x, home_y, home_yaw = fsm.home_pose
            target_yaw = math.atan2(home_y - cy, home_x - cx)
            fsm.publish_goal(home_x, home_y, target_yaw)

            if fsm.distance(cx, cy, home_x, home_y) < MissionConfig.HOME_POSITION_TOLERANCE:
                fsm.hold_yaw = home_yaw
                fsm.hover_timer = 0
                fsm.transition("HOME_HOVER")
                fsm.get_logger().info("Tiba di titik takeoff. Hover sebelum landing.")

        elif fsm.state == "HOME_HOVER":
            home_x, home_y, _ = fsm.home_pose
            fsm.publish_goal(home_x, home_y, fsm.hold_yaw)
            fsm.hover_timer += 1
            required_ticks = int(MissionConfig.HOME_HOVER_TIME / 0.1)
            if fsm.hover_timer >= required_ticks:
                fsm.transition("LANDING")
                fsm.get_logger().info("Hover home selesai. Memulai pendaratan.")

        elif fsm.state == "FINAL_SPIN":
            qx = fsm.current_pose.pose.orientation.x
            qy = fsm.current_pose.pose.orientation.y
            qz = fsm.current_pose.pose.orientation.z
            qw = fsm.current_pose.pose.orientation.w
            current_yaw = math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
            
            delta = current_yaw - fsm.last_yaw
            if delta > math.pi: delta -= 2 * math.pi
            elif delta < -math.pi: delta += 2 * math.pi
            
            fsm.spin_accumulated += abs(delta)
            fsm.last_yaw = current_yaw
            
            fsm.target_tree = fsm.find_uninspected_tree()
            if fsm.target_tree is not None:
                fsm.verification_retries = 0
                fsm.target_tree = deepcopy(fsm.target_tree)
                fsm.frozen_target_tree = deepcopy(fsm.target_tree)
                fsm.transition("ALIGN_TO_TREE")
                fsm.get_logger().info("Pohon terlewat ditemukan saat Final Spin!")
                return
                
            if fsm.spin_accumulated >= 2 * math.pi:
                fsm.transition("LANDING")
                fsm.get_logger().info("Area bersih. Memulai Pendaratan.")
            else:
                target_yaw = current_yaw + 0.2
                fsm.publish_goal(cx, cy, target_yaw)

        elif fsm.state == "LANDING":
            now = fsm.get_clock().now()
            last_command_age = (
                float('inf') if fsm.last_landing_command_time is None else
                (now - fsm.last_landing_command_time).nanoseconds * 1e-9)
            if fsm.landing_command_due(fsm.current_mode, last_command_age, fsm.landing_retry_interval):
                land_msg = Bool(); land_msg.data = True
                fsm.cmd_land_pub.publish(land_msg)
                fsm.last_landing_command_time = now
            if not fsm.is_armed:
                fsm.transition("DONE")
                fsm.get_logger().info("Landing selesai. Misi DONE.")

        elif fsm.state == "DONE":
            fsm.publish_setpoint_enabled(False)

        elif fsm.state == 'ABORT':
            fsm.publish_setpoint_enabled(False)
            stop = Bool(); stop.data = False
            fsm.orbit_start_pub.publish(stop)
            if not fsm.abort_command_sent:
                mode = String(); mode.data = 'BRAKE'
                fsm.cmd_mode_pub.publish(mode)
                fsm.abort_command_sent = True

        elif fsm.state in ('MANUAL_OVERRIDE', 'MANUAL_SPRAY'):
            fsm.publish_setpoint_enabled(False)

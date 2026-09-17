from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
import math
from copy import deepcopy
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, String, Float32
from std_srvs.srv import SetBool
from uav_interfaces.msg import Tree
from beehive_drone.mission_params import MissionConfig
from beehive_drone.missions.base_mission import BaseMissionStrategy

if TYPE_CHECKING:
    from mission_state_machine import MissionStateMachine


class ApproachFlowerMission(BaseMissionStrategy):
    @property
    def navigation_states(self) -> Sequence[str]:
        return (
            "POST_TAKEOFF_HOVER",
            "EXPLORE_ROW",
            "ALIGN_TO_TREE",
            "APPROACH_TREE",
            "VERIFY_TREE",
            "START_ORBIT",
            "ALIGN_TO_LAST_ORBIT",
            "WAIT_ORBIT",
            "POST_ORBIT_HOVER",
            "ALIGN_HOME",
            "END_OF_ROW",
            "DETECT_FLOWER",
            "ALIGN_TO_FLOWER",
            "APPROACH_FLOWER",
            "CRAB_SCAN",
            "RETURN_TO_HOME",
            "HOME_HOVER",
            "FINAL_SPIN",
        )

    @property
    def timeout_exempt_states(self) -> Sequence[str]:
        return (
            "WAIT_START",
            "EXPLORE_ROW",
            "WAIT_ORBIT",
            "LANDING",
            "DONE",
            "ABORT",
            "MANUAL_OVERRIDE",
        )
        
    def call_sprayer(self, fsm: MissionStateMachine, state: bool):
        """Asynchronously send spray command without blocking the FSM thread."""
        req = SetBool.Request()
        req.data = state
        self.spray_in_progress = True
        
        future = fsm.sprayer_service.call_async(req)
        # Attach callback method defined inside this strategy class
        future.add_done_callback(lambda fut: self._sprayer_response_cb(fsm, fut, state))
    
    def _sprayer_response_cb(self, fsm: MissionStateMachine, future, requested_state: bool):
        """Callback executed in strategy class when sprayer service completes."""
        self.spray_in_progress = False
        try:
            response = future.result()
            self.last_spray_success = response.success
            action = "ENABLE" if requested_state else "DISABLE"
            fsm.get_logger().info(
                f"Sprayer {action} response: success={response.success}, msg='{response.message}'"
            )
        except Exception as e:
            fsm.get_logger().error(f"Sprayer service call failed with exception: {e}")

    def execute(
        self,
        fsm: MissionStateMachine,
        active: bool,
        elapsed: float,
        cx: float,
        cy: float,
    ):
        # --- FASE PRE-FLIGHT ---
        if fsm.state == "WAIT_START":
            if fsm.start_requested and not fsm.frame_alignment_ready:
                fsm.get_logger().warning(
                    "WAIT_START: menunggu alignment ZED-FC terkunci.",
                    throttle_duration_sec=2.0,
                )
            elif fsm.start_requested and not fsm.vision_ready():
                fsm.log_vision_wait()
            elif fsm.start_requested and (fsm.safety_ok or not fsm.require_safety):
                fsm.home_pose = (cx, cy, fsm.current_yaw())
                fsm.transition("INIT")
                fsm.get_logger().info(
                    "Start diterima dan sensor sehat; memulai preflight."
                )

        elif fsm.state == "INIT":
            mode_msg = String()
            mode_msg.data = "GUIDED"
            fsm.cmd_mode_pub.publish(mode_msg)
            fsm.transition("WAIT_GUIDED")
            fsm.retry_counter = 0
            fsm.get_logger().info("Meminta transisi ke mode GUIDED...")

        elif fsm.state == "WAIT_GUIDED":
            if fsm.current_mode == "GUIDED":
                arm_msg = Bool()
                arm_msg.data = True
                fsm.takeoff_yaw = fsm.current_yaw()
                fsm.cmd_arm_pub.publish(arm_msg)
                fsm.transition("WAIT_ARM")
                fsm.retry_counter = 0
                fsm.get_logger().info("Mode GUIDED aktif. Meminta Arming Motor...")
            else:
                fsm.retry_counter += 1
                if fsm.retry_counter > 20:
                    mode_msg = String()
                    mode_msg.data = "GUIDED"
                    fsm.cmd_mode_pub.publish(mode_msg)
                    fsm.retry_counter = 0

        elif fsm.state == "WAIT_ARM":
            if fsm.is_armed:
                if not hasattr(fsm, "arm_delay_start"):
                    fsm.arm_delay_start = fsm.get_clock().now()
                elapsed_ns = (fsm.get_clock().now() - fsm.arm_delay_start).nanoseconds
                if elapsed_ns >= 350_000_000:
                    takeoff_msg = Float32()
                    takeoff_msg.data = fsm.flight_altitude
                    fsm.cmd_takeoff_pub.publish(takeoff_msg)
                    fsm.transition("WAIT_TAKEOFF")
                    fsm.retry_counter = 0
                    fsm.get_logger().info(
                        f"Motor Bersenjata (Armed). Takeoff ke ketinggian {fsm.flight_altitude}m..."
                    )
            else:
                fsm.retry_counter += 1
                if fsm.retry_counter > 20:
                    arm_msg = Bool()
                    arm_msg.data = True
                    fsm.cmd_arm_pub.publish(arm_msg)
                    fsm.get_logger().info(
                        "Mencoba Arming ulang... (Menunggu Pre-arm good dari ArduPilot)"
                    )
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
                    f"{fsm.post_takeoff_hover_time:.1f} detik."
                )

        elif fsm.state == "POST_TAKEOFF_HOVER":
            fsm.publish_goal(fsm.hold_x, fsm.hold_y, fsm.hold_yaw)
            if elapsed >= fsm.post_takeoff_hover_time:
                fsm.transition("EXPLORE_ROW")
                fsm.get_logger().info(
                    "Hover pasca-takeoff selesai. Mulai EXPLORE_ROW (mencari pohon)."
                )

        # --- FASE MISI UTAMA ---
        elif fsm.state == "EXPLORE_ROW":
            fsm.target_tree = fsm.find_uninspected_tree()
            if fsm.target_tree is not None:
                fsm.verification_retries = 0
                fsm.target_tree = deepcopy(fsm.target_tree)
                fsm.frozen_target_tree = deepcopy(fsm.target_tree)
                fsm.transition("ALIGN_TO_TREE")
                fsm.get_logger().info(
                    f"Pohon ditemukan di ({fsm.target_tree.x:.1f}, {fsm.target_tree.y:.1f})"
                )
            else:
                target_x = cx + (fsm.explore_speed * fsm.explore_dir_x)
                target_yaw = 0.0 if fsm.explore_dir_x > 0 else math.pi
                fsm.publish_goal(target_x, cy, target_yaw)

                dist_from_last = fsm.distance(cx, cy, fsm.last_tree_x, fsm.last_tree_y)
                if dist_from_last > fsm.end_of_row_dist:
                    fsm.transition("END_OF_ROW")
                    fsm.get_logger().info("Lorong Habis. Bersiap pindah lorong.")

        elif fsm.state == "ALIGN_TO_TREE":
            tree = fsm.frozen_target_tree or fsm.target_tree
            target_yaw = math.atan2(tree.y - cy, tree.x - cx)
            fsm.publish_goal(cx, cy, target_yaw)
            if fsm.yaw_aligned(fsm.current_yaw(), target_yaw, fsm.align_yaw_tolerance):
                if fsm.align_yaw_since is None:
                    fsm.align_yaw_since = fsm.get_clock().now()
                held = (fsm.get_clock().now() - fsm.align_yaw_since).nanoseconds * 1e-9
                if held >= fsm.align_yaw_hold_time:
                    fsm.transition("APPROACH_TREE")
                    fsm.get_logger().info(
                        "Yaw ke pohon stabil; memulai translasi approach."
                    )
            else:
                fsm.align_yaw_since = None

        elif fsm.state == "APPROACH_TREE":
            tree = fsm.frozen_target_tree or fsm.target_tree
            target_yaw = math.atan2(tree.y - cy, tree.x - cx)
            stop_x = tree.x - (fsm.approach_safe_dist * math.cos(target_yaw))
            stop_y = tree.y - (fsm.approach_safe_dist * math.sin(target_yaw))
            dist_to_stop = fsm.distance(cx, cy, stop_x, stop_y)

            if dist_to_stop > fsm.approach_goal_tolerance:
                fsm.publish_goal(stop_x, stop_y, target_yaw)
            else:
                fsm.transition("VERIFY_TREE")
                fsm.hover_timer = 0
                fsm.get_logger().info(
                    "Titik pengereman tercapai. Hovering 4 detik untuk stabilisasi..."
                )

        elif fsm.state == "VERIFY_TREE":
            target_yaw = math.atan2(fsm.target_tree.y - cy, fsm.target_tree.x - cx)
            fsm.publish_goal(cx, cy, target_yaw)
            fsm.hover_timer += 1

            if fsm.hover_timer >= 40:
                target_matched_tree = None
                for tree in fsm.trees:
                    if tree.id == fsm.target_tree.id:
                        target_matched_tree = tree
                        break

                if target_matched_tree is not None:
                    actual_dist_to_tree = fsm.distance(
                        cx, cy, target_matched_tree.x, target_matched_tree.y
                    )
                    min_verify = max(
                        0.1, fsm.approach_safe_dist - fsm.tree_distance_tolerance
                    )
                    max_verify = fsm.approach_safe_dist + fsm.tree_distance_tolerance

                    if min_verify <= actual_dist_to_tree <= max_verify:
                        fsm.target_tree = target_matched_tree
                        update_msg = Tree()
                        update_msg.id = target_matched_tree.id
                        update_msg.x = target_matched_tree.x
                        update_msg.y = target_matched_tree.y
                        update_msg.z = target_matched_tree.z
                        update_msg.confidence = target_matched_tree.confidence
                        update_msg.inspected = target_matched_tree.inspected
                        update_msg.validated = True
                        update_msg.orbit_count = target_matched_tree.orbit_count
                        fsm.tree_update_pub.publish(update_msg)
                        fsm.transition("START_ORBIT")
                        fsm.get_logger().info(
                            f"Verifikasi sukses! Pohon ID:{target_matched_tree.id} valid. Memulai orbit."
                        )
                    else:
                        fsm.verification_retries += 1
                        if fsm.verification_retries <= fsm.verification_retry_limit:
                            fsm.target_tree = deepcopy(target_matched_tree)
                            fsm.frozen_target_tree = deepcopy(target_matched_tree)
                            fsm.hover_timer = 0
                            fsm.transition("ALIGN_TO_TREE")
                            fsm.get_logger().warning(
                                f"Pohon ID:{target_matched_tree.id} di luar rentang toleransi; retry "
                                f"{fsm.verification_retries}/{fsm.verification_retry_limit}."
                            )
                        else:
                            fsm.get_logger().warning(
                                f"Pohon ID:{target_matched_tree.id} gagal verifikasi; dihapus."
                            )
                            update_msg = Tree()
                            update_msg.id = target_matched_tree.id
                            update_msg.confidence = -1.0
                            fsm.tree_update_pub.publish(update_msg)
                            fsm.target_tree = None
                            fsm.transition("EXPLORE_ROW")
                else:
                    fsm.get_logger().warn(
                        "Pohon hilang dari peta saat hovering! Membatalkan orbit."
                    )
                    if fsm.target_tree is not None:
                        update_msg = Tree()
                        update_msg.id = fsm.target_tree.id
                        update_msg.confidence = -1.0
                        fsm.tree_update_pub.publish(update_msg)
                    fsm.target_tree = None
                    fsm.transition("EXPLORE_ROW")

        elif fsm.state == "START_ORBIT":
            target_msg = Point()
            target_msg.x = fsm.target_tree.x
            target_msg.y = fsm.target_tree.y
            target_msg.z = float(fsm.navigation_altitude)
            fsm.orbit_target_pub.publish(target_msg)

            start_msg = Bool()
            start_msg.data = True
            fsm.orbit_start_pub.publish(start_msg)
            fsm.transition("WAIT_ORBIT")

        elif fsm.state == "WAIT_ORBIT":
            if fsm.receiving_flower_pose and not fsm.done_receiving_flower_pose:
                fsm.transition("DETECT_FLOWER")
                fsm.get_logger().info("Flower DETECTED, Align dengan flower")
            elif fsm.orbit_status == "ORBIT_COMPLETED":
                stop_msg = Bool()
                stop_msg.data = False
                fsm.orbit_start_pub.publish(stop_msg)

                if fsm.target_tree is not None:
                    update_msg = Tree()
                    update_msg.id = fsm.target_tree.id
                    update_msg.x = fsm.target_tree.x
                    update_msg.y = fsm.target_tree.y
                    update_msg.z = fsm.target_tree.z
                    update_msg.confidence = fsm.target_tree.confidence
                    update_msg.inspected = True
                    update_msg.validated = True
                    update_msg.orbit_count = min(
                        255, int(fsm.target_tree.orbit_count) + 1
                    )
                    fsm.tree_update_pub.publish(update_msg)
                    fsm.get_logger().info(
                        f"Pohon ID:{fsm.target_tree.id} ditandai SELESAI."
                    )

                fsm.hold_x = cx
                fsm.hold_y = cy
                fsm.hold_yaw = fsm.current_yaw()
                fsm.hover_timer = 0
                fsm.target_tree = None
                fsm.transition("POST_ORBIT_HOVER")
                fsm.get_logger().info(
                    "Orbit selesai. Hover sebelum kembali ke titik takeoff."
                )
            elif fsm.orbit_status.startswith("ORBIT_FAILED"):
                fsm.transition("ABORT")
                fsm.get_logger().error(f"ABORT: {fsm.orbit_status}")

        elif fsm.state == "DETECT_FLOWER":
            pause_msg = Bool()
            pause_msg.data = True
            fsm.orbit_pause_pub.publish(pause_msg)
            fsm.transition("ALIGN_TO_FLOWER")

        elif fsm.state == "ALIGN_TO_FLOWER":
            if fsm.flower_pose is None or fsm.target_tree is None:
                fsm.get_logger().error("Data flower atau target_tree tidak ada! ABORT.")
                fsm.transition("ABORT")
                return
            tree_x, tree_y = fsm.target_tree.x, fsm.target_tree.y
            flower_x, flower_y = fsm.flower_pose.position.x, fsm.flower_pose.position.y

            dx = flower_x - tree_x
            dy = flower_y - tree_y
            target_angle = math.atan2(dy, dx)

            goal_x = tree_x + fsm.orbit_radius * math.cos(target_angle)
            goal_y = tree_y + fsm.orbit_radius * math.sin(target_angle)
            target_yaw = math.atan2(flower_y - cy, flower_x - cx)

            fsm.publish_goal(goal_x, goal_y, target_yaw)

            dist_to_goal = fsm.distance(cx, cy, goal_x, goal_y)
            if dist_to_goal <= fsm.approach_goal_tolerance and fsm.yaw_aligned(
                fsm.current_yaw(), target_yaw, fsm.align_yaw_tolerance
            ):
                fsm.transition("APPROACH_FLOWER")
                fsm.hover_timer = 0
                fsm.get_logger().info(
                    f"Berhasil Aligned dengan Bunga pada ({goal_x:.2f}, {goal_y:.2f})"
                )

        elif fsm.state == "APPROACH_FLOWER":
            tree_x, tree_y = fsm.target_tree.x, fsm.target_tree.y
            flower_x, flower_y = fsm.flower_pose.position.x, fsm.flower_pose.position.y

            target_angle = math.atan2(flower_y - tree_y, flower_x - tree_x)
            goal_x = tree_x + 1 * math.cos(target_angle)
            goal_y = tree_y + 1 * math.sin(target_angle)
            target_yaw = math.atan2(tree_y - cy, tree_x - cx)

            fsm.publish_goal(goal_x, goal_y, target_yaw)

            dist_to_goal = fsm.distance(cx, cy, goal_x, goal_y)
            if dist_to_goal <= fsm.approach_goal_tolerance and fsm.yaw_aligned(
                fsm.current_yaw(), target_yaw, fsm.align_yaw_tolerance
            ):
                fsm.hover_timer += 1
                
                # Turn ON sprayer ONCE
                if fsm.hover_timer == 20:
                    fsm.get_logger().info("Mendekati Bunga - Menyalakan Sprayer...")
                    self.call_sprayer(fsm, True)
                    
                # Turn OFF sprayer ONCE
                elif fsm.hover_timer == 40:
                    fsm.get_logger().info("Penyemprotan selesai - Mematikan Sprayer...")
                    self.call_sprayer(fsm, False)
                    fsm.hover_timer = 0
                    fsm.transition("ALIGN_TO_LAST_ORBIT")
            else:
                fsm.hover_timer = 0

        elif fsm.state == "POST_ORBIT_HOVER":
            fsm.publish_goal(fsm.hold_x, fsm.hold_y, fsm.hold_yaw)
            fsm.hover_timer += 1
            required_ticks = int(MissionConfig.POST_ORBIT_HOVER_TIME / 0.1)
            if fsm.hover_timer >= required_ticks:
                fsm.hover_timer = 0
                fsm.transition("ALIGN_HOME")
                fsm.get_logger().info("Hover stabil. Menyesuaikan yaw menuju home.")

        elif fsm.state == "ALIGN_HOME":
            fsm.done_receiving_flower_pose = False
            if fsm.home_pose is None:
                fsm.get_logger().error("Home belum tersimpan; menahan posisi.")
                fsm.publish_goal(fsm.hold_x, fsm.hold_y, fsm.hold_yaw)
                return

            home_x, home_y, _ = fsm.home_pose
            yaw_to_home = math.atan2(home_y - cy, home_x - cx)
            fsm.publish_goal(fsm.hold_x, fsm.hold_y, yaw_to_home)

            yaw_error = abs(fsm.normalize_angle(yaw_to_home - fsm.current_yaw()))
            aligned = yaw_error <= MissionConfig.HOME_YAW_TOLERANCE
            fsm.hover_timer = fsm.hover_timer + 1 if aligned else 0

            required_ticks = int(MissionConfig.HOME_ALIGN_TIME / 0.1)
            if fsm.hover_timer >= required_ticks:
                fsm.hover_timer = 0
                fsm.transition("RETURN_TO_HOME")
                fsm.get_logger().info(
                    "Arah ke home stabil. Mulai kembali ke titik takeoff."
                )

        elif fsm.state == "END_OF_ROW":
            retreat_x = fsm.last_tree_x - (fsm.approach_safe_dist * fsm.explore_dir_x)
            target_yaw = 0.0 if fsm.explore_dir_x > 0 else math.pi
            fsm.publish_goal(retreat_x, fsm.last_tree_y, target_yaw)

            if abs(cx - retreat_x) < 0.5:
                fsm.explore_dir_x *= -1.0
                fsm.crab_start_y = cy
                fsm.transition("CRAB_SCAN")
                fsm.get_logger().info("Mundur selesai. Memulai Crab Scan 90 derajat.")

        elif fsm.state == "CRAB_SCAN":
            target_y = cy + (fsm.crab_speed * fsm.explore_dir_y)
            target_yaw = 0.0 if fsm.explore_dir_x > 0 else math.pi
            fsm.publish_goal(cx, target_y, target_yaw)

            fsm.target_tree = fsm.find_uninspected_tree()
            if fsm.target_tree is not None:
                fsm.last_tree_y = fsm.target_tree.y
                fsm.verification_retries = 0
                fsm.target_tree = deepcopy(fsm.target_tree)
                fsm.frozen_target_tree = deepcopy(fsm.target_tree)
                fsm.transition("ALIGN_TO_TREE")
                fsm.get_logger().info("Lorong baru ditemukan!")
            else:
                if abs(cy - fsm.crab_start_y) > fsm.end_of_farm_dist:
                    fsm.transition("RETURN_TO_HOME")
                    fsm.get_logger().info("Lahan habis. Cari jalur untuk pulang (RTH).")

        elif fsm.state == "RETURN_TO_HOME":
            if fsm.home_pose is None:
                return
            home_x, home_y, home_yaw = fsm.home_pose
            target_yaw = math.atan2(home_y - cy, home_x - cx)
            fsm.publish_goal(home_x, home_y, target_yaw)

            if (
                fsm.distance(cx, cy, home_x, home_y)
                < MissionConfig.HOME_POSITION_TOLERANCE
            ):
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
            current_yaw = math.atan2(
                2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)
            )

            delta = current_yaw - fsm.last_yaw
            if delta > math.pi:
                delta -= 2 * math.pi
            elif delta < -math.pi:
                delta += 2 * math.pi

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
                float("inf")
                if fsm.last_landing_command_time is None
                else (now - fsm.last_landing_command_time).nanoseconds * 1e-9
            )
            if fsm.landing_command_due(
                fsm.current_mode, last_command_age, fsm.landing_retry_interval
            ):
                land_msg = Bool()
                land_msg.data = True
                fsm.cmd_land_pub.publish(land_msg)
                fsm.last_landing_command_time = now
            if not fsm.is_armed:
                fsm.transition("DONE")
                fsm.get_logger().info("Landing selesai. Misi DONE.")

        elif fsm.state == "DONE":
            fsm.publish_setpoint_enabled(False)

        elif fsm.state == "ABORT":
            fsm.publish_setpoint_enabled(False)
            stop = Bool()
            stop.data = False
            fsm.orbit_start_pub.publish(stop)
            if not fsm.abort_command_sent:
                mode = String()
                mode.data = "BRAKE"
                fsm.cmd_mode_pub.publish(mode)
                fsm.abort_command_sent = True

        elif fsm.state in ("MANUAL_OVERRIDE", "MANUAL_SPRAY"):
            fsm.publish_setpoint_enabled(False)

        elif fsm.state == "ALIGN_TO_LAST_ORBIT":
            pause_msg = Bool()
            pause_msg.data = False
            fsm.orbit_pause_pub.publish(pause_msg)
            fsm.transition("WAIT_ORBIT")
            fsm.done_receiving_flower_pose = True
            fsm.receiving_flower_pose = False
            fsm.flower_pose = None

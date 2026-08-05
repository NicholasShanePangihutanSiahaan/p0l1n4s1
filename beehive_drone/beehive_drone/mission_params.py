#!/usr/bin/env python3
"""Default configuration shared by all beehive mission nodes."""

import math


class MissionConfig:
    # Internal navigation frame for the real vehicle. The mission, PCL map,
    # and MAVROS local pose must all be aligned to the same local odometry frame.
    WORLD_FRAME = "odom"
    PCL_FRAME = "odom"

    # Flight lifecycle
    FLIGHT_MODE = "GUIDED"
    FLIGHT_ALTITUDE = 3.0
    PRESTREAM_SEC = 0.0
    COMMAND_RETRY_SEC = 2.5
    LAND_RETRY_SEC = 5.0
    TAKEOFF_TIMEOUT_SEC = 45.0
    TAKEOFF_PROGRESS_CHECK_SEC = 20.0
    MIN_TAKEOFF_PROGRESS = 0.20
    HOME_REACHED_TOLERANCE = 0.8
    HOVER_WAIT_TIMEOUT_SEC = 15.0
    LAND_COMPLETE_ALTITUDE = 0.25
    POSE_TIMEOUT_SEC = 1.5
    MAP_TIMEOUT_SEC = 6.0
    STATE_TIMEOUT_SEC = 90.0
    DISCONNECT_GRACE_SEC = 8.0

    # Map safety gate
    REQUIRE_TREE_MAP = True
    MAP_STARTUP_TIMEOUT_SEC = 35.0
    MAP_LOSS_GRACE_SEC = 3.0
    MIN_READY_TREES = 1

    # Stable hover detection
    HOVER_ALT_TOLERANCE = 0.25
    HOVER_SPEED_TOLERANCE = 0.20
    HOVER_STABLE_SEC = 1.5

    # Plantation exploration
    EXPLORE_STEP = 1.0
    CRAB_STEP = 1.0
    END_OF_ROW_DIST = 10.0
    END_OF_FARM_DIST = 20.0
    TREE_SEARCH_RADIUS = 18.0
    TREE_MIN_CONFIDENCE = 0.35
    VERIFY_DURATION_SEC = 1.5
    VERIFY_POSITION_TOLERANCE = 1.0

    # Single-tree approach/hover sequence
    APPROACH_DISTANCE = 3.0
    APPROACH_TOLERANCE = 0.25
    PRE_ORBIT_HOVER_SEC = 3.0
    POST_ORBIT_HOVER_SEC = 2.0
    RETURN_HOVER_SEC = 1.5
    HOME_HOVER_SEC = 2.0

    # Orbit
    ORBIT_RADIUS = 3.0
    ORBIT_ALTITUDE = FLIGHT_ALTITUDE
    ORBIT_VELOCITY = 0.30
    ORBIT_DIRECTION = 1.0
    ORBIT_LOOKAHEAD_DISTANCE = 0.35
    ORBIT_RADIAL_GAIN = 0.35
    ORBIT_RADIUS_TOLERANCE = 0.30
    ORBIT_START_TOLERANCE = 0.45
    ORBIT_COMPLETION_MARGIN = 0.0
    ORBIT_TIMEOUT_SEC = 90.0
    YAW_OFFSET = 0.0

    # Obstacle avoidance
    OBSTACLE_INFLUENCE_RADIUS = 4.5
    OBSTACLE_HARD_RADIUS = 2.2
    ACTIVE_TARGET_KEEP_OUT_RADIUS = 2.2
    EMERGENCY_STOP_RADIUS = 1.8
    REPULSIVE_GAIN = 2.5
    VORTEX_GAIN = 0.8
    ATTRACTION_GAIN = 0.6
    MAX_TARGET_SHIFT = 0.6

    # Velocity controller
    KP_XY = 0.45
    KP_Z = 0.80
    KP_YAW = 1.0
    MAX_VELOCITY_XY = 0.30
    MAX_VELOCITY_Z = 0.60
    MAX_VELOCITY_YAW = 0.45
    MAX_ACCELERATION_XY = 0.20
    MAX_ACCELERATION_Z = 0.50
    GOAL_THRESHOLD_XY = 0.15
    GOAL_THRESHOLD_Z = 0.10
    TARGET_TIMEOUT_SEC = 1.0

    # PCL tree map
    PCL_TOPIC = "/perception/pcl/tracked_cylinders"
    PCL_MIN_SEEN_COUNT = 2
    PCL_MAX_MISSED_COUNT = 20
    PCL_MIN_CONFIDENCE = 0.25
    PCL_MIN_RADIUS = 0.08
    PCL_MAX_RADIUS = 1.20
    PCL_MIN_HEIGHT = 0.8
    PCL_MAX_HEIGHT = 20.0
    TREE_STALE_SEC = 60.0
    PCL_STREAM_TIMEOUT_SEC = 6.0
    TREE_REJECT_HOLD_SEC = 20.0
    TREE_FALLBACK_MERGE_DISTANCE = 2.0
    ALLOW_IDENTITY_FRAME_RELABEL = True

    # YOLO fallback
    YOLO_CONFIDENCE = 0.35
    YOLO_IOU = 0.45
    YOLO_MAX_DEPTH = 20.0
    YOLO_MIN_DEPTH = 0.25
    YOLO_DEVICE = "0"
    YOLO_TARGET_LABELS = ["palm", "palm_tree", "tree", "oil_palm"]

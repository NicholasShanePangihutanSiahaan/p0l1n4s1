# Revision Summary

## Active integration

- ZED pose -> `vision_to_mavros` -> `/mavros/vision_pose/pose`.
- ZED registered point cloud + `/mavros/odometry/out` -> `pcl_proc_node`.
- PCL tracked cylinders -> `pcl_tree_mapper` -> `/map/trees`.
- Single-tree FSM from the Gazebo-tested mission flow.
- One orbit, return to pre-orbit point, return to captured home, hover, land.
- Detailed CSV and 2D/3D PNG autosave analyzer.

## Hardware-specific changes

- Navigation frame changed from `map` to `odom`.
- PCL output frame changed from hardcoded `plantation` to configurable `odom`.
- Camera mounting translation and rotation are configurable.
- ZED pose and point-cloud topics are launch arguments.
- VSLAM bridge is optional to avoid duplicate ExternalNav publishers.
- Default full mission is disabled by `hold_after_takeoff:=true`.

## Preserved source

The original controller and mission files from `drone_real.zip` are retained under:

```text
beehive_drone/legacy_real_nodes/
```

They are not used by the integrated launch.

## Validation performed

- Python syntax compilation for active Python and launch files.
- YAML parsing.
- package.xml parsing.
- C++ brace/integration checks.
- ZIP integrity test.

A full ROS 2/Jetson build and hardware flight test were not possible in this environment.

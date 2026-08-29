# Polinasi — stack drone nyata

Workspace ROS 2 Humble untuk Jetson + ZED2i + Pixhawk 6C. Jalur persepsi real
menggunakan custom object detection ZED, bukan fitting silinder point cloud:

```text
ZED2i ObjectsStamped + pose
          |
          +-> bb_pcl_proc_node -> /global_cylinders -> tree_mapper
          +-> vision_to_mavros -> Pixhawk ExternalNav
                                                |
                                      controller + mission FSM
```

Komponen Gazebo, `sim_zed_adapter`, dan parameter SITL tidak disertakan dalam
repository deployment ini. Jangan menjalankan `pcl_proc_node` bersamaan dengan
`bb_pcl_proc_node`, karena `/global_cylinders` harus memiliki tepat satu
publisher.

## Build di Jetson

Pasang ROS 2 Humble, ZED SDK/ROS 2 wrapper yang sesuai dengan JetPack, MAVROS,
`mavros_extras`, dan `diagnostic_updater`. Build/source workspace ZED lebih
dahulu agar package `zed_msgs` tersedia.

```bash
cd ~/polinasi
source /opt/ros/humble/setup.bash
source /PATH/WORKSPACE_ZED/install/setup.bash

rosdep install --from-paths . --ignore-src -r -y
colcon build --symlink-install --packages-select \
  uav_interfaces pcl_cstm_msg point-cloud-test beehive_drone
source install/setup.bash
```

Model `beehive_drone/models/best_detection_palm_oil.onnx` sudah disertakan.
Launch ZED memetakan class 0 (`Trunk`) menjadi label `pohon`, yaitu label yang
diterima BB node.

## Menjalankan drone nyata

Gunakan lima terminal terpisah dan biarkan semuanya aktif.

Terminal 1 — Pixhawk/MAVROS; sesuaikan device dan baud rate:

```bash
source /opt/ros/humble/setup.bash
source ~/polinasi/install/setup.bash
ros2 launch mavros apm.launch fcu_url:=/dev/ttyACM0:921600
```

Terminal 2 — ZED2i tracking dan custom detector:

```bash
source /opt/ros/humble/setup.bash
source /PATH/WORKSPACE_ZED/install/setup.bash
source ~/polinasi/install/setup.bash
ros2 launch beehive_drone real_zed2i.launch.py
```

Terminal 3 — ExternalNav ZED ke Pixhawk:

```bash
source /opt/ros/humble/setup.bash
source ~/polinasi/install/setup.bash
ros2 launch beehive_drone vision_to_mavros.launch.py
```

Terminal 4 — bounding box menjadi landmark global:

```bash
source /opt/ros/humble/setup.bash
source ~/polinasi/install/setup.bash
ros2 launch point-cloud-test bb_proc_node.launch.py
```

Terminal 5 — mapper, controller, safety monitor, analyzer, dan FSM:

```bash
source /opt/ros/humble/setup.bash
source ~/polinasi/install/setup.bash
ros2 launch beehive_drone real_mission.launch.py auto_start:=false
```

## Gate sebelum start

Lakukan pemeriksaan pertama tanpa propeller:

```bash
ros2 topic echo --once /mavros/state
ros2 topic hz /zed/zed_node/pose
ros2 topic hz /zed/zed_node/obj_det/objects
ros2 topic hz /mavros/vision_pose/pose
ros2 topic hz /mavros/local_position/pose
ros2 topic echo --once /mavros/rangefinder/rangefinder
ros2 topic echo --once /global_cylinders
ros2 topic echo --once /map/trees
ros2 topic info /global_cylinders -v
ros2 topic echo --once /mission/safety_reason
ros2 topic echo --once /mission/safety_ok
```

Pastikan MAVROS connected, vision/local pose stabil, label pohon terdeteksi,
rangefinder valid, koordinat pohon tidak bergerak ketika drone digeser/yaw,
publisher `/global_cylinders` tepat satu, dan `safety_ok.data` bernilai `true`.

Setelah seluruh gate lulus dan area aman:

```bash
ros2 topic pub --once /mission/start std_msgs/msg/Bool "{data: true}"
```

Checklist kalibrasi, TF kamera, EKF, takeover RC, pengujian bertahap, dan rosbag
tersedia di [beehive_drone/REAL_FLIGHT.md](beehive_drone/REAL_FLIGHT.md).

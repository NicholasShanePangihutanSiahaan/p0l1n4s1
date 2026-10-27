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
ros2 launch beehive_drone real_mission.launch.py \
  auto_start:=false \
  mission_type:=basic_orbit \
  mission_mode:=single_tree
```

Untuk multi-tree dengan batas dua pohon:

```bash
ros2 launch beehive_drone real_mission.launch.py \
  auto_start:=false \
  mission_type:=basic_orbit \
  mission_mode:=multi_tree \
  max_trees:=2
```

`mission_type` memilih strategi penerbangan/spraying, sedangkan
`mission_mode` memilih apakah strategi tersebut berhenti setelah satu pohon
atau melanjutkan ke pohon berikutnya. `max_trees` hanya berlaku pada
`multi_tree`; nilai `0` berarti tanpa batas jumlah pohon.

Untuk menguji satu pohon hasil deteksi lalu satu target virtual dengan
koordinat relatif terhadap home:

```bash
ros2 launch beehive_drone real_mission.launch.py \
  auto_start:=false \
  mission_type:=virtual_tree_test \
  mission_mode:=multi_tree \
  max_trees:=2 \
  enable_flower_detection:=false \
  require_tree_ahead:=false \
  virtual_tree_position_mode:=home_relative \
  virtual_tree_position_x:=6.0 \
  virtual_tree_position_y:=3.0
```

`home_relative` menghitung pusat virtual sebagai
`(home_x + position_x, home_y + position_y)`. Alternatifnya,
`virtual_tree_position_mode:=map` menggunakan X/Y absolut pada local map,
sedangkan `toward_home` menggunakan `virtual_tree_offset_toward_home` sebagai
jarak dari pohon pertama menuju home. Target virtual hanya hidup di dalam FSM;
target tersebut tidak dipublikasikan sebagai hasil AI ke tree mapper.

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

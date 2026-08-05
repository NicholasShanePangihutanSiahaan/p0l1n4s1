# Drone Real + ZED 2i VSLAM + PCL Segmentation

Target perangkat:

- ZED 2i
- Jetson Orin Nano Super
- Pixhawk 6C dengan ArduCopter dan MAVROS
- ROS 2 Humble

## Arsitektur aktif

```text
ZED 2i positional tracking
/zed/zed_node/pose
        |
        v
vision_to_mavros.py
/mavros/vision_pose/pose
        |
        v
Pixhawk EKF ExternalNav -> /mavros/odometry/out + local_position/pose

ZED registered point cloud
/zed/zed_node/point_cloud/cloud_registered
        + /mavros/odometry/out
        |
        v
PCL_Segmentation/pcl_proc_node
voxel -> outlier removal -> global transform -> ground removal
-> clustering -> cylinder fitting -> tracking
        |
        v
/perception/pcl/tracked_cylinders
        |
        v
pcl_tree_mapper -> /map/trees + /map/trees_ready
        |
        v
single-tree FSM -> orbit controller -> vortex safety -> velocity controller
        |
        v
/mavros/setpoint_velocity/cmd_vel
```

## Alur misi

```text
WAIT_CONNECTION -> GUIDED -> ARM -> TAKEOFF -> HOVER
-> SEARCH_TREE -> APPROACH_TREE (3 m)
-> HOVER_BEFORE_ORBIT -> ORBIT 360 deg
-> POST_ORBIT_HOVER -> RETURN_PRE_ORBIT -> HOVER
-> RETURN_HOME -> HOME_HOVER -> LAND -> DONE
```

Home direkam dari `/mavros/local_position/pose` sebelum arming. Posisi "sebelum takeoff"
adalah koordinat local/home tersebut. Setelah orbit, drone kembali dahulu ke titik sebelum
orbit, lalu ke home, hover, dan landing.

## Isi paket

- `beehive_drone`: VSLAM bridge, FSM, controller, mapper, analyzer.
- `point-cloud-test`: kode PCL Segmentation yang telah diberi parameter untuk perangkat nyata.
- `pcl_cstm_msg`: custom messages dari PCL Segmentation.
- `uav_interfaces`: message pohon untuk mission stack.
- `open3d_vis`: visualisasi opsional.
- `beehive_drone/legacy_real_nodes`: salinan node asli `drone_real.zip` sebelum integrasi.

## Perubahan PCL untuk drone nyata

`pcl_proc_node` sekarang memiliki parameter:

- `output_frame`
- `process_period_ms`
- `max_buffered_pairs`
- `voxel_leaf_size`
- `max_merged_points`
- `sor_mean_k`, `sor_stddev`
- `camera_mount_x/y/z`
- `camera_mount_roll/pitch/yaw`

Point cloud optik diubah menjadi sumbu badan ROS, kemudian diberi extrinsic pemasangan
kamera, lalu ditransformasikan menggunakan `/mavros/odometry/out` ke frame `odom`.

## Extrinsic ZED 2i

Isi bagian berikut pada `beehive_drone/config/mission_real_pcl.yaml`:

```yaml
pcl_proc_node:
  ros__parameters:
    camera_mount_x: 0.0
    camera_mount_y: 0.0
    camera_mount_z: 0.0
    camera_mount_roll: 0.0
    camera_mount_pitch: 0.0
    camera_mount_yaw: 0.0
```

Konvensi:

- translasi dalam meter dari pusat badan/autopilot ke pusat kamera;
- rotasi dalam radian;
- kamera menghadap depan dan rata: roll/pitch/yaw mendekati nol;
- nilai harus konsisten dengan konfigurasi `VISO_POS_X/Y/Z` pada ArduPilot.

Jangan melakukan flight test sebelum arah sumbu diverifikasi dengan menggerakkan drone
secara manual tanpa propeller dan mengamati posisi pohon di RViz.

## Build

Salin empat package ke `src` workspace:

```bash
cd ~/ProjekAtaka/gazebo_sim/src
cp -r /lokasi/drone_real_pcl_integrated/beehive_drone .
cp -r /lokasi/drone_real_pcl_integrated/uav_interfaces .
cp -r /lokasi/drone_real_pcl_integrated/pcl_cstm_msg .
cp -r /lokasi/drone_real_pcl_integrated/point-cloud-test .
```

Dependencies umum Ubuntu 22.04 / ROS 2 Humble:

```bash
sudo apt update
sudo apt install -y \
  libpcl-dev libvtk9-dev \
  ros-humble-pcl-conversions \
  ros-humble-message-filters \
  ros-humble-mavros-msgs \
  ros-humble-tf2-ros
```

Build bersih:

```bash
cd ~/ProjekAtaka/gazebo_sim
rm -rf build/beehive_drone build/uav_interfaces build/pcl_cstm_msg build/point-cloud-test
rm -rf install/beehive_drone install/uav_interfaces install/pcl_cstm_msg install/point-cloud-test
colcon build --symlink-install --packages-select \
  uav_interfaces pcl_cstm_msg point-cloud-test beehive_drone
source install/setup.bash
```

## Pastikan nama topic ZED

```bash
ros2 topic list | grep -E 'zed.*(pose|point_cloud|depth)'
ros2 topic type /zed/zed_node/pose
ros2 topic type /zed/zed_node/point_cloud/cloud_registered
ros2 topic hz /zed/zed_node/pose
ros2 topic hz /zed/zed_node/point_cloud/cloud_registered
```

Jika namespace kamera adalah `/zed2i/zed_node`, berikan override saat launch.

## Tahap 1: uji perception tanpa propeller

ZED wrapper dan MAVROS harus sudah berjalan.

```bash
source ~/ProjekAtaka/gazebo_sim/install/setup.bash
ros2 launch beehive_drone perception_test.launch.py \
  point_cloud_topic:=/zed/zed_node/point_cloud/cloud_registered \
  odom_topic:=/mavros/odometry/out
```

Periksa:

```bash
ros2 topic echo /perception/pcl/tracked_cylinders --once
ros2 topic echo /map/trees --once
ros2 topic echo /map/trees_ready
```

Gerakkan drone secara manual. Koordinat pohon yang sama harus relatif tetap di frame
`odom`. Jika pohon berputar atau bergeser mengikuti kamera, extrinsic atau frame odometri
belum benar.

## Tahap 2: uji VSLAM ke Pixhawk tanpa motor

```bash
ros2 run beehive_drone vision_to_mavros --ros-args \
  -p input_pose_topic:=/zed/zed_node/pose
```

Periksa:

```bash
ros2 topic hz /mavros/vision_pose/pose
ros2 topic hz /mavros/odometry/out
ros2 topic echo /mavros/local_position/pose --once
```

Pastikan gerakan maju drone menghasilkan arah posisi yang sama antara ZED pose,
MAVROS odometry, dan local pose.

## Tahap 3: takeoff dan hold

Default launch sengaja `hold_after_takeoff:=true`:

```bash
ros2 launch beehive_drone mission.launch.py \
  hold_after_takeoff:=true \
  point_cloud_topic:=/zed/zed_node/point_cloud/cloud_registered \
  zed_pose_topic:=/zed/zed_node/pose \
  odom_topic:=/mavros/odometry/out
```

Gunakan area terbuka, propeller guard, pilot siap mengambil alih, dan mode manual/AltHold
pada transmitter.

## Tahap 4: misi satu pohon

Setelah tahap sebelumnya lolos:

```bash
ros2 launch beehive_drone mission.launch.py \
  hold_after_takeoff:=false \
  point_cloud_topic:=/zed/zed_node/point_cloud/cloud_registered \
  zed_pose_topic:=/zed/zed_node/pose \
  odom_topic:=/mavros/odometry/out
```

## Parameter keselamatan awal

Konfigurasi awal sengaja konservatif:

- altitude 3 m;
- approach/orbit radius 3 m dari pusat cylinder;
- kecepatan horizontal maksimum 0.30 m/s;
- orbit velocity 0.30 m/s;
- emergency stop radius 1.8 m;
- orbit tidak dilanjutkan jika hover tidak terkonfirmasi.

Radius tersebut belum memperhitungkan diameter tajuk, cabang, ukuran propeller, error VSLAM,
dan error cylinder fit. Naikkan radius untuk pohon nyata bila ada cabang/tajuk pada altitude
orbit.

## Analyzer

Hasil disimpan langsung dan autosave ke:

```text
/home/shane/beehive_mission_results/mission_*/
```

Termasuk CSV detail, event state, summary, `map_2d.png`, dan `map_3d.png`.

## Catatan ExternalNav Pixhawk

Konfigurasi EKF3, VISO camera position, EKF origin, arming checks, dan failsafe harus
diverifikasi pada Mission Planner. Jangan mengaktifkan dua sumber VSLAM yang mengirim ke
Pixhawk secara bersamaan. Jika bridge ExternalNav lain sudah berjalan, jalankan launch dengan:

```bash
use_vslam_bridge:=false
```

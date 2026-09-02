# Real-flight checklist

`real_mission.launch.py` hanya menjalankan mapper, controller, FSM, safety
monitor, dan analyzer. MAVROS, ZED wrapper, `vision_to_mavros`, dan
`bb_proc_node.launch.py` dijalankan sebagai empat proses terpisah sebelum
misi. Pemisahan ini mencegah dua node persepsi menerbitkan
`/global_cylinders` secara bersamaan. Default real launch adalah
`auto_start:=false`; pertahankan nilai ini sampai seluruh pemeriksaan
pra-terbang lulus.

Persepsi nyata tidak lagi memakai fitting silinder dari point cloud.
`bb_pcl_proc_node` menerima 3D bounding box ZED berlabel tepat `pohon`,
mengubahnya dari frame kamera ke `base_link` melalui TF, lalu dari `base_link`
ke `map` menggunakan pose ZED. Keluaran `/global_cylinders` tetap sama sehingga
`tree_mapper` dan alur pemilihan/approach/orbit pohon tidak perlu diganti.

Konfigurasi misi utama saat ini adalah:

- target takeoff dan terrain-follow AGL: 1.5 m,
- titik approach: 1.5 m secara horizontal dari pusat batang,
- radius orbit: 1.5 m,
- hover sesudah takeoff: 2 detik.

## Arsitektur altitude dan terrain following

- Pixhawk/EKF menyediakan posisi lokal yang stabil untuk navigasi XY dan Z.
- Rangefinder bawah menyediakan tinggi AGL (jarak drone terhadap permukaan).
- `position_setpoint_controller` mengubah target Z lokal secara perlahan agar
  `desired_agl` tetap tercapai ketika tanah naik atau turun.
- Rangefinder tidak menggantikan seluruh local pose. Target yang dikirim ke
  MAVROS tetap `PoseStamped` dalam frame `map`; koreksinya adalah
  `Z_target = Z_local + (desired_agl - range_terfilter)`.
- Jika rangefinder stale/tidak valid, controller menahan setpoint terakhir dan
  tidak melanjutkan XY selama `hold_position_on_range_loss: true`. Ia tidak
  menebak ketinggian tanah.

Konfigurasi utama ada di blok `position_setpoint_controller` pada
`config/real.yaml`. `flight_manager.altitude_source: rangefinder` hanya
menentukan gate selesai takeoff/hover, sedangkan terrain following berlangsung
di controller setpoint selama approach, orbit, dan kembali ke home.

## Kalibrasi yang wajib

1. Isi transform pemasangan ZED2i terhadap `base_link` pada URDF/launch ZED.
   BB node membaca TF dari `ObjectsStamped.header.frame_id` ke `base_link`;
   parameter `camera_offset_*` milik PCL lama tidak digunakan pada jalur BB.
2. Pastikan TF tersedia dan pohon pada `/global_cylinders` tidak berpindah
   ketika drone digeser atau diputar secara manual:

   ```bash
   ros2 run tf2_ros tf2_echo base_link zed_left_camera_frame
   ```
3. Pastikan `/mavros/rangefinder/rangefinder` bertipe `sensor_msgs/msg/Range`,
   finite, lebih besar dari `rangefinder_min_valid`, dan frekuensinya stabil.
   Nilai maksimum sensor/parameter FC harus lebih tinggi dari 1.5 m dengan
   margin memadai; jangan terbang bila sensor berhenti pada nilai maksimum.
4. Tuning batas radius/tinggi batang, orbit, dan slew limit pada `real.yaml`
   dengan propeller dilepas terlebih dahulu. Radius BB adalah radius seluruh
   3D bounding box hasil model, sedangkan approach/orbit saat ini berjarak
   1.5 m dari pusat XY—bukan 1.5 m dari tepi bounding box. Jangan terbang jika
   bounding box mencakup tajuk dengan radius mendekati/melebihi 1.5 m; gunakan
   model box batang atau naikkan jarak approach/orbit lebih dahulu.
5. Pastikan hanya satu node yang menerbitkan setpoint MAVROS. Launch nyata
   menjalankan `position_setpoint_controller`; jangan jalankan
   `velocity_controller` pada saat yang sama.
6. Pastikan estimator Pixhawk sudah menerima VisualOdom dan topic
   `/mavros/local_position/pose` stabil sebelum mission launch dijalankan.
   FSM juga menunggu `/mavros/vision_pose/pose` kontinu selama 5 detik. Gate
   ini mencegah `auto_start` dimulai dari satu sampel vision pertama, tetapi
   tidak dapat membuktikan bahwa EKF telah menerima fusion; pesan pre-arm FC
   tetap harus bersih.
   Node alignment membutuhkan referensi yaw/local pose FC sebelum meneruskan
   pose ZED. Jika FC hanya dikonfigurasi ExternalNav tanpa sumber yaw awal,
   terjadi bootstrap loop dan `/alignment/ready` tidak akan pernah true.
7. Untuk konfigurasi EKF, gunakan Barometer sebagai sumber POSZ utama dan
   ExternalNav/Vision untuk posisi horizontal sesuai konfigurasi kendaraan.
   Rangefinder digunakan sebagai pengukuran AGL oleh program; jangan memilih
   RangeFinder sebagai sumber EKF POSZ hanya karena terrain following aktif.

## Pemeriksaan tanpa propeller

```bash
ros2 topic hz /mavros/local_position/pose
ros2 topic hz /mavros/rangefinder/rangefinder
ros2 topic echo /mavros/rangefinder/rangefinder
ros2 topic echo /mavros/statustext/recv
```

Angkat dan turunkan drone secara manual. Rangefinder harus berubah sesuai jarak
ke tanah, local pose tidak boleh melompat, dan setelah mission launch dengan
`auto_start:=false` status terrain harus `TRACKING`:

```bash
ros2 topic echo /control/terrain/status
ros2 topic echo /control/terrain/measured_agl
ros2 topic echo /control/terrain/target_z
```

## Menjalankan dalam urutan yang benar

Build workspace lebih dahulu. `bb_pcl_proc_node` bergantung pada `zed_msgs`.
Pada komputer ini source interface ZED berada di workspace PalmBee, sehingga
sertakan path tersebut saat build (pada Jetson, gunakan path instalasi/source
ZED yang sesuai):

```bash
cd /home/shane/ProjekAtaka/gazebo_sim
source /opt/ros/humble/setup.bash
colcon build --symlink-install --base-paths src \
  /home/shane/ProjekAtaka/palmbee_project/PalmBee/palmbee_ws/palmbee_ws/src/zed-ros2-interfaces
source install/setup.bash
```

Gunakan lima terminal terpisah. Setiap terminal harus men-source ROS dan
`install/setup.bash`. Jangan menjalankan proses yang sama dua kali.

Terminal 1 — MAVROS/APM (gunakan `fcu_url` yang sesuai perangkat):

```bash
ros2 launch mavros apm.launch fcu_url:=/dev/ttyACM0:921600
```

Terminal 2 — ZED2i dengan positional tracking dan custom object detection:

```bash
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i \
  ros_params_override_path:=/PATH/ABSOLUT/zed2i_pohon.yaml
```

File override tersebut wajib mengaktifkan `object_detection.od_enabled`, model
custom pohon, tracking, serta menghasilkan label persis `pohon`. Launch ZED2i
default menonaktifkan object detection, sehingga command tanpa override tidak
cukup untuk BB node.

Terminal 3 — auto-alignment dan bridge ExternalNav. Drone wajib diam, disarmed,
tracking ZED stabil, dan compass/EKF FC sudah aligned. Biarkan mengumpulkan
setidaknya 50 sampel:

```bash
source /opt/ros/humble/setup.bash
source /home/shane/ProjekAtaka/gazebo_sim/install/setup.bash
ros2 launch beehive_drone vision_to_mavros.launch.py
```

Transform `map <- zed_map_raw` dikunci sekali per penerbangan dan tidak dihitung
ulang ketika terbang. Verifikasi:

```bash
ros2 topic echo --once /alignment/ready
ros2 topic echo --once /alignment/yaw_offset_deg
ros2 run tf2_ros tf2_echo map zed_map_raw
```

Jangan arm bila `ready: false`. Jika pose mentah ZED meloncat lebih dari 1 m
atau yaw lebih dari 30 derajat antarsampel, alignment berhenti meneruskan pose,
`ready` menjadi false, dan FSM masuk `ABORT`. Land/takeover, lalu restart ZED,
alignment, mapper, dan misi; jangan kalibrasi ulang saat terbang.

Validasi sebelum membuka misi:

```bash
ros2 topic hz /zed/zed_node/pose
ros2 topic hz /zed/zed_node/obj_det/objects
ros2 topic echo --once /zed/zed_node/obj_det/objects
ros2 topic hz /mavros/vision_pose/pose
ros2 topic hz /mavros/local_position/pose
ros2 topic echo --once /mavros/state
ros2 topic echo --once /mavros/rangefinder/rangefinder
```

Pada pesan `ObjectsStamped`, cek `header.frame_id`, `label: pohon`, confidence,
posisi, dan seluruh corner `bounding_box_3d`. Tanpa pesan ini BB tidak dapat
membuat landmark.

Terminal 4 — konversi bounding box menjadi landmark global:

```bash
source /opt/ros/humble/setup.bash
source /home/shane/ProjekAtaka/gazebo_sim/install/setup.bash
ros2 launch point-cloud-test bb_proc_node.launch.py
```

Default BB sekarang memakai `/zed/aligned_pose`, sehingga landmark pohon dan
ExternalNav berada pada frame yang sama:

```bash
ros2 topic info /zed/aligned_pose -v
```

Validasi BB sebelum mission:

```bash
ros2 node info /bb_pcl_proc_node
ros2 param get /bb_pcl_proc_node object_label_target
ros2 topic echo --once /global_cylinders
```

Terminal 5 — mapper, controller, FSM, safety monitor, dan analyzer:

```bash
source /opt/ros/humble/setup.bash
source /home/shane/ProjekAtaka/gazebo_sim/install/setup.bash
ros2 launch beehive_drone real_mission.launch.py auto_start:=false
```

Saat pohon dipilih, FSM membekukan landmark, masuk `ALIGN_TO_TREE`, menahan XY,
dan mensyaratkan yaw error di bawah 7.5 derajat selama 1.5 detik. Setelah itu
barulah `APPROACH_TREE` berjalan pada slew maksimum 0.015 m per tick x 20 Hz =
0.30 m/s. Tanpa obstacle dalam safety zone, vortex controller meneruskan goal
final langsung; koreksi repulsive/vortex hanya aktif di dalam safety zone.

Jangan menjalankan `pcl_proc_node` atau `bb_pcl_proc_node` kedua. Pastikan
`/global_cylinders` berisi track dan `/map/trees` muncul sebelum mengharapkan
drone mengunci pohon. Selama validasi orientasi, gerakkan drone ke belakang,
kanan, lalu yaw; koordinat XY pohon harus tetap hampir konstan.

Setelah seluruh data sehat dan area aman, mulai misi:

```bash
ros2 topic pub --once /mission/start std_msgs/msg/Bool "{data: true}"
```

Periksa watchdog sebelum start:

```bash
ros2 topic echo --once /mission/safety_reason
ros2 topic echo --once /mission/safety_ok
```

Untuk pengujian yang memang menghendaki start otomatis:

```bash
ros2 launch beehive_drone real_mission.launch.py auto_start:=true
```

Perpindahan mode melalui RC menghentikan publikasi setpoint otomatis. Pilot
tetap harus siap mengambil alih melalui RC. Uji takeover pada ketinggian rendah
dan area terbuka sebelum menjalankan misi dekat pohon.

## Tahapan validasi lapangan

Keberhasilan Gazebo bukan jaminan drone nyata aman. Lakukan bertahap:

1. Propeller dilepas: validasi topic, frame, range, dan setpoint.
2. Tether/area terbuka: takeoff 1.5 m, hover, takeover, lalu land; tanpa
   menjalankan persepsi/approach pohon.
3. Gerak maju-mundur di permukaan datar sambil mengecek AGL dan local Z.
4. Lintasan lereng ringan tanpa pohon; target awal kecepatan rendah.
5. Approach dan orbit objek lunak/aman, baru kemudian pohon sawit.

Setelah setiap uji, periksa `mission_summary.json`,
`altitude_diagnostics.csv`, `diagnostic_events.csv`, dan rosbag. Tolak uji
berikutnya bila `tracking_availability_percent` tidak mendekati 100%, terdapat
range dropout, EKF/status text bermasalah, atau AGL error melampaui batas uji.

`mission_summary.json` juga berisi `flight_geometry`, `state_durations_s`, dan
`sensor_availability`. `flight_geometry` melaporkan jarak verifikasi approach,
statistik radius orbit, serta error akhir terhadap home.

Untuk menyimpan data ROS mentah bersamaan dengan analyzer:

```bash
ros2 bag record -o ~/beehive_bags/mission_$(date +%Y%m%d_%H%M%S) \
  /mavros/state /mavros/extended_state /mavros/statustext/recv \
  /mavros/local_position/pose /mavros/local_position/odom \
  /mavros/local_position/velocity_local /mavros/global_position/rel_alt \
  /mavros/rangefinder/rangefinder /mavros/vision_pose/pose \
  /zed/zed_node/pose /zed/zed_node/obj_det/objects \
  /perception/bb/cylinders /global_cylinders /map/trees \
  /mission/fsm_state \
  /control/terrain/measured_agl /control/terrain/agl_error \
  /control/terrain/target_z /control/terrain/status \
  /mavros/setpoint_position/local
```

Point cloud tidak diperlukan oleh BB node dan sengaja tidak dimasukkan ke
command default karena ukurannya besar. Tambahkan
`/zed/zed_node/point_cloud/cloud_registered` hanya untuk diagnosis depth bila
media penyimpanan dan bandwidth Jetson mencukupi.

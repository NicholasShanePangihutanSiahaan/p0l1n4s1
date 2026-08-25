# Real-flight checklist

`real_mission.launch.py` menjalankan PCL dan seluruh node misi, tetapi tidak
menjalankan MAVROS, ZED wrapper, atau `vision_to_mavros`. Ketiga bagian itu
harus hidup lebih dahulu. Default launch saat ini adalah
`auto_start:=true`; untuk uji lapangan bertahap selalu override menjadi
`auto_start:=false` sampai seluruh pemeriksaan pra-terbang lulus.

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

1. Isi `camera_offset_*` dan `camera_mount_*` pada `config/real.yaml`
   berdasarkan pemasangan ZED2i terhadap `base_link` (meter dan radian).
2. Pastikan `header.frame_id` point cloud benar dan pohon global tidak berpindah
   ketika drone digeser atau diputar secara manual.
3. Pastikan `/mavros/rangefinder/rangefinder` bertipe `sensor_msgs/msg/Range`,
   finite, lebih besar dari `rangefinder_min_valid`, dan frekuensinya stabil.
   Nilai maksimum sensor/parameter FC harus lebih tinggi dari 1.5 m dengan
   margin memadai; jangan terbang bila sensor berhenti pada nilai maksimum.
4. Tuning batas radius/tinggi batang, orbit, dan slew limit pada `real.yaml`
   dengan propeller dilepas terlebih dahulu.
5. Pastikan hanya satu node yang menerbitkan setpoint MAVROS. Launch nyata
   menjalankan `position_setpoint_controller`; jangan jalankan
   `velocity_controller` pada saat yang sama.
6. Pastikan estimator Pixhawk sudah menerima VisualOdom dan topic
   `/mavros/local_position/pose` stabil sebelum mission launch dijalankan.
   FSM juga menunggu `/mavros/vision_pose/pose` kontinu selama 5 detik. Gate
   ini mencegah `auto_start` dimulai dari satu sampel vision pertama, tetapi
   tidak dapat membuktikan bahwa EKF telah menerima fusion; pesan pre-arm FC
   tetap harus bersih.
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

Gunakan terminal terpisah. Jangan menjalankan `vision_to_mavros` kedua kali
karena `real_mission.launch.py` sengaja tidak lagi memuat node tersebut.

Terminal 1 — MAVROS/APM (gunakan `fcu_url` yang sesuai perangkat):

```bash
ros2 launch mavros apm.launch fcu_url:=/dev/ttyACM0:921600
```

Terminal 2 — ZED2i:

```bash
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
```

Terminal 3 — bridge ExternalNav, lalu biarkan mengalir setidaknya 5 detik:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch beehive_drone vision_to_mavros.launch.py
```

Validasi sebelum membuka misi:

```bash
ros2 topic hz /mavros/vision_pose/pose
ros2 topic hz /mavros/local_position/pose
ros2 topic echo --once /mavros/state
ros2 topic echo --once /mavros/rangefinder/rangefinder
```

Terminal 4 — PCL, mapper, controller, FSM, dan analyzer:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch beehive_drone real_mission.launch.py auto_start:=false
```

PCL sudah dimuat oleh `real_mission.launch.py` dengan cloud ZED dan odometri
MAVROS. Jangan menjalankan `pcl_proc_node` terpisah pada saat yang sama.
Pastikan `/global_cylinders` dan `/map/trees` muncul sebelum mengharapkan drone
mengunci pohon.

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
2. Tether/area terbuka: takeoff 1.5 m, hover, takeover, lalu land; tanpa PCL.
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
  /zed/zed_node/pose /map/trees /mission/fsm_state \
  /control/terrain/measured_agl /control/terrain/agl_error \
  /control/terrain/target_z /control/terrain/status \
  /mavros/setpoint_position/local
```

Point cloud sengaja tidak dimasukkan ke command default karena ukurannya besar.
Tambahkan `/zed/zed_node/point_cloud/cloud_registered` hanya bila media
penyimpanan dan bandwidth Jetson mencukupi.

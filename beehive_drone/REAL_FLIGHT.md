# Real-flight checklist

`real_mission.launch.py` tidak menjalankan MAVROS atau ZED wrapper dan tidak
otomatis memulai misi. Jalankan keduanya lebih dahulu.

## Kalibrasi yang wajib

1. Isi `camera_offset_*` dan `camera_mount_*` pada `config/real.yaml`
   berdasarkan pemasangan ZED2i terhadap `base_link` (meter dan radian).
2. Pastikan `header.frame_id` point cloud benar dan pohon global tidak berpindah
   ketika drone digeser atau diputar secara manual.
3. Pastikan `/mavros/rangefinder/rangefinder` bertipe `sensor_msgs/msg/Range`,
   finite, dan frekuensinya stabil.
4. Tuning batas radius/tinggi batang, orbit, dan slew limit pada `real.yaml`
   dengan propeller dilepas terlebih dahulu.
5. Pastikan hanya satu node yang menerbitkan setpoint MAVROS. Launch nyata
   menjalankan `position_setpoint_controller`; jangan jalankan
   `velocity_controller` pada saat yang sama.
6. Pastikan estimator Pixhawk sudah menerima VisualOdom dan topic
   `/mavros/local_position/pose` stabil sebelum mission launch dijalankan.

## Menjalankan

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch beehive_drone real_mission.launch.py
```

Periksa watchdog sebelum start:

```bash
ros2 topic echo --once /mission/safety_reason
ros2 topic echo --once /mission/safety_ok
```

Misi hanya dimulai oleh perintah eksplisit berikut:

```bash
ros2 topic pub --once /mission/start std_msgs/msg/Bool "{data: true}"
```

Untuk pengujian terkendali yang memang menghendaki start otomatis setelah
watchdog sehat:

```bash
ros2 launch beehive_drone real_mission.launch.py auto_start:=true
```

Perpindahan mode melalui RC menghentikan publikasi setpoint otomatis. Watchdog
sensor saat misi aktif meminta mode `BRAKE`; pilot tetap harus siap mengambil
alih melalui RC.

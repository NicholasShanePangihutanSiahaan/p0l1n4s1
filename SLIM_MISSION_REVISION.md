# Revisi Mission Slim

## Penyebab log lama

`flight_manager` hanya menulis "takeoff ditolak" dan membuang `response.result`, sehingga alasan MAVLink tidak terlihat. Revisi ini menerbitkan:

- `/flight/response/takeoff_success`
- `/flight/response/takeoff_result`
- `/flight/response/takeoff_detail`

State machine lama tetap masuk `TAKEOFF` walaupun command ditolak, lalu setelah 20 detik mengirim `LAND`. Revisi baru tidak melakukan itu.

## Arsitektur baru

Node kontrol penerbangan utama:

1. `vision_to_mavros`
2. `flight_manager`
3. `simple_single_tree_mission`
4. `pcl_proc_node` dan `pcl_tree_mapper` hanya ketika persepsi pohon dibutuhkan

Tidak digunakan dalam `mission_slim.launch.py`:

- `dynamic_orbit_controller`
- `vortex_avoidance_controller`
- `velocity_controller`

Gerak dikirim langsung ke `/mavros/setpoint_position/local`, sama seperti node pengujian yang sudah berhasil.

## Safety yang dipertahankan

- Perubahan mode RC keluar dari `GUIDED` langsung menghentikan setpoint.
- Setelah takeover, misi tidak aktif kembali sampai node direstart.
- Hanya satu publisher setpoint posisi.
- Pose stale menghentikan pengiriman target baru.
- Jika takeoff ditolak saat masih dekat tanah, program meminta DISARM, bukan LAND.
- Jika kendaraan benar-benar sudah terangkat, fallback tetap LAND.

## Perubahan penting pada paket ini

- `mission.launch.py` sekarang identik dengan launch slim. Perintah lama tidak lagi menjalankan controller berlapis.
- `mission_slim.launch.py` tetap tersedia sebagai nama eksplisit.
- Auto-disarm saat takeoff gagal hanya dilakukan bila FCU menyatakan `ON_GROUND`, relative altitude dekat nol, dan local Z juga dekat titik awal.
- Hanya hasil `TEMPORARILY_REJECTED`/timeout yang dicoba ulang. `DENIED`, `UNSUPPORTED`, dan `FAILED` langsung masuk abort.

## Build

```bash
cd ~/polinasi_control_core
colcon build --symlink-install --packages-select beehive_drone
source install/setup.bash
```

## Uji 1: takeoff dan hold saja

```bash
ros2 launch beehive_drone mission_slim.launch.py \
  hold_after_takeoff:=true \
  use_vslam_bridge:=true \
  use_pcl:=false \
  use_analyzer:=false \
  zed_pose_topic:=/zed/zed_node/pose
```

Pantau hasil command:

```bash
ros2 topic echo /flight/response/takeoff_detail
ros2 topic echo /mavros/statustext/recv
ros2 topic echo /mission/fsm_state
```

## Uji 2: misi satu pohon

Aktifkan setelah uji takeoff/hold stabil dan `pcl_proc_node` tidak lagi error:

```bash
ros2 launch beehive_drone mission_slim.launch.py \
  hold_after_takeoff:=false \
  use_vslam_bridge:=true \
  use_pcl:=true \
  use_analyzer:=false \
  point_cloud_topic:=/zed/zed_node/point_cloud/cloud_registered \
  zed_pose_topic:=/zed/zed_node/pose \
  odom_topic:=/mavros/odometry/out
```

## Pastikan publisher tidak ganda

```bash
ros2 topic info /mavros/setpoint_position/local --verbose
ros2 topic info /mavros/setpoint_velocity/cmd_vel --verbose
```

Untuk launch slim, publisher posisi seharusnya hanya `simple_single_tree_mission`, dan tidak ada publisher velocity dari paket misi.

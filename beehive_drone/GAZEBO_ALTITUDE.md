# Uji altitude X → Y → X di Gazebo

Gunakan source misi terbaru dari `/home/shane/polinasi`, bukan salinan lama
`beehive_drone` di workspace Gazebo. Panduan ini memakai environment Gazebo/SITL
yang sudah tersedia di `/home/shane/ProjekAtaka/gazebo_sim`.

`config/sim.yaml` sama dengan real untuk altitude: X (`flight_altitude`) = 1.9 m,
Y (`pre_orbit_altitude`) = 2.0 m terhadap home, toleransi 0.15 m selama 1.5 detik.
Controller mengikuti target Z dari FSM. Untuk melihat
perubahan lebih jelas, boleh ubah X ke 1.5 dan Y ke 2.0 sebelum build; samakan
`mission_analyzer.expected_takeoff_altitude` dengan X.

## 1. Build source terbaru

```bash
cd /home/shane/polinasi
source /opt/ros/humble/setup.bash
source /home/shane/ProjekAtaka/gazebo_sim/install/setup.bash
colcon build --symlink-install --packages-select beehive_drone
```

## 2. Terminal 1: Gazebo

```bash
source /opt/ros/humble/setup.bash
source /home/shane/ProjekAtaka/gazebo_sim/install/setup.bash
ros2 launch uav_plantation_sim plantation_sim.launch.py world:=plantation.sdf
```

## 3. Terminal 2: SITL

```bash
cd /home/shane/ProjekAtaka/gazebo_sim
sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON -w \
  --add-param-file=/home/shane/ProjekAtaka/gazebo_sim/src/beehive_drone/config/sitl_sim.parm \
  --console --out=udp:127.0.0.1:14551
```

`-w` mereset EEPROM simulasi. Profil `sitl_sim.parm` memakai AHRS simulasi;
pengujian ini untuk alur misi, bukan validasi tuning EKF drone nyata.

## 4. Terminal 3: MAVROS

```bash
source /opt/ros/humble/setup.bash
source /home/shane/ProjekAtaka/gazebo_sim/install/setup.bash
ros2 launch mavros apm.launch fcu_url:=udp://127.0.0.1:14551@
```

## 5. Terminal 4: misi dari repo polinasi

```bash
source /opt/ros/humble/setup.bash
source /home/shane/ProjekAtaka/gazebo_sim/install/setup.bash
source /home/shane/polinasi/install/setup.bash
ros2 pkg prefix beehive_drone
ros2 launch beehive_drone simulation_mission.launch.py \
  auto_start:=false mission_type:=basic_orbit mission_mode:=single_tree \
  odometry_topic:=/mavros/odometry/out
```

Prefix `beehive_drone` harus menunjuk ke install `/home/shane/polinasi`.
Launch ini menggunakan segmentasi point cloud, adapter rangefinder, dan dummy
sprayer. Jangan jalankan misi lain atau `gazebo_stack.launch.py` bersamaan.
World Gazebo ini menyediakan odometry sintetis di `/mavros/odometry/out`.
Jika environment lain menyediakan `/simulation/ground_truth/odom`, ubah argumen
`odometry_topic` ke topic tersebut. Periksa topic benar-benar menerbitkan data.

## 6. Terminal 5: periksa dan mulai

```bash
source /opt/ros/humble/setup.bash
source /home/shane/ProjekAtaka/gazebo_sim/install/setup.bash
source /home/shane/polinasi/install/setup.bash
ros2 topic echo --once /mavros/state
ros2 topic echo --once /simulation/local_position/pose
ros2 topic echo --once /simulation/rangefinder
ros2 topic echo --once /map/trees
ros2 param get /mission_state_machine pre_orbit_altitude
ros2 topic pub --once /mission/start std_msgs/msg/Bool '{data: true}'
```

Tunggu MAVROS `connected: true`, pose/range valid, dan pohon terpetakan.
Y harus `2.0`.

## 7. Amati fase altitude

Di terminal terpisah, source environment yang sama, lalu jalankan salah satu:

```bash
ros2 topic echo /mission/fsm_state
ros2 topic echo /mavros/setpoint_position/local --field pose.position.z
ros2 topic echo /simulation/local_position/pose --field pose.position.z
```

Urutan yang diharapkan:

- Takeoff X → approach → `VERIFY_TREE`: target Z hasil takeoff tetap.
- `PRE_ORBIT_ALTITUDE`: XY ditahan, target Z menjadi home Z + Y.
- Setelah error Z dalam toleransi selama 1.5 detik: `START_ORBIT` → `WAIT_ORBIT`.
- Orbit tetap mengirim target Y tanpa gate altitude tambahan di FSM.
- `POST_ORBIT_HOVER` → `RESTORE_FLIGHT_ALTITUDE`: kembali ke Z hasil takeoff.
- Setelah stabil: `ALIGN_HOME` → `RETURN_TO_HOME` → `HOME_HOVER` → `LANDING`.

Bandingkan Z terhadap home Z; angka Z absolut tidak harus sama dengan X/Y.
Laporan analyzer disimpan di `~/beehive_mission_reports/sim`.

## Memilih mode terrain

Ubah `altitude_mode` di blok `/**/ros__parameters` pada `config/sim.yaml`
menjadi `terrain_following`, build ulang, lalu restart launch misi. Untuk
perilaku semula gunakan `home_relative`. Verifikasi kedua node memakai mode
yang sama:

```bash
ros2 param get /mission_state_machine altitude_mode
ros2 param get /position_setpoint_controller altitude_mode
ros2 topic echo /control/target_agl
ros2 topic echo /control/measured_agl
```

Mode terrain menggunakan X/Y sebagai AGL; mode home_relative menggunakan
referensi home. Hindari restart FSM ketika drone masih melayang karena home
baru akan direkam dari posisi itu.

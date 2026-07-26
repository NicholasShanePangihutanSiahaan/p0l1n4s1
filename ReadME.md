# Autonomous Plantation Drone Navigation System (`beehive_drone`)

Sistem navigasi otonom berbasis ROS 2 (Humble/Iron/Jazzy) untuk drone perkebunan sawit. Sistem ini dirancang untuk berjalan pada *companion computer* (seperti Jetson Orin Nano Super) yang terintegrasi dengan kamera stereo ZED 2i dan *flight controller* Pixhawk 6C (ArduPilot) via MAVROS.

---

## 📂 Struktur Direktori Workspace (`polinasi/`)

```text
polinasi/
├── beehive_drone/                # Paket ROS 2 Utama
│   ├── beehive_drone/            # Modul Skrip Python (Nodes)
│   │   ├── __init__.py
│   │   ├── dynamic_orbit_controller.py
│   │   ├── flight_manager.py
│   │   ├── mission_analyzer.py
│   │   ├── mission_params.py
│   │   ├── mission_state_machine.py
│   │   ├── tree_detector.py
│   │   ├── tree_localizer.py
│   │   ├── tree_mapper.py
│   │   ├── velocity_controller.py
│   │   ├── vision_to_mavros.py
│   │   └── vortex_avoidance_controller.py
│   ├── launch/                   # Folder Launch File ROS 2
│   │   └── system.launch.py      # (Contoh nama file launch utama)
│   ├── resource/
│   ├── test/
│   ├── package.xml
│   ├── setup.cfg
│   └── setup.py
└── uav_interfaces/               # Paket Custom Custom Messages ROS 2
    ├── include/
    ├── msg/
    │   ├── InspectionTarget.msg
    │   ├── Tree.msg
    │   └── TreeArray.msg
    ├── CMakeLists.txt
    ├── package.xml
    └── src/
```

# Penjelasan Modul Node

1. `mission_state_machine.py` (*The Brain*): Mengatur FSM (Finite State Machine) utama misi, mulai dari inisialisasi, takeoff, menyusuri lorong (explore row), verifikasi pohon (anti-pohon hantu), orbit, hingga kembali ke home (RTH) dan pendaratan.

2. `flight_manager.py` (*Hardware Abstraction Layer*): Menjembatani perintah tingkat tinggi dari FSM (arming, ganti mode, takeoff, land) menjadi layanan MAVROS ke Pixhawk, sekaligus mempublikasikan telemetri status.

3. `tree_detector.py` (*Persepsi Visual*): Memproses gambar RGB (HSV thresholding) dan depth map dari kamera ZED 2i untuk mendeteksi posisi piksel pohon.

4. `tree_localizer.py` (*Lokalisasi 3D*): Mengubah koordinat piksel dan kedalaman kamera menjadi koordinat global 3D dunia (world frame / `odom`) berdasarkan posisi dan orientasi yaw drone.

5. `tree_mapper.py` (*Database Spasial*): Mengelola database peta pohon, menggabungkan deteksi yang berdekatan (`merge_distance`), menghitung tingkat confidence, serta menangani penghapusan pohon hantu.

6. `dynamic_orbit_controller.py` (*Kontrol Orbit*): Menghitung lintasan melingkar 360 derajat di sekeliling pohon target menggunakan teknik lookahead serta memberikan offset orientasi kamera 45 derajat.

7. `vortex_avoidance_controller.py` (*Penghindaran Rintangan*): Menerapkan Medan Potensial Artifisial (Artificial Potential Field) untuk memadukan gaya tarik tujuan dengan gaya tolak dan pusaran (vortex) guna menghindari rintangan secara mulus.

8. `velocity_controller.py` (*Kontrol Kecepatan Rendah*): Menerjemahkan target posisi aman menjadi komando kecepatan linear dan angular (`TwistStamped`) yang dikirim ke MAVROS.

9. `vision_to_mavros.py` (*Jembatan VSLAM*): Mengirimkan data pose Visual Odometry dari kamera ZED langsung ke ArduPilot (`/mavros/vision_pose/pose`).

10. `mission_analyzer.py` (*Logging & Evaluasi*): Merekam data lintasan dan statistik inspeksi secara real-time, serta otomatis menghasilkan laporan file CSV dan peta visual PNG saat misi selesai.

11. `mission_params.py` (*Pusat Konfigurasi*): File konfigurasi global untuk menyetel kecepatan, radius orbit, gain kontroler, dan parameter batas aman lainnya.



# Alur Kerja Sistem (Workflow)

1. *Pre-Arm & Takeoff*: FlightManager meminta mode `GUIDED` ke Pixhawk -> Melakukan Arming -> Takeoff ke ketinggian target (`FLIGHT_ALTITUDE = 3.0m`).

2. *Eksplorasi & Deteksi*: Drone maju menyusuri lorong perkebunan (EXPLORE_ROW). Kamera ZED 2i mendeteksi pohon via `tree_detector.py`, dilokalisasikan oleh `tree_localizer.py`, dan dicatat ke dalam peta oleh `tree_mapper.py`.

3. *Pendekatan & Verifikasi*: Ketika pohon tak terinspeksi ditemukan, FSM memerintahkan drone mendekat ke titik pengereman (APPROACH_TREE), lalu melakukan hovering selama 4 detik guna memverifikasi apakah pohon tersebut valid.

4. *Orbit & Inspeksi*: Jika valid, dynamic_orbit_controller mengambil alih untuk memutar drone 360 derajat mengitari pohon dengan yaw offset 45 derajat. Setelah selesai, pohon ditandai inspected = True di mapper.

5. *Navigasi Lorong & RTH*: Setelah lorong habis (`END_OF_ROW`), drone bergeser menyamping (`CRAB_SCAN`) ke lorong berikutnya. Jika seluruh lahan selesai dieksplorasi, drone kembali ke titik awal (`RETURN_TO_HOME`), melakukan spin akhir, lalu mendarat otomatis (`LANDING`).

# Cara Menjalankan Program
Prasyarat

- Pastikan ROS 2 (Humble/Iron/Jazzy) sudah terinstal bersama workspace yang aktif.

- Pastikan paket pengendali MAVROS dan driver ZED ROS 2 (`zed_wrapper`) sudah terpasang dan berjalan di sistem.

## 1. Kompilasi Workspace (Build)

Buka terminal, arahkan ke direktori root workspace (`polinasi/`), lalu build paket menggunakan colcon:
```Bash

cd ~/polinasi
colcon build --symlink-install
```
## 2. Sumber Environment (Source)

Aktifkan overlay workspace ke dalam terminal Anda:
```Bash

source install/setup.bash
```
## 3. Menjalankan Komunikasi MAVROS & Sensor ZED (Eksternal)

Sebelum menjalankan node misi, pastikan jembatan komunikasi ke flight controller dan kamera sudah aktif di terminal terpisah:

- Menyalakan MAVROS (Hubungkan ke Pixhawk 6C):
```Bash

ros2 launch mavros apm.launch fcu_url:=/dev/ttyACM0:57600
```

- (Opsional) Jika ingin dihubungkan dengan Ground Control System (GCS) seperti Mission Planner atau QGroundControl, dapat menambahkan gcs_url:
```Bash
ros2 launch mavros apm.launch fcu_url:=/dev/ttyACM0:115200 gcs_url:="udp://@(IP_LAPTOP):14550"
```
- Menyalakan Driver ZED 2i:
```Bash

ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
```

- Menyalakan Jembatan VSLAM
```Bash
ros2 run beehive_drone vision_to_mavros
```
## 4. Menjalankan Seluruh Misi Menggunakan Launch File

Karena seluruh node (termasuk Flight Manager, FSM, kontroler, pemetaan, dan analyzer) sudah didaftarkan ke dalam skrip launch, Anda cukup menjalankan satu perintah ini untuk menginisialisasi seluruh sistem secara bersamaan:
```Bash

ros2 launch beehive_drone mission.launch.py
```
## 5. Hasil Output Evaluasi Misi

Ketika misi selesai atau node `mission_analyzer` dihentikan (shutdown via `Ctrl+C`), program akan secara otomatis mengonversi data penerbangan dan menyimpannya di direktori aktif berupa:

- File Log Statistik & Trajektori: `mission_result_YYYYMMDD_HHMMSS.csv`

- File Gambar Visualisasi Peta Misi: `mission_map_YYYYMMDD_HHMMSS.png`
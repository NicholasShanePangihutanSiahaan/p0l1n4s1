# Integrasi eksperimental orbit dengan ketinggian adaptif

Program ini merupakan hasil integrasi dan belum dinyatakan siap untuk penerbangan
nyata. Pengaturan bawaan `real_mission` dan `basic_orbit` tidak diubah.
Strategi baru `mission_type: adaptive_orbit` menggunakan fase
`PRE_ORBIT_ALTITUDE` dan `RESTORE_FLIGHT_ALTITUDE` yang sudah ada di `polinasi`.
Strategi ini memerlukan mode `home_relative` serta posisi lokal FC dari estimasi
lokalisasi nyata. Launch ini tidak menjalankan penerbit odometri simulasi.

## Pemeriksaan awal di darat

Jalankan tanpa memulai misi atau melakukan arming otomatis:

```bash
ros2 launch beehive_drone real_adaptive_orbit.launch.py \
  lidar_topic:=TOPIK_POINTCLOUD_LIDAR auto_start:=false
```

Jangan mengirim perintah `/mission/start` selama pemeriksaan awal. Pastikan:

- `/mavros/local_position/pose` berisi estimasi posisi nyata yang masih baru.
- Transformasi TF dari `map` ke frame LiDAR tersedia pada waktu pengambilan
  scan. Arah sumbu, titik nol, skala meter, dan koordinat kandidat kamera harus
  sesuai dengan koordinat lokal FC.
- Posisi dan orientasi pemasangan LiDAR telah diukur. Launch `real_mission`
  sebelumnya menggunakan translasi nol dan sudut pitch 0,349 radian. Nilai
  tersebut dipertahankan, tetapi belum divalidasi. Dua penerbit TF statisnya
  juga memakai nama node yang sama; periksa TF saat program berjalan.
- `/mavros/rangefinder/rangefinder` memberikan pembacaan yang valid dan baru.
- Argumen `lidar_topic` sesuai dengan topik `PointCloud2` dari driver LiDAR
  sebenarnya, bukan topik yang diasumsikan.

## Alur penyesuaian ketinggian

1. Kamera mencari kandidat batang dan drone mendekatinya.
2. Pada `VERIFY_TREE`, LiDAR memverifikasi bentuk batang, mengoreksi pusatnya,
   dan memperkirakan tinggi bagian batang yang terlihat.
3. Hasil LiDAR harus masih baru, disertai pembacaan sensor jarak bawah yang
   masih baru dan minimal 0,8 meter. Syarat tersebut harus bertahan selama
   satu detik sebelum melanjutkan fase penyesuaian ketinggian.
4. Jika validasi tidak berhasil dalam delapan detik, misi masuk `ABORT`.
   Program tidak mengganti hasil yang gagal dengan estimasi tinggi kamera.
5. Target Z dihitung dari estimasi Z bawah batang ditambah setengah tinggi
   batang teramati. Ketinggian relatif terhadap bagian bawah batang dibatasi
   1–3 meter pada profil uji awal ini.
6. Perubahan target dari Z lokal drone saat itu dibatasi: turun maksimal
   1 meter atau naik maksimal 1,5 meter. Target di luar batas menyebabkan
   `ABORT`.
7. Fase `PRE_ORBIT_ALTITUDE` menahan posisi horizontal dan menunggu ketinggian
   sesuai syarat stabilitas. Selama orbit, target Z lokal dipertahankan tetap.
8. Setelah orbit, strategi yang sudah ada memulihkan ketinggian penerbangan
   melalui fase `RESTORE_FLIGHT_ALTITUDE`.

Estimasi ini adalah tinggi segmen batang yang teramati, bukan tinggi seluruh
pohon atau jaminan bahwa bagian bawah segmen merupakan pangkal batang.

Urutan pengujian: periksa sensor dan TF di darat, uji menggunakan rekaman data
atau pengujian di darat, kemudian lakukan uji satu pohon yang dibatasi dengan
kendali manual tersedia. Integrasi ini belum diuji terbang pada drone nyata.

## Mengaktifkan PRIEST melalui YAML

Edit `config/real_adaptive.yaml`:

```yaml
real_adaptive_orbit_launch:
  ros__parameters:
    use_priest: true
```

Nilai bawaan tetap `false`. Launch membaca bagian tersebut saat mulai, lalu
meneruskan nilainya ke `real_mission.launch.py`. Nilai `true` menjalankan
`priest_dyn_node` dan `priest_avoidance_controller`, serta menonaktifkan vortex.
Opsi terminal mengalahkan nilai YAML untuk satu kali pemanggilan:

```bash
ros2 launch beehive_drone real_adaptive_orbit.launch.py \
  lidar_topic:=TOPIK_POINTCLOUD_LIDAR_ANDA use_priest:=true auto_start:=false
```

Tanpa opsi terminal `use_priest`, nilai YAML digunakan. Setelah mengubah YAML,
jalankan ulang launch. Jika instalasi tidak memakai symlink, bangun ulang paket
dan muat hasil build dengan `source install/setup.bash`.

Launch `real_mission` biasa tidak membaca opsi launch khusus ini. Gunakan
`real_adaptive_orbit` untuk memakai pengaturan bawaan dari bagian YAML tersebut.

### Pembagian tugas

- Kamera mencari kandidat batang. Verifier LiDAR mengoreksi posisi dan tinggi
  batang teramati sebelum orbit.
- FSM adaptive menentukan target ketinggian dan menggunakan fase penyesuaian
  ketinggian milik `polinasi` sebelum dan sesudah orbit.
- PRIEST menerima target navigasi dan lintasan referensi orbit pada
  `WAIT_ORBIT`. PRIEST tidak hanya digunakan saat mendekati pohon.
- Pengendali orbit tetap menyediakan referensi orbit dan yaw. Keluaran
  lintasan PRIEST digunakan melalui `priest_avoidance_controller`.

### Parameter yang perlu diperiksa di real_adaptive.yaml

| Bagian / parameter | Nilai awal | Makna |
| --- | --- | --- |
| `priest_dyn_node.device` | `cuda` | Perangkat komputasi yang harus tersedia |
| `v_max` / `v_des` | 0,9 / 0,55 | Batas dan kecepatan referensi (m/s) |
| `a_max` | 1,3 | Batas percepatan (m/s²) |
| `robot_radius` / `safety_margin` | 0,4 / 0,2 | Ukuran dan margin drone (m) |
| `plan_rate_hz` | 1,0 | Frekuensi perencanaan (Hz) |
| `horizon_seconds` | 2,0 | Rentang waktu perencanaan (detik) |
| `odom_timeout` / `obstacle_timeout` | 20,0 / 20,0 | Batas umur data masukan (detik) |
| `max_solve_age` | 30,0 | Batas umur solusi (detik) |
| `require_obstacles` | `false` | Perencana dapat berjalan tanpa data rintangan |
| `use_safety` | `false` | Pemeriksaan keselamatan internal perencana belum aktif |
| `priest_avoidance_controller.path_timeout` | 2,0 | Batas umur lintasan (detik) |
| `priest_avoidance_controller.orbit_radius` | 2,7 | Radius referensi orbit (m) |

Angka desimal pada tabel menggunakan penulisan Indonesia. Pada YAML, tetap
pakai titik, misalnya `v_max: 0.9`.

Nilai di atas adalah konfigurasi awal yang diwarisi, bukan rekomendasi siap
terbang. Selaraskan radius orbit dengan `dynamic_orbit_controller`, ukuran
drone, kemampuan komputasi, frame `map`, dan kesegaran sensor. Jangan hanya
melonggarkan batas waktu ketika perencana lambat.

### Memastikan data rintangan masuk ke PRIEST

PRIEST membaca `/ellipsoids`. Pada adapter `priest_avoidance_controller` saat
integrasi ini dibuat, subscription `/global_cylinders` dan publisher
`/ellipsoids` dikomentari. Karena itu, pastikan ada penerbit lain yang
menyediakan data rintangan.

Mengaktifkan PRIEST tidak otomatis menjamin data rintangan tersedia atau
penghindaran rintangan bekerja. Verifier LiDAR tidak menerbitkan `/ellipsoids`.

```bash
ros2 topic info /ellipsoids -v
ros2 topic hz /ellipsoids
ros2 topic hz /navigation/priest/trajectory/out
ros2 node list | rg 'priest|vortex'
```

Saat PRIEST aktif, seharusnya kedua node PRIEST muncul dan node vortex dari
launch yang sama tidak dijalankan. Periksa juga `/mavros/local_position/pose`,
TF sensor ke `map`, dan rangefinder. Jangan menjalankan dua stack misi bersamaan.

Kombinasi adaptive altitude dan PRIEST belum diuji terbang nyata. `auto_start`
tetap `false`; pemeriksaan launch atau perubahan YAML tidak mengirim perintah
arming.

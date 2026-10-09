# PRIEST 3D local path planner.

Node ini mengambil input goal atau reference trajectory yang nantinya akan dimodifikasi menjadi hasilnya, 
yaitu local reference path yang menghindari segala jenis obstacle yang diproses dari data point cloud.

Message dependencies package ini:
- **pcl_cstm_msg** (Custom package dari **pcl-segment-sawit**)
- **nav_msgs**
- **geometry_msgs**

Library dependencies package ini:
- numpy
- JAX

**Note:** jika ingin pakai GPU maka harus compile atau install JAX untuk GPU. Disarankan karena biar proses kalkulasi trajectory lebih cepat

Topic yang disubscribe:
- **/mission/fsm_state** (std_msgs/string):\
  State dari drone sekarang, berpengaruh referensi goal mana yang dipakai drone nya nanti

- **/mavros/local_position/odom** (nav_msgs/Odom):\
  Update posisi drone paling terbaru

- **/navigation/priest/linear_goal** (geometry_msgs/Pose):\
  Satu posisi goal yang diinginkan oleh drone tanpa ada referensi sehingga hanya gerak lurus saja

- **/navigation/priest/trajectory/in** (nav_msgs/Path):\
  Trajectory reference yang dipakai oleh algroritma PRIEST untuk diikuti contohnya pada saat gerak orbit drone.

- **/ellipsoids** (pcl_cstm_msg/AxisAlignedElipsoidArray):\
  Kumpulan obstacle dari hasil filter point cloud dibentuk menjadi ellipsoids
  
Topic yang dipublish:
- **/navigation/priest/trajectory/out** (nav_msgs/Path):\
  Trajectory yang dikoreksi agar tidak bertabrakan dengan obstacle.

## Pengujian
PRIEST dapat divisualisasi dan dipelajari dahulu dengan menjalankan `ros2 launch priest_dyn_traj rviz_test_visualization.launch.py`. Dia bakal munculin visualisasi RVIZ dalam membuat trajectory. 

Untuk PRIEST ini ada dua jenis goal, berbasis referensi trajectory atau satu titik goal (trajectory linear). Bisa diuji dengan:
```
# Single goal pose 
ros2 launch priest_dyn_traj rviz_test_visualization.launch.py type:=linear

# Trajectory reference
ros2 launch priest_dyn_traj rviz_test_visualization.launch.py type:=by_ref_trajectory
```

### Tipe Linear
Untuk `type:=linear` node `priest_dyn_node` akan menggunakan goal dari subscribe `/navigation/priest/linear_goal` yang bertipe **(geometry_msgs/Pose)** dimana mengambil posisi awal dari local odometry drone dan posisi akhir di posisi yang disubscribe dari `/navigation/priest/linear_goal`

`priest_dyn_node` akan publish **(nav_msgs/Path)** ke `/navigation/priest/trajectory/out` dimana hasilnya mendekati garis lurus antara posisi odom sekarang dan posisi `linear_goal`. Path tidak akan sampai ke goal jika goal terlalu jauh dari posisi odom sekarang. Jadinya dia hanya menghitung path lokal dalam window waktu tertentu.

### Tipe Reference Trajectory
Untuk `type:=by_ref_trajectory` node `priest_dyn_node` akan menggunakan goal dari subscribe `/navigation/priest/trajectory/in` yang bertipe **(nav_msgs/Path)** dimana mengambil posisi awal dari local odometry drone dan path reference dari `/navigation/priest/trajectory/in`.

`priest_dyn_node` akan publish **(nav_msgs/Path)** ke `/navigation/priest/trajectory/out` dimana hasil path akan mendekati `/navigation/priest/trajectory/in` dan terhindar dari obstacle. Path tidak sepenuhnya lengkap seperti `/navigation/priest/trajectory/in` karena Path yang dihasilkan hanya mengambil segmen local atau lookahead goal yang diatur oleh parameter di `config/priest_node_config.yaml`:

```
priest_dyn_node:
  ros__parameters:
    ...

    goal_tolerance: 0.15

    # v_des * horizon_seconds = lookahead_distance pada trajectory
    horizon_seconds: 10.0 

    ...

    v_max: 0.90
    v_min: 0.0
    a_max: 1.30
    v_des: 0.55 # velocity yang diinginkan
    
    ... 
```

## GPU Acceleration
ada parameter di `config/priest_node_config.yaml` bernama `device: 'cpu'` bisa diubah ke `device: 'gpu'` misalnya sudah menginstall JAX untuk GPU.\
**NOTE**: JAX perlu waktu delay sebelum jalan tergantung device lama waktu delaynya, sebab optimisasi dari JAX ny sendiri
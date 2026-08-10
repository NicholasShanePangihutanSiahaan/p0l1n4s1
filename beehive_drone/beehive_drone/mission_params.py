#!/usr/bin/env python3

import math

class MissionConfig:
    """
    Pusat Konfigurasi Parameter UAV Plantation
    Ubah nilai-nilai di sini untuk melakukan tuning performa drone.
    """

    # ==========================================
    # 1. Parameter Misi & Eksplorasi (mission_state_machine.py)
    # ==========================================
    FLIGHT_ALTITUDE = 3.0        # meter
    EXPLORE_SPEED = 1.0          # m/s
    CRAB_SPEED = 0.5             # m/s
    END_OF_ROW_DIST = 10.0       # meter
    END_OF_FARM_DIST = 20.0      # meter
    APPROACH_SAFE_DIST = 2.0     # meter
    HOVERING_PERIODE = 30.0      # periode dikali 1.0 adalah waktu hovering sebelum loitering
    POST_ORBIT_HOVER_TIME = 3.0  # detik stabilisasi setelah satu orbit selesai
    HOME_ALIGN_TIME = 2.0        # detik mempertahankan arah menuju titik takeoff
    HOME_HOVER_TIME = 3.0        # detik hover di atas titik takeoff sebelum land
    HOME_POSITION_TOLERANCE = 0.7  # meter
    HOME_YAW_TOLERANCE = math.radians(10.0)
    MINIMUM_DISTANCE_TO_DELETE = 0.5  # meter; jarak minimum untuk menghapus pohon dari database
    MAXIMUM_DISTANCE_TO_DELETE = 2.5  # meter; jarak maksimum untuk menghapus pohon dari database

    # ==========================================
    # 2. Parameter Orbit (dynamic_orbit_controller.py)
    # ==========================================
    ORBIT_RADIUS = 3.0           # meter (Diperkecil dari 2.5 agar aman dari rintangan)
    ORBIT_ALTITUDE = FLIGHT_ALTITUDE 
    ORBIT_VELOCITY = 1.0         # m/s
    YAW_OFFSET = math.pi / 4     # radian (45 derajat)

    # ==========================================
    # 3. Parameter Penghindaran (vortex_avoidance_controller.py)
    # ==========================================
    SAFETY_RADIUS = 0.35          # meter (Diperkecil dari 1.5 agar luwes di celah sempit)
    REPULSIVE_GAIN = 0.25
    VORTEX_GAIN = 1.0
    ATTRACTION_GAIN = 0.5
    MAX_SHIFT = 1.5              # meter

    # ==========================================
    # 4. Parameter Kontrol (velocity_controller.py)
    # ==========================================
    KP_XY = 0.5                  # Gain Proposional Translasi
    KP_Z = 0.3                   # Gain Proposional Ketinggian
    KP_YAW = 0.8                 # Gain Proposional Rotasi
    MAX_VELOCITY_XY = 1.0        # m/s
    MAX_VELOCITY_Z = 0.5         # m/s
    MAX_VELOCITY_YAW = 0.5       # rad/s
    GOAL_THRESHOLD = 0.5         # meter

    # ==========================================
    # 5. Parameter Pemetaan Pohon (tree_mapper.py)
    # ==========================================
    TREE_MERGE_DISTANCE = 1.0           # meter; 6 m dapat menggabungkan dua pohon berbeda
    TREE_MAX_CONFIDENCE = 1.0           # Maksimum confidence untuk pohon
    TREE_NEW_CONFIDENCE = 0.2      # Confidence awal untuk pohon baru
    TREE_CONFIDENCE_INCREMENT = 0.25    # Penambahan confidence setiap deteksi
    TREE_CONFIDENCE_DECAY = 0.01         # Penurunan confidence setiap deteksi hilang
    TREE_TIMEOUT = 30.0                 # detik

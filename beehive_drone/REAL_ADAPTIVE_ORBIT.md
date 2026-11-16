# Experimental adaptive orbit integration

This is integration code, not certification of real flight readiness. Existing
real_mission and basic_orbit defaults are unchanged. New mission_type adaptive_orbit
uses the existing PRE_ORBIT_ALTITUDE / RESTORE_FLIGHT_ALTITUDE phases. It requires
home_relative mode and real FC local position, and never starts simulation odometry.

Bench launch (no automatic arming):
ros2 launch beehive_drone real_adaptive_orbit.launch.py lidar_topic:=ACTUAL_POINTCLOUD_TOPIC

Do not request /mission/start during bench validation. Before flight, verify:
- /mavros/local_position/pose is a fresh real localization estimate.
- TF map -> actual LiDAR frame at sensor acquisition timestamps agrees with FC
  local axes/origin, metric scale, and camera candidate coordinates.
- Actual measured LiDAR extrinsics are correct. Existing real_mission publishes
  mounts with zero translations and 0.349 rad pitch; these were preserved, not
  validated. Its two static publishers share the same node name; inspect runtime TF.
- /mavros/rangefinder/rangefinder is fresh and valid.
- Confirm the actual driver PointCloud2 topic; it is mandatory, not guessed.

During VERIFY_TREE, corrected center + visible trunk height require fresh LiDAR
geometry and a fresh downward range measurement >=0.8m, stable for 1s. Missing
verification aborts after 8s rather than enabling camera-height fallback. Initial
altitude delta bounds: -1.0m to +1.5m from current local Z; relative trunk target
clamped 1–3m. Orbit target stays fixed local Z. Existing strategy restores flight
altitude afterward. This estimates visible segment height, not true tree height.

First validate sensors and TF on the ground, then recorded-data/bench behavior,
then a bounded single-tree flight with manual takeover available. No real-drone
flight has been performed as part of this integration.

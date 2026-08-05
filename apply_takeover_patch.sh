#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/ros2_workspace"
  exit 2
fi

WS="$(realpath "$1")"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$WS/src/beehive_drone"

if [[ ! -d "$SRC/beehive_drone" ]]; then
  echo "Package beehive_drone tidak ditemukan di $SRC"
  exit 1
fi

for FILE in \
  mission_params.py \
  mission_state_machine.py \
  mission_state_machine_single_tree.py \
  dynamic_orbit_controller.py \
  vortex_avoidance_controller.py \
  velocity_controller.py
do
  cp "$HERE/beehive_drone/beehive_drone/$FILE" "$SRC/beehive_drone/$FILE"
done

cp "$HERE/beehive_drone/config/mission_real_pcl.yaml" \
   "$SRC/config/mission_real_pcl.yaml"

echo "Patch copied. Building beehive_drone..."
cd "$WS"
colcon build --symlink-install --packages-select beehive_drone

echo "Done. Run: source $WS/install/setup.bash"

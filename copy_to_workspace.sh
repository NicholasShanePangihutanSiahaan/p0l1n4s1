#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/ros2_workspace"
  exit 2
fi

WS="$(realpath "$1")"
SRC="$WS/src"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SRC"

for PKG in beehive_drone uav_interfaces pcl_cstm_msg point-cloud-test; do
  rm -rf "$SRC/$PKG"
  cp -a "$HERE/$PKG" "$SRC/$PKG"
done

echo "Packages copied to $SRC"
echo "Build with:"
echo "  cd $WS"
echo "  colcon build --symlink-install --packages-select uav_interfaces pcl_cstm_msg point-cloud-test beehive_drone"

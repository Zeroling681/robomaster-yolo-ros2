#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "请使用: source scripts/activate_wsl.sh"
  exit 1
fi

source /opt/ros/humble/setup.bash
source /home/tonyt/.venvs/robomaster/bin/activate
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

echo "已启用 ROS 2 ${ROS_DISTRO}、Python 环境 ${VIRTUAL_ENV}"


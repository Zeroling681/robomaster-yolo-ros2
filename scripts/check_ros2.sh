#!/usr/bin/env bash
set -eo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_dir}/scripts/activate_wsl.sh"
set -u

check_dir="$(mktemp -d)"
cleanup() {
  rm -rf -- "${check_dir}"
}
trap cleanup EXIT

timeout 12s ros2 topic echo /robomaster_env_check std_msgs/msg/String --once \
  >"${check_dir}/subscriber.log" 2>&1 &
subscriber_pid=$!
sleep 2

ros2 topic pub --once /robomaster_env_check std_msgs/msg/String \
  "{data: environment_ok}" >"${check_dir}/publisher.log" 2>&1

if ! wait "${subscriber_pid}"; then
  cat "${check_dir}/subscriber.log"
  echo "ROS 2 话题互通失败" >&2
  exit 1
fi

grep -q "environment_ok" "${check_dir}/subscriber.log"
echo "ROS2_PUB_SUB=PASS"

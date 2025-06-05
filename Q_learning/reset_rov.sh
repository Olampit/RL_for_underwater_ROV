#!/bin/bash

echo "Respawning robot..."

ros2 service call /stonefish_ros2/respawn_robot stonefish_ros2/srv/Respawn "{name: 'bluerov', origin: {position: {x: 0.0, y: 50.0, z: 10.0}, orientation: {x: 0.0, y: 0.0, z: 1.0, w: 1.0}}}"


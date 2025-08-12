#!/bin/bash

# File to store the commit counter
COUNTER_FILE=".commit_counter"

# Initialize counter file if it doesn't exist
if [ ! -f "$COUNTER_FILE" ]; then
  echo 1 > "$COUNTER_FILE"
fi

# Read the current commit number
count=$(cat "$COUNTER_FILE")

# Git commands
git add .
git commit --author="Olampit <olampit@gmail.com>" -m "no model changes but better results - fixed issue in imu_reader for sampling at least 2"
git push -u origin main

# Increment and save back
echo $((count + 1)) > "$COUNTER_FILE"






#!/usr/bin/env bash
cd "$(dirname "$0")"
source mavlink/venv/bin/activate

# Start all in background with &
python3 state_server.py &
PID1=$!
python3 executor.py &
PID2=$!
python3 imu_reader.py &
PID3=$!
python3 run_training.py &
PID4=$!

# Wait all & clean up on Ctrl+C
trap "echo 'Ctrl+C reçu, arrêt des scripts...'; kill $PID1 $PID2 $PID3 $PID4; exit" SIGINT

wait

#!/usr/bin/env bash
cd "$(dirname "$0")"
source venv/bin/activate

# Kill anything already using port 8080 (e.g. old state_server.py)
echo "Checking for existing processes on port 8080..."
PID_8080=$(lsof -ti tcp:8080)
if [ -n "$PID_8080" ]; then
    echo "Port 8080 is in use by PID(s): $PID_8080. Killing..."
    kill -9 $PID_8080
fi

# Start all scripts in the background
python3 state_server.py &
PID1=$!
python3 executor.py &
PID2=$!
python3 imu_reader.py &
PID3=$!
python3 run_training.py &
PID4=$!

# Trap Ctrl+C to kill all background processes
trap "echo 'Ctrl+C reçu, arrêt des scripts...'; kill $PID1 $PID2 $PID3 $PID4; exit" SIGINT

# Wait for all background processes to finish
wait

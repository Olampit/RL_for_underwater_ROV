#!/usr/bin/env bash
cd "$(dirname "$0")"
source venv/bin/activate

# Kill anything already using port 311 (e.g. old state_server.py)
echo "Checking for existing processes on port 311..."
PID_311=$(lsof -ti tcp:311)
if [ -n "$PID_311" ]; then
    echo "Port 311 is in use by PID(s): $PID_311. Killing..."
    kill -9 $PID_311
fi

# Start all scripts in the background
python3 state_server.py &
PID1=$!
python3 executor.py &
PID2=$!
python3 run_training.py &
PID3=$!

# Trap Ctrl+C to kill all background processes
trap "echo 'Ctrl+C reçu, arrêt des scripts...'; kill $PID1 $PID2 $PID3; exit" SIGINT

# Wait for all background processes to finish
wait

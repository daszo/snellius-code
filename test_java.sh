#!/usr/bin/env bash
#SBATCH --account=gisr113267
#SBATCH --partition=genoa
#SBATCH -t 00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=debug_java.log

echo "=== CHECKING AVAILABLE JAVA MODULES ==="
module avail Java 2>&1 | grep "Java"

echo -e "\n=== ATTEMPTING TO LOAD A COMMON JAVA MODULE ==="
# Try loading the default Java module (often just 'Java')
module load Java
echo "Loaded default 'Java'. Checking JAVA_HOME..."
echo "JAVA_HOME: $JAVA_HOME"
echo "EBROOTJAVA: $EBROOTJAVA"
which javac

echo -e "\n=== CHECKING PYTHON PATHS ==="
source /gpfs/work5/0/prjs1828/DSI-QG/.venv/bin/activate
which python
python --version

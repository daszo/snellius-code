#!/usr/bin/env bash
#SBATCH --account=gisr113267
#SBATCH --partition=genoa       # Explicitly target CPU nodes (commonly 'genoa', 'rome', or 'batch')
#SBATCH -t 1:00:00
#SBATCH --nodes=1               # Force all tasks/cpus onto a single node
#SBATCH --ntasks=1              # Run 1 main task (the python script)
#SBATCH --cpus-per-task=16      # Give that task 16 cores for multiprocessing
#SBATCH --mem=64G               # Request explicit RAM (BM25 can be memory hungry)
#SBATCH --job-name=test_bm25
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/gpfs/work5/0/prjs1828/DSI-QG/logs/count_null.log

exec > >(ts '[%Y-%m-%d %H:%M:%S]') 2>&1


sqlite3 -header -csv data/enron.db "SELECT COUNT(*) FROM text_rank_thread  WHERE text_rank_query IS NOT NULL;"

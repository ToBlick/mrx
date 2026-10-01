#!/bin/bash
# usage: waitjob.sh JOBID LOGFILE  -- blocks until the slurm job leaves the queue, prints head/tail of the log
J=$1; L=$2
while squeue -h -j "$J" 2>/dev/null | grep -q .; do sleep 30; done
echo "job $J finished"
sacct -j "$J" -o JobID,State,Elapsed,MaxRSS -n 2>/dev/null | head -3
head -3 "$L"; echo ...; tail -4 "$L"

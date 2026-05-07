#!/bin/bash -l
#PBS -N ohc_compute
#PBS -A UCUC0007
#PBS -l select=1:ncpus=4:mem=32GB:ngpus=0
#PBS -l walltime=0:20:00
#PBS -q casper
#PBS -j oe
# #PBS -o /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/job_logs

# Kill all casper jobs with: qselect -u jonahshaw | xargs qdel

echo $PBS_JOBID

if [[ -z "$INPUT_FILE" ]] || [[ -z "$OUTPUT_FILE" ]]; then
    echo "ERROR: missing variables INPUT_FILE or OUTPUT_FILE"
    exit 2
fi

mamba run -p /glade/work/jonahshaw/conda-envs/hackathon_extended \
    python /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/J09_OHC_calculation_single.py "$INPUT_FILE" "$OUTPUT_FILE"

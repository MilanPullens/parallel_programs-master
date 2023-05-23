#!/bin/sh

#SBATCH --account=csmpistud
#SBATCH --cpus-per-task=32
#SBATCH --partition=csmpi_fpga_short
#SBATCH --time=00:10:00
#SBATCH --output=stencil_avx.out

# Compile on the machine, not the head node
make bin/stencil_avx2_omp
make clean -C util
make -C util

printf "P,mean,min,max\n" > results/stencil_avx2.csv

for P in 1 2 4 8 16 32; do
    run=1
    while [ "$run" -le 10 ]; do
        {
            OMP_NUM_THREADS="$P" bin/stencil_avx2_omp 999000000 300
            printf "\n"
        } >> results/stencil_avx2_temp.csv
        run=$(( run + 1 ))
    done

    {
        printf "%s," "$P"
        util/stat results/stencil_avx2_temp.csv
        printf "\n"
    } >> results/stencil_avx2.csv

    rm results/stencil_avx2_temp.csv
done

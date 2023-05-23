#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <mpi.h>
#include <sys/time.h>
#include <immintrin.h>

#define REAL double

/* You may need a different method of timing if you are not on Linux. */
#define TIME(duration, fncalls)                                        \
    do {                                                               \
        struct timeval tv1, tv2;                                       \
        gettimeofday(&tv1, NULL);                                      \
        fncalls                                                        \
        gettimeofday(&tv2, NULL);                                      \
        duration = (REAL) (tv2.tv_usec - tv1.tv_usec) / 1000000 +    \
         (REAL) (tv2.tv_sec - tv1.tv_sec);                           \
    } while (0)

const REAL a = 0.1;
const REAL b = 0.2;
const REAL c = 0.3;

void Stencil(REAL **in, REAL **out, size_t n, int iterations)
{
    (*out)[0] = (*in)[0];
    (*out)[n - 1] = (*in)[n - 1];

    for (int t = 1; t <= iterations; t++) {
        for (int i = 1; i < n - 1; i++) {
            (*out)[i] = a * (*in)[i - 1] +
                        b * (*in)[i] +
                        c * (*in)[i + 1];
        }

        if (t != iterations) {
            REAL *temp = *in;
            *in = *out;
            *out = temp;
        }
    }
}

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        printf("Please specify 2 arguments (n, iterations).\n");
        return EXIT_FAILURE;
    }

    size_t n = atoll(argv[1]);
    int iterations = atoi(argv[2]);

    size_t local_n = n / size;
    if (rank < n % size) {
        local_n++;
    }
        

    REAL *in = calloc(n, sizeof(REAL));
    in[0] = 100;
    in[n - 1] = 1000;
    REAL *out = malloc(n * sizeof(REAL));

    double duration;
    TIME(duration, Stencil(&in, &out, local_n, iterations););

    REAL *result = NULL;
    if (rank == 0) {
        result = malloc(n * sizeof(REAL));
    }
    MPI_Gather(out, local_n, MPI_DOUBLE, result, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double total_duration;
        MPI_Reduce(&duration, &total_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        double gflopsS = 5.0 * (n - 2) * iterations / 1e9 / total_duration;
        printf("This took %lf seconds, or %lf GFLOPS/s\n", total_duration, gflopsS);

        free(result);
    }

    
    free(in);
    free(out);

    MPI_Finalize();

    return EXIT_SUCCESS;
}

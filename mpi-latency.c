#define BENCHMARK "OSU MPI Latency Test"
/*
 * Copyright (C) 2002-2012 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University. 
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */

#include <mpi.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

#define MESSAGE_ALIGNMENT 64
#define MAX_MSG_SIZE (1<<22)
#define MYBUFSIZE (MAX_MSG_SIZE + MESSAGE_ALIGNMENT)
#define SKIP_LARGE  10
#define LOOP_LARGE  100
#define LARGE_MESSAGE_SIZE  8192


char s_buf_original[MYBUFSIZE];
char r_buf_original[MYBUFSIZE];

int skip = 1000;
int loop = 10000;
#ifdef PACKAGE_VERSION
#   define HEADER "# " BENCHMARK " v" PACKAGE_VERSION "\n"
#else
#   define HEADER "# " BENCHMARK "\n"
#endif

#ifndef FIELD_WIDTH
#   define FIELD_WIDTH 20
#endif

#ifndef FLOAT_PRECISION
#   define FLOAT_PRECISION 2
#endif

int main(int argc, char *argv[])
{
    int myid, numprocs, i;
    int size;
    MPI_Status reqstat;
    char *s_buf, *r_buf;
    int align_size;
    double t_start = 0.0, t_end = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if(numprocs != 2) {
        if(myid == 0) {
            fprintf(stderr, "This test requires exactly two processes\n");
        }

        MPI_Finalize();

        return EXIT_FAILURE;
    }

    align_size = MESSAGE_ALIGNMENT;

    /**************Allocating Memory*********************/

    s_buf =
        (char *) (((unsigned long) s_buf_original + (align_size - 1)) /
                align_size * align_size);

    r_buf =
        (char *) (((unsigned long) r_buf_original + (align_size - 1)) /
                align_size * align_size);

    /**************Memory Allocation Done*********************/

    if(myid == 0) {
        fprintf(stdout, HEADER);
        fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Latency (us)");
        fflush(stdout);
    }

    for(size = 0; size <= MAX_MSG_SIZE; size = (size ? size * 2 : 1)) {
        /* touch the data */
        for(i = 0; i < size; i++) {
            s_buf[i] = 'a';
            r_buf[i] = 'b';
        }

        if(size > LARGE_MESSAGE_SIZE) {
            loop = LOOP_LARGE;
            skip = SKIP_LARGE;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if(myid == 0) {
            for(i = 0; i < loop + skip; i++) {
                if(i == skip) t_start = MPI_Wtime();

                MPI_Send(s_buf, size, MPI_CHAR, 1, 1, MPI_COMM_WORLD);
                MPI_Recv(r_buf, size, MPI_CHAR, 1, 1, MPI_COMM_WORLD, &reqstat);
            }

            t_end = MPI_Wtime();
        }

        else if(myid == 1) {
            for(i = 0; i < loop + skip; i++) {
                MPI_Recv(r_buf, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &reqstat);
                MPI_Send(s_buf, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
            }
        }

        if(myid == 0) {
            double latency = (t_end - t_start) * 1e6 / (2.0 * loop);

            fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                    FLOAT_PRECISION, latency);
            fflush(stdout);
        }
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

/* vi: set sw=4 sts=4 tw=80: */

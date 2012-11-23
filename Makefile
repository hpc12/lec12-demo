EXECUTABLES = \
	mpi-bandwidth mpi-bi-bandwidth mpi-latency \
	transpose-soln

all: $(EXECUTABLES)

ifdef OPENCL_INC
  CL_CFLAGS = -I$(OPENCL_INC)
endif

ifdef OPENCL_LIB
  CL_LDFLAGS = -L$(OPENCL_LIB)
endif

numa-test: numa-test.c
	gcc -O3 -std=gnu99 -fopenmp $(DEBUG_FLAGS) -lrt -lnuma -o$@ $^

transpose-soln: transpose-soln.c cl-helper.o
	gcc $(CL_CFLAGS) $(CL_LDFLAGS) -std=gnu99 -o$@ $^ -lrt -lOpenCL

%.o : %.c %.h
	gcc -c $(CL_CFLAGS) -std=gnu99 $<

mpi%: mpi%.c
	mpicc -std=gnu99 $(DEBUG_FLAGS) -lrt -o$@ $^

clean:
	rm -f $(EXECUTABLES) *.o mpe-*

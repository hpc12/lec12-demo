import pyopencl as cl
import pyopencl.array
import numpy as np
import sys

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

n = 5 * (2**10)**2

mf = cl.mem_flags
a = cl.array.zeros(queue, n, np.float32)

arg = int(sys.argv[1])

prg = cl.Program(ctx, """//CL//
    #define ARGUMENT myarg

    kernel void fill_vec(global volatile float *a, long int n)
    {
      local float loc_array[2048];
      volatile local float *loc = loc_array;

      long int li = get_local_id(0) % 32;
      long int gi = get_global_id(0);

      loc[li] = 0;

      float x = 0;
      for (int j = 0; j < 100; ++j)
      {
        #pragma unroll
        for (int k = 0; k < 10; ++k)
          x += loc[ARGUMENT * li];
      }
      loc[li] = x;
    }

    """.replace("myarg", str(arg))).build()

from time import time

ntrips = 10

queue.finish()
t1 = time()

for i in xrange(ntrips):
    prg.fill_vec(queue, (n,), (128,), a.data, np.int64(n))
queue.finish()
t2 = time()
print "arg %d elapsed: %g s" % (arg, (t2-t1)/ntrips)

# vim: filetype=pyopencl


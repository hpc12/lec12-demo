import pyopencl as cl
import pyopencl.array
import numpy as np
import sys

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

n = 200 * (2**10)**2

mf = cl.mem_flags
a = cl.array.zeros(queue, n, np.float32)

prg = cl.Program(ctx, """//CL//
    #define ARGUMENT %s

    kernel void fill_vec(global volatile float *a, long int n)
    {
      long int i = ARGUMENT*get_global_id(0);

      if (i < n)
      {
        a[i] *= 2;
        a[i] *= 3;
        a[i] *= 4;
        a[i] *= 2;
        a[i] *= 5;

        a[i] *= 4;
        a[i] *= 3;
        a[i] *= 1.5;
        a[i] *= 2;
        a[i] *= 3;
      }
    }

    """ % sys.argv[1]).build()

from time import time

ntrips = 10

queue.finish()
t1 = time()

for i in xrange(ntrips):
    prg.fill_vec(queue, (n,), (128,), a.data, np.int64(n))
queue.finish()
t2 = time()
print "elapsed: %g s" % ((t2-t1)/ntrips)

# vim: filetype=pyopencl

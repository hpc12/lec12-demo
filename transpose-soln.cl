#pragma OPENCL EXTENSION cl_khr_fp64: enable

#define lid(N) ((int) get_local_id(N))
#define gid(N) ((int) get_group_id(N))


kernel void transpose(
    global float *a,
    global float *b,
    long n)
{
   __local float a_fetch_0[16][16];

   a_fetch_0[lid(1)][lid(0)] = a[n * (lid(1) + 16 * gid(1)) + lid(0) + 16 * gid(0)];
   barrier(CLK_LOCAL_MEM_FENCE);
   b[n * (lid(1) + gid(0) * 16) + lid(0) + gid(1) * 16] = a_fetch_0[lid(0)][lid(1)];
}

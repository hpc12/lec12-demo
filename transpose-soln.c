#include "timing.h"
#include "cl-helper.h"




typedef float value_type;

// #define USE_PINNED

int main(int argc, char **argv)
{
  if (argc != 3)
  {
    fprintf(stderr, "need two arguments!\n");
    abort();
  }

  const long n = atol(argv[1]);
  const long size = n*n;
  const int ntrips = atoi(argv[2]);

  cl_context ctx;
  cl_command_queue queue;
  create_context_on(CHOOSE_INTERACTIVELY, CHOOSE_INTERACTIVELY, 0, &ctx, &queue, 0);

  cl_int status;

  // --------------------------------------------------------------------------
  // load kernels 
  // --------------------------------------------------------------------------
  char *knl_text = read_file("transpose-soln.cl");
  cl_kernel knl = kernel_from_string(ctx, knl_text, "transpose", NULL);
  free(knl_text);

  // --------------------------------------------------------------------------
  // allocate and initialize CPU memory
  // --------------------------------------------------------------------------
#ifdef USE_PINNED
  cl_mem buf_a_host = clCreateBuffer(ctx,
      CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
      sizeof(value_type) * size, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");
  cl_mem buf_b_host = clCreateBuffer(ctx,
      CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
      sizeof(value_type) * size, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  value_type *a = (value_type *) clEnqueueMapBuffer(queue, buf_a_host,
      /*blocking*/ CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 
      /*offs*/ 0, sizeof(value_type)*size, 0, NULL, NULL, &status);
  CHECK_CL_ERROR(status, "clEnqueueMapBuffer");
  value_type *b = (value_type *) clEnqueueMapBuffer(queue, buf_b_host,
      /*blocking*/ CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 
      /*offs*/ 0, sizeof(value_type)*size, 0, NULL, NULL, &status);
  CHECK_CL_ERROR(status, "clEnqueueMapBuffer");

#else
  value_type *a = (value_type *) malloc(sizeof(value_type) * size);
  if (!a) { perror("alloc x"); abort(); }
  value_type *b = (value_type *) malloc(sizeof(value_type) * size);
  if (!b) { perror("alloc y"); abort(); }
#endif

  for (size_t j = 0; j < n; ++j)
    for (size_t i = 0; i < n; ++i)
      a[i + j*n] = i + j*n;

  // --------------------------------------------------------------------------
  // allocate device memory
  // --------------------------------------------------------------------------
  cl_mem buf_a = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
      sizeof(value_type) * size, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem buf_b = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
      sizeof(value_type) * size, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  // --------------------------------------------------------------------------
  // transfer to device
  // --------------------------------------------------------------------------
  CALL_CL_GUARDED(clFinish, (queue));

  timestamp_type time1, time2;
  get_timestamp(&time1);

  CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        queue, buf_a, /*blocking*/ CL_FALSE, /*offset*/ 0,
        size * sizeof(value_type), a,
        0, NULL, NULL));

  CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        queue, buf_b, /*blocking*/ CL_FALSE, /*offset*/ 0,
        size * sizeof(value_type), b,
        0, NULL, NULL));

  get_timestamp(&time2);
  double elapsed = timestamp_diff_in_seconds(time1,time2);
  printf("transfer: %f s\n", elapsed);
  printf("transfer: %f GB/s\n",
      2*size*sizeof(value_type)/1e9/elapsed);


  // --------------------------------------------------------------------------
  // run code on device
  // --------------------------------------------------------------------------

  CALL_CL_GUARDED(clFinish, (queue));

  get_timestamp(&time1);

  for (int trip = 0; trip < ntrips; ++trip)
  {
    SET_3_KERNEL_ARGS(knl, buf_a, buf_b, n);
    size_t ldim[] = { 16, 16 };
    size_t gdim[] = { n, n };
    CALL_CL_GUARDED(clEnqueueNDRangeKernel,
        (queue, knl,
         /*dimensions*/ 2, NULL, gdim, ldim,
         0, NULL, NULL));
  }

  CALL_CL_GUARDED(clFinish, (queue));

  get_timestamp(&time2);
  elapsed = timestamp_diff_in_seconds(time1,time2)/ntrips;
  printf("%f s\n", elapsed);
  printf("%f GB/s\n",
      2*size*sizeof(value_type)/1e9/elapsed);

  CALL_CL_GUARDED(clEnqueueReadBuffer, (
        queue, buf_b, /*blocking*/ CL_FALSE, /*offset*/ 0,
        size * sizeof(value_type), b,
        0, NULL, NULL));

  CALL_CL_GUARDED(clFinish, (queue));

  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j)
      if (a[i + j*n] != b[j + i*n])
      {
        printf("bad %d %d\n", i, j);
        abort();
      }

  // --------------------------------------------------------------------------
  // clean up
  // --------------------------------------------------------------------------
  CALL_CL_GUARDED(clFinish, (queue));
  CALL_CL_GUARDED(clReleaseMemObject, (buf_a));
  CALL_CL_GUARDED(clReleaseMemObject, (buf_b));
  CALL_CL_GUARDED(clReleaseKernel, (knl));
  CALL_CL_GUARDED(clReleaseCommandQueue, (queue));
  CALL_CL_GUARDED(clReleaseContext, (ctx));

#ifdef USE_PINNED
  CALL_CL_GUARDED(clReleaseMemObject, (buf_a_host));
  CALL_CL_GUARDED(clReleaseMemObject, (buf_b_host));
#else
  free(a);
  free(b);
#endif
  return 0;
}

#include "hip/hip_runtime.h"

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  int deviceCount = 0;
  int N = 1<<20;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  for (int i = 0; i < N; i++)
    x[i] = 1.0f;

  HIP_ASSERT(hipGetDeviceCount(&deviceCount));

  for (int j = 0; j < deviceCount; j++) {
    for (int i = 0; i < N; i++)
      y[i] = 2.0f;

    HIP_ASSERT(hipSetDevice(j));
    HIP_ASSERT(hipMalloc(&d_x, N*sizeof(float)));
    HIP_ASSERT(hipMalloc(&d_y, N*sizeof(float)));

    HIP_ASSERT(hipMemcpy(d_x, x, N*sizeof(float), hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(d_y, y, N*sizeof(float), hipMemcpyHostToDevice));

    // Perform SAXPY on 1M elements
    hipLaunchKernelGGL(saxpy,(N+255)/256, 256,0,0,N, 2.0f, d_x, d_y );

    HIP_ASSERT(hipMemcpy(y, d_y, N*sizeof(float), hipMemcpyDeviceToHost));

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
      maxError = ( maxError > abs(y[i]-4.0f) ) ? maxError : abs(y[i]-4.0f) ;
    printf("Max error: %f\n", maxError);

    HIP_ASSERT(hipFree(d_x));
    HIP_ASSERT(hipFree(d_y));
  }
  free(x);
  free(y);
}

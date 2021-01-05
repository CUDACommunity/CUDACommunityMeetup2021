#include <iostream>
#include <numeric>
#include <memory>
#include <array>

__forceinline__ __device__ int ldg (const int * p)
{
  int out;
  asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(out) : "l"(p));
  return out;
}

union data_flag
{
  struct 
  {
    int32_t data;
    int32_t flag;
  } fields;

  int64_t vec;
};

__global__ void kernel (
    int n,
    volatile data_flag * df, 
    int *result)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  data_flag tmp;

  if (bid == 0)
    {
      if (tid == 0)
        {
          tmp.fields.data = 1;
          tmp.fields.flag = 1;

          df->vec = tmp.vec;
        }
    }
  else
    {
      while (true)
      {
        tmp.vec = df->vec;
        if (tmp.fields.flag)
          break;
      }

      if (tmp.fields.data == 0)
        *result = 0;
    }
}

int main ()
{
  data_flag *data {};
  int *r {};

  constexpr int n = 1;
  unsigned int count = 0;

  const int max_iterations = 10;

  for (int blocks_th = 1; blocks_th < 32 * 1024; blocks_th += 256)
    {
      float sum = 0;

      for (int iteration = 0; iteration < max_iterations; iteration++)
        {
          cudaDeviceReset ();
          cudaMalloc (&r, sizeof (int));

          cudaEvent_t begin, end;
          cudaEventCreate (&begin);
          cudaEventCreate (&end);

          cudaMalloc (&data, n * sizeof (data_flag));
          cudaMemset (data, n * sizeof (data_flag), 0);
          cudaMemset (r, sizeof (data_flag), 1);

          cudaEventRecord (begin);
          kernel<<<blocks_th * 32, 32>>> (n, data, r);
          cudaEventRecord (end);
          cudaEventSynchronize (end);

          float elapsed = 0;
          cudaEventElapsedTime (&elapsed, begin, end);
          sum += elapsed;

          cudaFree (data);

          int cpu_r {};
          cudaMemcpy (&cpu_r, r, sizeof (int), cudaMemcpyDeviceToHost);

          if (cpu_r == 0)
            count++;

          cudaEventDestroy (begin);
          cudaEventDestroy (end);

          cudaFree (r);
        }

      std::cout << blocks_th * 32 << ", " << sum / max_iterations << std::endl;
    }

  return 0;
}

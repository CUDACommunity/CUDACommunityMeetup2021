#include <iostream>
#include <numeric>
#include <memory>
#include <array>

__global__ void kernel (
    int n,
    volatile int * flag, 
    int * data, 
    int *result)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if (bid == 0)
    {
      if (tid == 0)
        {
          for (int i = 0; i < n; i++)
            data[i] = i + 1;
          __threadfence ();

          *flag = 1;
        }
    }
  else
    {
      while (*flag == 0);
      __threadfence ();

      for (int i = 0; i < n; i++)
        if (data[i] == 0)
          *result = 0;
    }
}

int main ()
{
  int *data {};
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

          cudaMalloc (&data, (n + 1) * sizeof (int));
          cudaMemset (data, (n + 1) * sizeof (int), 0);
          cudaMemset (r, sizeof (int), 1);

          cudaEventRecord (begin);
          kernel<<<blocks_th * 32, 32>>> (n, data + 0, data + 1, r);
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

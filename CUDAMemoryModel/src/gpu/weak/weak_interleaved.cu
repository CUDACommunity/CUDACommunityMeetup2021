#include <iostream>
#include <numeric>
#include <memory>

__device__ void producer (
    volatile int * const v_1, 
    volatile int * const v_2,  
    volatile int * const v_3, 
    volatile int * const v_4)
{
  *v_1 = 1;
  *v_2 = 2;
  __threadfence();
  *v_3 = 3;
  *v_4 = 4;
}

__device__ void consumer (
    volatile int * const v_1, 
    volatile int * const v_2,  
    volatile int * const v_3, 
    volatile int * const v_4, 
    int *result)
{
  int reg_1 = *v_1;
  int reg_2 = *v_2;
  __threadfence();
  int reg_3 = *v_3;
  int reg_4 = *v_4;

  if (reg_1 == 1 && reg_2 == 0 && reg_3 == 3 && reg_4 == 0)
    *result = 0;
}

__global__ void kernel (
    volatile int * const v_1, 
    volatile int * const v_2, 
    volatile int * const v_3, 
    volatile int * const v_4, 
    int *result)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if (bid == 0)
    {
      if (tid == 0)
        producer (v_1, v_2, v_3, v_4);
    }
  else
    consumer (v_1, v_2, v_3, v_4, result);
}

int main ()
{
  int *x {};
  int *y {};
  int *r {};

  // for (bool single_segment: {false, true})
  const bool single_segment = false;

  for (unsigned int i = 64; i < 128 * 1024; i += 32)
    {
      int violations_count = 0;

      cudaMalloc (&r, sizeof (int));
      cudaMalloc (&x, 4 * sizeof (int));
      cudaMalloc (&y, 2 * sizeof (int));

      cudaMemset (r, 1, sizeof (int));
      cudaMemset (x, 0, 4 * sizeof (int));
      cudaMemset (y, 0, 2 * sizeof (int));

      cudaEvent_t begin, end;

      cudaEventCreate (&begin);
      cudaEventCreate (&end);

      const unsigned int max_iters = 10;

      cudaEventRecord (begin);
      int iteration = 0;
      for (; iteration < max_iters; iteration++)
        {
          if (single_segment)
            kernel<<<i, 64>>> (x + 0, x + 1, x + 2, x + 3, r);
          else
            kernel<<<i, 64>>> (x + 0, y + 0, x + 1, y + 1, r);
        }
      cudaEventRecord (end);
      cudaEventSynchronize (end);

      float ms = 0;

      cudaEventElapsedTime (&ms, begin, end);

      std::cout << i << ", " << ms / max_iters << "\n";

      cudaEventDestroy (end);
      cudaEventDestroy (begin);

      int cpu_r {};
      cudaMemcpy (&cpu_r, r, sizeof (int), cudaMemcpyDeviceToHost);

      cudaFree (x);
      cudaFree (y);
      cudaFree (r);

      if (0)
        {
          if (cpu_r == 0)
            {
              violations_count++;
              std::cout << "fail on " << iteration << " iteration\n";
              // return 0;
            }

          std::cout << "SS: " << single_segment << "; VC: " << violations_count << "\n";
        }
    }

  return 0;
}

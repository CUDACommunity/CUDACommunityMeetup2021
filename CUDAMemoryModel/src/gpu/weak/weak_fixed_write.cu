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

      // __threadfence ();

      for (int i = 0; i < n; i++)
         if (ldg (data + i) == 0) // if (data[i] == 0)
          *result = 0;
    }
}

int main ()
{
  int *flag {};
  int *data {};
  int *r {};

  constexpr int max_n = 32;
  std::array<int, max_n> stat;
  stat.fill (0);

  for (int iteration = 0; iteration < 100; iteration++)
    {
      cudaDeviceReset ();
      cudaMalloc (&r, sizeof (int));

      for (int n = 1; n < max_n; n++)
        {
          cudaMalloc (&data, (n + 1) * sizeof (int));
          cudaMalloc (&flag, sizeof (int));
          cudaMemset (data, (n + 1) * sizeof (int), 0);
          cudaMemset (flag, sizeof (int), 0);
          cudaMemset (r, sizeof (int), 1);

          kernel<<<1024, 512>>> (n, flag, data + 1, r);

          cudaFree (data);
          cudaFree (flag);

          int cpu_r {};
          cudaMemcpy (&cpu_r, r, sizeof (int), cudaMemcpyDeviceToHost);

          if (cpu_r == 0)
            stat[n]++;
        }

      cudaFree (r);
    }

  for (int i = 1; i < max_n; i++)
      std::cout << "n=" << i << ": " << stat[i] << "\n";

  return 0;
}

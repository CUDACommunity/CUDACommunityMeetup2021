#include <iostream>
#include <stdexcept>
#include <numeric>
#include <memory>
#include <array>

void chck (cudaError_t status)
{
  if (status != cudaSuccess)
    throw std::runtime_error ("Error!");
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

          *flag = 1;
        }
    }
  else
    {
      while (*flag == 0);

      for (int i = 0; i < n; i++)
        if (data[i] == 0)
          *result = 0;
    }
}

int main ()
{
  int *data {};
  int *r {};

  constexpr int max_n = 9;
  std::array<int, max_n> stat;
  stat.fill (0);

  for (int iteration = 0; iteration < 10000; iteration++)
    {
      cudaMalloc (&r, sizeof (int));

      for (int n = 1; n < max_n; n++)
        {
          cudaMalloc (&data, (n + 1) * sizeof (int));
          cudaMemset (data, (n + 1) * sizeof (int), 0);
          cudaMemset (r, sizeof (int), 1);

          kernel<<<1024, 32>>> (n, data + 0, data + 1, r);

          chck (cudaFree (data));

          int cpu_r {};
          chck (cudaMemcpy (&cpu_r, r, sizeof (int), cudaMemcpyDeviceToHost));

          if (cpu_r == 0)
          {
            std::cerr << "fail\n";
            stat[n]++;
          }
        }

      cudaFree (r);
    }

  for (int i = 1; i < max_n; i++)
      std::cout << "n=" << i << ": " << stat[i] << "\n";

  return 0;
}

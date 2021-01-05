#include <iostream>
#include <numeric>
#include <memory>

__global__ void kernel (
    volatile int * const __restrict__ x, 
    int * const __restrict__ y, int *result)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if (bid == 0 && tid == 0)
    {
      *y = 2;
      *x = 1;
    }
  else
    {
      int reg = *y;
      while (*x == reg);

      if (*y == 0)
        *result = 0;
    }
}

int main ()
{
  int *x {};
  int *y {};
  int *r {};

  cudaMalloc (&r, sizeof (int));
  cudaMalloc (&x, sizeof (int));
  cudaMalloc (&y, sizeof (int));

  cudaMemset (x, sizeof (int), 0);
  cudaMemset (y, sizeof (int), 0);
  cudaMemset (r, sizeof (int), 1);

  kernel<<<2056, 1024>>> (x, y, r);

  int cpu_r {};
  cudaMemcpy (&cpu_r, r, sizeof (int), cudaMemcpyDeviceToHost);

  std::cout << "passed = " << cpu_r << "\n";

  return 0;
}

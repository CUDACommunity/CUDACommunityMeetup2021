#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <cstdlib>

// test

void chck (cudaError_t status)
{
  if (status != cudaSuccess)
    throw std::runtime_error (cudaGetErrorString (status));
}

#define BLOCK_SIZE 1024

__global__ void coherence_of_read_read_shared (
  int rthread, int wthread, int *x, 
  int *reordering_1, int *reordering_2, int *violated)
{
  __shared__ int cache[BLOCK_SIZE];

  const int tid = threadIdx.x;
  const int a1 = reordering_1[tid];
  const int a2 = reordering_2[tid];

  cache[tid] = x[tid];
  __syncthreads ();

  if (tid == wthread)
    {
      cache[tid] = 1;
    }
  else 
    {
      const int r1 = cache[a1];
      const int r2 = cache[a2];

      if (r1 == 1 && r2 == 0)
	      if (tid == rthread)
          *violated = 1;
    }
}

int main ()
{
  int *x = 0;
  int *device_flag = 0;
  int *device_reordering_1 = 0;
  int *device_reordering_2 = 0;
  int host_flag = 0;

      chck (cudaMalloc (&x, BLOCK_SIZE * sizeof (int)));
      chck (cudaMalloc (&device_reordering_1, BLOCK_SIZE * sizeof (int)));
      chck (cudaMalloc (&device_reordering_2, BLOCK_SIZE * sizeof (int)));
      chck (cudaMalloc (&device_flag, sizeof (int)));

  int reader_reordering_1[BLOCK_SIZE];
  int reader_reordering_2[BLOCK_SIZE];


  int violations_count = 0;
  for (int iteration = 0; iteration < 100000; iteration++)
    {
      // cudaDeviceReset ();
      cudaMemset (device_flag, 0, sizeof (int));
      cudaMemset (x, 0, BLOCK_SIZE * sizeof (int));

      int rthread = std::rand () % BLOCK_SIZE;
      int wthread = std::rand () % BLOCK_SIZE;
      
      for (int t = 0; t < BLOCK_SIZE; t++)
        {
          reader_reordering_1[t] = std::rand () % BLOCK_SIZE;
          reader_reordering_2[t] = std::rand () % BLOCK_SIZE;
        }
      reader_reordering_1[rthread] = wthread;
      reader_reordering_2[rthread] = wthread;

      cudaMemcpy (device_reordering_1, reader_reordering_1, sizeof (int) * BLOCK_SIZE, cudaMemcpyHostToDevice);
      cudaMemcpy (device_reordering_2, reader_reordering_2, sizeof (int) * BLOCK_SIZE, cudaMemcpyHostToDevice);

      coherence_of_read_read_shared<<<1, BLOCK_SIZE>>> (rthread, wthread, x, device_reordering_1, device_reordering_2, device_flag);

      cudaMemcpy (&host_flag, device_flag, sizeof (int), cudaMemcpyDeviceToHost);

      if (host_flag)
        {
          std::cout << "=== violation (r=" << rthread << "; w=" << wthread << ") ===\n";
	  for (int t = 0; t < BLOCK_SIZE; t++)
              std::cout << std::setfill ('0') << std::setw (3) << t << " ";
          std::cout << "\n";
	  for (int t = 0; t < BLOCK_SIZE; t++)
              std::cout << std::setfill ('0') << std::setw (3) << reader_reordering_1[t] << " ";
          std::cout << "\n";
	  for (int t = 0; t < BLOCK_SIZE; t++)
              std::cout << std::setfill ('0') << std::setw (3) << reader_reordering_2[t] << " ";
          std::cout << "\n";
          violations_count++;
        }

    }

      cudaFree (device_reordering_1);
      cudaFree (device_reordering_2);
      cudaFree (device_flag);
      cudaFree (x);

  if (violations_count)
    std::cout << "violated " << violations_count << " times\n";
  else
    std::cout << "ok\n";
  return 0;
}

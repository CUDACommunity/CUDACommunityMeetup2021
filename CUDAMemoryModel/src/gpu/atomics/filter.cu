#include <iostream>
#include <numeric>
#include <memory>

template <typename action_type, typename filter_type>
__global__ void transform_and_count (
    const int n,
    const int *data,
    int *result,
    action_type action,
    filter_type filter)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < n && filter (action (data[i])))
    atomicAdd (result, 1);
}


int main ()
{
  const int n = 10'000'000;
  std::unique_ptr<int[]> cpu_data (new int[n]);
  std::iota (cpu_data.get (), cpu_data.get () + n, 0);

  int *gpu_data {};
  cudaMalloc (&gpu_data, n * sizeof (int));
  cudaMemcpy (gpu_data, cpu_data.get (), n * sizeof (int), cudaMemcpyHostToDevice);
  
  int *gpu_result {};
  cudaMalloc (&gpu_result, sizeof (int));
  cudaMemset (gpu_result, sizeof (int), 0);

  cudaEvent_t begin, end;
  cudaEventCreate (&begin);
  cudaEventCreate (&end);

  cudaEventRecord (begin);
  transform_and_count<<<(n + 127) / 128, 128>>> (
      n, gpu_data, gpu_result, 
      [] __device__ (int value) { return value * value; },
      [] __device__ (int value) { return value == 10'000'000; });
  cudaEventRecord (end);

  int cpu_result {};
  cudaMemcpy (&cpu_result, gpu_result, sizeof (int), cudaMemcpyDeviceToHost);

  cudaEventSynchronize (end);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, begin, end);

  std::cout << cpu_result << " (complete in " << milliseconds << ")\n";
}

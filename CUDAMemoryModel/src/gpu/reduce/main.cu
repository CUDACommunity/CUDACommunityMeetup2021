#include <iostream>
#include <cuda/atomic>
#include <cuda/std/atomic>
#include <cub/cub.cuh>

constexpr int block_size = 128;
constexpr int thread_data_size = 12;

template <typename data_type>
__global__ void fill (int n, data_type val, data_type *data)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < n)
    data[i] = val;
}

template <typename data_type>
__global__ void reduce (const data_type *in, data_type *block_results)
{
  typedef cub::BlockLoad<data_type, block_size, thread_data_size, cub::BLOCK_LOAD_TRANSPOSE> BlockLoad;
  typedef cub::BlockReduce<data_type, block_size> BlockReduce;

  __shared__ union
  {
    typename BlockLoad::TempStorage load;
    typename BlockReduce::TempStorage reduce;
  } temp_storage;

  const int block_offset = blockIdx.x * block_size * thread_data_size;

  data_type thread_data[thread_data_size];
  BlockLoad (temp_storage.load).Load (in + block_offset, thread_data);
  __syncthreads ();

  const data_type block_result = BlockReduce (temp_storage.reduce).Sum (thread_data);
  __syncthreads ();

  if (threadIdx.x == 0)
    block_results[blockIdx.x] = block_result;
}

template <typename data_type>
__global__ void reduce_block_results (int blocks, const data_type *in, data_type *result)
{
  typedef cub::BlockReduce<data_type, block_size> BlockReduce;

  __shared__ typename BlockReduce::TempStorage temp_storage;

  data_type thread_data = 0;

  for (int i = threadIdx.x; i < blocks; i += blockDim.x)
    thread_data += in[i];

  const data_type block_result = BlockReduce (temp_storage).Sum (thread_data);

  if (threadIdx.x == 0)
    result[0] = block_result;
}

template <typename data_type>
__global__ void reduce_single_kernel (int *count, const data_type *in, data_type *block_results)
{
  typedef cub::BlockLoad<data_type, block_size, thread_data_size, cub::BLOCK_LOAD_TRANSPOSE> BlockLoad;
  typedef cub::BlockReduce<data_type, block_size> BlockReduce;

  __shared__ union
  {
    typename BlockLoad::TempStorage load;
    typename BlockReduce::TempStorage reduce;
    bool need_to_perform_final_reduce;
  } temp_storage;

  const int block_offset = blockIdx.x * block_size * thread_data_size;

  {
    data_type thread_data[thread_data_size];
    BlockLoad (temp_storage.load).Load (in + block_offset, thread_data);
    __syncthreads ();

    const data_type block_result = BlockReduce (temp_storage.reduce).Sum (thread_data);
    __syncthreads ();

    if (threadIdx.x == 0)
      {
        block_results[blockIdx.x] = block_result;
        __threadfence ();

        const int prev_count = atomicAdd (count, 1);

        temp_storage.need_to_perform_final_reduce = prev_count == gridDim.x - 1;
      }
    __syncthreads ();
  }

  if (temp_storage.need_to_perform_final_reduce)
    {
      data_type thread_data = 0;

      for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x)
        thread_data += block_results[i];

      const data_type block_result = BlockReduce (temp_storage.reduce).Sum (thread_data);

      if (threadIdx.x == 0)
        block_results[0] = block_result;
    }
}

template <typename data_type>
__global__ void reduce_single_kernel_atom (cuda::atomic<int, cuda::thread_scope_device> *count, const data_type *in, data_type *block_results)
{
  typedef cub::BlockLoad<data_type, block_size, thread_data_size, cub::BLOCK_LOAD_TRANSPOSE> BlockLoad;
  typedef cub::BlockReduce<data_type, block_size> BlockReduce;

  __shared__ union
  {
    typename BlockLoad::TempStorage load;
    typename BlockReduce::TempStorage reduce;
    bool need_to_perform_final_reduce;
  } temp_storage;

  const int block_offset = blockIdx.x * block_size * thread_data_size;

  {
    data_type thread_data[thread_data_size];
    BlockLoad (temp_storage.load).Load (in + block_offset, thread_data);
    __syncthreads ();

    const data_type block_result = BlockReduce (temp_storage.reduce).Sum (thread_data);
    __syncthreads ();

    if (threadIdx.x == 0)
    {
      block_results[blockIdx.x] = block_result;

      const int prev_count = count->fetch_add (1, cuda::std::memory_order_release);
      temp_storage.need_to_perform_final_reduce = prev_count == gridDim.x - 1;
    }
    __syncthreads ();
  }

  if (temp_storage.need_to_perform_final_reduce)
  {
    data_type thread_data = 0;

    for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x)
      thread_data += block_results[i];

    const data_type block_result = BlockReduce (temp_storage.reduce).Sum (thread_data);

    if (threadIdx.x == 0)
      block_results[0] = block_result;
  }
}

template <typename data_type>
__global__ void reduce_single_kernel_atom (
    cuda::std::atomic<int> *count, const data_type *in, data_type *block_results)
{
  typedef cub::BlockLoad<data_type, block_size, thread_data_size, cub::BLOCK_LOAD_TRANSPOSE> BlockLoad;
  typedef cub::BlockReduce<data_type, block_size> BlockReduce;

  __shared__ union
  {
    typename BlockLoad::TempStorage load;
    typename BlockReduce::TempStorage reduce;
    bool need_to_perform_final_reduce;
  } temp_storage;

  const int block_offset = blockIdx.x * block_size * thread_data_size;

  {
    data_type thread_data[thread_data_size];
    BlockLoad (temp_storage.load).Load (in + block_offset, thread_data);
    __syncthreads ();

    const data_type block_result = BlockReduce (temp_storage.reduce).Sum (thread_data);
    __syncthreads ();

    if (threadIdx.x == 0)
    {
      block_results[blockIdx.x] = block_result;

      const int prev_count = count->fetch_add (1, cuda::std::memory_order_release);
      temp_storage.need_to_perform_final_reduce = prev_count == gridDim.x - 1;
    }
    __syncthreads ();
  }

  if (temp_storage.need_to_perform_final_reduce)
  {
    data_type thread_data = 0;

    for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x)
      thread_data += block_results[i];

    const data_type block_result = BlockReduce (temp_storage.reduce).Sum (thread_data);

    if (threadIdx.x == 0)
      block_results[0] = block_result;
  }
}

enum class reduce_method
{
  two_kernels,
  single_kernel,
  single_kernel_atom,
  single_kernel_std_atom,
};

const char *method_to_str (reduce_method method)
{
  switch (method)
  {
    case reduce_method::two_kernels: return "two kernels";
    case reduce_method::single_kernel: return "single kernel";
  }

  return "unknown";
}

void run (reduce_method method)
{
  // std::cout << method_to_str (method) << "\n";

  for (int blocks = 1; blocks < 1026; blocks += 32)
    {
      const int n = blocks * block_size * thread_data_size;

      int *in = nullptr;
      cudaMalloc (&in, n * sizeof (int));

      int *block_results = nullptr;
      cudaMalloc (&block_results, blocks * sizeof (int));

      int *count = nullptr;
      cuda::std::atomic<int> *count_std_atom = nullptr;
      cuda::atomic<int, cuda::thread_scope_device> *count_atom = nullptr;
      cudaMalloc (&count, sizeof (int));
      cudaMalloc (&count_atom, sizeof (cuda::atomic<int>));
      cudaMalloc (&count_std_atom, sizeof (cuda::std::atomic<int>));

      {
        const int bs = 128;
        const int gs = (n + bs - 1) / bs;
        fill<<<gs, bs>>> (n, 1, in);
      }

      cudaEvent_t begin, end;

      cudaEventCreate (&begin);
      cudaEventCreate (&end);

      cudaEventRecord (begin);

      const int max_iters = 100;

      for (int iter = 0; iter < max_iters; iter++)
        {
          if (method == reduce_method::two_kernels)
            {
              reduce<<<blocks, block_size>>> (in, block_results);
              reduce_block_results<<<1, block_size>>> (blocks, block_results, block_results);
            }
          else if (method == reduce_method::single_kernel_atom)
            {
              cudaMemset (count_atom, 0, sizeof (cuda::atomic<int>));
              reduce_single_kernel_atom<<<blocks, block_size>>> (count_atom, in, block_results);
            }
          else if (method == reduce_method::single_kernel_std_atom)
            {
              cudaMemset (count_std_atom, 0, sizeof (cuda::atomic<int>));
              reduce_single_kernel_atom<<<blocks, block_size>>> (count_std_atom, in, block_results);
            }
          else
            {
              cudaMemset (count, 0, sizeof (int));
              reduce_single_kernel<<<blocks, block_size>>> (count, in, block_results);
            }
        }

      cudaEventRecord (end);
      cudaEventSynchronize (end);

      float ms = 0;
      cudaEventElapsedTime (&ms, begin, end);

      std::cout << n << ", " << ms / max_iters << "\n";

      int result = 0;
      cudaMemcpy (&result, block_results, sizeof (int), cudaMemcpyDeviceToHost);

      if (result != n)
        std::cerr << "error, result " << result << " != " << n << std::endl;

      cudaEventDestroy (end);
      cudaEventDestroy (begin);

      cudaFree (block_results);
      cudaFree (count_atom);
      cudaFree (count_std_atom);
      cudaFree (count);
      cudaFree (in);
    }
}

int main()
{
  // run (reduce_method::two_kernels);
  // run (reduce_method::single_kernel);
  // run (reduce_method::single_kernel_atom);
  run (reduce_method::single_kernel_std_atom);
  return 0;
}

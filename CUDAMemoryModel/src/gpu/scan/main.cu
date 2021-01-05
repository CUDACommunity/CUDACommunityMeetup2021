#include <cub/cub.cuh>
#include <iostream>
#include <stdexcept>

#define WARP_MASK 0xFFFFFFFF
#define WARP_SIZE 32

enum class implementation_type
{
  cub, copy, hierarchical, single_kernel, single_kernel_decoupled_lookback
};

void chk (cudaError_t status)
{
  if (cudaSuccess != status)
    throw std::runtime_error (std::string ("Error: ") + cudaGetErrorName (status));
}

constexpr int hierarchical_block_size = 128;
constexpr int thread_data_size = 12;

template <typename data_type>
__global__ void hierarchical_partial_scan (const data_type *in, data_type *out, data_type *blocks_results)
{
  typedef cub::BlockLoad<data_type, hierarchical_block_size, thread_data_size, cub::BLOCK_LOAD_TRANSPOSE> BlockLoad;
  typedef cub::BlockStore<data_type, hierarchical_block_size, thread_data_size, cub::BLOCK_STORE_TRANSPOSE> BlockStore;
  typedef cub::BlockScan<data_type, hierarchical_block_size> BlockScan;

  __shared__ union
  {
    typename BlockLoad::TempStorage load;
    typename BlockScan::TempStorage scan;
    typename BlockStore::TempStorage store;
  } temp_storage;

  const int block_offset = blockIdx.x * hierarchical_block_size * thread_data_size;

  data_type block_aggregate {};

  data_type thread_data[thread_data_size];
  BlockLoad (temp_storage.load).Load (in + block_offset, thread_data);
  __syncthreads ();

  BlockScan (temp_storage.scan).ExclusiveSum (thread_data, thread_data, block_aggregate);
  __syncthreads ();

  BlockStore (temp_storage.store).Store (out + block_offset, thread_data);
  __syncthreads ();

  if (threadIdx.x == blockDim.x - 1)
    blocks_results[blockIdx.x] = block_aggregate;
}

template <typename data_type>
__global__ void hierarchical_blocks_results_scan (
    int size,
    const data_type *in,
    data_type *blocks_results)
{
  typedef cub::BlockScan<data_type, 1024> BlockScan;

  __shared__ typename BlockScan::TempStorage temp_storage;

  data_type result_block_aggregate {};
  for (int segment_offset = 0; segment_offset < size; segment_offset += blockDim.x)
    {
      const int i = segment_offset + threadIdx.x;
      data_type thread_data = i < size ? in[i] : data_type ();

      data_type block_aggregate {};
      BlockScan (temp_storage).ExclusiveSum (thread_data, thread_data, block_aggregate);

      thread_data += result_block_aggregate;
      __syncthreads ();

      if (i < size)
        {
          result_block_aggregate += block_aggregate;
          blocks_results[i] = thread_data;
        }
    }
}

template <typename data_type>
__global__ void hierarchical_final_adjust (data_type *out, data_type *blocks_results)
{
  typedef cub::BlockLoad<data_type, hierarchical_block_size, thread_data_size, cub::BLOCK_LOAD_TRANSPOSE> BlockLoad;
  typedef cub::BlockStore<data_type, hierarchical_block_size, thread_data_size, cub::BLOCK_STORE_TRANSPOSE> BlockStore;

  if (blockIdx.x == 0)
    return;

  const data_type prev_aggregate = blocks_results[blockIdx.x];

  __shared__ union
  {
    typename BlockLoad::TempStorage load;
    typename BlockStore::TempStorage store;
  } temp_storage;

  const int block_offset = blockIdx.x * hierarchical_block_size * thread_data_size;

  data_type thread_data[thread_data_size];
  BlockLoad (temp_storage.load).Load (out + block_offset, thread_data);
  __syncthreads ();

  for (int i = 0; i < thread_data_size; i++)
    thread_data[i] += prev_aggregate;

  __syncthreads ();

  BlockStore (temp_storage.store).Store (out + block_offset, thread_data);
  __syncthreads ();
}

template <typename data_type>
void measure_hierarchical (
    int size,
    void *helper_buffer_ptr,
    const data_type *in,
    data_type *out)
{
  data_type *helper_buffer = reinterpret_cast<data_type*> (helper_buffer_ptr);
  const int elements_per_block = thread_data_size * hierarchical_block_size;
  const int blocks = (size + elements_per_block - 1) / elements_per_block;

  hierarchical_partial_scan<<<blocks, hierarchical_block_size>>> (in, out, helper_buffer);
  hierarchical_blocks_results_scan<<<1, 1024>>> (blocks, helper_buffer, helper_buffer);
  hierarchical_final_adjust<<<blocks, hierarchical_block_size>>> (out, helper_buffer);
}

struct block_status_word
{
  int32_t flag;
  int32_t data;
};

__device__ int64_t ld_cg (const int64_t *p)
{
  int64_t out;
  asm volatile("ld.global.cg.s64 %0, [%1];" : "=l"(out) : "l"(p));
  return out;
}

__device__ void st_cg (int64_t *p, int64_t val)
{
  asm volatile("st.global.cg.s64 [%1], %0;" : : "l"(val), "l"(p));
}

__device__ int32_t sm_id ()
{
  int32_t result;
  asm volatile("mov.u32 %0, %smid;" : "=r"(result) : );

  return result;
}

__device__ block_status_word load (volatile block_status_word *ptr)
{
  int64_t tmp = *reinterpret_cast<volatile int64_t*> (ptr);
  block_status_word &rhs = reinterpret_cast<block_status_word&> (tmp);
  return rhs;
}

__device__ void store (volatile block_status_word *ptr, int32_t flag, int32_t data)
{
  int64_t tmp;
  block_status_word &rhs = reinterpret_cast<block_status_word&> (tmp);

  rhs.flag = flag;
  rhs.data = data;

  *reinterpret_cast<volatile int64_t*> (ptr) = tmp;
}

template <typename data_type>
__global__ void single_kernel_scan (const data_type *in, data_type *out, volatile block_status_word *block_statuses)
{
  typedef cub::BlockLoad<data_type, hierarchical_block_size, thread_data_size, cub::BLOCK_LOAD_TRANSPOSE> BlockLoad;
  typedef cub::BlockStore<data_type, hierarchical_block_size, thread_data_size, cub::BLOCK_STORE_TRANSPOSE> BlockStore;
  typedef cub::BlockScan<data_type, hierarchical_block_size> BlockScan;
  typedef cub::BlockReduce<data_type, hierarchical_block_size> BlockReduce;

  __shared__ union
  {
    typename BlockLoad::TempStorage load;
    typename BlockScan::TempStorage scan;
    typename BlockReduce::TempStorage reduce;
    typename BlockStore::TempStorage store;
  } temp_storage;

  const int block_offset = blockIdx.x * hierarchical_block_size * thread_data_size;

  data_type thread_data[thread_data_size];
  BlockLoad (temp_storage.load).Load (in + block_offset, thread_data);
  __syncthreads ();

  data_type block_reduce = BlockReduce (temp_storage.reduce).Sum (thread_data);

  if (threadIdx.x == 0)
    {
      if (blockIdx.x == 0)
        {
          store (block_statuses + blockIdx.x, 1 /* flag */, block_reduce /* data */);
        }
      else
        {
          block_status_word prev_block_status;

          do
          {
            prev_block_status = load (block_statuses + blockIdx.x - 1);
          } while (prev_block_status.flag == 0);

          thread_data[0] += prev_block_status.data;
          store (block_statuses + blockIdx.x, 1 /* flag */, prev_block_status.data + block_reduce /* data */);
        }
    }
  __syncthreads ();

  BlockScan (temp_storage.scan).ExclusiveSum (thread_data, thread_data);
  __syncthreads ();

  BlockStore (temp_storage.store).Store (out + block_offset, thread_data);
}

template <typename data_type>
void measure_single_kernel (
    int size,
    void *helper_buffer_ptr,
    const data_type *in,
    data_type *out)
{
  block_status_word *helper_buffer = reinterpret_cast<block_status_word*> (helper_buffer_ptr);
  const int elements_per_block = thread_data_size * hierarchical_block_size;
  const int blocks = (size + elements_per_block - 1) / elements_per_block;

  cudaMemsetAsync (helper_buffer, 0, blocks * sizeof (block_status_word));
  single_kernel_scan<<<blocks, hierarchical_block_size>>> (in, out, helper_buffer);
}

template <typename data_type>
__global__ void single_kernel_decoupled_lookback_scan (const data_type *in, data_type *out, volatile block_status_word *block_statuses)
{
  typedef cub::WarpReduce<data_type> WarpReduce;
  typedef cub::BlockLoad<data_type, hierarchical_block_size, thread_data_size, cub::BLOCK_LOAD_TRANSPOSE> BlockLoad;
  typedef cub::BlockStore<data_type, hierarchical_block_size, thread_data_size, cub::BLOCK_STORE_TRANSPOSE> BlockStore;
  typedef cub::BlockScan<data_type, hierarchical_block_size> BlockScan;

  __shared__ union
  {
    typename WarpReduce::TempStorage warp_reduce;
    typename BlockLoad::TempStorage load;
    typename BlockScan::TempStorage scan;
    typename BlockStore::TempStorage store;
  } temp_storage;

	__shared__ data_type prev_aggregate;

  const int block_offset = blockIdx.x * hierarchical_block_size * thread_data_size;

  data_type thread_data[thread_data_size];
  BlockLoad (temp_storage.load).Load (in + block_offset, thread_data);
  __syncthreads ();

	data_type block_aggregate {}; 

  BlockScan (temp_storage.scan).ExclusiveSum (thread_data, thread_data, block_aggregate);
  __syncthreads ();

  if (blockIdx.x == 0)
    {
      if (threadIdx.x == 0)
        store (block_statuses + blockIdx.x, 2, block_aggregate);
    }
  else
    {
      if (threadIdx.x == 0)
        store (block_statuses + blockIdx.x, 1, block_aggregate);

      block_status_word prev_block_status;

      if (threadIdx.x / WARP_SIZE == 0)
        {
          int predecessor_idx = blockIdx.x - WARP_SIZE + threadIdx.x;
          data_type thread_sum = data_type ();

					while (true)
          {
            prev_block_status = load (block_statuses + predecessor_idx);

            while (prev_block_status.flag == 0)
              prev_block_status = load (block_statuses + predecessor_idx);

            if (__all_sync (WARP_MASK, prev_block_status.flag < 2))
							{
								thread_sum += prev_block_status.data;
                predecessor_idx -= WARP_SIZE;
						    continue;
							}
						else
							{
								const int final_mask = __ballot_sync (WARP_MASK, prev_block_status.flag > 1);
								const int rightmost_thread = WARP_SIZE - 1 - __clz (final_mask);

								thread_sum += threadIdx.x % WARP_SIZE < rightmost_thread ? 0 : prev_block_status.data;
								break;
							}
          } 

					data_type aggregate = WarpReduce (temp_storage.warp_reduce).Sum (thread_sum);

          if (threadIdx.x == 0)
            {
              store (block_statuses + blockIdx.x, 2, aggregate + block_aggregate);
							prev_aggregate = aggregate;
            }
        }
    }
  __syncthreads ();

	const data_type reg_prev_aggregate = prev_aggregate;
	for (int i = 0; i < thread_data_size; i++)
		thread_data[i] += reg_prev_aggregate;

  BlockStore (temp_storage.store).Store (out + block_offset, thread_data);
}

__global__ void kernel_init (int blocks_count, block_status_word *block_statuses)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < WARP_SIZE)
		{
			block_statuses[i].flag = 42;
			block_statuses[i].data = 0;
		}
  else if (i < blocks_count)
    block_statuses[i].flag = 0;
}

template <typename data_type>
void measure_single_decoupled_lookback_kernel (
    int size,
    void *helper_buffer_ptr,
    const data_type *in,
    data_type *out)
{
  block_status_word *helper_buffer = reinterpret_cast<block_status_word*> (helper_buffer_ptr);
  const int elements_per_block = thread_data_size * hierarchical_block_size;
  const int blocks = (size + elements_per_block - 1) / elements_per_block;

	{
		const int bs = 128;
		const int gs = (WARP_SIZE + blocks + bs - 1) / bs;

		kernel_init<<<gs, bs>>> (blocks, helper_buffer);
	}

  single_kernel_decoupled_lookback_scan<<<blocks, hierarchical_block_size>>> (in, out, helper_buffer + WARP_SIZE);
}

template <typename data_type>
size_t calculate_helper_storage_size (int size, implementation_type implementation)
{
  switch (implementation)
  {
	case implementation_type::cub:
	{
		size_t temp_storage_bytes = 0;
		cub::DeviceScan::ExclusiveSum (
				static_cast<data_type*> (nullptr),
				temp_storage_bytes,
				static_cast<data_type*> (nullptr),
				static_cast<data_type*> (nullptr),
				size);

		return temp_storage_bytes;
	}
  case implementation_type::hierarchical:
  {
    const int elements_per_block = thread_data_size * hierarchical_block_size;
    const int blocks = (size + elements_per_block - 1) / elements_per_block;
    return blocks * sizeof (data_type);
  }
  case implementation_type::single_kernel:
  case implementation_type::single_kernel_decoupled_lookback:
  {
    const int elements_per_block = thread_data_size * hierarchical_block_size;
    const int blocks = WARP_SIZE + (size + elements_per_block - 1) / elements_per_block;
    return blocks * sizeof (block_status_word);
  }
  }

  return 0;
}

template <typename data_type>
__global__ void fill (int n, data_type val, data_type *data)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < n)
    data[i] = val;
}

template <typename data_type>
void measure_cub (
    int size,
    size_t helper_buffer_size,
    void *helper_buffer,
    const data_type *in,
          data_type *out)
{
  cub::DeviceScan::ExclusiveSum (
      helper_buffer,
      helper_buffer_size,
      in, out, size);
}

template <typename data_type>
void measure (implementation_type implementation, bool check = false)
{
  float ms = 0;
  const int max_size = thread_data_size * hierarchical_block_size * 2100;
  data_type *in = nullptr;
  data_type *out = nullptr;

  const int iterations = 10;

  cudaMalloc (&in, max_size * sizeof (data_type));
  cudaMalloc (&out, max_size * sizeof (data_type));
  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  // for (int blocks_count : { 1024 })
  for (int blocks_count = 1; blocks_count < 2 * 1024; blocks_count += 32)
  {
    const int size = hierarchical_block_size * thread_data_size * blocks_count;
    const int threads_in_block = 256;
    const int blocks = (size + threads_in_block - 1) / threads_in_block;

    const size_t helper_buffer_size = calculate_helper_storage_size<data_type> (size, implementation);
    void *helper_buffer = nullptr;
    cudaMalloc (&helper_buffer, helper_buffer_size);

    fill<<<blocks, threads_in_block>>> (size, 1, in);
    fill<<<blocks, threads_in_block>>> (size, 0, out);

    cudaEventRecord (start);

    for (int iteration = 0; iteration < iterations; iteration++)
    {
      switch (implementation)
      {
      case implementation_type::copy: cudaMemcpyAsync (out, in, size * sizeof (data_type), cudaMemcpyDeviceToDevice); break;
      case implementation_type::hierarchical: measure_hierarchical (size, helper_buffer, in, out); break;
      case implementation_type::single_kernel: measure_single_kernel (size, helper_buffer, in, out); break;
      case implementation_type::single_kernel_decoupled_lookback: measure_single_decoupled_lookback_kernel (size, helper_buffer, in, out); break;
      case implementation_type::cub: measure_cub (size, helper_buffer_size, helper_buffer, in, out); break;
      }
    }

    cudaEventRecord (stop);
    cudaEventSynchronize (stop);
    cudaEventElapsedTime (&ms, start, stop);
    cudaFree (helper_buffer);

    ms /= iterations;

    if (check)
    {
      int result = 0;
      cudaMemcpy (&result, out + size - 1, sizeof (data_type), cudaMemcpyDeviceToHost);

      if (result != size - 1)
        std::cerr << "Check violation! " << result << " != " << size - 1 << std::endl;
    }

    std::cout << size << ", " << ms << "\n";
  }

  cudaEventDestroy (start);
  cudaEventDestroy (stop);
  cudaFree (out);
  cudaFree (in);
}

int main ()
{
  cudaSetDevice (0);

  /*
  std::cout << "copy:\n";
  measure<int> (implementation_type::copy);

  std::cout << "\nhierarchical:\n";
  measure<int> (implementation_type::hierarchical, true);

  std::cout << "\nsingle kernel:\n";
  measure<int> (implementation_type::single_kernel, true);
   */

  std::cout << "\ndecoupled:\n";
  measure<int> (implementation_type::single_kernel_decoupled_lookback, true);

  return 0;

  std::cout << "\ncub:\n";
  measure<int> (implementation_type::cub, true);

  return 0;
}

#include <iostream>
#include <stdexcept>
#include <memory>

#include <cub/cub.cuh>

#define WARP_SIZE 32
#define WARP_MASK 0xFFFFFFFF

void chk (cudaError_t status)
{
  if (cudaSuccess != status)
    throw std::runtime_error ("CUDA Error!");
}

struct block_status_word
{
  int32_t flag;
  int32_t data;
};

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

__device__ int32_t sm_id ()
{
  int32_t result;
  asm volatile("mov.u32 %0, %smid;" : "=r"(result) : );

  return result;
}

__global__ void kernel (volatile block_status_word *block_statuses, int32_t *inv_spin_cnt)
{
  const int bid = blockIdx.x;

  int32_t invalid_spin_count = 0;

  if (bid == 0)
    {
      store (block_statuses + bid, 1, bid);
    }
  else
    {
      block_status_word prev_block_status;

      do
      {
        prev_block_status = load (block_statuses + bid - 1);
        invalid_spin_count++;
      } while (prev_block_status.flag == 0);

      store (block_statuses + bid, 1, prev_block_status.data + bid);
    }

  if (threadIdx.x == 0)
    if (inv_spin_cnt)
      inv_spin_cnt[blockIdx.x] = invalid_spin_count;
}

__global__ void kernel_init (int blocks_count, block_status_word *block_statuses)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < WARP_SIZE)
    block_statuses[i].flag = 42;
  else if (i < blocks_count)
    block_statuses[i].flag = 0;
}

__global__ void kernel_warp (volatile block_status_word *block_statuses, int32_t *inv_spin_cnt, int32_t *fin_spin_cnt)
{
  const int bid = blockIdx.x;

  int32_t invalid_spin_count = 0;
  int32_t final_spin_count = 0;

  if (bid == 0)
    {
      if (threadIdx.x == 0)
        {
          store (block_statuses + bid, 2, bid);
        }
      __syncthreads ();
    }
  else
    {
      if (threadIdx.x == 0)
        store (block_statuses + bid, 1, bid);

      block_status_word prev_block_status;

      if (threadIdx.x / WARP_SIZE == 0)
        {
          int predecessor_idx = bid - threadIdx.x - 1;

          do
            {
              prev_block_status = load (block_statuses + predecessor_idx);

              while (prev_block_status.flag == 0)
              {
                invalid_spin_count++;
                prev_block_status = load (block_statuses + predecessor_idx);
              }

              final_spin_count++;
              predecessor_idx -= WARP_SIZE;
            } while (__all_sync (WARP_MASK, prev_block_status.flag < 2));
        }

      if (threadIdx.x == 0)
        store (block_statuses + bid, 2, prev_block_status.data + bid);
      __syncthreads ();
    }

  typedef cub::WarpReduce<int32_t> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage;
  int aggregate = WarpReduce(temp_storage).Reduce (invalid_spin_count, cub::Max ());


  if (threadIdx.x == 0)
  {
    if (inv_spin_cnt)
      inv_spin_cnt[blockIdx.x] = aggregate;

    if (fin_spin_cnt)
      fin_spin_cnt[blockIdx.x] = final_spin_count;
  }
}

int main()
{
  cudaSetDevice (1);
  cudaDeviceReset ();

  // for (int blocks = 1; blocks < 23000; blocks++)
  for (int blocks = 1; blocks < 2 * 1024; blocks += 32)
  {
    dim3 grid (blocks, 1, 1);
    const unsigned int block_size = 1;

    const unsigned int n = block_size * grid.x * grid.y * grid.z;

    block_status_word *block_statuses = 0;
    chk (cudaMalloc (&block_statuses, (n + WARP_SIZE) * sizeof (block_status_word)));

    int32_t *inv_spin_cnt = 0;
    int32_t *fin_spin_cnt = 0;

    if (0)
    {
      chk (cudaMalloc (&inv_spin_cnt, blocks * sizeof (int32_t)));
      chk (cudaMalloc (&fin_spin_cnt, blocks * sizeof (int32_t)));
    }

    // chk (cudaMemset (block_statuses, 0, (n + WARP_SIZE) * sizeof (block_status_word)));
    {
      const int bs = 128;
      const int gs = (blocks + bs - 1) / bs;

      kernel_init<<<gs, bs>>> (blocks, block_statuses);
    }

    cudaEvent_t begin;
    cudaEvent_t end;

    chk (cudaEventCreate (&begin));
    chk (cudaEventCreate (&end));

    chk (cudaEventRecord (begin));

    if (1)
      kernel<<<grid, block_size>>> (block_statuses + WARP_SIZE, inv_spin_cnt);
    else
      kernel_warp<<<grid, 32>>> (block_statuses + WARP_SIZE, inv_spin_cnt, fin_spin_cnt);

    chk (cudaEventRecord (end));
    chk (cudaEventSynchronize (end));

    float elapsed;
    chk (cudaEventElapsedTime (&elapsed, begin, end));

    chk (cudaEventDestroy (end));
    chk (cudaEventDestroy (begin));

    std::cout << blocks * 1536 << ", " << elapsed << std::endl;

    if (0)
    {
      std::unique_ptr<int32_t[]> host_inv_spin_cnt (new int32_t[blocks]);
      std::unique_ptr<int32_t[]> host_fin_spin_cnt (new int32_t[blocks]);

      cudaMemcpy (host_inv_spin_cnt.get (), inv_spin_cnt, blocks * sizeof (int32_t), cudaMemcpyDeviceToHost);
      cudaMemcpy (host_fin_spin_cnt.get (), fin_spin_cnt, blocks * sizeof (int32_t), cudaMemcpyDeviceToHost);

      for (int i = 0; i < blocks; i++)
        std::cout << host_inv_spin_cnt[i] << "\n";
      // std::cout << host_fin_spin_cnt[i] << "\n";
      // std::cout << "B" << i << ": inv(" << host_inv_spin_cnt[i] << ") fin(" << host_fin_spin_cnt[i] << ")\n";
    }

    chk (cudaFree (fin_spin_cnt));
    chk (cudaFree (inv_spin_cnt));
    chk (cudaFree (block_statuses));
  }

  return 0;
}

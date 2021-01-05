#include <algorithm>
#include <iostream>
#include <thread>
#include <vector>

#include <type_traits>

#include <cuda/atomic>

#include <atomic>
#include <xmmintrin.h>

class barrier_class
{
public:
  barrier_class () = delete;
  explicit barrier_class (int threads_count_arg)
      : threads_count (threads_count_arg)
      , barrier_epoch (0)
      , threads_in_barrier (0)
  { }

  void operator ()()
  {
    if (threads_count == 1)
      return;

    const unsigned int thread_epoch = barrier_epoch.load ();

    if (threads_in_barrier.fetch_add (1) == threads_count - 1)
    {
      threads_in_barrier.store (0);
      barrier_epoch.fetch_add (1);
    }
    else
    {
      while (thread_epoch == barrier_epoch.load ())
      {
        _mm_pause ();
      }
    }
  }

private:
  int threads_count {};
  std::atomic<unsigned int> barrier_epoch;
  std::atomic<unsigned int> threads_in_barrier;
};

class status_word
{
public:
  int32_t flag;
  int32_t data;

  status_word () = default;
  __device__ status_word (
      int32_t flag_arg,
      int32_t data_arg)
    : flag (flag_arg)
    , data (data_arg)
  {}

  __device__ status_word (const cuda::std::__3::__cxx_atomic_alignment_wrapper_t<status_word> &rhs)
    : flag (rhs.__a_held.flag)
    , data (rhs.__a_held.data)
  {
  }
};

static_assert(std::is_trivially_copyable<status_word>::value, "ohoh"    );

__device__ bool operator ==(const status_word &lhs, const status_word &rhs)
{
  return lhs.data == rhs.data;
}

__device__ status_word load (volatile status_word *ptr)
{
  int64_t tmp = *reinterpret_cast<volatile int64_t*> (ptr);
  status_word &rhs = reinterpret_cast<status_word&> (tmp);
  return rhs;
}

__device__ void store (volatile status_word *ptr, int32_t flag, int32_t data)
{
  int64_t tmp;
  status_word &rhs = reinterpret_cast<status_word&> (tmp);

  rhs.flag = flag;
  rhs.data = data;

  *reinterpret_cast<volatile int64_t*> (ptr) = tmp;
}

__global__ void kernel (
    unsigned int n,
    float value,
    float threshold,
    float *data)
{
  bool stop = false;

  unsigned int special_value = (unsigned int) -1;
  unsigned int last_iteration = special_value;

  for (unsigned iteration = 0; iteration < last_iteration; iteration++)
    {
      for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x)
        {
          data[i] += value;

          if (data[i] > threshold)
            stop = true;
        }

      if (stop)
        last_iteration = iteration;
    }
}

__global__ void kernel_global_volatile (
    unsigned int n,
    unsigned int sensor,
    unsigned int offset,
    float value,
    float threshold,
    float *data,
    volatile status_word *flag)
{
  bool stop = false;

  unsigned int special_value = (unsigned int) -1;
  unsigned int last_iteration = special_value;

  for (unsigned iteration = 0; iteration < last_iteration; iteration++)
    {
      bool sensor_owner = false;

      for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x)
        {
          data[i] += value;

          if (i + offset == sensor)
            {
              sensor_owner = true;

              if (data[i] > threshold)
                stop = true;

              store (flag, stop, iteration + 1);
            }
        }

      if (sensor_owner)
        {
          if (stop)
            last_iteration = iteration;
        }
      else
        {
          if (special_value == last_iteration)
            {
              status_word iteration_state = load (flag);

              while (iteration_state.data <= iteration)
                iteration_state = load (flag);

              if (iteration_state.flag)
                last_iteration = iteration_state.data;
            }
        }
    }
}

__global__ void kernel_global_volatile_block (
    unsigned int n,
    unsigned int sensor,
    unsigned int offset,
    float value,
    float threshold,
    float *data,
    volatile status_word *flag)
{
  bool stop = false;

  unsigned int special_value = -1;
  unsigned int last_iteration = special_value;

  __shared__ unsigned int block_last_iteration;

  if (threadIdx.x == 0)
    block_last_iteration = special_value;
  __syncthreads ();

  for (int iteration = 0; iteration < last_iteration; iteration++)
    {
      for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x)
        {
          data[i] += value;

          if (i + offset == sensor)
            {
              if (data[i] > threshold)
                stop = true;

              store (flag, stop, iteration + 1);
            }
        }

      if (last_iteration == special_value)
        {
          if (threadIdx.x == 0)
            {
              status_word iteration_state = load (flag);

              while (iteration_state.data <= iteration)
                iteration_state = load (flag);

              if (iteration_state.flag)
                block_last_iteration = iteration_state.data;
            }
          __syncthreads ();

          last_iteration = block_last_iteration;
        }
    }
}

__global__ void kernel_global_volatile_block_sleep (
    unsigned int n,
    unsigned int sensor,
    unsigned int offset,
    float value,
    float threshold,
    float *data,
    volatile status_word *flag)
{
  unsigned int special_value = -1;
  unsigned int last_iteration = special_value;

  __shared__ unsigned int block_last_iteration;

  if (threadIdx.x == 0)
    block_last_iteration = special_value;
  __syncthreads ();

  for (int iteration = 0; iteration < last_iteration; iteration++)
    {
      for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x)
        {
          data[i] += value;

          if (i + offset == sensor)
            {
              bool stop = data[i] > threshold;

              store (flag, stop, iteration + 1);
            }
        }

      if (last_iteration == special_value)
        {
          if (threadIdx.x == 0)
            {
              status_word iteration_state = load (flag);

              while (iteration_state.data <= iteration)
                {
                  __nanosleep (250);
                  iteration_state = load (flag);
                }

              if (iteration_state.flag)
                block_last_iteration = iteration_state.data;
            }
          __syncthreads ();

          last_iteration = block_last_iteration;
        }
    }
}

__global__ void kernel_atomic (
    unsigned int n,
    unsigned int sensor,
    unsigned int offset,
    float value,
    float threshold,
    float *data,
    cuda::atomic<status_word, cuda::thread_scope_device> *flag)
{
  bool stop = false;

  unsigned int special_value = -1;
  unsigned int last_iteration = special_value;

  __shared__ unsigned int block_last_iteration;

  if (threadIdx.x == 0)
    block_last_iteration = special_value;
  __syncthreads ();

  for (int iteration = 0; iteration < last_iteration; iteration++)
    {
      for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x)
        {
          data[i] += value;

          if (i + offset == sensor)
            {
              if (data[i] > threshold)
                stop = true;

              flag->store (status_word {stop, iteration + 1}, cuda::memory_order_relaxed);
              flag->notify_all ();
            }
        }

      if (last_iteration == special_value)
        {
          if (threadIdx.x == 0)
            {
              status_word iteration_state = flag->load (cuda::memory_order_relaxed);

              if (true)
                {
                  if (false)
                    {
                      if (iteration_state.data <= iteration)
                        {
                          flag->wait (iteration_state, cuda::memory_order_relaxed);
                          iteration_state = flag->load (cuda::memory_order_relaxed);
                        }
                    }
                  else
                    {
                      if (iteration_state.data <= iteration)
                        iteration_state = flag->wait_with_ret (iteration_state, cuda::memory_order_relaxed);
                    }
                }
              else
                {
                  while (iteration_state.data <= iteration)
                    iteration_state = flag->load (cuda::memory_order_relaxed);
                }

              /*
              flag->wait (
                  iteration_state,
                  [&] (const status_word &loaded, const status_word &local)
                  {
                    iteration_state = loaded;
                    return loaded.data <= iteration;
                  },
                  cuda::memory_order_relaxed);
                  */

              if (iteration_state.flag)
                block_last_iteration = iteration_state.data;
            }
          __syncthreads ();

          last_iteration = block_last_iteration;
        }
    }
}

__global__ void initialize_atomic (cuda::atomic<status_word, cuda::thread_scope_device> *flag)
{
  new (flag) cuda::atomic<status_word, cuda::thread_scope_device> ({ 0, 0 });
}

enum class mode_enum
{
  basic,
  volatile_global,
  volatile_global_block,
  volatile_global_block_sleep,
  atomic,
  COUNT
};

__global__ void validate (const float *data, const float reference, float value, int* result)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (fabs (data[i] - reference) > (value / 4.0f))
    {
      printf ("data[%u] = %g != %g\n", i, data[i], reference);
      *result = 1;
    }
}

float run (mode_enum mode, bool use_second_gpu, bool print = false)
{
  int threads = 64;
  int blocks_per_sm = 0;

  int dev = 0;

  cudaDeviceProp device_prop {};
  cudaGetDeviceProperties (&device_prop, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor (&blocks_per_sm, kernel_global_volatile, threads, 0);

  int blocks = device_prop.multiProcessorCount * blocks_per_sm;

  int cpu_threads = use_second_gpu ? 2 : 1;
  int multiplier = 19 / cpu_threads;
  int n = multiplier * blocks * threads;


  if (print)
    {
      std::cout << "SMs count = " << device_prop.multiProcessorCount << "\n";
      std::cout << "Blocks per SM = " << blocks_per_sm << "\n";
      std::cout << "Total blocks count = " << blocks << "\n";
    }


  barrier_class barrier (cpu_threads);

  status_word *first_gpu_word = nullptr;
  cuda::atomic<status_word, cuda::thread_scope_device> *first_atomic = nullptr;

  std::vector<std::thread> workers;
  std::vector<float> mss (cpu_threads, 0.0f);

  float *reference = nullptr;

  for (int tid = 0; tid < cpu_threads; tid++)
    workers.push_back(std::thread ([&, tid] () {
      cudaSetDevice (tid);

      float *data {};
      cudaMalloc (&data, sizeof (float) * n);
      cudaMemset (data, 0, sizeof (float) * n);

      status_word *flag {};

      int *result {};

      cudaMalloc (&flag, sizeof (status_word));
      cudaMemset (flag, 0, sizeof (status_word));

      if (tid == 0)
        first_gpu_word = flag;

      cudaMalloc (&result, sizeof (int));
      cudaMemset (result, 0, sizeof (int));

      cuda::atomic<status_word, cuda::thread_scope_device> *atomic = nullptr;
      cudaMalloc (&atomic, sizeof (cuda::atomic<status_word, cuda::thread_scope_device>));

      if (tid == 0)
        first_atomic = atomic;

      initialize_atomic<<<1, 1>>> (atomic);

      int sensor = n / 2;

      if (tid == 0)
        reference = data + sensor;

      float threshold = 14.0f;
      float value = 1.0f;

      cudaEvent_t begin, end;

      cudaEventCreate (&begin);
      cudaEventCreate (&end);

      cudaDeviceSynchronize ();
      barrier ();

      cudaEventRecord (begin);

      int offset = tid * n;

      switch (mode)
      {
      case mode_enum::basic: kernel<<<blocks, threads>>> (n, value, threshold, data); break;
      case mode_enum::volatile_global: kernel_global_volatile<<<blocks, threads>>> (n, sensor, offset, value, threshold, data, first_gpu_word); break;
      case mode_enum::volatile_global_block: kernel_global_volatile_block<<<blocks, threads>>> (n, sensor, offset, value, threshold, data, first_gpu_word); break;
      case mode_enum::volatile_global_block_sleep: kernel_global_volatile_block_sleep<<<blocks, threads>>> (n, sensor, offset, value, threshold, data, first_gpu_word); break;
      case mode_enum::atomic: kernel_atomic<<<blocks, threads>>> (n, sensor, offset, value, threshold, data, first_atomic);
      }

      cudaEventRecord (end);
      cudaEventSynchronize (end);

      float ms = 0;
      cudaEventElapsedTime (&ms, begin, end);

      if (print)
        std::cout << "\nElapsed: " << ms << " ms" << std::endl;

      float cpu_ref = 0.0f;
      cudaMemcpy (&cpu_ref, reference, sizeof (float), cudaMemcpyDeviceToHost);

      validate<<<blocks * multiplier, threads>>> (data, cpu_ref, value, result);

      int host_result = 0;
      cudaMemcpy (&host_result, result, sizeof (int), cudaMemcpyDeviceToHost);

      barrier ();

      if (host_result)
        throw std::runtime_error ("Invalid result");

      cudaEventDestroy (end);
      cudaEventDestroy (begin);

      cudaFree (atomic);
      cudaFree (result);

      cudaFree (flag);
      cudaFree (data);

      mss[tid] = ms;
    }));

  for (auto &worker: workers)
    worker.join ();

  return *std::max_element (mss.begin (), mss.end ());
}

int main ()
{
  int max_iterations = 1;
  std::vector<float> elapsed (max_iterations);

  // cudaSetDevice (1);
  // cudaDeviceEnablePeerAccess (0, 0);
  // cudaSetDevice (0);
  // cudaDeviceEnablePeerAccess (1, 0);

  for (int iteration = 0; iteration < max_iterations; iteration++)
    elapsed[iteration] = run (mode_enum::volatile_global_block_sleep, false, true);

  for (float &ms: elapsed)
    std::cout << ms << "\n";

  return 0;
}

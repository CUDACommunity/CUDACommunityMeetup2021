#include <cuda/std/atomic>

using atomic = cuda::std::atomic<int>;
using atomic_ptr = atomic*;

__global__ void initialize_atomic_kernel (atomic_ptr ptr, int value)
{
	new (ptr) atomic (value);
}

__global__ void count_kernel (atomic_ptr count)
{
	count->fetch_add (1, cuda::std::memory_order_relaxed);
}


int main ()
{
	atomic_ptr count;	
	cudaMalloc (&count, sizeof (atomic));
	initialize_atomic_kernel<<<1, 1>>> (count, 0);
	count_kernel<<<1, 1>>> (count);

	cudaFree (count);
}

#include "not_interesting.cuh"
#include <cuda/runtime_api.hpp>
#include <memory>

int main()
{
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	int numElements = 50000;
	size_t size = numElements * sizeof(float);

	auto h_A = std::make_unique<float[]>(numElements);
	auto h_B = std::make_unique<float[]>(numElements);
	auto h_C = std::make_unique<float[]>(numElements);

	// Initialize the host input vectors
	for (int i = 0; i < numElements; ++i)
	{
		h_A.get()[i] = rand()/(float)RAND_MAX;
		h_B.get()[i] = rand()/(float)RAND_MAX;
	}

	auto device = cuda::device::current::get();
	auto d_A = cuda::memory::device::make_unique<float[]>(device, numElements);
	auto d_B = cuda::memory::device::make_unique<float[]>(device, numElements);
	auto d_C = cuda::memory::device::make_unique<float[]>(device, numElements);

	cuda::memory::copy(d_A.get(), h_A.get(), size);
	cuda::memory::copy(d_B.get(), h_B.get(), size);

	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A.get(), d_B.get(), d_C.get(), numElements);
	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cuda::memory::copy(h_C.get(), d_C.get(), size);

	verify_correctness(h_A.get(), h_B.get(), h_C.get(), numElements);
}


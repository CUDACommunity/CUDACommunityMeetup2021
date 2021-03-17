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

	cuda::grid::block_dimension_t threadsPerBlock {256};
	auto blocksPerGrid = static_cast<cuda::grid::dimension_t>(
		(numElements + threadsPerBlock - 1) / threadsPerBlock);
	auto launch_config = cuda::make_launch_config(blocksPerGrid, threadsPerBlock);
	device.launch( // No chevrons nor optional arguments. Launching is always just:
		vectorAdd,                                     // 1. kernel name
		launch_config,                                 // 2. launch config
		d_A.get(), d_B.get(), d_C.get(), numElements); // 3. kernel arguments

	cuda::memory::copy(h_C.get(), d_C.get(), size);

	verify_correctness(h_A.get(), h_B.get(), h_C.get(), numElements);
}


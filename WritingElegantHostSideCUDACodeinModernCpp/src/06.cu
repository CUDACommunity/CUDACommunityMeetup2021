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

	// Copy the host input vectors A and B in host memory to the device input vectors in
	// device memory
	err = cudaMemcpy(d_A.get(), h_A.get(), size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_B.get(), h_B.get(), size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

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

	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	err = cudaMemcpy(h_C.get(), d_C.get(), size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	verify_correctness(h_A.get(), h_B.get(), h_C.get(), numElements);
}


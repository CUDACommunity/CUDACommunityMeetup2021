#include "not_interesting.cuh"

int main()
{
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	int numElements = 50000;
	size_t size = numElements * sizeof(float);

	// Allocate the host input vector A
	float *h_A = (float *)malloc(size);

	// Allocate the host input vector B
	float *h_B = (float *)malloc(size);

	// Allocate the host output vector C
	float *h_C = (float *)malloc(size);

	// Verify that allocations succeeded
	if (h_A == NULL || h_B == NULL || h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialize the host input vectors
	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = rand()/(float)RAND_MAX;
		h_B[i] = rand()/(float)RAND_MAX;
	}

	// Allocate the device input vector A
	float *d_A = NULL;
	err = cudaMalloc((void **)&d_A, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
	float *d_B = NULL;
	err = cudaMalloc((void **)&d_B, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	float *d_C = NULL;
	err = cudaMalloc((void **)&d_C, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the host input vectors A and B in host memory to the device input vectors in
	// device memory
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	verify_correctness(h_A, h_B, h_C, numElements);

	// Free device global memory
	err = cudaFree(d_A);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_B);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_C);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);
}


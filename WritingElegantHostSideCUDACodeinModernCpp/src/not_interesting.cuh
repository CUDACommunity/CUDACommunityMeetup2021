/**
 * The original code (especially in vectorAdd.cu) is:
 *
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#include <stdio.h>
#include <cuda_runtime.h>
#include <exception>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		C[i] = A[i] + B[i];
	}
}

void fill_with_random_data(float* h_A, float* h_B, int numElements)
{
	// Initialize the host input vectors
	for (int i = 0; i < numElements; ++i) {
		h_A[i] = rand()/(float)RAND_MAX;
		h_B[i] = rand()/(float)RAND_MAX;
	}
}

void verify_correctness(float* h_A, float* h_B, float* h_C, int numElements)
{
	// Verify that the result vector is correct
	for (int i = 0; i < numElements; ++i)
	{
		if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
		{
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}
}

constexpr inline int div_rounding_up(int dividend, int divisor)
{
	// This can't overflow but is not the fastest solution possible.
	// Actually, implementing this is an interesting problem, see:
	// https://stackoverflow.com/q/2745074/1593077
	return (dividend / divisor) + ((dividend % divisor == 0) ? 0 : 1);
}

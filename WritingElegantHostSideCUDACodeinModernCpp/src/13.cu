#include "not_interesting.cuh"
#include <cuda/runtime_api.hpp>
#include <memory>
#include <algorithm>
#include <random>

int main()
{
	int numElements = 50000;
	size_t size = numElements * sizeof(float);

	auto h_A = std::make_unique<float[]>(numElements);
	auto h_B = std::make_unique<float[]>(numElements);
	auto h_C = std::make_unique<float[]>(numElements);

	std::random_device randomDevice{};
	std::mt19937 randomEngine{randomDevice()};
	std::uniform_real_distribution<float> distribution{0.0, 1.0};
	auto dataGenerator = [&]() { return distribution(randomEngine); };
	std::generate_n(h_A.get(), numElements, dataGenerator);
	std::generate_n(h_B.get(), numElements, dataGenerator);

	auto device = cuda::device::current::get();
	auto d_A = cuda::memory::device::make_unique<float[]>(device, numElements);
	auto d_B = cuda::memory::device::make_unique<float[]>(device, numElements);
	auto d_C = cuda::memory::device::make_unique<float[]>(device, numElements);

	cuda::memory::copy(d_A.get(), h_A.get(), size);
	cuda::memory::copy(d_B.get(), h_B.get(), size);

	cuda::grid::block_dimension_t threadsPerBlock {256};
	auto blocksPerGrid = static_cast<cuda::grid::dimension_t>(
		div_rounding_up(numElements, threadsPerBlock));
	auto launch_config = cuda::make_launch_config(blocksPerGrid, threadsPerBlock);
	device.launch(vectorAdd, launch_config,
		d_A.get(), d_B.get(), d_C.get(), numElements);

	cuda::memory::copy(h_C.get(), d_C.get(), size);

	verify_correctness(h_A.get(), h_B.get(), h_C.get(), numElements);
}


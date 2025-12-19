
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <random>
#include <vector>

#ifndef __CUDACC__  
#define __CUDACC__
#endif

#include <crt/device_functions.h>

#include <chrono>

using Clock = std::chrono::steady_clock;
using ms = std::chrono::duration<double, std::milli>;


#define CHECK_CUDA_STATUS(status) { if (status != cudaSuccess) { \
    fprintf(stderr, "CUDA Error: %s (error code %d) at %s:%d\n", \
        cudaGetErrorString(status), status, __FILE__, __LINE__); exit(EXIT_FAILURE); } }


#define BLOCK_DIM 1024
#define COARSE_FACTOR 2


void reduction_cpu(float* out, float* in, int length)
{
	double sum = 0.0;
	for (int i = 0; i < length; i++)
	{
		sum += (double)in[i];
	}
	*out = sum;
}


__global__ void reduction_naive_kernel(float* out, float* in, int length)
{
	int tx = threadIdx.x * 2;

	int segment_start = 2 * blockDim.x * blockIdx.x;

	for (int stride = 1; stride <= blockDim.x; stride *= 2)
	{
		if (tx + segment_start + stride < length && threadIdx.x % stride == 0)
		{
			in[tx + segment_start] += in[tx + segment_start + stride];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
		atomicAdd(out, in[segment_start]);
}

__global__ void reduction_improved_kernel(float* out, float* in, int length)
{
	int tx = threadIdx.x;
	int segment_start = blockDim.x * 2 * blockIdx.x;

	for (int offset = blockDim.x; offset >= 1; offset /= 2)
	{
		if (tx < offset)
		{
			in[tx + segment_start] += in[offset + tx + segment_start];
		}
		__syncthreads();
	}

	if (tx == 0)
		atomicAdd(out, in[segment_start]);
}


__global__ void reduction_shared_kernel(float* out, const float* in, int length)
{
	__shared__ float smem[BLOCK_DIM];

	int segment_start = blockDim.x * 2 * blockIdx.x;

	smem[threadIdx.x] = in[threadIdx.x + segment_start] + in[threadIdx.x + segment_start + blockDim.x];

	for (int offset = blockDim.x / 2; offset >= 1; offset /= 2)
	{
		__syncthreads();
		if (threadIdx.x < offset)
		{
			smem[threadIdx.x] += smem[threadIdx.x + offset];
		}
	}

	if (threadIdx.x == 0)
		atomicAdd(out, smem[0]);
}


__global__ void reduction_shared_coarse_kernel(float* out, const float* in, int length)
{
	__shared__ float smem[BLOCK_DIM];

	int segment_start = blockIdx.x * blockDim.x * 2 * COARSE_FACTOR;
	int tx = segment_start + threadIdx.x;

	float sum = in[tx];
	for (int tile = 1; tile < 2 * COARSE_FACTOR; tile++)
	{
		sum += in[tx + blockDim.x * tile];
	}

	smem[threadIdx.x] = sum;

	for (int offset = blockDim.x / 2; offset >= 1; offset /= 2)
	{
		__syncthreads();
		if (threadIdx.x < offset)
		{
			smem[threadIdx.x] += smem[threadIdx.x + offset];
		}
	}

	if (threadIdx.x == 0)
		atomicAdd(out, smem[0]);
}


__inline__ __device__ float reduce_warp(float value)
{
	for (int offset = 16; offset > 0; offset /= 2)
	{
		value += __shfl_down_sync(0xffffffff, value, offset);
	}
	return value;
}

__global__ void reduction_warp_shuffle_kernel(float* out, const float* in, int length)
{
	int tid = blockIdx.x * 2 * COARSE_FACTOR * blockDim.x + threadIdx.x;

	float value = tid < length ? in[tid] : 0.f;

	for (int i = 1; i < 2 * COARSE_FACTOR; i++)
	{
		if (tid + i * blockDim.x < length)
			value += in[tid + i * blockDim.x];
	}

	// 1. reduce warp
	value = reduce_warp(value);

	// 2. store warp reduced values in smem
	__shared__ float reduced_values[32];

	int lane_id = threadIdx.x & 0x1f;
	int warp_id = threadIdx.x >> 5;

	if (lane_id == 0)
	{
		reduced_values[warp_id] = value;
	}
	__syncthreads();

	// 3. pick warp 0 and init its threads with values from reduced_values
	// make sure that threads have a valid value from the corresponding warp (lane_id == warp_id)
	int num_warps = (blockDim.x + 31) / 32;

	if (warp_id == 0)
	{
		value = lane_id < num_warps ? reduced_values[lane_id] : 0.f;

		value = reduce_warp(value);
	}

	// 4. update global memory
	if (warp_id == 0 && lane_id == 0)
	{
		atomicAdd(out, value);
	}

	/*if (lane_id == 0 && warp_id == 0)
		printf("Thread %d from warp %d block %d has %f\n", threadIdx.x, warp_id, blockIdx.x, value);*/
}


std::vector<float> make_uniform(int n, unsigned int seed = 1234) {
	std::mt19937 rng(seed);                     // deterministic RNG
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	std::vector<float> v(n);
	for (auto& x : v) {
		x = dist(rng);
	}
	return v;
}

void benchmark_reduction_cpu(int length, unsigned int iters)
{
	std::vector<float> in = make_uniform(length);
	float out = 0.f;

	// warmup: make sure caches etc. are "hot"
	reduction_cpu(&out, in.data(), length);

	auto start = Clock::now();
	for (int i = 0; i < iters; ++i) {
		reduction_cpu(&out, in.data(), length);
	}
	auto end = Clock::now();

	ms total = end - start;
	double avg_ms = total.count() / iters;

	std::cout << "reduction_cpu" << "\n";
	std::cout << "  iters          = " << iters << "\n";
	std::cout << "  avg time       = " << avg_ms << " ms\n";
	std::cout << "  checksum       = " << out << "\n";
}


template <typename Kernel>
void benchmark_kernel(const char* name,
	dim3 gridSize,
	dim3 blockSize,
	int length,
	int iters,
	double bytes_per_iter,   // how many bytes the kernel touches once
	Kernel kernel)           // the kernel symbol
{
	std::vector<float> h_in = make_uniform(length);
	float h_out = 0.f;

	float* d_in = nullptr;
	float* d_out = nullptr;

	CHECK_CUDA_STATUS(cudaMalloc(&d_in, length * sizeof(float)));
	CHECK_CUDA_STATUS(cudaMalloc((void**)&d_out, sizeof(float)));
	CHECK_CUDA_STATUS(cudaMemset(d_out, 0, sizeof(float)));

	CHECK_CUDA_STATUS(cudaMemcpy(d_in, h_in.data(), length * sizeof(float), cudaMemcpyHostToDevice));

	// warm-up
	for (int i = 0; i < 5; ++i) {
		kernel << <gridSize, blockSize >> > (d_out, d_in, length);
	}
	CHECK_CUDA_STATUS(cudaGetLastError());
	CHECK_CUDA_STATUS(cudaDeviceSynchronize());

	cudaEvent_t start, stop;
	CHECK_CUDA_STATUS(cudaEventCreate(&start));
	CHECK_CUDA_STATUS(cudaEventCreate(&stop));

	CHECK_CUDA_STATUS(cudaEventRecord(start));

	for (int i = 0; i < iters; ++i) {
		kernel << <gridSize, blockSize >> > (d_out, d_in, length);
	}
	CHECK_CUDA_STATUS(cudaEventRecord(stop));
	CHECK_CUDA_STATUS(cudaEventSynchronize(stop));
	CHECK_CUDA_STATUS(cudaGetLastError());
	CHECK_CUDA_STATUS(cudaDeviceSynchronize());

	float ms = 0.f;
	CHECK_CUDA_STATUS(cudaEventElapsedTime(&ms, start, stop));

	CHECK_CUDA_STATUS(cudaEventDestroy(start));
	CHECK_CUDA_STATUS(cudaEventDestroy(stop));

	CHECK_CUDA_STATUS(cudaFree(d_out));
	CHECK_CUDA_STATUS(cudaFree(d_in));

	double avg_ms = ms / iters;
	double avg_s = avg_ms * 1e-3;

	double gbytes = bytes_per_iter / 1e9;
	double gbytes_s = gbytes / avg_s;

	std::cout << name << "\n";
	std::cout << "  iters          = " << iters << "\n";
	std::cout << "  avg time       = " << avg_ms << " ms\n";
	std::cout << "  Bandwidth      = " << gbytes_s << " GB/s\n";
}

template <typename Kernel>
void run_kernel(const char* name,
	dim3 gridSize,
	dim3 blockSize,
	Kernel kernel,
	int length,
	const float ref_out)
{
	std::cout << "Running " << name << " ...\n";
	
	std::vector<float> h_in = make_uniform(length);
	float h_out = 0.f;

	float* d_in = nullptr;
	float* d_out = nullptr;

	CHECK_CUDA_STATUS(cudaMalloc(&d_in, length * sizeof(float)));
	CHECK_CUDA_STATUS(cudaMalloc((void**)&d_out, sizeof(float)));
	CHECK_CUDA_STATUS(cudaMemset(d_out, 0, sizeof(float)));

	CHECK_CUDA_STATUS(cudaMemcpy(d_in, h_in.data(), length * sizeof(float), cudaMemcpyHostToDevice));

	kernel << <gridSize, blockSize >> > (d_out, d_in, length);

	CHECK_CUDA_STATUS(cudaGetLastError());
	CHECK_CUDA_STATUS(cudaDeviceSynchronize());

	CHECK_CUDA_STATUS(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

	float abs_err = std::fabs(h_out - ref_out);

	std::cout << "Ref out: " << ref_out << " Kernel out: " << h_out << " Diff: " << abs_err << std::endl;
	
	CHECK_CUDA_STATUS(cudaFree(d_out));
	CHECK_CUDA_STATUS(cudaFree(d_in));

	CHECK_CUDA_STATUS(cudaGetLastError());
	CHECK_CUDA_STATUS(cudaDeviceSynchronize());
}


int main()
{
	int block = BLOCK_DIM;
	int iters = 100;

	int length = 2 * BLOCK_DIM*1e4 + 5;

	std::vector<float> h_data = make_uniform(length);

	float ref_out = 0.f;
	float h_out = 0.f;

	reduction_cpu(&ref_out, h_data.data(), length);

	//benchmark_reduction_cpu(length, iters);

	dim3 block_size(block);
	dim3 grid_size((length + 2*block - 1) / (2*block));

	run_kernel("reduction_naive_kernel",
		grid_size,
		block_size,
		reduction_naive_kernel,
		length,
		ref_out);

	run_kernel("reduction_improved_kernel",
		grid_size,
		block_size,
		reduction_improved_kernel,
		length,
		ref_out);

	run_kernel("reduction_shared_kernel",
		grid_size,
		block_size,
		reduction_shared_kernel,
		length,
		ref_out);


	dim3 grid_size2((length + 2 * COARSE_FACTOR * block - 1) / (2 * COARSE_FACTOR * block));
	run_kernel("reduction_shared_coarse_kernel",
		grid_size2,
		block_size,
		reduction_shared_coarse_kernel,
		length,
		ref_out);

	run_kernel("reduction_warp_shuffle_kernel",
		grid_size2,
		block_size,
		reduction_warp_shuffle_kernel,
		length,
		ref_out);


	benchmark_kernel("reduction_naive_kernel",
		grid_size,
		block_size,
		length,
		iters,
		sizeof(float) * length,
		reduction_naive_kernel);

	benchmark_kernel("reduction_improved_kernel",
		grid_size,
		block_size,
		length,
		iters,
		sizeof(float) * length,
		reduction_improved_kernel);

	benchmark_kernel("reduction_shared_kernel",
		grid_size,
		block_size,
		length,
		iters,
		sizeof(float) * length,
		reduction_shared_kernel);

	benchmark_kernel("reduction_shared_coarse_kernel",
		grid_size2,
		block_size,
		length,
		iters,
		sizeof(float) * length,
		reduction_shared_coarse_kernel);

	benchmark_kernel("reduction_warp_shuffle_kernel",
		grid_size2,
		block_size,
		length,
		iters,
		sizeof(float) * length,
		reduction_warp_shuffle_kernel);

	return 0;
}
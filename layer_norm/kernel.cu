
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <memory>
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


__device__ void reduction_shared(float& sum, float& sum_sq, const float* in, int length)
{
	__shared__ float s_sum[BLOCK_DIM];
	__shared__ float s_sum_sq[BLOCK_DIM];

	float val = threadIdx.x < length ? in[threadIdx.x] : 0.f;
	sum = val;
	sum_sq = val * val;
	for (int offset = blockDim.x; offset + threadIdx.x < length; offset += blockDim.x)
	{
		val = in[threadIdx.x + offset];
		sum += val;
		sum_sq += val * val;
	}

	s_sum[threadIdx.x] = sum;
	s_sum_sq[threadIdx.x] = sum_sq;

	for (int offset = blockDim.x / 2; offset >= 1; offset /= 2)
	{
		__syncthreads();
		if (threadIdx.x < offset)
		{
			s_sum[threadIdx.x] += s_sum[threadIdx.x + offset];
			s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + offset];
		}
	}
	__syncthreads();

	sum = s_sum[0];
	sum_sq = s_sum_sq[0];
}

__global__ void layer_norm_kernel(float* out, const float* in, int embed_dim, int seq_length, float eps)
{
	int offset = blockIdx.y * embed_dim;

	float sum = 0.f;
	float sum_sq = 0.f;

	reduction_shared(sum, sum_sq, in + offset, embed_dim);

	/*if (threadIdx.x == 0)
		printf("Thread %d has value mean = %f mean_sq = %f\n", threadIdx.x, sum / embed_dim, sum_sq / embed_dim);*/

	int tx = threadIdx.x;
	if (tx >= embed_dim)
		return;

	float mean = sum / embed_dim;
	float variance = (sum_sq / embed_dim) - mean * mean;
	variance = fmaxf(variance, 0.f);

	for (int i = tx; i < embed_dim; i += blockDim.x)
	{
		out[offset + i] = (in[offset + i] - mean) / std::sqrtf(variance + eps);
	}
}


void layer_norm_cpu(float* out, const float* in, int batch_size, int seq_length, int embed_dim, float eps)
{
	for (int i = 0; i < batch_size * seq_length; i++)
	{
		const float* in_ptr = in + i * embed_dim;

		double sum = 0.0;
		double sum_sq = 0.0;

		for (int j = 0; j < embed_dim; j++)
		{
			sum += (double)in_ptr[j];
			sum_sq += (double)in_ptr[j] * (double)in_ptr[j];
		}
		double mean = sum / embed_dim;
		double variance = (sum_sq / embed_dim) - mean * mean;
		variance = fmaxf(variance, 0.f);

		float* out_ptr = out + i * embed_dim;
		for (int j = 0; j < embed_dim; j++)
		{
			out_ptr[j] = (in_ptr[j] - mean) / sqrtf(variance + eps);
		}
	}
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

std::vector<float> make_normal(int n, unsigned int seed = 1234) {
	std::mt19937 rng(seed);                     // deterministic RNG
	std::normal_distribution<float> dist(0.f, 1.f);

	std::vector<float> v(n);
	for (auto& x : v) {
		x = dist(rng);
	}
	return v;
}

std::vector<float> make_vector_with(int n, float value)
{
	std::vector<float> v(n, value);
	return v;
}


int main()
{
	int block = BLOCK_DIM;
	int iters = 100;

	int batch_size = 16;
	int seq_length = 1024;
	int embed_dim = 1024 + 5;

	//std::vector<float> h_data = make_vector_with(batch_size * seq_length * embed_dim, 1);
	std::vector<float> h_in = make_normal(batch_size * seq_length * embed_dim);

	std::vector<float> ref_out = make_vector_with(batch_size * seq_length * embed_dim, 0);

	layer_norm_cpu(ref_out.data(), h_in.data(), batch_size, seq_length, embed_dim, 1e-5f);

	dim3 block_size(block, 1);
	dim3 grid_size(1, batch_size * seq_length);


	std::vector<float> h_out = make_vector_with(batch_size * seq_length * embed_dim, 0);

	float* d_in = nullptr;
	float* d_out = nullptr;

	CHECK_CUDA_STATUS(cudaMalloc(&d_in, batch_size * seq_length * embed_dim * sizeof(float)));
	CHECK_CUDA_STATUS(cudaMalloc(&d_out, batch_size * seq_length * embed_dim * sizeof(float)));

	CHECK_CUDA_STATUS(cudaMemcpy(d_in, h_in.data(), batch_size * seq_length * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA_STATUS(cudaMemcpy(d_out, h_out.data(), batch_size * seq_length * embed_dim * sizeof(float), cudaMemcpyHostToDevice));

	layer_norm_kernel << <grid_size, block_size >> > (d_out, d_in, embed_dim, seq_length, 1e-5f);

	CHECK_CUDA_STATUS(cudaGetLastError());
	CHECK_CUDA_STATUS(cudaDeviceSynchronize());

	CHECK_CUDA_STATUS(cudaMemcpy(h_out.data(), d_out, batch_size * seq_length * embed_dim * sizeof(float), cudaMemcpyDeviceToHost));

	float abs_err = 0.f;
	for (size_t i = 0; i < batch_size * seq_length * embed_dim; i++)
	{
		abs_err = std::max(abs_err, std::fabs(h_out[i] - ref_out[i]));

	}
	std::cout << "Max abs diff: " << abs_err << std::endl;


	CHECK_CUDA_STATUS(cudaFree(d_out));
	CHECK_CUDA_STATUS(cudaFree(d_in));

	CHECK_CUDA_STATUS(cudaGetLastError());
	CHECK_CUDA_STATUS(cudaDeviceSynchronize());

	return 0;
}
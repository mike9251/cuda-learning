
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

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


#define BINS 256


void histogram_cpu(unsigned int* out, const unsigned char* in, int width, int height)
{
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			unsigned char pixel = in[row * width + col];
			out[pixel]++;
		}
	}
}


__global__ void histogram_naive_kernel(unsigned int* out, const unsigned char* in, int width, int height)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < height && col < width)
	{
		unsigned char pixel = in[row * width + col];
		atomicAdd(&out[pixel], 1);
	}
}


__global__ void histogram_priv_global_kernel(unsigned int* out, const unsigned char* in, int width, int height)
{
	int offset = blockIdx.y * gridDim.x * BINS + blockIdx.x * BINS;

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < height && col < width)
	{
		int pixel_value= (int)in[row * width + col];
		atomicAdd(&out[offset + pixel_value], 1);
	}

	// threads in block after 0, 0
	if (offset > 0)
	{
		__syncthreads();
		int tx = threadIdx.y * blockDim.x + threadIdx.x;
		for (int i = tx; i < BINS; i += blockDim.x * blockDim.y)
		{
			unsigned int bin_value = out[offset + i];
			if (bin_value > 0)
			{
				atomicAdd(&out[i], bin_value);
			}
		}
	}
}


__global__ void histogram_priv_shared_kernel(unsigned int* out, const unsigned char* in, int width, int height)
{
	__shared__ unsigned int hist_s[BINS];

	for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < BINS; i += blockDim.x * blockDim.y)
	{
		hist_s[i] = 0U;
	}
	__syncthreads();

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < height && col < width)
	{
		unsigned char pixel_value = in[row * width + col];
		atomicAdd(&hist_s[pixel_value], 1);
	}

	__syncthreads();

	for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < BINS; i += blockDim.x * blockDim.y)
	{
		unsigned int bin_value = hist_s[i];
		if (bin_value > 0)
		{
			atomicAdd(&out[i], bin_value);
		}
	}
}

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define COARSENING_FACTOR 4
__global__ void histogram_priv_shared_coarse_cont_kernel(unsigned int* out, const unsigned char* in, int width, int height)
{
	__shared__ unsigned int hist_s[BINS];

	for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < BINS; i += blockDim.x * blockDim.y)
	{
		hist_s[i] = 0U;
	}
	__syncthreads();

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < height)
	{
		#pragma unroll
		for (int i = col * COARSENING_FACTOR; i < MIN((col + 1) * COARSENING_FACTOR, width); i++)
		{
			unsigned char pixel_value = in[row * width + i];
			atomicAdd(&hist_s[pixel_value], 1);
		}
	}

	__syncthreads();

	for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < BINS; i += blockDim.x * blockDim.y)
	{
		unsigned int bin_value = hist_s[i];
		if (bin_value > 0)
		{
			atomicAdd(&out[i], bin_value);
		}
	}
}

__global__ void histogram_priv_shared_coarse_interleave_kernel(unsigned int* out, const unsigned char* in, int width, int height)
{
	__shared__ unsigned int hist_s[BINS];

	for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < BINS; i += blockDim.x * blockDim.y)
	{
		hist_s[i] = 0U;
	}
	__syncthreads();

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) * COARSENING_FACTOR + threadIdx.x;

	if (row < height)
	{
#pragma unroll
		for (int i = 0; i < COARSENING_FACTOR; i++)
		{
			if (col < width)
			{
				unsigned char pixel_value = in[row * width + col];
				atomicAdd(&hist_s[pixel_value], 1);
			}
			col += blockDim.x;
		}
	}

	__syncthreads();

	for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < BINS; i += blockDim.x * blockDim.y)
	{
		unsigned int bin_value = hist_s[i];
		if (bin_value > 0)
		{
			atomicAdd(&out[i], bin_value);
		}
	}
}


__global__ void histogram_priv_shared_agg_kernel(unsigned int* out, const unsigned char* in, int width, int height)
{
	__shared__ unsigned int hist_s[BINS];

	int prev_bin = -1;
	int accumulator = 0;

	for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < BINS; i += blockDim.x * blockDim.y)
	{
		hist_s[i] = 0U;
	}
	__syncthreads();

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < height && col < width)
	{
		unsigned char pixel_value = in[row * width + col];

		if (pixel_value == prev_bin)
		{
			accumulator++;
		}
		else
		{
			if (accumulator > 0)
			{
				atomicAdd(&hist_s[prev_bin], accumulator);
			}
			prev_bin = pixel_value;
			accumulator = 1;
		}
	}

	if (accumulator > 0)
	{
		atomicAdd(&hist_s[prev_bin], accumulator);
	}

	__syncthreads();

	for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < BINS; i += blockDim.x * blockDim.y)
	{
		unsigned int bin_value = hist_s[i];
		if (bin_value > 0)
		{
			atomicAdd(&out[i], bin_value);
		}
	}
}


void show_images(cv::Mat image, std::string title)
{
	cv::namedWindow(title + "_GRAY", cv::WINDOW_NORMAL);
	imshow(title + "_GRAY", image);

	cv::waitKey(0);
}


double benchmark_histogram_cpu(
	unsigned int* out,
	const unsigned char* in,
	unsigned int width,
	unsigned int height,
	unsigned int iters)
{
	// warmup: make sure caches etc. are "hot"
	histogram_cpu(out, in, width, height);

	auto start = Clock::now();
	for (int i = 0; i < iters; ++i) {
		histogram_cpu(out, in, width, height);
	}
	auto end = Clock::now();

	float checksum = 0.0f;
	for (int i = 0; i < BINS; ++i) checksum += out[i];
	std::cout << "checksum = " << checksum << "\n";

	ms total = end - start;
	double avg_ms = total.count() / iters;
	return avg_ms;
}

template <typename Kernel, typename... Args>
void benchmark_kernel(const char* name,
	dim3 gridSize,
	dim3 blockSize,
	int iters,
	double bytes_per_iter,   // how many bytes the kernel touches once
	Kernel kernel,           // the kernel symbol
	Args... args)            // kernel arguments
{
	// warm-up
	for (int i = 0; i < 5; ++i) {
		kernel << <gridSize, blockSize >> > (args...);
	}
	CHECK_CUDA_STATUS(cudaDeviceSynchronize());

	cudaEvent_t start, stop;
	CHECK_CUDA_STATUS(cudaEventCreate(&start));
	CHECK_CUDA_STATUS(cudaEventCreate(&stop));

	CHECK_CUDA_STATUS(cudaEventRecord(start));

	for (int i = 0; i < iters; ++i) {
		kernel << <gridSize, blockSize >> > (args...);
	}
	CHECK_CUDA_STATUS(cudaEventRecord(stop));
	CHECK_CUDA_STATUS(cudaEventSynchronize(stop));

	float ms = 0.f;
	CHECK_CUDA_STATUS(cudaEventElapsedTime(&ms, start, stop));

	CHECK_CUDA_STATUS(cudaEventDestroy(start));
	CHECK_CUDA_STATUS(cudaEventDestroy(stop));

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
	int hist_size,
	int width,
	int height,
	const unsigned char* d_image,
	const unsigned int* ref_hist)
{
	std::cout << "Running " << name << " ...\n";

	unsigned int* h_hist = (unsigned int*)malloc(hist_size);
	unsigned int* d_hist = nullptr;

	CHECK_CUDA_STATUS(cudaMalloc((void**)&d_hist, hist_size));
	CHECK_CUDA_STATUS(cudaMemset(d_hist, 0, hist_size));

	kernel <<<gridSize, blockSize >>> (d_hist, d_image, width, height);

	CHECK_CUDA_STATUS(cudaGetLastError());
	CHECK_CUDA_STATUS(cudaDeviceSynchronize());

	CHECK_CUDA_STATUS(cudaMemcpy(h_hist, d_hist, hist_size, cudaMemcpyDeviceToHost));

	for (int i = 0; i < BINS; i++)
	{
		if (h_hist[i] != ref_hist[i])
		{
			std::cout << "Mismatch at bin " << i << ": GPU=" << h_hist[i] << " CPU=" << ref_hist[i] << "\n";
			break;
		}
		if (i == BINS - 1)
		{
			std::cout << "Results match!\n";
		}
	}
	free(h_hist);
	CHECK_CUDA_STATUS(cudaFree(d_hist));
}


int main()
{
	int block = 16;
	int iters = 100;

	cv::Mat image;
	image = imread("C:\\Users\\petrush\\Downloads\\painting-mountain-lake-with-mountain-background.jpg", cv::IMREAD_GRAYSCALE);

	if (image.empty())
	{
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	int height = image.rows;
	int width = image.cols;
	int channels = image.channels();

	std::cout << "H=" << height << " W=" << width << " C=" << channels << "\n";

	size_t image_size = width * height * channels * sizeof(unsigned char);
	size_t hist_size = BINS * sizeof(unsigned int);

	unsigned char* h_image = (unsigned char*)image.data;
	unsigned int* h_hist= (unsigned int*)malloc(hist_size);
	unsigned int* ref_hist = (unsigned int*)malloc(hist_size);
	unsigned int* test_hist = (unsigned int*)malloc(hist_size);

	std::memset(h_hist, 0, hist_size);
	std::memset(ref_hist, 0, hist_size);
	std::memset(test_hist, 0, hist_size);

	histogram_cpu(ref_hist, h_image, width, height);

	unsigned char* d_image = nullptr;
	unsigned int* d_hist = nullptr;

	CHECK_CUDA_STATUS(cudaMalloc((void**)&d_image, image_size));
	CHECK_CUDA_STATUS(cudaMalloc((void**)&d_hist, hist_size));

	CHECK_CUDA_STATUS(cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice));
	CHECK_CUDA_STATUS(cudaMemset(d_hist, 0, hist_size));

	dim3 block_size(block, block);
	dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

	run_kernel("histogram_naive_kernel",
		grid_size,
		block_size,
		histogram_naive_kernel,
		hist_size,
		width,
		height,
		d_image,
		ref_hist);

	size_t hist_size_priv_global = grid_size.x * grid_size.y * BINS * sizeof(unsigned int);
	run_kernel("histogram_priv_global_kernel",
		grid_size,
		block_size,
		histogram_priv_global_kernel,
		hist_size_priv_global,
		width,
		height,
		d_image,
		ref_hist);

	run_kernel("histogram_priv_shared_kernel",
		grid_size,
		block_size,
		histogram_priv_shared_kernel,
		hist_size,
		width,
		height,
		d_image,
		ref_hist);

	run_kernel("histogram_priv_shared_agg_kernel",
		grid_size,
		block_size,
		histogram_priv_shared_agg_kernel,
		hist_size,
		width,
		height,
		d_image,
		ref_hist);
	
	dim3 grid_size2((width + COARSENING_FACTOR * block_size.x - 1) / (COARSENING_FACTOR * block_size.x), (height + block_size.y - 1) / block_size.y);

	run_kernel("histogram_priv_shared_coarse_cont_kernel",
		grid_size2,
		block_size,
		histogram_priv_shared_coarse_cont_kernel,
		hist_size,
		width,
		height,
		d_image,
		ref_hist);

	run_kernel("histogram_priv_shared_coarse_interleave_kernel",
		grid_size2,
		block_size,
		histogram_priv_shared_coarse_interleave_kernel,
		hist_size,
		width,
		height,
		d_image,
		ref_hist);

	// Benchmark kernels

	double t = benchmark_histogram_cpu(test_hist, h_image, width, height, 100);
	std::cout << "CPU kernel took: " << t << "ms\n";

	double bytes_naive =
		width * height * sizeof(unsigned char) +     // input
		BINS * sizeof(unsigned int);                 // global hist

	benchmark_kernel("Histogram naive kernel:",
		grid_size, block_size, iters, bytes_naive,
		histogram_naive_kernel,
		d_hist, d_image, width, height);
	CHECK_CUDA_STATUS(cudaMemset(d_hist, 0, hist_size));

	double bytes_priv_global =
		width * height * sizeof(unsigned char) + grid_size.x * grid_size.y * BINS * sizeof(unsigned int);

	unsigned int* d_hist_priv_global = nullptr;
	CHECK_CUDA_STATUS(cudaMalloc(&d_hist_priv_global, hist_size_priv_global));
	CHECK_CUDA_STATUS(cudaMemset(d_hist_priv_global, 0, hist_size_priv_global));

	benchmark_kernel("Histogram privatization (global) kernel:",
		grid_size, block_size, iters, bytes_priv_global,
		histogram_priv_global_kernel,
		d_hist_priv_global, d_image, width, height);

	double bytes_priv_shared =
		width * height * sizeof(unsigned char) + BINS * sizeof(unsigned int);

	benchmark_kernel("Histogram privatization (shared) kernel:",
		grid_size, block_size, iters, bytes_priv_shared,
		histogram_priv_shared_kernel,
		d_hist, d_image, width, height);
	CHECK_CUDA_STATUS(cudaMemset(d_hist, 0, hist_size));

	benchmark_kernel("Histogram privatization (shared) with aggregation kernel:",
		grid_size, block_size, iters, bytes_priv_shared,
		histogram_priv_shared_agg_kernel,
		d_hist, d_image, width, height);
	CHECK_CUDA_STATUS(cudaMemset(d_hist, 0, hist_size));

	benchmark_kernel("Histogram privatization (shared) thread coarsening (continous) kernel:",
		grid_size2, block_size, iters, bytes_priv_shared,
		histogram_priv_shared_coarse_cont_kernel,
		d_hist, d_image, width, height);
	CHECK_CUDA_STATUS(cudaMemset(d_hist, 0, hist_size));

	benchmark_kernel("Histogram privatization (shared) thread coarsening (thread interleaving) kernel:",
		grid_size2, block_size, iters, bytes_priv_shared,
		histogram_priv_shared_coarse_interleave_kernel,
		d_hist, d_image, width, height);
	CHECK_CUDA_STATUS(cudaMemset(d_hist, 0, hist_size));

	CHECK_CUDA_STATUS(cudaFree(d_image));
	CHECK_CUDA_STATUS(cudaFree(d_hist));
	CHECK_CUDA_STATUS(cudaFree(d_hist_priv_global));

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	CHECK_CUDA_STATUS(cudaDeviceReset());

	free(h_hist);
	free(ref_hist);
	free(test_hist);

	return 0;
}
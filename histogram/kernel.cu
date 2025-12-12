
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


__global__ void histogram_kernel(unsigned int* out, const unsigned char* in, int width, int height)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < height && col < width)
	{
		unsigned char pixel = in[row * width + col];
		atomicAdd(&out[pixel], 1);
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

void benchmark_historgram_kernel(
	unsigned int* d_out,
	const unsigned char* d_in,
	unsigned int width,
	unsigned int height,
	unsigned int iters)
{
	dim3 block_size(16, 16);
	dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
	for (int i = 0; i < 5; i++)
	{
		histogram_kernel << <grid_size, block_size >> > (d_out, d_in, width, height);
	}
	CHECK_CUDA_STATUS(cudaDeviceSynchronize());

	cudaEvent_t start, stop;
	CHECK_CUDA_STATUS(cudaEventCreate(&start));
	CHECK_CUDA_STATUS(cudaEventCreate(&stop));

	CHECK_CUDA_STATUS(cudaEventRecord(start));
	for (int i = 0; i < iters; i++)
	{
		histogram_kernel << <grid_size, block_size >> > (d_out, d_in, width, height);
	}
	CHECK_CUDA_STATUS(cudaEventRecord(stop));
	CHECK_CUDA_STATUS(cudaEventSynchronize(stop));

	float ms = 0.f;
	CHECK_CUDA_STATUS(cudaEventElapsedTime(&ms, start, stop));

	CHECK_CUDA_STATUS(cudaEventDestroy(start));
	CHECK_CUDA_STATUS(cudaEventDestroy(stop));

	// Average kernel time
	double avg_ms = ms / iters;
	double avg_s = avg_ms / 1e3;

	double bytes = height * width + BINS * sizeof(unsigned int);
	double gbytes = bytes / 1e9;

	double gbytes_s = gbytes / avg_s;

	std::cout << "Histogram naive kernel:\n";
	std::cout << "  N              = " << width * height << "\n";
	std::cout << "  iters          = " << iters << "\n";
	std::cout << "  avg time       = " << avg_ms << " ms\n";
	std::cout << "  Bandwidth      = " << gbytes_s << " GB/s\n";
}

int main()
{
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

	for (int i = 0; i < BINS; i++)
	{
		h_hist[i] = 0;
		ref_hist[i] = 0;
		test_hist[i] = 0;
	}

	double t = benchmark_histogram_cpu(test_hist, h_image, width, height, 100);
	std::cout << "CPU kernel took: " << t << "ms\n";

	histogram_cpu(ref_hist, h_image, width, height);

	unsigned char* d_image = nullptr;
	unsigned int* d_hist = nullptr;

	CHECK_CUDA_STATUS(cudaMalloc((void**)&d_image, image_size));
	CHECK_CUDA_STATUS(cudaMalloc((void**)&d_hist, hist_size));

	CHECK_CUDA_STATUS(cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice));
	CHECK_CUDA_STATUS(cudaMemcpy(d_hist, h_hist, hist_size, cudaMemcpyHostToDevice));

	// global access
	dim3 block_size(16, 16);
	dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
	histogram_kernel << <grid_size, block_size >> > (d_hist, d_image, width, height);
	CHECK_CUDA_STATUS(cudaMemcpy(h_hist, d_hist, hist_size, cudaMemcpyDeviceToHost));

	// verify results
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

	benchmark_historgram_kernel(d_hist, d_image, width, height, 100);

	CHECK_CUDA_STATUS(cudaFree(d_image));
	CHECK_CUDA_STATUS(cudaFree(d_hist));

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	CHECK_CUDA_STATUS(cudaDeviceReset());

	free(h_hist);
	free(ref_hist);

	return 0;

}
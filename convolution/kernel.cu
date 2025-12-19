
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

using namespace cv;
using namespace std;

#define CHECK_CUDA_STATUS(status) { if (status != cudaSuccess) { \
    fprintf(stderr, "CUDA Error: %s (error code %d) at %s:%d\n", \
        cudaGetErrorString(status), status, __FILE__, __LINE__); exit(EXIT_FAILURE); } }


#define RADIUS 2
#define OUT_TILE 16


__global__ void conv_kernel(float* out, float* in, float* filter, int height, int width, int kernel_radius)
{
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	int kernel_size = 2 * kernel_radius + 1;

	if (row < height && col < width)
	{	
		float res = 0.f;
		int count = 0;
		for (int y = 0; y < kernel_size; y++)
		{
			for (int x = 0; x < kernel_size; x++)
			{
				int in_row = row - kernel_radius + y;
				int in_col = col - kernel_radius + x;
				if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width)
				{
					res += filter[y * kernel_size + x] * in[in_row * width + in_col];
					//count++;
				}
			}
		}

		out[row * width + col] = res;// / count;
	}
}


__constant__ float FILTER[(RADIUS*2 + 1) * (RADIUS * 2 + 1)];

__global__ void conv_kernel_const_mem(float* out, float* in, int height, int width, int kernel_radius)
{
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	int kernel_size = 2 * kernel_radius + 1;

	if (row < height && col < width)
	{
		float res = 0.f;
		for (int y = 0; y < kernel_size; y++)
		{
			for (int x = 0; x < kernel_size; x++)
			{
				int in_row = row - kernel_radius + y;
				int in_col = col - kernel_radius + x;
				if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width)
				{
					res += FILTER[y * kernel_size + x] * in[in_row * width + in_col];
				}
			}
		}
		out[row * width + col] = res;
	}
}

/*The number of threads in a block matches the number of elements + padding*/
#define IN_TILE OUT_TILE + 2 * RADIUS
__global__ void conv_kernel_shared_mem(float* out, float* in, int height, int width, int kernel_radius)
{
	int row = blockIdx.y * OUT_TILE + threadIdx.y - RADIUS; // nuymber of threads in the block for input tile, but we skip padding threads to perform a write to the out
	int col = blockIdx.x * OUT_TILE + threadIdx.x - RADIUS;

	__shared__ float shmem[IN_TILE][IN_TILE];

	if (row >= 0 && row < height && col >= 0 && col < width)
	{
		shmem[threadIdx.y][threadIdx.x] = in[row * width + col];
	}
	else
	{
		shmem[threadIdx.y][threadIdx.x] = 0.f;
	}

	__syncthreads();

	int tile_row = threadIdx.y - RADIUS;
	int tile_col = threadIdx.x - RADIUS;

	int kernel_size = 2 * RADIUS + 1;

	if (row >= 0 && row < height && col >= 0 && col < width)
	{
		if (tile_row >= 0 && tile_row < OUT_TILE && tile_col >= 0 && tile_col < OUT_TILE)
		{
			float res = 0.f;

			for (int y = 0; y < kernel_size; y++)
			{
				for (int x = 0; x < kernel_size; x++)
				{
					res += FILTER[y * kernel_size + x] * shmem[tile_row + y][tile_col + x];
				}
			}
			out[row * width + col] = res;
		}
	}
}

/*Read tile into shared mem, load padding from global memory (should be cached)*/
__global__ void conv_kernel_shared_mem_cache(float* out, float* in, int height, int width, int kernel_radius)
{
	int row = blockIdx.y * OUT_TILE + threadIdx.y;
	int col = blockIdx.x * OUT_TILE + threadIdx.x;

	__shared__ float shmem[OUT_TILE][OUT_TILE];

	if (row >= 0 && row < height && col >= 0 && col < width)
	{
		shmem[threadIdx.y][threadIdx.x] = in[row * width + col];
	}
	else
	{
		shmem[threadIdx.y][threadIdx.x] = 0.f;
	}

	__syncthreads();

	int kernel_size = 2 * RADIUS + 1;

	if (row < height && col < width)
	{
			float res = 0.f;

			for (int y = 0; y < kernel_size; y++)
			{
				for (int x = 0; x < kernel_size; x++)
				{
					int s_row = threadIdx.y - RADIUS + y;
					int s_col = threadIdx.x - RADIUS + x;

					if (s_row >= 0 && s_row < OUT_TILE && s_col >= 0 && s_col < OUT_TILE)
					{
						res += FILTER[y * kernel_size + x] * shmem[s_row][s_col];
					}
					else
					{
						int g_row = row - RADIUS + y;
						int g_col = col - RADIUS + x;
						if (g_row >= 0 && g_row < height && g_col >= 0 && g_col < width)
						{
							res += FILTER[y * kernel_size + x] * in[g_row * width + g_col];

						}
					}
				}
			}
			out[row * width + col] = res;
	}
}

#define SH OUT_TILE + 2 * RADIUS

__global__ void conv_kernel_shared_mem_2(float* out, float* in, int height, int width, int kernel_radius)
{
	int base_row = blockIdx.y * blockDim.y;
	int base_col = blockIdx.x * blockDim.x;

	__shared__ float shmem[SH][SH];

	for (int sy = threadIdx.y; sy < SH; sy += blockDim.y)
	{
		for (int sx = threadIdx.x; sx < SH; sx += blockDim.x)
		{
			int g_row = base_row + sy - RADIUS;
			int g_col = base_col + sx - RADIUS;

			if (g_row >= 0 && g_row < height && g_col >= 0 && g_col < width)
			{
				shmem[sy][sx] = in[g_row * width + g_col];
			}
			else
			{
				shmem[sy][sx] = 0.f;
			}
		}
	}

	__syncthreads();

	int out_row = base_row + threadIdx.y;
	int out_col = base_col + threadIdx.x;

	if (out_row < height && out_col < width)
	{
		int filter_size = RADIUS * 2 + 1;

		float res = 0.f;
		for (int fy = -RADIUS; fy <= RADIUS; fy++)
		{
			for (int fx = -RADIUS; fx <= RADIUS; fx++)
			{
				int sy = threadIdx.y + RADIUS + fy;
				int sx = threadIdx.x + RADIUS + fx;
				res += FILTER[(fy + RADIUS) * filter_size + (fx + RADIUS)] * shmem[sy][sx];
			}
		}

		out[out_row * width + out_col] = res;
	}
}


void show_images(Mat image, float* data, int height, int width, string title)
{
	Mat blur_image_f(height, width, CV_32FC1, data);
	Mat blur_image;
	blur_image_f.convertTo(blur_image, CV_8UC1, 255);

	namedWindow(title + "_GRAY", WINDOW_NORMAL);
	imshow(title + "_GRAY", image);

	namedWindow(title + "_BLUR", WINDOW_NORMAL);
	imshow(title + "_BLUR", blur_image);

	waitKey(0);
}

int main()
{
	int device_id{ 0 };
	cudaGetDevice(&device_id);
	cudaDeviceProp device_prop;
	int mem_clock = 0;
	cudaDeviceGetAttribute(&mem_clock, cudaDevAttrMemoryClockRate, device_id);
	cudaGetDeviceProperties(&device_prop, device_id);
	std::cout << "Device Name: " << device_prop.name << std::endl;
	float const memory_size{ static_cast<float>(device_prop.totalGlobalMem) /
							(1 << 30) };
	std::cout << "Memory Size: " << memory_size << " GB" << std::endl;
	std::cout << "mem_clock = " << mem_clock << " device_prop.memoryBusWidth = " << device_prop.memoryBusWidth << std::endl;
	float const peak_bandwidth{
		static_cast<float>(2.0f * 1e3 * mem_clock *
						   (device_prop.memoryBusWidth / 8) / 1e9) };
	std::cout << "Peak Bandwitdh: " << peak_bandwidth << " GB/s" << std::endl;

	Mat image;
	image = imread("C:\\Users\\petrush\\Downloads\\twocats.png", cv::IMREAD_GRAYSCALE);

	if (image.empty())
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	int filter_size = RADIUS * 2 + 1;

	int height = image.rows;
	int width = image.cols;
	int channels = image.channels();

	cout << "H=" << height << " W=" << width << " C=" << channels << "\n";

	size_t size = width * height * channels * sizeof(float);

	Mat image_f;
	image.convertTo(image_f, CV_32F, 1.0 / 255);
	float* h_gray = (float*)image_f.data;
	float* h_blur = (float*)malloc(size);

	float* h_filter = (float*)malloc(filter_size * filter_size * sizeof(float));
	for (int i = 0; i < filter_size * filter_size; i++)
	{
		h_filter[i] = 1.f / (filter_size * filter_size);
	}

	float* d_gray = nullptr;
	float* d_blur = nullptr;
	float* d_filter = nullptr;

	CHECK_CUDA_STATUS(cudaMalloc((void**)&d_gray, size));
	CHECK_CUDA_STATUS(cudaMalloc((void**)&d_blur, size));
	CHECK_CUDA_STATUS(cudaMalloc((void**)&d_filter, filter_size * filter_size * sizeof(float)));

	CHECK_CUDA_STATUS(cudaMemcpy(d_gray, h_gray, size, cudaMemcpyHostToDevice));
	CHECK_CUDA_STATUS(cudaMemcpy(d_filter, h_filter, filter_size * filter_size * sizeof(float), cudaMemcpyHostToDevice));

	// global access
	dim3 block_size(OUT_TILE, OUT_TILE);
	dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
	conv_kernel << <grid_size, block_size >> > (d_blur, d_gray, d_filter, height, width, RADIUS);
	CHECK_CUDA_STATUS(cudaMemcpy(h_blur, d_blur, size, cudaMemcpyDeviceToHost));
	show_images(image, h_blur, height, width, "glob");

	// filter in cache
	block_size = dim3(OUT_TILE, OUT_TILE);
	grid_size = ((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
	conv_kernel_const_mem << <grid_size, block_size >> > (d_blur, d_gray, height, width, RADIUS);
	CHECK_CUDA_STATUS(cudaMemcpy(h_blur, d_blur, size, cudaMemcpyDeviceToHost));

	// shared mem for tile
	dim3 in_block_size = (IN_TILE, IN_TILE);
	block_size = (OUT_TILE, OUT_TILE);
	grid_size = ((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
	conv_kernel_shared_mem << <grid_size, in_block_size >> > (d_blur, d_gray, height, width, RADIUS);
	CHECK_CUDA_STATUS(cudaMemcpy(h_blur, d_blur, size, cudaMemcpyDeviceToHost));

	// shared meme for tile but global for padding
	block_size = (OUT_TILE, OUT_TILE);
	grid_size = ((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
	conv_kernel_shared_mem_cache << <grid_size, block_size >> > (d_blur, d_gray, height, width, RADIUS);
	CHECK_CUDA_STATUS(cudaMemcpy(h_blur, d_blur, size, cudaMemcpyDeviceToHost));

	// shared memory for tile but in tile == out tile
	block_size = (OUT_TILE, OUT_TILE);
	grid_size = ((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
	conv_kernel_shared_mem_2 << <grid_size, block_size >> > (d_blur, d_gray, height, width, RADIUS);
	CHECK_CUDA_STATUS(cudaMemcpy(h_blur, d_blur, size, cudaMemcpyDeviceToHost));

	show_images(image, h_blur, height, width, "shared_2");


	CHECK_CUDA_STATUS(cudaFree(d_gray));
	CHECK_CUDA_STATUS(cudaFree(d_blur));

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	CHECK_CUDA_STATUS(cudaDeviceReset());

	free(h_blur);

	return 0;

}
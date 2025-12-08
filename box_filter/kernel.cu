
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


/* almost coalesced memory access pattern
Within a warp each thread loads 1 byte with stride = 3
Data is loaded in 32B sectors inside a 128B cache line. For the entire warp I need to load 96B - only 3 mem transactions for the entire warp.
If the stride was bigger, then every thread could request data in a separate 32B sector - 32 transactions per warp - fully uncoalesced.*/
__global__ void blur_box_kernel(unsigned char* out, unsigned char* in, int height, int width, int kernel_size)
{
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	if (row < height && col < width)
	{
		int out_offset = (row * width + col) * 3;
		int num_pixels = 0;
		int3 curr_value = { 0 };

		int radius = kernel_size / 2;

		for (int y = -radius; y < radius + 1; y++)
		{
			for (int x = -radius; x < radius + 1; x++)
			{
				int curr_row = row + y;
				int curr_col = col + x;

				if ((curr_row >= 0 && curr_row < height) && (curr_col >= 0 && curr_col < width))
				{
					int in_offset = (curr_row * width + curr_col) * 3;
					curr_value.x += in[in_offset];
					curr_value.y += in[in_offset + 1];
					curr_value.z += in[in_offset + 2];
					num_pixels += 1;
				}
			}
		}

		out[out_offset] = (unsigned char)(curr_value.x / num_pixels);
		out[out_offset + 1] = (unsigned char)(curr_value.y / num_pixels);
		out[out_offset + 2] = (unsigned char)(curr_value.z / num_pixels);
	}
}


__global__ void blur_box_kernel_coalesced(unsigned char* out_r, unsigned char* out_g, unsigned char* out_b, unsigned char* in_r, unsigned char* in_g, unsigned char* in_b, int height, int width, int kernel_size)
{
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	if (row < height && col < width)
	{
		int num_pixels = 0;
		int curr_r = 0;
		int curr_g = 0;
		int curr_b = 0;

		int radius = kernel_size / 2;

		for (int y = -radius; y < radius + 1; y++)
		{
			for (int x = -radius; x < radius + 1; x++)
			{
				int curr_row = row + y;
				int curr_col = col + x;

				if ((curr_row >= 0 && curr_row < height) && (curr_col >= 0 && curr_col < width))
				{
					int in_offset = curr_row * width + curr_col;
					curr_r += in_r[in_offset];
					curr_g += in_g[in_offset];
					curr_b += in_b[in_offset];					
					num_pixels += 1;
				}
			}
		}

		int out_offset = row * width + col;
		out_r[out_offset] = (unsigned char)(curr_r / num_pixels);
		out_g[out_offset] = (unsigned char)(curr_g / num_pixels);
		out_b[out_offset] = (unsigned char)(curr_b / num_pixels);
	}
}

#define TILE_WIDTH 16
#define RADIUS 5
#define SH (TILE_WIDTH + 2*RADIUS)

__global__ void blur_box_kernel_shared(unsigned char* out,
	const unsigned char* in,
	int height, int width, int kernel_size)
{
	__shared__ unsigned char shmem[SH][SH];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int base_row = TILE_WIDTH * by;
	int base_col = TILE_WIDTH * bx;

	for (int y = ty; y < SH; y += TILE_WIDTH)
	{
		for (int x = tx; x < SH; x += TILE_WIDTH)
		{
			int row = base_row + y - RADIUS;
			int col = base_col + x - RADIUS;

			if (row >= 0 && row < height && col >= 0 && col < width)
			{
				shmem[y][x] = in[row * width + col];
			}
			else
			{
				shmem[y][x] = 0;
			}

		}
	}

	__syncthreads();

	int out_row = base_row + ty;
	int out_col = base_col + tx;
	if (out_row < height && out_col < width)
	{
		int sum = 0;
		int count = 0;

		for (int y = -RADIUS; y <= RADIUS; y++)
		{
			for (int x = -RADIUS; x <= RADIUS; x++)
			{
				sum += shmem[ty + RADIUS - y][tx + RADIUS - x];
				count++;
			}
		}
		out[out_row * width + out_col] = (unsigned char)(sum / count);
	}
	
}


int main()
{

	{
		Mat image;
		image = imread("C:\\Users\\petrush\\Downloads\\twocats.png", cv::IMREAD_COLOR);

		if (image.empty())
		{
			cout << "Could not open or find the image" << std::endl;
			return -1;
		}

		int height = image.rows;
		int width = image.cols;
		int channels = image.channels();

		cout << "H=" << height << " W=" << width << " C=" << channels << "\n";

		size_t size = width * height * channels * sizeof(unsigned char);

		unsigned char* h_bgr = (unsigned char*)image.data;
		unsigned char* h_blur = (unsigned char*)malloc(size);

		unsigned char* d_bgr = nullptr;
		unsigned char* d_blur = nullptr;

		CHECK_CUDA_STATUS(cudaMalloc((void**)&d_bgr, size));
		CHECK_CUDA_STATUS(cudaMalloc((void**)&d_blur, size));

		CHECK_CUDA_STATUS(cudaMemcpy(d_bgr, h_bgr, size, cudaMemcpyHostToDevice));

		int filter_size = RADIUS * 2 + 1;
		dim3 block_size(TILE_WIDTH, TILE_WIDTH);
		dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
		blur_box_kernel << <grid_size, block_size >> > (d_blur, d_bgr, height, width, filter_size);

		CHECK_CUDA_STATUS(cudaMemcpy(h_blur, d_blur, size, cudaMemcpyDeviceToHost));

		CHECK_CUDA_STATUS(cudaFree(d_bgr));
		CHECK_CUDA_STATUS(cudaFree(d_blur));

		Mat blur_image(height, width, CV_8UC3, h_blur);

		/*namedWindow("RGB", WINDOW_NORMAL);
		imshow("RGB", image);

		namedWindow("BLUR", WINDOW_NORMAL);
		imshow("BLUR", blur_image);

		waitKey(0);*/

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		CHECK_CUDA_STATUS(cudaDeviceReset());

		free(h_blur);
	}

	{
		Mat image;
		image = imread("C:\\Users\\petrush\\Downloads\\twocats.png", cv::IMREAD_COLOR);

		if (image.empty())
		{
			cout << "Could not open or find the image" << std::endl;
			return -1;
		}

		int height = image.rows;
		int width = image.cols;
		int channels = image.channels();

		cout << "H=" << height << " W=" << width << " C=" << channels << "\n";

		size_t size = width * height * sizeof(unsigned char);

		Mat channel[3];
		Mat blur_image;
		split(image, channel);

		unsigned char* h_b = (unsigned char*)channel[0].data;
		unsigned char* h_g = (unsigned char*)channel[1].data;
		unsigned char* h_r = (unsigned char*)channel[2].data;

		unsigned char* h_b_blur = (unsigned char*)malloc(size);
		unsigned char* h_g_blur = (unsigned char*)malloc(size);
		unsigned char* h_r_blur = (unsigned char*)malloc(size);

		unsigned char* d_b = nullptr;
		unsigned char* d_g = nullptr;
		unsigned char* d_r = nullptr;
		unsigned char* d_b_blur = nullptr;
		unsigned char* d_g_blur = nullptr;
		unsigned char* d_r_blur = nullptr;

		CHECK_CUDA_STATUS(cudaMalloc((void**)&d_b, size));
		CHECK_CUDA_STATUS(cudaMalloc((void**)&d_b_blur, size));

		CHECK_CUDA_STATUS(cudaMalloc((void**)&d_g, size));
		CHECK_CUDA_STATUS(cudaMalloc((void**)&d_g_blur, size));

		CHECK_CUDA_STATUS(cudaMalloc((void**)&d_r, size));
		CHECK_CUDA_STATUS(cudaMalloc((void**)&d_r_blur, size));

		CHECK_CUDA_STATUS(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
		CHECK_CUDA_STATUS(cudaMemcpy(d_g, h_g, size, cudaMemcpyHostToDevice));
		CHECK_CUDA_STATUS(cudaMemcpy(d_r, h_r, size, cudaMemcpyHostToDevice));

		int filter_size = RADIUS * 2 + 1;
		dim3 block_size(TILE_WIDTH, TILE_WIDTH);
		dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
		blur_box_kernel_coalesced << <grid_size, block_size >> > (d_b_blur, d_g_blur, d_r_blur, d_b, d_g, d_r, height, width, filter_size);

		CHECK_CUDA_STATUS(cudaMemcpy(h_b_blur, d_b_blur, size, cudaMemcpyDeviceToHost));
		CHECK_CUDA_STATUS(cudaMemcpy(h_g_blur, d_g_blur, size, cudaMemcpyDeviceToHost));
		CHECK_CUDA_STATUS(cudaMemcpy(h_r_blur, d_r_blur, size, cudaMemcpyDeviceToHost));

		CHECK_CUDA_STATUS(cudaFree(d_b));
		CHECK_CUDA_STATUS(cudaFree(d_b_blur));
		CHECK_CUDA_STATUS(cudaFree(d_g));
		CHECK_CUDA_STATUS(cudaFree(d_g_blur));
		CHECK_CUDA_STATUS(cudaFree(d_r));
		CHECK_CUDA_STATUS(cudaFree(d_r_blur));

		channel[0].data = h_b_blur;
		channel[1].data = h_g_blur;
		channel[2].data = h_r_blur;

		merge(channel, 3, blur_image);

		//Mat blur_image(height, width, CV_8UC3, h_blur);

	/*	namedWindow("RGB 2", WINDOW_NORMAL);
		imshow("RGB 2", image);

		namedWindow("BLUR 2", WINDOW_NORMAL);
		imshow("BLUR 2", blur_image);

		waitKey(0);*/

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		CHECK_CUDA_STATUS(cudaDeviceReset());

		free(h_b_blur);
		free(h_g_blur);
		free(h_r_blur);
	}

	{
		Mat image;
		image = imread("C:\\Users\\petrush\\Downloads\\twocats.png", cv::IMREAD_GRAYSCALE);

		if (image.empty())
		{
			cout << "Could not open or find the image" << std::endl;
			return -1;
		}

		int height = image.rows;
		int width = image.cols;
		int channels = image.channels();

		cout << "H=" << height << " W=" << width << " C=" << channels << "\n";

		size_t size = width * height * channels * sizeof(unsigned char);

		unsigned char* h_gray = (unsigned char*)image.data;
		unsigned char* h_blur = (unsigned char*)malloc(size);

		unsigned char* d_gray = nullptr;
		unsigned char* d_blur = nullptr;

		CHECK_CUDA_STATUS(cudaMalloc((void**)&d_gray, size));
		CHECK_CUDA_STATUS(cudaMalloc((void**)&d_blur, size));

		CHECK_CUDA_STATUS(cudaMemcpy(d_gray, h_gray, size, cudaMemcpyHostToDevice));

		int filter_size = RADIUS * 2 + 1;
		dim3 block_size(TILE_WIDTH, TILE_WIDTH);
		dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
		blur_box_kernel_shared << <grid_size, block_size >> > (d_blur, d_gray, height, width, filter_size);

		CHECK_CUDA_STATUS(cudaMemcpy(h_blur, d_blur, size, cudaMemcpyDeviceToHost));

		Mat blur_image(height, width, CV_8UC1, h_blur);
		cout << "H=" << blur_image.rows << " W=" << blur_image.cols<< " C=" << blur_image.channels() << "\n";

		//namedWindow("GRAY", WINDOW_NORMAL);
		//imshow("GRAY", image);

		//namedWindow("BLUR", WINDOW_NORMAL);
		//imshow("BLUR", blur_image);

		//waitKey(0);

		CHECK_CUDA_STATUS(cudaFree(d_gray));
		CHECK_CUDA_STATUS(cudaFree(d_blur));

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		CHECK_CUDA_STATUS(cudaDeviceReset());

		free(h_blur);
	}
	
	return 0;

}
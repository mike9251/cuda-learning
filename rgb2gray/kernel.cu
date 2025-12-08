
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


#define CHECK_CUDA_STATUS(status) { if (status != cudaSuccess) { \
    fprintf(stderr, "CUDA Error: %s (error code %d) at %s:%d\n", \
        cudaGetErrorString(status), status, __FILE__, __LINE__); exit(EXIT_FAILURE); } }

__global__ void bgr2gray_kernel(unsigned char *gray, const unsigned char *bgr, unsigned int height, unsigned int width)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < height && col < width)
    {
        int offset = row * width + col;

        //unsigned char b = bgr[offset * 3];
        //unsigned char g = bgr[offset * 3 + 1];
        //unsigned char r = bgr[offset * 3 + 2];
        uchar3 pix = *reinterpret_cast<const uchar3*>(&bgr[offset * 3]);
        float value = 0.299f * pix.z + 0.587f * pix.y + 0.114f * pix.x;

        //float value = 0.299f * r + 0.587f * g + 0.114f * b;
        gray[offset] = static_cast<unsigned char>(value);
    }
}


void benchmark_rgb_to_gray(unsigned char* d_out, const unsigned char* d_in, int height, int width, int block_size, int iters)
{
    dim3 blockSize(block_size, block_size);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    for (int i = 0; i < 5; i++)
    {
        bgr2gray_kernel << <gridSize, blockSize >> > (d_out, d_in, height, width);
    }
    CHECK_CUDA_STATUS(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA_STATUS(cudaEventCreate(&start));
    CHECK_CUDA_STATUS(cudaEventCreate(&stop));

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++)
    {
        bgr2gray_kernel << <gridSize, blockSize >> > (d_out, d_in, height, width);
    }
    cudaEventRecord(stop);
    CHECK_CUDA_STATUS(cudaEventSynchronize(stop));

    float ms = 0.0;
    CHECK_CUDA_STATUS(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA_STATUS(cudaEventDestroy(start));
    CHECK_CUDA_STATUS(cudaEventDestroy(stop));

    // Average kernel time
    double avg_ms = ms / iters;
    double avg_s = avg_ms * 1e-3;

    // Bytes moved per kernel call: 1 3B load + 1 1B store = 4 * height * width
    double bytes = 4 * height * width;
    double gbytes = bytes / 1e9;

    double bw_GBps = gbytes / avg_s;          // achieved bandwidth
    double flops = 5 * height * width;        // 5 FLOP per element
    double gflops = (flops / avg_s) / 1e9;    // FLOP/s → GFLOP/s
    double intensity = flops / bytes;         // FLOP/byte

    std::cout << "RGB -> Grayscale kernel benchmark\n";
    std::cout << "  N              = " << height * width << "\n";
    std::cout << "  blockSize      = " << block_size * block_size << "\n";
    std::cout << "  iters          = " << iters << "\n";
    std::cout << "  avg time       = " << avg_ms << " ms\n";
    std::cout << "  bandwidth      = " << bw_GBps << " GB/s\n";
    std::cout << "  GFLOP/s        = " << gflops << "\n";
    std::cout << "  arithmetic I   = " << intensity << " FLOP/B\n";
}


int main()
{
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    std::cout << "Device name: " << device_prop.name << std::endl;
    std::cout << "Max threads per block: " << device_prop.maxThreadsPerBlock << std::endl;
    std::cout << "Shared memory per block: " << device_prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "regsPerBlock: " << device_prop.regsPerBlock << std::endl;
    std::cout << "maxBlocksPerMultiProcessor: " << device_prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "maxThreadsPerMultiProcessor: " << device_prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "maxWarpsPerMultiProcessor: " << device_prop.maxThreadsPerMultiProcessor / 32 << std::endl;

    int block_size = 16;
    int iters = 100;

    cv::Mat image;
    image = cv::imread("C:\\Users\\petrush\\Downloads\\painting-mountain-lake-with-mountain-background.jpg", cv::IMREAD_COLOR); // Read the file

    if (image.empty()) // Check for invalid input
    {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

	int width = image.cols * 4;
	int height = image.rows * 4;
	int channels = image.channels();

    // Create an empty Mat object for the resized image
    cv::Mat resized_image;

    // Resize the image
    cv::resize(image, resized_image, cv::Size(width, height));

    image = resized_image;

    size_t size = width * height * channels * sizeof(unsigned char);

    unsigned char* h_gray = (unsigned char*)malloc(size / 3);
    unsigned char* h_bgr = (unsigned char*)image.data;

    unsigned char* d_bgr = nullptr;
    unsigned char* d_gray = nullptr;

    cudaEvent_t start, stop;
    CHECK_CUDA_STATUS(cudaEventCreate(&start));
    CHECK_CUDA_STATUS(cudaEventCreate(&stop));

    CHECK_CUDA_STATUS(cudaMalloc((void**)&d_bgr, size));
    CHECK_CUDA_STATUS(cudaMalloc((void**)&d_gray, size / channels));
    
    cudaEventRecord(start);
	CHECK_CUDA_STATUS(cudaMemcpy(d_bgr, h_bgr, size, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    CHECK_CUDA_STATUS(cudaEventSynchronize(stop));

    float h2d_ms = 0.0f;
    CHECK_CUDA_STATUS(cudaEventElapsedTime(&h2d_ms, start, stop));

    benchmark_rgb_to_gray(d_gray, d_bgr, height, width, block_size, iters);

	cudaEventRecord(start);
	CHECK_CUDA_STATUS(cudaMemcpy(h_gray, d_gray, size / channels, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    CHECK_CUDA_STATUS(cudaEventSynchronize(stop));

    float d2h_ms = 0.0f;
    CHECK_CUDA_STATUS(cudaEventElapsedTime(&d2h_ms, start, stop));
    std::cout << "Host to Device transfer time: " << h2d_ms << " ms" << std::endl;
    std::cout << "Device to Host transfer time: " << d2h_ms << " ms" << std::endl;
	std::cout << "Total time for data transfer: " << (h2d_ms + d2h_ms) << " ms" << std::endl;

	cv::Mat gray_image(height, width, CV_8UC1, h_gray);

    cv::namedWindow("RGB", cv::WINDOW_NORMAL); // Create a window for display.
    cv::imshow("RGB", image); // Show our image inside it.
    cv::namedWindow("GRAY", cv::WINDOW_NORMAL); // Create a window for display.
    cv::imshow("GRAY", gray_image); // Show our image inside it.
    cv::waitKey(0); // Wait for a keystroke in the window

    // Cleanup
    cudaFree(d_bgr);
    cudaFree(d_gray);
	free(h_gray);
    CHECK_CUDA_STATUS(cudaEventDestroy(start));
    CHECK_CUDA_STATUS(cudaEventDestroy(stop));

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    CHECK_CUDA_STATUS(cudaDeviceReset());

    return 0;
}
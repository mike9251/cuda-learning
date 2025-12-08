
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#ifndef __CUDACC__  
    #define __CUDACC__
#endif

#include <crt/device_functions.h>

#define CHECK_CUDA_STATUS(status) { if (status != cudaSuccess) { \
    fprintf(stderr, "CUDA Error: %s (error code %d) at %s:%d\n", \
        cudaGetErrorString(status), status, __FILE__, __LINE__); exit(EXIT_FAILURE); } }


__global__ void matmul_kernel_1(const float* A, const float* B, float* C, int m, int n, int k)
{
    // (m, k) @ (k, n)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < m) && (col < n))
    {
        float val = 0.f;
        for (int i = 0; i < k; i++)
        {
            val += A[row * k + i] * B[col + i * n];
        }
        C[row * n + col] = val;
    }
}

#define TILE_WIDTH 16

__global__ void matmul_kernel_tiled(const float* A, const float* B, float* C, int m, int n, int k)
{
	// A - m x k
    // B - k x n
    // C - m x n
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float P = 0.f;

    for (int ph = 0; ph < (k + TILE_WIDTH - 1) / TILE_WIDTH; ph++)
    {
        if ((row < m) && (ph * TILE_WIDTH + tx < k))
        {
            As[ty][tx] = A[row * k + ph * TILE_WIDTH + tx];
        }
        else
        {
            As[ty][tx] = 0.f;
        }
        
        if ((col < n) && (ph * TILE_WIDTH + ty < k))
        {
            Bs[ty][tx] = B[(ph * TILE_WIDTH + ty) * n + col];
        }
        else
        {
            Bs[ty][tx] = 0.f;
        }
        
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++)
        {
            P += As[ty][i] * Bs[i][tx];
        }

		__syncthreads();
    }

    if ((row < m) && (col < n))
    {
        C[row * n + col] = P;
    }
}

void read_data(std::string path, float* data)
{
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) {
        perror("fopen");
        return;
    }

    fseek(fp, 0, SEEK_END);
    long bytes = ftell(fp);
    rewind(fp);

    size_t count = bytes / sizeof(float);
	printf("Reading %zu floats from %s\n", count, path.c_str());

    if (fread(data, sizeof(float), count, fp) != count) {
        perror("fread");
        free(data);
        fclose(fp);
        return;
    }

    fclose(fp);
}

int main()
{
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
	std::cout << "Device name: " << device_prop.name << std::endl;
	std::cout << "Max threads per block: " << device_prop.maxThreadsPerBlock << std::endl;
	std::cout << "Shared memory per block: " << device_prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
	std::cout << "regsPerBlock: " << device_prop.regsPerBlock << std::endl;
	std::cout << "maxThreadsPerBlock: " << device_prop.maxThreadsPerBlock << std::endl;

    int m = 512;
    int n = 2500;
    int k = 1000;
    size_t size_a = m * k * sizeof(float);
    size_t size_b = k * n * sizeof(float);
    size_t size_c = m * n * sizeof(float);

    float* h_a = (float*)malloc(size_a);
    float* h_b = (float*)malloc(size_b);
    float* h_c = (float*)malloc(size_c);

    float* h_c_ref = (float*)malloc(size_c);

    /*read_data("C:\\Users\\petrush\\source\\repos\\cuda_1\\matmul_tile\\x64\\Debug\\a_2048_2048.npy", h_a);
    read_data("C:\\Users\\petrush\\source\\repos\\cuda_1\\matmul_tile\\x64\\Debug\\b_2048_2048.npy", h_b);
    read_data("C:\\Users\\petrush\\source\\repos\\cuda_1\\matmul_tile\\x64\\Debug\\c_2048_2048.npy", h_c_ref);*/

    read_data("C:\\Users\\petrush\\source\\repos\\cuda_1\\x64\\Release\\a_512_1000.npy", h_a);
    read_data("C:\\Users\\petrush\\source\\repos\\cuda_1\\x64\\Release\\b_1000_2500.npy", h_b);
    read_data("C:\\Users\\petrush\\source\\repos\\cuda_1\\x64\\Release\\c_512_2500.npy", h_c_ref);
    

    //for (int i = 0; i < m; i++)
    //{
    //    for (int j = 0; j < n; j++)
    //    {
    //        std::cout << h_b[i * n + j] << " ";
    //    }
    //    std::cout << std::endl;
    //}

    //float h_a[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    //float h_b[] = { 1, 1, 1, 2, 2, 2, 3, 3, 3 };
    //float h_c[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    float* d_a, * d_b, * d_c;
    CHECK_CUDA_STATUS(cudaMalloc((void**)&d_a, size_a));
    CHECK_CUDA_STATUS(cudaMalloc((void**)&d_b, size_b));
    CHECK_CUDA_STATUS(cudaMalloc((void**)&d_c, size_c));

    CHECK_CUDA_STATUS(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    CHECK_CUDA_STATUS(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));

    {
        dim3 blockDim(16, 16);
        dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);
        std::cout << "Num blocks: " << gridDim.x << " x " << gridDim.y << " = " << gridDim.x * gridDim.y << std::endl;
        matmul_kernel_1 << <gridDim, blockDim >> > (d_a, d_b, d_c, m, n, k);

        cudaDeviceSynchronize();
    }

    {
        dim3 blockDim(16, 16);
        dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);
        std::cout << "Num blocks: " << gridDim.x << " x " << gridDim.y << " = " << gridDim.x * gridDim.y << std::endl;
        matmul_kernel_tiled << <gridDim, blockDim >> > (d_a, d_b, d_c, m, n, k);

        cudaDeviceSynchronize();
    }


    CHECK_CUDA_STATUS(cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost));

    unsigned int errors = 0;
    float max_error = 0.f;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            float delta = abs(h_c[i * n + j] - h_c_ref[i * n + j]);
            if (delta > 1e-5)
            {
                max_error = fmax(max_error, delta);
                errors++;
            }
            //std::cout << h_c[i * n + j] << " ";
        }
        //std::cout << std::endl;
    }

    std::cout << "Number of errors: " << errors << " Max error: " << max_error << std::endl;

    CHECK_CUDA_STATUS(cudaFree(d_a));
    CHECK_CUDA_STATUS(cudaFree(d_b));
    CHECK_CUDA_STATUS(cudaFree(d_c));

    CHECK_CUDA_STATUS(cudaDeviceReset());

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_ref);

    return 0;
}

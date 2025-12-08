
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>


#define CHECK_CUDA_STATUS(status) { if (status != cudaSuccess) { \
    fprintf(stderr, "CUDA Error: %s (error code %d) at %s:%d\n", \
        cudaGetErrorString(status), status, __FILE__, __LINE__); exit(EXIT_FAILURE); } }


__global__ void addKernel(float*c, const float*a, const float*b, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
        c[i] = a[i] + b[i];
}


void benchmark_vector_add(float* d_c, const float* d_a, const float* d_b,
    long N, int blockSize, int iters)
{
    dim3 block(blockSize);
    dim3 grid((N + block.x - 1) / block.x);

    // Warm-up (not timed)
    for (int i = 0; i < 5; ++i) {
        addKernel << <grid, block >> > (d_c, d_a, d_b, N);
    }
    CHECK_CUDA_STATUS(cudaDeviceSynchronize());

    // Create events
    cudaEvent_t start, stop;
    CHECK_CUDA_STATUS(cudaEventCreate(&start));
    CHECK_CUDA_STATUS(cudaEventCreate(&stop));

    // Record start
    CHECK_CUDA_STATUS(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        addKernel << <grid, block >> > (d_c, d_a, d_b, N);
    }
    CHECK_CUDA_STATUS(cudaEventRecord(stop));
    CHECK_CUDA_STATUS(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA_STATUS(cudaEventElapsedTime(&ms, start, stop));

    // Cleanup events
    CHECK_CUDA_STATUS(cudaEventDestroy(start));
    CHECK_CUDA_STATUS(cudaEventDestroy(stop));

    // Average kernel time
    double avg_ms = ms / iters;
    double avg_s = avg_ms * 1e-3;

    // Bytes moved per kernel call: 2 loads + 1 store = 3 * N * sizeof(float)
    double bytes = 3.0 * N * sizeof(float);
    double gbytes = bytes / 1e9;

    double bw_GBps = gbytes / avg_s;          // achieved bandwidth
    double flops = 1.0 * N;                 // 1 FLOP per element
    double gflops = (flops / avg_s) / 1e9;   // FLOP/s → GFLOP/s
    double intensity = flops / bytes;           // FLOP/byte

    std::cout << "Vector add kernel benchmark\n";
    std::cout << "  N              = " << N << "\n";
    std::cout << "  blockSize      = " << blockSize << "\n";
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

    const long N = 10'000'000;
    const int blockSize = 256;
    const int iters = 100;

    size_t bytes = N * sizeof(float);

    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);

    // Init host data
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    // Init device
    CHECK_CUDA_STATUS(cudaSetDevice(0));
    float* d_a = nullptr, * d_b = nullptr, * d_c = nullptr;
    CHECK_CUDA_STATUS(cudaMalloc((void**)&d_a, bytes));
    CHECK_CUDA_STATUS(cudaMalloc((void**)&d_b, bytes));
    CHECK_CUDA_STATUS(cudaMalloc((void**)&d_c, bytes));

    cudaEvent_t start, stop;
    CHECK_CUDA_STATUS(cudaEventCreate(&start));
    CHECK_CUDA_STATUS(cudaEventCreate(&stop));

    cudaEventRecord(start);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Copy inputs to device (not timed for kernel benchmark)
    CHECK_CUDA_STATUS(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_STATUS(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    CHECK_CUDA_STATUS(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA_STATUS(cudaEventElapsedTime(&ms, start, stop));

    // Benchmark kernel
    benchmark_vector_add(d_c, d_a, d_b, N, blockSize, iters);

    // Copy back once and verify correctness
    cudaEventRecord(start);
    CHECK_CUDA_STATUS(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    CHECK_CUDA_STATUS(cudaEventSynchronize(stop));

    float ms_d2h = 0.0f;
    CHECK_CUDA_STATUS(cudaEventElapsedTime(&ms_d2h, start, stop));

    std::cout << "Vector add memory cpy\n";
    std::cout << "  time       = " << ms + ms_d2h << " ms\n";

    int errors = 0;
    for (long i = 0; i < N; ++i) {
        float ref = h_a[i] + h_b[i];
        if (h_c[i] != ref) {
            ++errors;
        }
    }
    std::cout << "Number of errors: " << errors << std::endl;

    // Cleanup
    CHECK_CUDA_STATUS(cudaEventDestroy(start));
    CHECK_CUDA_STATUS(cudaEventDestroy(stop));
    CHECK_CUDA_STATUS(cudaFree(d_a));
    CHECK_CUDA_STATUS(cudaFree(d_b));
    CHECK_CUDA_STATUS(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    CHECK_CUDA_STATUS(cudaDeviceReset());
    return 0;
}

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <memory>
#include <stdio.h>
#include <iostream>

#include <chrono>

#ifndef __CUDACC__  
    #define __CUDACC__
#endif

#include <crt/device_functions.h>


#define CHECK_CUDA_STATUS(status) { if (status != cudaSuccess) { \
    fprintf(stderr, "CUDA Error: %s (error code %d) at %s:%d\n", \
        cudaGetErrorString(status), status, __FILE__, __LINE__); exit(EXIT_FAILURE); } }


using Clock = std::chrono::steady_clock;
using ms = std::chrono::duration<double, std::milli>;


// Stencil coefficients (3d order = 2)
static constexpr float a0 = 1.f;    // center
static constexpr float a1 = 0.5f;   // right + 1
static constexpr float a2 = 0.25f;  // right + 2
static constexpr float a3 = 0.5f;   // left - 1
static constexpr float a4 = 0.25f;  // left - 2
static constexpr float a5 = 0.5f;   // top + 1
static constexpr float a6 = 0.25f;  // top + 2
static constexpr float a7 = 0.5f;   // bottom - 1
static constexpr float a8 = 0.25f;  // bottom - 2
static constexpr float a9 = 0.5f;   // front + 1
static constexpr float a10 = 0.25f; // front + 2
static constexpr float a11 = 0.5f;  // back + 1
static constexpr float a12 = 0.25f; // back + 2


void stencil3d_cpu(
    float* out,
    const float* in,
    unsigned int width,
    unsigned int height,
    unsigned int depth)
{
    for (unsigned int ch = 0; ch < depth; ch++)
    {
        for (unsigned int row = 0; row < height; row++)
        {
            for (unsigned int col = 0; col < width; col++)
            {
                // to preserve boundary conditions
                if (ch >= 2 && ch < depth - 2 && row >= 2 && row < height - 2 && col >= 2 && col < width - 2)
                {
                    out[ch * height * width + row * width + col] = a0 * in[ch * height * width + row * width + col] + \
                        a1 * in[ch * height * width + row * width + col + 1] + \
                        a2 * in[ch * height * width + row * width + col + 2] + \
                        a3 * in[ch * height * width + row * width + col - 1] + \
                        a4 * in[ch * height * width + row * width + col - 2] + \
                        a5 * in[ch * height * width + (row + 1) * width + col] + \
                        a6 * in[ch * height * width + (row + 2) * width + col] + \
                        a7 * in[ch * height * width + (row - 1) * width + col] + \
                        a8 * in[ch * height * width + (row - 2) * width + col] + \
                        a9 * in[(ch + 1) * height * width + row * width + col] + \
                        a10 * in[(ch + 2) * height * width + row * width + col] + \
                        a11 * in[(ch - 1) * height * width + row * width + col] + \
                        a12 * in[(ch - 2) * height * width + row * width + col];
                }
                else
                {
                    out[ch * height * width + row * width + col] = in[ch * height * width + row * width + col];
                }
            }
        }
    }
}



__global__ void stencil3d_kernel(
    float* out,
    const float* in,
    unsigned int width,
    unsigned int height,
    unsigned int channels)
{
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int tz = blockIdx.z * blockDim.z + threadIdx.z;

    if (tx >= width || ty >= height || tz >= channels)
        return;

    unsigned int depth_stride = height * width;

    if (tx >= 2 && tx < width - 2 && ty >= 2 && ty < height - 2 && tz >= 2 && tz < channels - 2)
    {
        out[tz * depth_stride + ty * width + tx] = a0 * in[tz * depth_stride + ty * width + tx] + \
            a1 * in[tz * depth_stride + ty * width + tx + 1] + \
            a2 * in[tz * depth_stride + ty * width + tx + 2] + \
            a3 * in[tz * depth_stride + ty * width + tx - 1] + \
            a4 * in[tz * depth_stride + ty * width + tx - 2] + \
            a5 * in[tz * depth_stride + (ty + 1) * width + tx] + \
            a6 * in[tz * depth_stride + (ty + 2) * width + tx] + \
            a7 * in[tz * depth_stride + (ty - 1) * width + tx] + \
            a8 * in[tz * depth_stride + (ty - 2) * width + tx] + \
            a9 * in[(tz + 1) * depth_stride + ty * width + tx] + \
            a10 * in[(tz + 2) * depth_stride + ty * width + tx] + \
            a11 * in[(tz - 1) * depth_stride + ty * width + tx] + \
            a12 * in[(tz - 2) * depth_stride + ty * width + tx];
    }
    else
    {
        out[tz * depth_stride + ty * width + tx] = in[tz * depth_stride + ty * width + tx];
    }
}


#define STENCIL_ORDER 2
#define TILE_IN1 8
#define TILE_OUT1 (TILE_IN1 - 2 * STENCIL_ORDER)

__global__ void stencil3d_tiled_kernel(
    float* out,
    const float* in,
    unsigned int width,
    unsigned int height,
    unsigned int channels)
{
	// 1. get thread position within the input tile
    int in_col = blockIdx.x * TILE_OUT1 + threadIdx.x - STENCIL_ORDER;
    int in_row = blockIdx.y * TILE_OUT1 + threadIdx.y - STENCIL_ORDER;
    int in_ch = blockIdx.z * TILE_OUT1 + threadIdx.z - STENCIL_ORDER;

    unsigned int depth_stride = height * width;
    
	// 2. load input tile to shared memory
    __shared__ float smem[TILE_IN1][TILE_IN1][TILE_IN1];

    if (in_ch >= 0 && in_ch < channels && in_row >= 0 && in_row < height && in_col >= 0 && in_col < width) // skip ghost elements
    {
        smem[threadIdx.z][threadIdx.y][threadIdx.x] = in[in_ch * depth_stride + in_row * width + in_col];
    }

    __syncthreads();

    float temp = 0.f;

	// 3. select threads that will compute output elements
    if (threadIdx.x >= STENCIL_ORDER && threadIdx.x < TILE_OUT1 + STENCIL_ORDER && \
        threadIdx.y >= STENCIL_ORDER && threadIdx.y < TILE_OUT1 + STENCIL_ORDER && \
        threadIdx.z >= STENCIL_ORDER && threadIdx.z < TILE_OUT1 + STENCIL_ORDER)
    {
        unsigned int sx = threadIdx.x;
        unsigned int sy = threadIdx.y;
        unsigned int sz = threadIdx.z;

        // 4. check that corresponding input elements are not ghost elements (map working threads back to global output grid)
        if (in_col >= STENCIL_ORDER && in_col < width - STENCIL_ORDER &&
            in_row >= STENCIL_ORDER && in_row < height - STENCIL_ORDER &&
            in_ch >= STENCIL_ORDER && in_ch < channels - STENCIL_ORDER)
        {
            // 5. perform stencil computation using shared memory
            temp = a0 * smem[sz][sy][sx] + \
                a1 * smem[sz][sy][sx + 1] + \
                a2 * smem[sz][sy][sx + 2] + \
                a3 * smem[sz][sy][sx - 1] + \
                a4 * smem[sz][sy][sx - 2] + \
                a5 * smem[sz][sy + 1][sx] + \
                a6 * smem[sz][sy + 2][sx] + \
                a7 * smem[sz][sy - 1][sx] + \
                a8 * smem[sz][sy - 2][sx] + \
                a9 * smem[sz + 1][sy][sx] + \
                a10 * smem[sz + 2][sy][sx] + \
                a11 * smem[sz - 1][sy][sx] + \
                a12 * smem[sz - 2][sy][sx];

        }
        else
        {
            temp = smem[sz][sy][sx];
        }
        if (in_ch < channels && in_row < height && in_col < width)
         out[in_ch * depth_stride + in_row * width + in_col] = temp;
    }

}


#define STENCIL_ORDER 2
#define TILE_IN2 16
#define TILE_OUT2 (TILE_IN2 - 2 * STENCIL_ORDER)

__global__ void stencil3d_tiled_thread_coarsening_kernel(
    float* out,
    const float* in,
    unsigned int width,
    unsigned int height,
    unsigned int channels)
{
    int gz = blockIdx.z * TILE_OUT2;
	int gy = blockIdx.y * TILE_OUT2 + threadIdx.y - STENCIL_ORDER;
    int gx = blockIdx.x * TILE_OUT2 + threadIdx.x - STENCIL_ORDER;

    int channel_stride = height * width;

    __shared__ float prev[TILE_IN2][TILE_IN2];
    __shared__ float prev2[TILE_IN2][TILE_IN2];
    __shared__ float curr[TILE_IN2][TILE_IN2];
    __shared__ float next[TILE_IN2][TILE_IN2];
    __shared__ float next2[TILE_IN2][TILE_IN2];

    if (gz - 2 >= 0 && gz - 2 < channels && gy >= 0 && gy < height && gx >= 0 && gx < width)
    {
        prev[threadIdx.y][threadIdx.x] = in[(gz - 2) * channel_stride + gy * width + gx];
    }

    if (gz - 1 >= 0 && gz - 1 < channels && gy >= 0 && gy < height && gx >= 0 && gx < width)
    {
        prev2[threadIdx.y][threadIdx.x] = in[(gz - 1) * channel_stride + gy * width + gx];
    }

    if (gz >= 0 && gz < channels && gy >= 0 && gy < height && gx >= 0 && gx < width)
    {
        curr[threadIdx.y][threadIdx.x] = in[gz * channel_stride + gy * width + gx];
    }

    if (gz + 1 < channels && gy >= 0 && gy < height && gx >= 0 && gx < width)
    {
        next[threadIdx.y][threadIdx.x] = in[(gz + 1) * channel_stride + gy * width + gx];
    }

    float temp = 0.f;
    for (int z = gz; z < gz + TILE_OUT2; z++)
    {
        if (z + 2 < channels && gy >= 0 && gy < height && gx >= 0 && gx < width)
        {
            next2[threadIdx.y][threadIdx.x] = in[(z + 2) * channel_stride + gy * width + gx];
        }

        __syncthreads();

        bool is_center = threadIdx.x >= STENCIL_ORDER && threadIdx.x < TILE_OUT2 + STENCIL_ORDER && \
            threadIdx.y >= STENCIL_ORDER && threadIdx.y < TILE_OUT2 + STENCIL_ORDER;

        bool is_interior = z >= STENCIL_ORDER && z < channels - STENCIL_ORDER && gy >= STENCIL_ORDER && gy < height - STENCIL_ORDER && \
            gx >= STENCIL_ORDER && gx < width - STENCIL_ORDER;

        if (is_center)
        {
            if (is_interior)
            {
                temp = a0 * curr[threadIdx.y][threadIdx.x] + \
                    a1 * curr[threadIdx.y][threadIdx.x + 1] + \
                    a2 * curr[threadIdx.y][threadIdx.x + 2] + \
                    a3 * curr[threadIdx.y][threadIdx.x - 1] + \
                    a4 * curr[threadIdx.y][threadIdx.x - 2] + \
                    a5 * curr[threadIdx.y + 1][threadIdx.x] + \
                    a6 * curr[threadIdx.y + 2][threadIdx.x] + \
                    a7 * curr[threadIdx.y - 1][threadIdx.x] + \
                    a8 * curr[threadIdx.y - 2][threadIdx.x] + \
                    a9 * next[threadIdx.y][threadIdx.x] + \
                    a10 * next2[threadIdx.y][threadIdx.x] + \
                    a11 * prev2[threadIdx.y][threadIdx.x] + \
                    a12 * prev[threadIdx.y][threadIdx.x];
            }
            else
            {
                temp = curr[threadIdx.y][threadIdx.x];
            }

            if (z < channels && gy >= 0 && gy < height && gx >= 0 && gx < width)
                out[z * channel_stride + gy * width + gx] = temp;
        }
        
        __syncthreads();

        prev[threadIdx.y][threadIdx.x] = prev2[threadIdx.y][threadIdx.x];
        prev2[threadIdx.y][threadIdx.x] = curr[threadIdx.y][threadIdx.x];
        curr[threadIdx.y][threadIdx.x] = next[threadIdx.y][threadIdx.x];
        next[threadIdx.y][threadIdx.x] = next2[threadIdx.y][threadIdx.x];
    }
}


__global__ void stencil3d_tiled_thread_coarsening_register_tiling_kernel(
    float* out,
    const float* in,
    unsigned int width,
    unsigned int height,
    unsigned int channels)
{
    int gz = blockIdx.z * TILE_OUT2;
    int gy = blockIdx.y * TILE_OUT2 + threadIdx.y - STENCIL_ORDER;
    int gx = blockIdx.x * TILE_OUT2 + threadIdx.x - STENCIL_ORDER;

    int channel_stride = height * width;

    float prev = 0.f;
    float prev2 = 0.f;
    __shared__ float curr[TILE_IN2][TILE_IN2];
    float next = 0.f;
    float next2 = 0.f;

    if (gz - 2 >= 0 && gz - 2 < channels && gy >= 0 && gy < height && gx >= 0 && gx < width)
    {
        prev = in[(gz - 2) * channel_stride + gy * width + gx];
    }

    if (gz - 1 >= 0 && gz - 1 < channels && gy >= 0 && gy < height && gx >= 0 && gx < width)
    {
        prev2 = in[(gz - 1) * channel_stride + gy * width + gx];
    }

    if (gz >= 0 && gz < channels && gy >= 0 && gy < height && gx >= 0 && gx < width)
    {
        curr[threadIdx.y][threadIdx.x] = in[gz * channel_stride + gy * width + gx];
    }

    if (gz + 1 < channels && gy >= 0 && gy < height && gx >= 0 && gx < width)
    {
        next = in[(gz + 1) * channel_stride + gy * width + gx];
    }

    float temp = 0.f;
    for (int z = gz; z < gz + TILE_OUT2; z++)
    {
        if (z + 2 < channels && gy >= 0 && gy < height && gx >= 0 && gx < width)
        {
            next2 = in[(z + 2) * channel_stride + gy * width + gx];
        }

        __syncthreads();

        bool is_center = threadIdx.x >= STENCIL_ORDER && threadIdx.x < TILE_OUT2 + STENCIL_ORDER && \
            threadIdx.y >= STENCIL_ORDER && threadIdx.y < TILE_OUT2 + STENCIL_ORDER;

        bool is_interior = z >= STENCIL_ORDER && z < channels - STENCIL_ORDER && gy >= STENCIL_ORDER && gy < height - STENCIL_ORDER && \
            gx >= STENCIL_ORDER && gx < width - STENCIL_ORDER;

        if (is_center)
        {
            if (is_interior)
            {
                temp = a0 * curr[threadIdx.y][threadIdx.x] + \
                    a1 * curr[threadIdx.y][threadIdx.x + 1] + \
                    a2 * curr[threadIdx.y][threadIdx.x + 2] + \
                    a3 * curr[threadIdx.y][threadIdx.x - 1] + \
                    a4 * curr[threadIdx.y][threadIdx.x - 2] + \
                    a5 * curr[threadIdx.y + 1][threadIdx.x] + \
                    a6 * curr[threadIdx.y + 2][threadIdx.x] + \
                    a7 * curr[threadIdx.y - 1][threadIdx.x] + \
                    a8 * curr[threadIdx.y - 2][threadIdx.x] + \
                    a9 * next + \
                    a10 * next2 + \
                    a11 * prev2 + \
                    a12 * prev;
            }
            else
            {
                temp = curr[threadIdx.y][threadIdx.x];
            }

            if (z < channels && gy >= 0 && gy < height && gx >= 0 && gx < width)
                out[z * channel_stride + gy * width + gx] = temp;
        }

        __syncthreads();

        prev = prev2;
        prev2 = curr[threadIdx.y][threadIdx.x];
        curr[threadIdx.y][threadIdx.x] = next;
        next = next2;
    }
}


void test_ref_code()
{
    std::cout << "--------------- Stencil 3D --------------\n";
    unsigned int N = 10;

    float* h_in = (float*)malloc(N * N * N * sizeof(float));

    for (unsigned int z = 0; z < N; z++)
    {
        for (unsigned int y = 0; y < N; y++)
        {
            for (unsigned int x = 0; x < N; x++)
            {
                h_in[z * N * N + y * N + x] = (float)(x + 1);
            }
        }
    }

    float* h_out = (float*)malloc(N * N * N * sizeof(float));

    stencil3d_cpu(h_out, h_in, N, N, N);

    unsigned int z = 2;
    std::cout << "Input: " << std::endl;
    for (unsigned int y = 0; y < N; y++)
    {
        for (unsigned int x = 0; x < N; x++)
        {
            std::cout << h_in[z * N * N + y * N + x] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Output: " << std::endl;

    for (unsigned int y = 0; y < N; y++)
    {
        for (unsigned int x = 0; x < N; x++)
        {
            std::cout << h_out[z * N * N + y * N + x] << " ";
        }
        std::cout << std::endl;
    }

    free(h_in);
    free(h_out);
}

double benchmark_stencil3d_cpu(
    float* out,
    const float* in,
    unsigned int width,
    unsigned int height,
    unsigned int channels,
    unsigned int iters)
{
    // warmup: make sure caches etc. are "hot"
    stencil3d_cpu(out, in, width, height, channels);

    auto start = Clock::now();
    for (int i = 0; i < iters; ++i) {
        stencil3d_cpu(out, in, width, height, channels);
    }
    auto end = Clock::now();

    float checksum = 0.0f;
    for (int i = 0; i < width * height * channels; ++i) checksum += out[i];
    std::cout << "checksum = " << checksum << "\n";

    ms total = end - start;
    double avg_ms = total.count() / iters;
    return avg_ms;
}



unsigned int cdiv(unsigned int n, unsigned int x)
{
    return (n + x - 1) / x;
}


void benchmark_stencil3d_kernel(
    float* d_out,
    const float* d_in,
    unsigned int width,
    unsigned int height,
    unsigned int channels,
    int blockSize,
    int iters)
{
    dim3 block_size(blockSize, blockSize, blockSize);
    dim3 grid_size(cdiv(width, block_size.x), cdiv(height, block_size.y), cdiv(channels, block_size.z));

    // Warm-up (not timed)
    for (int i = 0; i < 5; ++i) {
        stencil3d_kernel << <grid_size, block_size >> > (d_out, d_in, width, height, channels);
    }
    CHECK_CUDA_STATUS(cudaDeviceSynchronize());

    // Create events
    cudaEvent_t start, stop;
    CHECK_CUDA_STATUS(cudaEventCreate(&start));
    CHECK_CUDA_STATUS(cudaEventCreate(&stop));

    // Record start
    CHECK_CUDA_STATUS(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        stencil3d_kernel << <grid_size, block_size >> > (d_out, d_in, width, height, channels);
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

    unsigned int N = height * width * channels;
    // Bytes moved per kernel call
    double bytes = N * 14 * sizeof(float);
    double gbytes = bytes / 1e9;

    double bw_GBps = gbytes / avg_s;
    double flops = N * 25;
    double gflops = (flops / avg_s) / 1e9;
    double intensity = flops / bytes;

    std::cout << "Stencil 3D kernel benchmark\n";
    std::cout << "  N              = " << N << "\n";
    std::cout << "  blockSize      = " << blockSize << "\n";
    std::cout << "  iters          = " << iters << "\n";
    std::cout << "  avg time       = " << avg_ms << " ms\n";
    std::cout << "  bandwidth      = " << bw_GBps << " GB/s\n";
    std::cout << "  GFLOP/s        = " << gflops << "\n";
    std::cout << "  arithmetic I   = " << intensity << " FLOP/B\n";
}

void benchmark_stencil3d_tiled_kernel(
    float* d_out,
    const float* d_in,
    unsigned int width,
    unsigned int height,
    unsigned int channels,
    int iters)
{
    dim3 block_size(TILE_IN1, TILE_IN1, TILE_IN1);
    dim3 grid_size(cdiv(width, TILE_OUT1), cdiv(height, TILE_OUT1), cdiv(channels, TILE_OUT1));

    // Warm-up (not timed)
    for (int i = 0; i < 5; ++i) {
        stencil3d_tiled_kernel << <grid_size, block_size >> > (d_out, d_in, width, height, channels);
    }
    CHECK_CUDA_STATUS(cudaDeviceSynchronize());

    // Create events
    cudaEvent_t start, stop;
    CHECK_CUDA_STATUS(cudaEventCreate(&start));
    CHECK_CUDA_STATUS(cudaEventCreate(&stop));

    // Record start
    CHECK_CUDA_STATUS(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        stencil3d_tiled_kernel << <grid_size, block_size >> > (d_out, d_in, width, height, channels);
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

    unsigned int N = height * width * channels;
    // Bytes moved per kernel call
    double bytes = grid_size.x * grid_size.y * grid_size.z * (TILE_IN1 * TILE_IN1 * TILE_IN1 + TILE_OUT1 * TILE_OUT1 * TILE_OUT1) * sizeof(float);
    double gbytes = bytes / 1e9;

    double bw_GBps = gbytes / avg_s;
    double flops = grid_size.x * grid_size.y * grid_size.z * TILE_OUT1 * TILE_OUT1 * TILE_OUT1 * 25;
    double flops_per_tile = TILE_OUT1 * TILE_OUT1 * TILE_OUT1 * 25;
    double gflops = (flops / avg_s) / 1e9;
    double intensity = flops / bytes;

    std::cout << "Stencil 3D tiled kernel benchmark\n";
    std::cout << "  N              = " << N << "\n";
    std::cout << "  iters          = " << iters << "\n";
    std::cout << "  avg time       = " << avg_ms << " ms\n";
    std::cout << "  bandwidth      = " << bw_GBps << " GB/s\n";
    std::cout << "  GFLOP/s        = " << gflops << "\n";
    std::cout << "  FLOPPerTile/s  = " << flops_per_tile << "\n";
    std::cout << "  arithmetic I   = " << intensity << " FLOP/B\n";
}


void benchmark_stencil3d_tiled_thread_coarsening_kernel(
    float* d_out,
    const float* d_in,
    unsigned int width,
    unsigned int height,
    unsigned int channels,
    int iters)
{
    dim3 block_size(TILE_IN2, TILE_IN2, 1);
    dim3 grid_size(cdiv(width, TILE_OUT2), cdiv(height, TILE_OUT2), cdiv(channels, TILE_OUT2));

    // Warm-up (not timed)
    for (int i = 0; i < 5; ++i) {
        stencil3d_tiled_thread_coarsening_kernel << <grid_size, block_size >> > (d_out, d_in, width, height, channels);
    }
    CHECK_CUDA_STATUS(cudaDeviceSynchronize());

    // Create events
    cudaEvent_t start, stop;
    CHECK_CUDA_STATUS(cudaEventCreate(&start));
    CHECK_CUDA_STATUS(cudaEventCreate(&stop));

    // Record start
    CHECK_CUDA_STATUS(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        stencil3d_tiled_thread_coarsening_kernel << <grid_size, block_size >> > (d_out, d_in, width, height, channels);
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

    unsigned int N = height * width * channels;
    // Bytes moved per kernel call
    double bytes = grid_size.x * grid_size.y * grid_size.z * (TILE_IN2 * TILE_IN2 * TILE_IN2 + TILE_OUT2 * TILE_OUT2 * TILE_OUT2) * sizeof(float);
    double gbytes = bytes / 1e9;

    double bw_GBps = gbytes / avg_s;
    double flops = grid_size.x * grid_size.y * grid_size.z * TILE_OUT2 * TILE_OUT2 * TILE_OUT2 * 25;
    double flops_per_tile = TILE_OUT2 * TILE_OUT2 * TILE_OUT2 * 25;
    double gflops = (flops / avg_s) / 1e9;
    double intensity = flops / bytes;

    std::cout << "Stencil 3D tiled thread coarsening kernel benchmark\n";
    std::cout << "  N              = " << N << "\n";
    std::cout << "  iters          = " << iters << "\n";
    std::cout << "  avg time       = " << avg_ms << " ms\n";
    std::cout << "  bandwidth      = " << bw_GBps << " GB/s\n";
    std::cout << "  GFLOP/s        = " << gflops << "\n";
    std::cout << "  FLOPPerTile/s  = " << flops_per_tile << "\n";
    std::cout << "  arithmetic I   = " << intensity << " FLOP/B\n";
}

void benchmark_stencil3d_tiled_thread_coarsening_register_tiling_kernel(
    float* d_out,
    const float* d_in,
    unsigned int width,
    unsigned int height,
    unsigned int channels,
    int iters)
{
    dim3 block_size(TILE_IN2, TILE_IN2, 1);
    dim3 grid_size(cdiv(width, TILE_OUT2), cdiv(height, TILE_OUT2), cdiv(channels, TILE_OUT2));

    // Warm-up (not timed)
    for (int i = 0; i < 5; ++i) {
        stencil3d_tiled_thread_coarsening_register_tiling_kernel << <grid_size, block_size >> > (d_out, d_in, width, height, channels);
    }
    CHECK_CUDA_STATUS(cudaDeviceSynchronize());

    // Create events
    cudaEvent_t start, stop;
    CHECK_CUDA_STATUS(cudaEventCreate(&start));
    CHECK_CUDA_STATUS(cudaEventCreate(&stop));

    // Record start
    CHECK_CUDA_STATUS(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        stencil3d_tiled_thread_coarsening_register_tiling_kernel << <grid_size, block_size >> > (d_out, d_in, width, height, channels);
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

    unsigned int N = height * width * channels;
    // Bytes moved per kernel call
    double bytes = grid_size.x * grid_size.y * grid_size.z * (TILE_IN2 * TILE_IN2 * TILE_IN2 + TILE_OUT2 * TILE_OUT2 * TILE_OUT2) * sizeof(float);
    double gbytes = bytes / 1e9;

    double bw_GBps = gbytes / avg_s;
    double flops = grid_size.x * grid_size.y * grid_size.z * TILE_OUT2 * TILE_OUT2 * TILE_OUT2 * 25;
    double flops_per_tile = TILE_OUT2 * TILE_OUT2 * TILE_OUT2 * 25;
    double gflops = (flops / avg_s) / 1e9;
    double intensity = flops / bytes;

    std::cout << "Stencil 3D tiled thread coarsening register tiling kernel benchmark\n";
    std::cout << "  N              = " << N << "\n";
    std::cout << "  iters          = " << iters << "\n";
    std::cout << "  avg time       = " << avg_ms << " ms\n";
    std::cout << "  bandwidth      = " << bw_GBps << " GB/s\n";
    std::cout << "  GFLOP/s        = " << gflops << "\n";
    std::cout << "  FLOPPerTile/s  = " << flops_per_tile << "\n";
    std::cout << "  arithmetic I   = " << intensity << " FLOP/B\n";
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


    //test_ref_code();

    unsigned int width = 1000;
    unsigned int height = 100;
    unsigned int channels = 100;
    unsigned int order = 2;

    unsigned int N = width * height * channels;

    std::unique_ptr<float[]> h_in(new float[N]);

    for (unsigned int z = 0; z < channels; z++)
    {
        for (unsigned int y = 0; y < height; y++)
        {
            for (unsigned int x = 0; x < width; x++)
            {
                h_in[z * width * height + y * width + x] = (float)(x + 1);
            }
        }
    }

    std::unique_ptr<float[]> h_out(new float[N]);
    std::unique_ptr<float[]> out_ref(new float[N]);

    stencil3d_cpu(out_ref.get(), h_in.get(), width, height, channels);

    //double t = benchmark_stencil3d_cpu(h_out.get(), h_in.get(), width, height, channels, 100);
    //std::cout << "CPU kernel took: " << t << "ms\n";

    float* d_in = nullptr;
    float* d_out = nullptr;

    size_t size = N * sizeof(float);
    size_t filter_size = (2.0 * order + 1) * sizeof(float);

    CHECK_CUDA_STATUS(cudaMalloc(&d_in, size));
    CHECK_CUDA_STATUS(cudaMalloc(&d_out, size));

    cudaEvent_t start, stop;
    CHECK_CUDA_STATUS(cudaEventCreate(&start));
    CHECK_CUDA_STATUS(cudaEventCreate(&stop));

    CHECK_CUDA_STATUS(cudaEventRecord(start));
    CHECK_CUDA_STATUS(cudaMemcpy(d_in, h_in.get(), size, cudaMemcpyHostToDevice));

    CHECK_CUDA_STATUS(cudaEventRecord(stop));
    CHECK_CUDA_STATUS(cudaEventSynchronize(stop));

    float h2d_ms = 0.0f;
    CHECK_CUDA_STATUS(cudaEventElapsedTime(&h2d_ms, start, stop));

    /*dim3 block_size(8, 8, 8);
    dim3 grid_size(cdiv(width, 8), cdiv(height, 8), cdiv(channels, 8));
    stencil3d_kernel << <grid_size, block_size >> > (d_out, d_in, width, height, channels);*/

    /*dim3 block_size(TILE_IN1, TILE_IN1, TILE_IN1);
    dim3 grid_size(cdiv(width, TILE_OUT1), cdiv(height, TILE_OUT1), cdiv(channels, TILE_OUT1));
    stencil3d_tiled_kernel << <grid_size, block_size >> > (d_out, d_in, width, height, channels);*/

    /*dim3 block_size(TILE_IN2, TILE_IN2, 1);
    dim3 grid_size(cdiv(width, TILE_OUT2), cdiv(height, TILE_OUT2), cdiv(channels, TILE_OUT2));
    stencil3d_tiled_thread_coarsening_kernel << <grid_size, block_size >> > (d_out, d_in, width, height, channels);*/

    dim3 block_size(TILE_IN2, TILE_IN2, 1);
    dim3 grid_size(cdiv(width, TILE_OUT2), cdiv(height, TILE_OUT2), cdiv(channels, TILE_OUT2));
    stencil3d_tiled_thread_coarsening_register_tiling_kernel << <grid_size, block_size >> > (d_out, d_in, width, height, channels);

    CHECK_CUDA_STATUS(cudaEventRecord(start));
    CHECK_CUDA_STATUS(cudaMemcpy(h_out.get(), d_out, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA_STATUS(cudaEventRecord(stop));
    CHECK_CUDA_STATUS(cudaEventSynchronize(stop));

    float d2h_ms = 0.0f;
    CHECK_CUDA_STATUS(cudaEventElapsedTime(&d2h_ms, start, stop));

    // Average kernel time
    double avg_ms = h2d_ms + d2h_ms;
    double avg_s = avg_ms * 1e-3;

    double bw_GBps = (2 * size + 3 * filter_size) / 1e9 / avg_s;

    std::cout << "Stencil 3D kernel data transfer\n";
    std::cout << "  Total          = " << h2d_ms + d2h_ms << "ms\n";
    std::cout << "  HostToDevice   = " << h2d_ms << "ms\n";
    std::cout << "  Device2Host    = " << d2h_ms << "ms\n";
    std::cout << "  bandwidth      = " << bw_GBps << " GB/s\n";

    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;

    for (int i = 0; i < N; ++i) {
        float ref = out_ref[i];
        float got = h_out[i];
        float absd = std::fabs(ref - got);
        float reld = (std::fabs(ref) > 1e-6f) ? absd / std::fabs(ref) : absd;

        max_abs_diff = std::max(max_abs_diff, absd);
        max_rel_diff = std::max(max_rel_diff, reld);

        // Optional debug: print first mismatch
        if (absd > 1e-3f) { std::cout << "mismatch at " << i << ": ref=" << ref << " got=" << got << "\n"; break; }
    }

    std::cout << "Max abs diff = " << max_abs_diff << "\n";
    std::cout << "Max rel diff = " << max_rel_diff << "\n";

    //benchmark_stencil3d_kernel(d_out, d_in, width, height, channels, 8, 100);
    //benchmark_stencil3d_tiled_kernel(d_out, d_in, width, height, channels, 100);
    //benchmark_stencil3d_tiled_thread_coarsening_kernel(d_out, d_in, width, height, channels, 100);
    benchmark_stencil3d_tiled_thread_coarsening_register_tiling_kernel(d_out, d_in, width, height, channels, 100);

    CHECK_CUDA_STATUS(cudaFree(d_in));
    CHECK_CUDA_STATUS(cudaFree(d_out));

    CHECK_CUDA_STATUS(cudaEventDestroy(start));
    CHECK_CUDA_STATUS(cudaEventDestroy(stop));

    CHECK_CUDA_STATUS(cudaDeviceReset());

    return 0;
}
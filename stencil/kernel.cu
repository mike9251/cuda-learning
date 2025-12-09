
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


using Clock = std::chrono::steady_clock;
using ms = std::chrono::duration<double, std::milli>;


#define CHECK_CUDA_STATUS(status) { if (status != cudaSuccess) { \
    fprintf(stderr, "CUDA Error: %s (error code %d) at %s:%d\n", \
        cudaGetErrorString(status), status, __FILE__, __LINE__); exit(EXIT_FAILURE); } }


void stencil1d_cpu(float* out, const float* in, const float* filter, unsigned int order, unsigned int length)
{
    for (unsigned int x = 0; x < length; x++)
    {
        float temp = 0.f;

        // to preserve boundary conditions
        if (x >= order && x < length - order)
        {
            for (unsigned int fx = 0; fx < 2 * order + 1; fx++)
            {
                temp += filter[fx] * in[x - order + fx];
            }
        }
        else
        {
            temp = in[x];
        }

        out[x] = temp;
    }
}

void stencil2d_cpu(
    float* out, 
    const float* in, 
    const float* filter_w, 
    const float* filter_h, 
    unsigned int order,
    unsigned int width, 
    unsigned int height)
{
    for (unsigned int row = 0; row < height; row++)
    {
        for (unsigned int col = 0; col < width; col++)
        {
            float temp = 0.f;

            // to preserve boundary conditions
            if (col >= order && col < width - order && row >= order && row < height - order)
            {
                for (unsigned int i = 0; i < 2 * order + 1; i++)
                {
                    temp += filter_w[i] * in[row * width + col - order + i] + \
                        filter_h[i] * in[(row - order + i) * width + col];
                }
            }
            else
            {
                temp = in[row * width + col];
            }

            out[row * width + col] = temp;
        }
    }
}

void stencil3d_cpu(
    float* out, 
    const float* in, 
    const float* filter_w, 
    const float* filter_h, 
    const float* filter_z, 
    unsigned int order, 
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
                float temp = 0.f;

                // to preserve boundary conditions
                if (ch >= order && ch < depth - order && row >= order && row < height - order && col >= order && col < width - order)
                {
                    for (unsigned int i = 0; i < 2 * order + 1; i++)
                    {

                        temp += filter_w[i] * in[ch * height * width + row * width + col - order + i] + \
                            filter_h[i] * in[ch * height * width + (row - order + i) * width + col] + \
                            filter_z[i] * in[(ch - order + i) * height * width + row * width + col];
                    }
                }
                else
                {
                    temp = in[ch * height * width + row * width + col];
                }

                out[ch * height * width + row * width + col] = temp;
            }
        }
    }
}


void test_ref_code()
{
    {
        std::cout << "--------------- Stencil 1D --------------\n";
        unsigned int N = 10;
        unsigned int order = 2;

        std::unique_ptr<float[]> h_in(new float[N]);

        for (unsigned int i = 0; i < N; i++)
        {
            h_in[i] = (float)(i + 1);
        }

        float filter[5] = { 0.5f, 1.f, 1.f, 1.f, 0.5f };

        std::unique_ptr<float[]> h_out(new float[N]);

        stencil1d_cpu(h_out.get(), h_in.get(), filter, 2, N);

        std::cout << "Input: " << std::endl;
        for (unsigned int i = 0; i < N; i++)
        {
            std::cout << h_in[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Output: " << std::endl;

        for (unsigned int i = 0; i < N; i++)
        {
            std::cout << h_out[i] << " ";
        }
        std::cout << std::endl;
    }

    {
        std::cout << "--------------- Stencil 2D --------------\n";
        unsigned int N = 10;
        unsigned int order = 2;

        float* h_in = (float*)malloc(N * N * sizeof(float));

        for (unsigned int y = 0; y < N; y++)
        {
            for (unsigned int x = 0; x < N; x++)
            {
                h_in[y * N + x] = (float)(x + 1);
            }
        }

        float filter_w[5] = { 0.5f, 1.f, 1.f, 1.f, 0.5f };
        float filter_h[5] = { 0.5f, 1.f, 0.f, 1.f, 0.5f };

        float* h_out = (float*)malloc(N * N * sizeof(float));

        stencil2d_cpu(h_out, h_in, filter_w, filter_h, order, N, N);

        std::cout << "Input: " << std::endl;
        for (unsigned int y = 0; y < N; y++)
        {
            for (unsigned int x = 0; x < N; x++)
            {
                std::cout << h_in[y * N + x] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Output: " << std::endl;

        for (unsigned int y = 0; y < N; y++)
        {
            for (unsigned int x = 0; x < N; x++)
            {
                std::cout << h_out[y * N + x] << " ";
            }
            std::cout << std::endl;
        }

        free(h_in);
        free(h_out);
    }

    {
        std::cout << "--------------- Stencil 3D --------------\n";
        unsigned int N = 10;
        unsigned int order = 1;

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

        float filter_w[3] = { 1.f, 1.f, 1.f };
        float filter_h[3] = { 1.f, 0.f, 1.f };
        float filter_z[3] = { 1.f, 0.f, 1.f };

        float* h_out = (float*)malloc(N * N * N * sizeof(float));

        stencil3d_cpu(h_out, h_in, filter_w, filter_h, filter_z, order, N, N, N);

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
}

double benchmark_stencil3d_cpu(
    float* out, 
    const float* in, 
    const float* filter_w, 
    const float* filter_h,
    const float* filter_z,
    unsigned int order,
    unsigned int width, 
    unsigned int height, 
    unsigned int channels, 
    unsigned int iters)
{
    // warmup: make sure caches etc. are "hot"
    stencil3d_cpu(out, in, filter_w, filter_h, filter_z, order, width, height, channels);

    auto start = Clock::now();
    for (int i = 0; i < iters; ++i) {
        stencil3d_cpu(out, in, filter_w, filter_h, filter_z, order, width, height, channels);
    }
    auto end = Clock::now();

    float checksum = 0.0f;
    for (int i = 0; i < width * height * channels; ++i) checksum += out[i];
    std::cout << "checksum = " << checksum << "\n";

    ms total = end - start;
    double avg_ms = total.count() / iters;
    return avg_ms;
}

__global__ void stencil3d_kernel(
    float* out,
    const float* in,
    const float* filter_w,
    const float* filter_h,
    const float* filter_z,
    unsigned int order,
    unsigned int width,
    unsigned int height,
    unsigned int channels)
{
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int tz = blockIdx.z * blockDim.z + threadIdx.z;

    unsigned int depth_stride = height * width;

    float temp = 0.f;
    if (tx >= order && tx < width - order && ty >= order && ty < height - order && tz >= order && tz < channels - order)
    {
        for (unsigned int i = 0; i < 2 * order + 1; i++)
        {
            temp += filter_w[i] * in[tz * depth_stride + ty * width + tx - order + i] + \
                filter_h[i] * in[tz * depth_stride + (ty - order + i) * width + tx] + \
                filter_z[i] * in[(tz - order + i) * depth_stride + ty * width + tx];
        }
    }
    else
    {
        temp = in[tz * depth_stride + ty * width + tx];
    }

    if (tz < channels && ty < height && tx < width)
        out[tz * depth_stride + ty * width + tx] = temp;
}


#define STENCIL_ORDER 5
#define TILE 8
#define SH 2 * STENCIL_ORDER + TILE

__global__ void stencil3d_tiled_kernel(
    float* out,
    const float* in,
    const float* filter_w,
    const float* filter_h,
    const float* filter_z,
    int order,
    unsigned int width,
    unsigned int height,
    unsigned int channels)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int ch = blockIdx.z * blockDim.z + threadIdx.z;

    unsigned int bx = blockIdx.x * blockDim.x;
    unsigned int by = blockIdx.y * blockDim.y;
    unsigned int bz = blockIdx.z * blockDim.z;

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int tz = threadIdx.z;

    unsigned int depth_stride = height * width;

    __shared__ float smem[SH][SH][SH];

    for (unsigned int sz = tz; sz < SH; sz += TILE)
    {
        for (unsigned int sy = ty; sy < SH; sy += TILE)
        {
            for (unsigned int sx = tx; sx < SH; sx += TILE)
            {
                unsigned int gz = bz + sz - order;
                unsigned int gy = by + sy - order;
                unsigned int gx = bx + sx - order;

                if (gz >= 0 && gz < channels && gy >= 0 && gy < height && gx >= 0 && gx < width)
                {
                    smem[sz][sy][sx] = in[gz * depth_stride + gy * width + gx];
                }
            }
        }
    }

    __syncthreads();

    float temp = 0.f;
    if (col >= order && col < width - order && row >= order && row < height - order && ch >= order && ch < channels - order)
    {

        unsigned int sx = tx + order;
        unsigned int sy = ty + order;
        unsigned int sz = tz + order;

        for (int i = -order; i <= order; i++)
        {
            unsigned int k = i + order;
            temp += filter_w[k] * smem[sz][sy][sx + i] + \
                filter_h[k] * smem[sz][sy + i][sx] + \
                filter_z[k] * smem[sz + i][sy][sx];
        }
    }
    else
    {
        temp = in[ch * depth_stride + row * width + col];
    }

    if (ch < channels && row < height && col < width)
        out[ch * depth_stride + row * width + col] = temp;
}


unsigned int cdiv(unsigned int n, unsigned int x)
{
    return (n + x - 1) / x;
}

void benchmark_stencil3d_kernel(
    float* d_out, 
    const float* d_in, 
    const float* d_filter_w,
    const float* d_filter_h,
    const float* d_filter_z,
    unsigned int order, 
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
        stencil3d_kernel << <grid_size, block_size >> > (d_out, d_in, d_filter_w, d_filter_h, d_filter_z, order, width, height, channels);
    }
    CHECK_CUDA_STATUS(cudaDeviceSynchronize());

    // Create events
    cudaEvent_t start, stop;
    CHECK_CUDA_STATUS(cudaEventCreate(&start));
    CHECK_CUDA_STATUS(cudaEventCreate(&stop));

    // Record start
    CHECK_CUDA_STATUS(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        stencil3d_kernel << <grid_size, block_size >> > (d_out, d_in, d_filter_w, d_filter_h, d_filter_z, order, width, height, channels);
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
    double bytes = 0.05 * N * (2.0 * 3.0 * (2.0 * order + 1) + 1) * sizeof(float);
    double gbytes = bytes / 1e9;

    double bw_GBps = gbytes / avg_s;                // achieved bandwidth
    double flops = N * 5.0 * (2.0 * order + 1);     // (2 * order + 1)*5 FLOP per element
    double gflops = (flops / avg_s) / 1e9;          // FLOP/s → GFLOP/s
    double intensity = flops / bytes;               // FLOP/byte

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
    const float* d_filter_w,
    const float* d_filter_h,
    const float* d_filter_z,
    unsigned int order,
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
        stencil3d_tiled_kernel << <grid_size, block_size >> > (d_out, d_in, d_filter_w, d_filter_h, d_filter_z, order, width, height, channels);
    }
    CHECK_CUDA_STATUS(cudaDeviceSynchronize());

    // Create events
    cudaEvent_t start, stop;
    CHECK_CUDA_STATUS(cudaEventCreate(&start));
    CHECK_CUDA_STATUS(cudaEventCreate(&stop));

    // Record start
    CHECK_CUDA_STATUS(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        stencil3d_tiled_kernel << <grid_size, block_size >> > (d_out, d_in, d_filter_w, d_filter_h, d_filter_z, order, width, height, channels);
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
    double bytes = 0.05 * N * (2.0 * 3.0 * (2.0 * order + 1) + 1) * sizeof(float);
    double gbytes = bytes / 1e9;

    double bw_GBps = gbytes / avg_s;                // achieved bandwidth
    double flops = N * 5.0 * (2.0 * order + 1);     // (2 * order + 1)*5 FLOP per element
    double gflops = (flops / avg_s) / 1e9;          // FLOP/s → GFLOP/s
    double intensity = flops / bytes;               // FLOP/byte

    std::cout << "Stencil 3D tiled kernel benchmark\n";
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
    //test_ref_code();

    unsigned int width = 1000;
    unsigned int height = 100;
    unsigned int channels = 100;
    unsigned int order = 5;

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

    //float filter_w[5] = { 0.5f, 1.f, 1.f, 1.f, 0.5f };
    //float filter_h[5] = { 0.5f, 1.f, 0.f, 1.f, 0.5f };
    //float filter_z[5] = { 0.5f, 1.f, 0.f, 1.f, 0.5f };

    float filter_w[11] = { 0.5f, 1.f, 1.f, 1.f, 0.5f, 1.f, 0.5f, 1.f, 1.f, 1.f, 0.5f };
    float filter_h[11] = { 0.5f, 1.f, 0.f, 1.f, 0.5f, 0.f, 0.5f, 1.f, 1.f, 1.f, 0.5f };
    float filter_z[11] = { 0.5f, 1.f, 0.f, 1.f, 0.5f, 0.f, 0.5f, 1.f, 1.f, 1.f, 0.5f };

    std::unique_ptr<float[]> h_out(new float[N]);
    std::unique_ptr<float[]> out_ref(new float[N]);

    stencil3d_cpu(out_ref.get(), h_in.get(), filter_w, filter_h, filter_z, order, width, height, channels);

    //double t = benchmark_stencil3d_cpu(h_out.get(), h_in.get(), filter_w, filter_h, filter_z, order, width, height, channels, 100);
    //std::cout << "CPU kernel took: " << t << "ms\n";

    float* d_in = nullptr;
    float* d_out = nullptr;
    float* d_filter_w = nullptr;
    float* d_filter_h = nullptr;
    float* d_filter_z = nullptr;

    size_t size = N * sizeof(float);
    size_t filter_size = (2.0 * order + 1) * sizeof(float);

    CHECK_CUDA_STATUS(cudaMalloc(&d_in, size));
    CHECK_CUDA_STATUS(cudaMalloc(&d_out, size));
    CHECK_CUDA_STATUS(cudaMalloc(&d_filter_w, filter_size));
    CHECK_CUDA_STATUS(cudaMalloc(&d_filter_h, filter_size));
    CHECK_CUDA_STATUS(cudaMalloc(&d_filter_z, filter_size));
    
    cudaEvent_t start, stop;
    CHECK_CUDA_STATUS(cudaEventCreate(&start));
    CHECK_CUDA_STATUS(cudaEventCreate(&stop));

    CHECK_CUDA_STATUS(cudaEventRecord(start));
    CHECK_CUDA_STATUS(cudaMemcpy(d_in, h_in.get(), size, cudaMemcpyHostToDevice));
    CHECK_CUDA_STATUS(cudaMemcpy(d_filter_w, filter_w, filter_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_STATUS(cudaMemcpy(d_filter_h, filter_h, filter_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_STATUS(cudaMemcpy(d_filter_z, filter_z, filter_size, cudaMemcpyHostToDevice));

    CHECK_CUDA_STATUS(cudaEventRecord(stop));
    CHECK_CUDA_STATUS(cudaEventSynchronize(stop));

    float h2d_ms = 0.0f;
    CHECK_CUDA_STATUS(cudaEventElapsedTime(&h2d_ms, start, stop));

    dim3 block_size(8, 8, 8);
    dim3 grid_size(cdiv(width, block_size.x), cdiv(height, block_size.y), cdiv(channels, block_size.z));
    stencil3d_tiled_kernel <<<grid_size, block_size >>>(d_out, d_in, d_filter_w, d_filter_h, d_filter_z, order, width, height, channels);

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

    benchmark_stencil3d_kernel(d_out, d_in, d_filter_w, d_filter_h, d_filter_z, order, width, height, channels, 8, 100);
    benchmark_stencil3d_tiled_kernel(d_out, d_in, d_filter_w, d_filter_h, d_filter_z, order, width, height, channels, 8, 100);

    CHECK_CUDA_STATUS(cudaFree(d_in));
    CHECK_CUDA_STATUS(cudaFree(d_out));
    CHECK_CUDA_STATUS(cudaFree(d_filter_w));
    CHECK_CUDA_STATUS(cudaFree(d_filter_h));
    CHECK_CUDA_STATUS(cudaFree(d_filter_z));

    CHECK_CUDA_STATUS(cudaEventDestroy(start));
    CHECK_CUDA_STATUS(cudaEventDestroy(stop));

    CHECK_CUDA_STATUS(cudaDeviceReset());

    return 0;
}
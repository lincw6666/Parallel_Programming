#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 400
#define GROUP_SIZE 4

__global__ void mandelKernel(
    int *d_out, size_t pitch,
    float lowerX, float lowerY,
    float stepX, float stepY,
    int maxIters
) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int i = blockIdx.x * GROUP_SIZE;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    for (int _i = i; _i < i+GROUP_SIZE; ++_i) {
        float c_re = lowerX + _i*stepX;
        float c_im = lowerY + j*stepY;
        float z_re = c_re;
        float z_im = c_im;

        int iter;

        for (iter = 0; iter < maxIters; ++iter) {
            if (z_re*z_re + z_im*z_im > 4.f)
                break;

            float new_re = z_re*z_re - z_im*z_im;
            float new_im = 2.f*z_re*z_im;
            z_re = c_re + new_re;
            z_im = c_im + new_im;
        }

        *((int*)((char*)d_out + j * pitch) + _i) = iter;
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int *h_out, *d_out; // Mandelbort result on host & device
    int size = resX * resY * sizeof(int);

    // Allocate memory on host & device. 
    size_t pitch;
    cudaHostAlloc((void **)&h_out, size, cudaHostAllocDefault);
    cudaMallocPitch((void **)&d_out, &pitch, resX*sizeof(int), resY);

    // CUDA kernel function.
    dim3 block_size(1, BLOCK_SIZE);
    dim3 num_block(resX / GROUP_SIZE, resY / BLOCK_SIZE);
    mandelKernel<<<num_block, block_size>>>(d_out, pitch,
                                            lowerX, lowerY,
                                            stepX, stepY,
                                            maxIterations);
    // Wait for all CUDA threads to finish.
    cudaDeviceSynchronize();

    // Store result from device to host.
    cudaMemcpy2D(h_out, resX*sizeof(int), d_out, pitch,
                 resX*sizeof(int), resY, cudaMemcpyDeviceToHost);
    memcpy(img, h_out, size);

    // Free memory.
    cudaFreeHost(h_out);
    cudaFree(d_out);
}

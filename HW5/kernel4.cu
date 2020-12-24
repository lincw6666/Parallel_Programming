#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 25

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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    float c_re = lowerX + i*stepX;
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

    *((int*)((char*)d_out + j * pitch) + i) = iter;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int *d_out; // Mandelbort result on device

    // Allocate memory on host & device. 
    size_t pitch;
    cudaMallocPitch((void **)&d_out, &pitch, resX*sizeof(int), resY);

    // CUDA kernel function.
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 num_block(resX / BLOCK_SIZE, resY / BLOCK_SIZE);
    mandelKernel<<<num_block, block_size>>>(d_out, pitch,
                                            lowerX, lowerY,
                                            stepX, stepY,
                                            maxIterations);
    // Wait for all CUDA threads to finish.
    cudaDeviceSynchronize();

    // Store result from device to host.
    cudaMemcpy2D(img, resX*sizeof(int), d_out, pitch,
                 resX*sizeof(int), resY, cudaMemcpyDeviceToHost);

    // Free memory.
    cudaFree(d_out);
}

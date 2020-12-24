#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

__global__ void mandelKernel(
    int *d_out,
    float lowerX, float lowerY,
    float stepX, float stepY,
    int resX, int maxIters
) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int index = j*resX + i;

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

    d_out[index] = iter;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int *h_out, *d_out; // Mandelbort result on host & device
    int size = resX * resY * sizeof(int);

    // Allocate memory on host & device. 
    h_out = (int *)malloc(size);
    cudaMalloc((void **)&d_out, size);

    // CUDA kernel function.
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 num_block(resX / BLOCK_SIZE, resY / BLOCK_SIZE);
    mandelKernel<<<num_block, block_size>>>(d_out,
                                            lowerX, lowerY,
                                            stepX, stepY,
                                            resX, maxIterations);
    // Wait for all CUDA threads to finish.
    cudaDeviceSynchronize();

    // Store result from device to host.
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    memcpy(img, h_out, size);

    // Free memory.
    free(h_out);
    cudaFree(d_out);
}

#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

int test = 0;

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;
    int imageSize = imageHeight * imageWidth;

    // Create a command queue.
    cl_command_queue command_queue = clCreateCommandQueue(*context, *device,
                                                          0, &status);

    // Create memory buffers on the device.
    cl_mem img_height_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY,
                                           sizeof(int), NULL, &status);
    cl_mem img_width_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY,
                                          sizeof(int), NULL, &status);
    cl_mem filter_width_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY,
                                             sizeof(int), NULL, &status);
    cl_mem filter_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY,
                                       filterSize * sizeof(float),
                                       NULL, &status);
    cl_mem in_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY,
                                   imageSize * sizeof(float),
                                   NULL, &status);
    cl_mem out_mem = clCreateBuffer(*context, CL_MEM_WRITE_ONLY,
                                    imageSize * sizeof(float),
                                    NULL, &status);

    // Copy the lists A and B to their respective memory buffers
    status = clEnqueueWriteBuffer(command_queue, img_height_mem, CL_TRUE, 0,
                                  sizeof(int), &imageHeight, 0, NULL, NULL);
    printf("%d ", status);
    status = clEnqueueWriteBuffer(command_queue, img_width_mem, CL_TRUE, 0,
                                  sizeof(int), &imageWidth, 0, NULL, NULL);
    printf("%d ", status);
    status = clEnqueueWriteBuffer(command_queue, filter_width_mem, CL_TRUE, 0,
                                  sizeof(int), &filterWidth, 0, NULL, NULL);
    printf("%d ", status);
    status = clEnqueueWriteBuffer(command_queue, filter_mem, CL_TRUE, 0,
                                  filterSize * sizeof(float), filter,
                                  0, NULL, NULL);
    printf("%d ", status);
    status = clEnqueueWriteBuffer(command_queue, in_mem, CL_TRUE, 0, 
                                  imageSize * sizeof(float), inputImage,
                                  0, NULL, NULL);
    printf("%d ", status);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);

    // Set the arguments of the kernel
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &img_height_mem);
    printf("%d ", status);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &img_width_mem);
    printf("%d ", status);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &filter_width_mem);
    printf("%d ", status);
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&filter_mem);
    printf("%d ", status);
    status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&in_mem);
    printf("%d ", status);
    status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&out_mem);
    printf("%d ", status);

    // Execute the OpenCL kernel on the list
    size_t global_work_size[2] = {(size_t) imageWidth, (size_t) imageHeight};
    status = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
                                    global_work_size, 0, NULL, NULL, NULL);
    printf("%d ", status);

    // Read the memory buffer C on the device to the local variable C
    status = clEnqueueReadBuffer(command_queue, out_mem, CL_TRUE, 0, 
                                 imageSize * sizeof(float), outputImage,
                                 0, NULL, NULL);
    printf("%d ", status);

    // Clean up
    status = clFinish(command_queue);
    printf("%d\n", status);
    status = clReleaseKernel(kernel);
    status = clReleaseMemObject(img_height_mem);
    status = clReleaseMemObject(img_width_mem);
    status = clReleaseMemObject(filter_width_mem);
    status = clReleaseMemObject(filter_mem);
    status = clReleaseMemObject(in_mem);
    status = clReleaseMemObject(out_mem);
    status = clReleaseCommandQueue(command_queue);
}
